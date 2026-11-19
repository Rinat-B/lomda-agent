"""Stage 4 checkers: find problems in a written lesson before it is saved.

Two kinds of checks, all returning Finding objects:
- code checks (fast, no model): answer length, decision-diagram structure, addressing the
  learner consistently, repeated lists and phrases, questions that test the same decision;
- model checks (role "checker" in models.yaml, run in parallel): Hebrew language, grounding
  in the source, question validity (more than one correct answer, inconsistent scenario,
  recall instead of decision, weak distractors) and pedagogy (repetition, enrichment used
  for core rules, illustrations that add nothing, diagram logic).

The model writes the problem and suggestion texts in Hebrew; they are lesson-review data
for the content owner, not CLI strings.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, field_validator

from agent.guards import SOURCE_DATA_RULE, chunk_text_for_prompt, wrap_sources
from agent.llm import LONG_REQUEST_TIMEOUT_SECONDS, LLMResult, call_llm
from agent.tools import normalize
from agent.writer import chunk_matches

WORDS_PATH = Path(__file__).parent / "assets" / "checker_words_he.json"

CATEGORIES = (
    "language",
    "duplicate_text",
    "address",
    "ungrounded",
    "multiple_correct",
    "inconsistent_scenario",
    "recall_question",
    "distractors",
    "answer_length",
    "diagram_logic",
    "repetition",
    "question_overlap",
    "enrichment_misuse",
    "illustration_value",
    "other",
)

# answer length: the correct option is this much longer than the longest / the average distractor
LONGEST_RATIO = 1.4
MEAN_RATIO = 1.8
# repetition: share of a list's items that appear in another screen's list
LIST_OVERLAP = 0.5
# questions on different screens with this similar a correct answer test the same decision
ANSWER_OVERLAP = 0.7
PHRASE_WORDS = 3
MODEL_ATTEMPTS = 2
QUESTION_BATCH_SCREENS = 3  # screens per question-checklist call: keeps each reply short enough
CHECK_TIMEOUT_SECONDS = 240  # like the writer: free models think for a long time
SOURCE_BUDGET_CHARS = 20_000

_WORD = re.compile(r"[\w׳״'\"-]+")
_VAV = "ו"


class Finding(BaseModel):
    check: str
    category: str
    screen: int | None = None  # 1-based; None = the whole lesson
    where: str = ""
    evidence: str = ""
    problem: str = ""
    suggestion: str = ""

    @field_validator("category")
    @classmethod
    def _known(cls, v: str) -> str:
        v = (v or "").strip().lower()
        return v if v in CATEGORIES else "other"


@dataclass
class CheckReport:
    findings: list[Finding] = field(default_factory=list)
    calls: list[LLMResult] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)  # check name -> error


# ---- walking the lesson -------------------------------------------------------------------
def questions(lesson: dict):
    """(screen number, block number, label, question dict) for every question and alternate."""
    for s_no, screen in enumerate(lesson["screens"], 1):
        for b_no, block in enumerate(screen["blocks"], 1):
            if block["type"] == "multiple_choice":
                yield s_no, b_no, "main", block
                if block.get("alternate"):
                    yield s_no, b_no, "alternate", block["alternate"]


def block_texts(block: dict) -> list[str]:
    """Everything the learner reads in a block."""
    t = block["type"]
    if t in ("paragraph", "did_you_know"):
        return [block["text"]]
    if t == "list":
        return list(block["items"])
    if t == "stop_and_think":
        return [block["prediction_question"], block["explanation"]]
    if t == "multiple_choice":
        out = []
        for q in (block, block.get("alternate")):
            if q:
                out += [q["question"], *q["options"], q["feedback_correct"], q["feedback_incorrect"]]
        return out
    if t == "illustration":
        return [block["title"], block["description"]]
    if t == "decision_diagram":
        return [block["title"], *(n["label"] for n in block["nodes"]),
                *(e.get("label") or "" for e in block["edges"])]
    return []


def words(text: str) -> list[str]:
    return [w.strip("-'\"") for w in _WORD.findall(normalize(text)) if w.strip("-'\"")]


def _where(b_no: int, block_type: str, part: str = "") -> str:
    return f"block {b_no} ({block_type})" + (f", {part}" if part else "")


# ---- code checks ------------------------------------------------------------------------------
def check_answer_length(lesson: dict) -> list[Finding]:
    """The correct option is clearly the longest: learners can guess it without knowing."""
    out = []
    for s_no, b_no, label, q in questions(lesson):
        lengths = [len(o) for o in q["options"]]
        correct = lengths[q["correct_index"]]
        others = [n for i, n in enumerate(lengths) if i != q["correct_index"]]
        mean = sum(others) / len(others)
        if correct >= LONGEST_RATIO * max(others) and correct >= MEAN_RATIO * mean:
            out.append(Finding(
                check="answer_length", category="answer_length", screen=s_no,
                where=_where(b_no, "multiple_choice", label), evidence=q["options"][q["correct_index"]],
                problem=f"correct option {correct} chars, distractors {min(others)}-{max(others)} chars "
                        f"(ratio {correct / max(others):.1f} to the longest)"))
    return out


def check_diagrams(lesson: dict) -> list[Finding]:
    """Structure of decision diagrams: real branches, no dead questions, everything reachable."""
    out = []
    for s_no, screen in enumerate(lesson["screens"], 1):
        for b_no, block in enumerate(screen["blocks"], 1):
            if block["type"] != "decision_diagram":
                continue
            where = _where(b_no, "decision_diagram")
            nodes = {n["id"]: n for n in block["nodes"]}
            outgoing: dict[str, list[dict]] = {nid: [] for nid in nodes}
            incoming = {nid: 0 for nid in nodes}
            for e in block["edges"]:
                if e["from"] in outgoing and e["to"] in incoming:
                    outgoing[e["from"]].append(e)
                    incoming[e["to"]] += 1
            for nid, node in nodes.items():
                edges = outgoing[nid]
                if node["kind"] == "question" and len(edges) < 2:
                    out.append(Finding(
                        check="diagrams", category="diagram_logic", screen=s_no, where=where,
                        evidence=f"{nid}: {node['label']}",
                        problem=f"question node {nid} has {len(edges)} outgoing branch(es); "
                                "a decision needs at least two (e.g. yes and no)"))
                elif node["kind"] == "question" and len({e["to"] for e in edges}) == 1:
                    out.append(Finding(
                        check="diagrams", category="diagram_logic", screen=s_no, where=where,
                        evidence=f"{nid}: {node['label']}",
                        problem=f"every branch of question node {nid} leads to the same node "
                                f"{edges[0]['to']}: the answer does not change the outcome"))
            roots = [nid for nid in nodes if incoming[nid] == 0]
            for nid in roots[1:]:
                out.append(Finding(
                    check="diagrams", category="diagram_logic", screen=s_no, where=where,
                    evidence=f"{nid}: {nodes[nid]['label']}",
                    problem=f"node {nid} is not reached from the start node {roots[0]}"))
    return out


def _address_words() -> tuple[set[str], set[str]]:
    data = json.loads(WORDS_PATH.read_text(encoding="utf-8"))
    return set(data["address_singular"]), set(data["address_plural"])


def _address_hits(text: str, vocabulary: set[str]) -> list[str]:
    hits = []
    for w in words(text):
        if w in vocabulary or (w.startswith(_VAV) and w[1:] in vocabulary):
            hits.append(w)
    return hits


def check_address(lesson: dict) -> list[Finding]:
    """The learner is addressed both in singular and in plural."""
    singular, plural = _address_words()
    per_screen = []
    total_s = total_p = 0
    for s_no, screen in enumerate(lesson["screens"], 1):
        text = " ".join(t for b in screen["blocks"] for t in block_texts(b))
        s_hits, p_hits = _address_hits(text, singular), _address_hits(text, plural)
        total_s, total_p = total_s + len(s_hits), total_p + len(p_hits)
        per_screen.append((s_no, s_hits, p_hits))
    if not total_s or not total_p:
        return []
    out = []
    for s_no, s_hits, p_hits in per_screen:
        if s_hits and p_hits:
            out.append(Finding(
                check="address", category="address", screen=s_no,
                evidence=f"singular: {', '.join(dict.fromkeys(s_hits))}; plural: {', '.join(dict.fromkeys(p_hits))}",
                problem="the learner is addressed in singular and in plural on the same screen"))
    out.append(Finding(
        check="address", category="address", screen=None,
        evidence=f"{total_s} singular forms, {total_p} plural forms",
        problem="the lesson mixes singular and plural address; pick one form"))
    return out


def _similarity(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def check_repetition(lesson: dict) -> list[Finding]:
    """Lists repeated on another screen, and phrases repeated inside one paragraph."""
    out = []
    lists = [(s_no, b_no, [" ".join(words(i)) for i in block["items"]])
             for s_no, screen in enumerate(lesson["screens"], 1)
             for b_no, block in enumerate(screen["blocks"], 1) if block["type"] == "list"]
    for i, (s1, b1, items1) in enumerate(lists):
        for s2, b2, items2 in lists[i + 1:]:
            if s1 == s2:
                continue
            shared = set(items1) & set(items2)
            if len(shared) >= LIST_OVERLAP * min(len(items1), len(items2)):
                out.append(Finding(
                    check="repetition", category="repetition", screen=s2, where=_where(b2, "list"),
                    evidence=f"{len(shared)} of {len(items2)} items also in screen {s1}, block {b1}",
                    problem=f"the list repeats screen {s1} instead of adding something new"))
    for s_no, screen in enumerate(lesson["screens"], 1):
        for b_no, block in enumerate(screen["blocks"], 1):
            if block["type"] != "paragraph":
                continue
            ws = [w for w in words(block["text"]) if not w.isdigit()]
            grams = [" ".join(ws[k:k + PHRASE_WORDS]) for k in range(len(ws) - PHRASE_WORDS + 1)]
            repeated = sorted({g for g in grams if grams.count(g) > 1})
            if repeated:
                out.append(Finding(
                    check="repetition", category="duplicate_text", screen=s_no,
                    where=_where(b_no, "paragraph"), evidence="; ".join(repeated),
                    problem="the same phrase appears twice in one paragraph"))
    return out


def check_question_overlap(lesson: dict) -> list[Finding]:
    """Questions on different screens whose correct answers are nearly the same."""
    out = []
    qs = [(s_no, b_no, label, words(q["options"][q["correct_index"]])) for s_no, b_no, label, q in questions(lesson)]
    for i, (s1, b1, l1, a1) in enumerate(qs):
        for s2, b2, l2, a2 in qs[i + 1:]:
            if s1 != s2 and _similarity(a1, a2) >= ANSWER_OVERLAP:
                out.append(Finding(
                    check="question_overlap", category="question_overlap", screen=s2,
                    where=_where(b2, "multiple_choice", l2),
                    evidence=f"same decision as screen {s1}, block {b1} ({l1})",
                    problem="two questions test the same decision with nearly the same correct answer"))
    return out


CODE_CHECKS: dict[str, Callable[[dict], list[Finding]]] = {
    "answer_length": check_answer_length,
    "diagrams": check_diagrams,
    "address": check_address,
    "repetition": check_repetition,
    "question_overlap": check_question_overlap,
}


# ---- model checks -----------------------------------------------------------------------------
def render_lesson(lesson: dict) -> str:
    """The lesson as labelled text for the checker prompts (no ids or source locators)."""
    lines = [f"Lesson title: {lesson['title']}"]
    for s_no, screen in enumerate(lesson["screens"], 1):
        lines.append(f"\n=== Screen {s_no}: {screen['title']}")
        for b_no, b in enumerate(screen["blocks"], 1):
            head = f"[screen {s_no}, block {b_no}, {b['type']}]"
            t = b["type"]
            if t in ("paragraph", "did_you_know"):
                lines.append(f"{head} {b['text']}")
            elif t == "list":
                lines.append(head)
                lines += [f"  - {item}" for item in b["items"]]
            elif t == "stop_and_think":
                lines.append(f"{head} Question: {b['prediction_question']}")
                lines.append(f"  Explanation: {b['explanation']}")
            elif t == "multiple_choice":
                for label, q in (("question", b), ("alternate question (shown on retry)", b.get("alternate"))):
                    if not q:
                        continue
                    lines.append(f"{head} {label}: {q['question']}")
                    for k, option in enumerate(q["options"]):
                        mark = " (marked correct)" if k == q["correct_index"] else ""
                        lines.append(f"  {chr(ord('A') + k)}. {option}{mark}")
                    lines.append(f"  Feedback when correct: {q['feedback_correct']}")
                    lines.append(f"  Feedback when wrong: {q['feedback_incorrect']}")
            elif t == "illustration":
                lines.append(f"{head} Title: {b['title']}")
                lines.append(f"  Planned interactive illustration: {b['description']}")
            elif t == "decision_diagram":
                lines.append(f"{head} Title: {b['title']}")
                lines += [f"  node {n['id']} ({n['kind']}): {n['label']}" for n in b["nodes"]]
                lines += [f"  edge {e['from']} -> {e['to']}" + (f" [{e['label']}]" if e.get("label") else "")
                          for e in b["edges"]]
    return "\n".join(lines)


def cited_chunks(lesson: dict, chunks: list[dict]) -> list[dict]:
    """The source chunks the lesson cites, in source order; all chunks if it cites none."""
    locators = [loc for s in lesson["screens"] for b in s["blocks"] for loc in b.get("sources", [])]
    picked = [c for c in chunks if any(chunk_matches(c["locator"], loc) or c["locator"] == loc for loc in locators)]
    return picked or list(chunks)


def render_source(chunks: list[dict], budget: int = SOURCE_BUDGET_CHARS) -> str:
    lines, size = [], 0
    for c in chunks:
        loc = c["locator"]
        where = ", ".join(f"{k} {loc[k]}" for k in ("slide", "page", "paragraph") if k in loc)
        text = chunk_text_for_prompt(c)
        if size + len(text) > budget:
            break
        lines.append(f"({loc.get('file', '')}, {where}) {text}")
        size += len(text)
    return wrap_sources("\n".join(lines))


COMMON = (
    "You review a Hebrew e-learning lesson written for employees from the source material of "
    "a content owner. The lesson is labelled with screen and block numbers. Report only real "
    "problems that a careful human reviewer would want fixed; do not report style preferences, "
    "and do not report the same problem twice. Quote the exact Hebrew text in \"evidence\". "
    "Write \"problem\" and \"suggestion\" in Hebrew, short and concrete."
)
FINDINGS_FORMAT = (
    ' Reply with only a JSON object: {"findings": [{"screen": <screen number>, '
    '"where": "block <n> (<type>)[, alternate]", "category": "<one of: CATS>", "evidence": "...", '
    '"problem": "...", "suggestion": "..."}]}. If there is nothing to report, reply {"findings": []}.'
)
QUESTIONS_FORMAT = (
    " Assess EVERY question and EVERY alternate question, in lesson order, even when it is fine. "
    'Reply with only a JSON object: {"questions": [{"screen": <n>, "where": "block <n> '
    '(multiple_choice)" or "block <n> (multiple_choice), alternate", "question": "<the first words '
    'of the question>", "kind": "decision" or "recall", "kind_reason": "...", "other_correct": '
    '"<exact text of another option that is also correct or defensible, or empty>", '
    '"other_correct_reason": "...", "scenario_problem": "<what in the scenario is ambiguous or '
    'contradicts the rule, the source or the marked answer, or empty>", "weak_distractors": '
    '["<exact text of each weak wrong option>"], "weak_distractors_reason": "..."}]}. '
    "All reasons in Hebrew."
)
ILLUSTRATIONS_FORMAT = (
    " Assess EVERY illustration, in lesson order. Reply with only a JSON object: "
    '{"illustrations": [{"screen": <n>, "where": "block <n> (illustration)", "title": "...", '
    '"adds_value": true or false, "reason": "<Hebrew>", "suggestion": "<Hebrew, when adds_value is '
    'false: what the illustration could let the learner practise instead>"}]}.'
)

MODEL_CHECKS: dict[str, dict] = {
    "language": {
        "categories": ("language", "duplicate_text"),
        "source": False,
        "format": "findings",
        "prompt": (
            "Your task: proofread the Hebrew. Read every block, including titles of illustrations, "
            "options and feedback. Report: spelling and grammar errors; wrong or non-existent words "
            "and wrong verb forms; a word that names the wrong thing for the context (for example "
            "one car part instead of another); unclear or awkward sentences; and a sentence or "
            "phrase that repeats what the same block already said (category duplicate_text). "
            "Do not report singular versus plural address; that is checked elsewhere."
        ),
    },
    "grounding": {
        "categories": ("ungrounded",),
        "source": True,
        "format": "findings",
        "prompt": (
            "Your task: check every claim in the lesson against the source material. Report each "
            "claim the source does not support: numbers, comparisons or multiples the source does "
            "not give (for example comparing two values the source never compares), reasons, "
            "intentions or explanations the source does not state, people, roles or procedures the "
            "source does not mention (for example telling the learner to contact someone), and "
            "anything that contradicts the source. Check every block, including did-you-know boxes, "
            "explanations, options, feedback and illustration descriptions. A faithful rephrasing "
            "or a direct consequence of what the source says is fine."
        ),
    },
    "questions": {
        "categories": ("multiple_correct", "inconsistent_scenario", "recall_question", "distractors"),
        "source": True,
        "format": "questions",
        "prompt": (
            "Your task: check the multiple-choice questions. A lesson question must present a "
            "realistic situation and ask for a decision, with exactly one correct option. For each "
            "question:\n"
            "- kind: \"recall\" when the learner only has to repeat a rule or a fact (the stem "
            "already states the key condition and asks what the rule says, what to turn on, what is "
            "true); \"decision\" when the learner must interpret details of the situation to choose "
            "an action.\n"
            "- other_correct: be strict. Imagine a learner who chose a different option and argues "
            "for it from the scenario and the source. If the argument holds, quote that option. "
            "Typical cases: a scenario detail that, by the source's own numbers, also meets the "
            "condition of another rule; a more cautious action that the source also requires or "
            "allows; two options that differ only in wording.\n"
            "- scenario_problem: a detail of the scenario that contradicts the rule or the marked "
            "answer, a detail the answer depends on that is left ambiguous (for example a place "
            "described in words that the rule treats differently), or a marked answer that adds an "
            "action the source does not mention.\n"
            "- weak_distractors: wrong options a learner would rule out without knowing the material "
            "because they are absurd, obviously dangerous, unrelated or careless."
        ),
    },
    "illustrations": {
        "categories": ("illustration_value",),
        "source": False,
        "format": "illustrations",
        "prompt": (
            "Your task: judge whether each planned interactive illustration is worth building. It "
            "adds value only when the learner manipulates something whose result they must judge: "
            "practising a decision of the lesson, or seeing a relation (one quantity changing "
            "another) that text alone conveys poorly. It adds nothing when it only lets the learner "
            "tick a box, click to reveal the rule the text already states, read a card or a "
            "timeline that repeats a list, or watch a result without making a choice."
        ),
    },
    "pedagogy": {
        "categories": ("repetition", "question_overlap", "enrichment_misuse", "diagram_logic"),
        "source": True,
        "format": "findings",
        "prompt": (
            "Your task: check the lesson as a teacher. Report:\n"
            "- repetition: a screen or a block that repeats content already taught on another "
            "screen without adding anything;\n"
            "- question_overlap: two questions (including alternates) on different screens that test "
            "the same decision;\n"
            "- enrichment_misuse: a did-you-know box that holds a core rule the learner must follow "
            "or that a question tests; did-you-know is only for extra, optional enrichment;\n"
            "- diagram_logic: a decision diagram whose branches are not mutually exclusive, mixes "
            "different kinds of decisions as parallel branches, misses a branch (for example only "
            "'yes'), leaves actions disconnected from the rest of the flow, or contradicts the source."
        ),
    },
}

_FORMATS = {"findings": FINDINGS_FORMAT, "questions": QUESTIONS_FORMAT, "illustrations": ILLUSTRATIONS_FORMAT}
# share of the lesson's questions / illustrations a checklist reply must cover
MIN_COVERAGE = 0.8


def build_check_messages(name: str, lesson_text: str, source_text: str,
                         screens: tuple[int, int] | None = None) -> list[BaseMessage]:
    spec = MODEL_CHECKS[name]
    system = (COMMON + " " + spec["prompt"] + _FORMATS[spec["format"]]).replace("CATS", ", ".join(spec["categories"]))
    user = f"The lesson:\n{lesson_text}"
    if screens:
        user += f"\n\nAssess only the questions on screens {screens[0]} to {screens[1]}; the rest is context."
    if spec["source"]:
        system += SOURCE_DATA_RULE
        user = f"Source material:\n{source_text}\n\n{user}"
    return [SystemMessage(system), HumanMessage(user)]


def _json_object(text: str, key: str) -> dict:
    """The first JSON object in the reply that has `key`; text or more JSON around it is ignored."""
    decoder = json.JSONDecoder()
    error = "no JSON object in the reply"
    for match in re.finditer(r"\{", text):
        try:
            data, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError as e:
            error = f"invalid JSON: {e.msg}"
            continue
        if isinstance(data, dict) and key in data:
            return data
    raise ValueError(error)


def _items(data: dict, key: str) -> list[dict]:
    items = data.get(key)
    if not isinstance(items, list):
        raise ValueError(f'the reply has no "{key}" list')
    return [i for i in items if isinstance(i, dict)]


def _screen(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _text(value) -> str:
    if isinstance(value, list):
        return "; ".join(str(v) for v in value if v)
    return str(value).strip() if value not in (None, False) else ""


def _parse_plain(data: dict, name: str, expected: int) -> list[Finding]:
    out = []
    allowed = MODEL_CHECKS[name]["categories"]
    for raw in _items(data, "findings"):
        category = str(raw.get("category", "")).strip().lower()
        if category not in allowed:
            category = allowed[0] if len(allowed) == 1 else "other"
        out.append(Finding(check=name, category=category, screen=_screen(raw.get("screen")),
                           **{k: _text(raw.get(k)) for k in ("where", "evidence", "problem", "suggestion")}))
    return out


def _check_coverage(items: list[dict], expected: int, what: str) -> None:
    if expected and len(items) < MIN_COVERAGE * expected:
        raise ValueError(f"you assessed {len(items)} of the {expected} {what}; assess every one")


def _parse_questions(data: dict, name: str, expected: int) -> list[Finding]:
    items = _items(data, "questions")
    _check_coverage(items, expected, "questions (main and alternate)")
    out = []
    for q in items:
        base = {"check": name, "screen": _screen(q.get("screen")), "where": _text(q.get("where"))}
        stem = _text(q.get("question"))
        if str(q.get("kind", "")).strip().lower() == "recall":
            out.append(Finding(category="recall_question", evidence=stem, problem=_text(q.get("kind_reason")), **base))
        if _text(q.get("other_correct")):
            out.append(Finding(category="multiple_correct", evidence=_text(q.get("other_correct")),
                               problem=_text(q.get("other_correct_reason")), **base))
        if _text(q.get("scenario_problem")):
            out.append(Finding(category="inconsistent_scenario", evidence=stem,
                               problem=_text(q.get("scenario_problem")), **base))
        if _text(q.get("weak_distractors")):
            out.append(Finding(category="distractors", evidence=_text(q.get("weak_distractors")),
                               problem=_text(q.get("weak_distractors_reason")), **base))
    return out


def _parse_illustrations(data: dict, name: str, expected: int) -> list[Finding]:
    items = _items(data, "illustrations")
    _check_coverage(items, expected, "illustrations")
    return [Finding(check=name, category="illustration_value", screen=_screen(i.get("screen")),
                    where=_text(i.get("where")), evidence=_text(i.get("title")),
                    problem=_text(i.get("reason")), suggestion=_text(i.get("suggestion")))
            for i in items if i.get("adds_value") is False or str(i.get("adds_value")).lower() == "false"]


_PARSERS = {"findings": _parse_plain, "questions": _parse_questions, "illustrations": _parse_illustrations}


def expected_items(lesson: dict, name: str, screens: tuple[int, int] | None = None) -> int:
    fmt = MODEL_CHECKS[name]["format"]
    if fmt == "questions":
        return sum(1 for s_no, *_ in questions(lesson) if not screens or screens[0] <= s_no <= screens[1])
    if fmt == "illustrations":
        return sum(1 for s in lesson["screens"] for b in s["blocks"] if b["type"] == "illustration")
    return 0


def parse_findings(text: str, name: str, expected: int = 0) -> list[Finding]:
    """The model's reply as findings; ValueError when it is not usable (sent back to the model)."""
    fmt = MODEL_CHECKS[name]["format"]
    return _PARSERS[fmt](_json_object(text, fmt), name, expected)


ModelCall = Callable[..., LLMResult]


def run_model_check(name: str, lesson: dict, lesson_text: str, source_text: str,
                    call: ModelCall = call_llm,
                    screens: tuple[int, int] | None = None) -> tuple[list[Finding], list[LLMResult]]:
    messages = build_check_messages(name, lesson_text, source_text, screens)
    expected = expected_items(lesson, name, screens)
    calls: list[LLMResult] = []
    error = ""
    for _ in range(MODEL_ATTEMPTS):
        result = call(messages, timeout=CHECK_TIMEOUT_SECONDS, role="checker")
        calls.append(result)
        try:
            return parse_findings(result.text, name, expected), calls
        except ValueError as e:
            error = str(e)
            messages = messages + [
                AIMessage(result.text),
                HumanMessage(f"That reply was invalid: {error}. Reply again with only the JSON object."),
            ]
    raise ValueError(f"{name}: no valid reply after {MODEL_ATTEMPTS} attempts: {error}")


def check_jobs(lesson: dict, model_checks: tuple[str, ...]) -> list[tuple[str, tuple[int, int] | None]]:
    """(check name, screen range) per model call: the question checklist goes in screen batches."""
    jobs: list[tuple[str, tuple[int, int] | None]] = []
    count = len(lesson["screens"])
    for name in model_checks:
        if MODEL_CHECKS[name]["format"] == "questions" and count > QUESTION_BATCH_SCREENS:
            jobs += [(name, (a, min(a + QUESTION_BATCH_SCREENS - 1, count)))
                     for a in range(1, count + 1, QUESTION_BATCH_SCREENS)]
        else:
            jobs.append((name, None))
    return jobs


def run_checkers(lesson: dict, chunks: list[dict], call: ModelCall = call_llm,
                 model_checks: tuple[str, ...] = tuple(MODEL_CHECKS), workers: int = 6) -> CheckReport:
    """All code checks, then the model checks in parallel. A failed model check is recorded, not raised."""
    report = CheckReport()
    for check in CODE_CHECKS.values():
        report.findings += check(lesson)
    lesson_text = render_lesson(lesson)
    source_text = render_source(cited_chunks(lesson, chunks))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {}
        for name, screens in check_jobs(lesson, model_checks):
            label = name if screens is None else f"{name} {screens[0]}-{screens[1]}"
            futures[label] = pool.submit(run_model_check, name, lesson, lesson_text, source_text, call, screens)
        for name, future in futures.items():
            try:
                findings, calls = future.result()
            except Exception as e:  # noqa: BLE001 - one check failing must not stop the others
                report.failed[name] = f"{type(e).__name__}: {e}"
                continue
            report.findings += findings
            report.calls += calls
    return report

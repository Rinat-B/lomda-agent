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
from agent.messages import msg
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
    "wrong_answer",
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
# repetition: a sentence whose content words appear (in any prefix form) in another block's sentence
FACT_SIMILARITY = 0.75
MIN_FACT_WORDS = 4
# questions on different screens with this similar a correct answer test the same decision
ANSWER_OVERLAP = 0.7
PHRASE_WORDS = 3
MODEL_ATTEMPTS = 2
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
                problem=msg("checks.answer_length", correct=correct, low=min(others), high=max(others))))
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
                        problem=msg("checks.diagram_branches", node=nid, count=len(edges))))
                elif node["kind"] == "question" and len({e["to"] for e in edges}) == 1:
                    out.append(Finding(
                        check="diagrams", category="diagram_logic", screen=s_no, where=where,
                        evidence=f"{nid}: {node['label']}",
                        problem=msg("checks.diagram_same_target", node=nid, target=edges[0]["to"])))
            roots = [nid for nid in nodes if incoming[nid] == 0]
            for nid in roots[1:]:
                out.append(Finding(
                    check="diagrams", category="diagram_logic", screen=s_no, where=where,
                    evidence=f"{nid}: {nodes[nid]['label']}",
                    problem=msg("checks.diagram_unreached", node=nid, root=roots[0])))
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
                evidence=msg("checks.address_evidence", singular=", ".join(dict.fromkeys(s_hits)),
                             plural=", ".join(dict.fromkeys(p_hits))),
                problem=msg("checks.address_screen")))
    out.append(Finding(
        check="address", category="address", screen=None,
        evidence=msg("checks.address_counts", singular=total_s, plural=total_p),
        problem=msg("checks.address_lesson")))
    return out


def _content_words(text: str) -> list[set[str]]:
    """Each content word of the text as its set of Hebrew prefix variants (stopwords dropped)."""
    from agent.tools import prefix_variants, stopwords

    stop = stopwords()
    return [set(prefix_variants(w)) for w in words(text)
            if w not in stop and (len(w) > 1 or w.isdigit())]


def _similarity(a: list[set[str]], b: list[set[str]]) -> float:
    """Share of the shorter text's words that appear (in any prefix form) in the other one."""
    if not a or not b:
        return 0.0
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    pool = set().union(*long_)
    return sum(1 for w in short if w & pool) / len(short)


def _number_words() -> dict[str, str]:
    return json.loads(WORDS_PATH.read_text(encoding="utf-8")).get("number_words", {})


def _numbers(text: str) -> set[str]:
    """The numbers of a text, with Hebrew number words as digits ("twice" counts as 2)."""
    found = set(re.findall(r"\d+(?:[.,]\d+)?", text))
    table = _number_words()
    for w in words(text):
        for form in (w, w[1:] if len(w) > 2 else w):  # one prefix letter, e.g. "and two"
            if form in table:
                found.add(table[form])
                break
    return found


_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+|\n+")
FACT_BLOCKS = ("paragraph", "list", "stop_and_think", "did_you_know")


def facts(lesson: dict) -> list[tuple[int, int, str, str]]:
    """(screen, block, block type, sentence) for every taught sentence: paragraphs, list
    items, stop-and-think explanations and did-you-know boxes (not questions or feedback)."""
    out = []
    for s_no, screen in enumerate(lesson["screens"], 1):
        for b_no, b in enumerate(screen["blocks"], 1):
            if b["type"] not in FACT_BLOCKS:
                continue
            if b["type"] == "list":
                parts = list(b["items"])
            else:
                text = b["explanation"] if b["type"] == "stop_and_think" else b["text"]
                parts = [p for p in _SENTENCE_END.split(text)]
            out += [(s_no, b_no, b["type"], p.strip()) for p in parts if p and p.strip()]
    return out


def check_repetition(lesson: dict) -> list[Finding]:
    """A fact taught in more than one block (also across screens), and a phrase repeated
    inside one paragraph. One finding per pair of blocks."""
    out = []
    items = [(s, b, t, text, _content_words(text), _numbers(text)) for s, b, t, text in facts(lesson)]
    items = [it for it in items if len(it[4]) >= MIN_FACT_WORDS]
    reported: set[tuple] = set()
    for i, (s2, b2, t2, text2, w2, n2) in enumerate(items):
        for s1, b1, t1, text1, w1, n1 in items[:i]:
            if (s1, b1) == (s2, b2):
                continue
            if _similarity(w1, w2) < FACT_SIMILARITY:
                continue
            if not (n1 <= n2 or n2 <= n1):  # different numbers: a different fact
                continue
            dyk_first = t1 == "did_you_know" and t2 != "did_you_know"
            screen, block, kind, other = (s1, b1, t1, (s2, b2)) if dyk_first else (s2, b2, t2, (s1, b1))
            key = (screen, block)
            if key in reported:
                continue
            reported.add(key)
            dyk = "did_you_know" in (t1, t2)
            out.append(Finding(
                check="repetition", category="enrichment_misuse" if dyk else "repetition", screen=screen,
                where=_where(block, kind), evidence=text1 if dyk_first else text2,
                problem=msg("checks.dyk_repeated" if dyk else "checks.repeated_fact",
                            screen=other[0], block=other[1])))
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
                    problem=msg("checks.phrase_repeated")))
    return out


def check_question_overlap(lesson: dict) -> list[Finding]:
    """Questions on different screens whose correct answers are nearly the same."""
    out = []
    qs = [(s_no, b_no, label, _content_words(q["options"][q["correct_index"]]))
          for s_no, b_no, label, q in questions(lesson)]
    for i, (s1, b1, l1, a1) in enumerate(qs):
        for s2, b2, l2, a2 in qs[i + 1:]:
            if s1 != s2 and _similarity(a1, a2) >= ANSWER_OVERLAP:
                out.append(Finding(
                    check="question_overlap", category="question_overlap", screen=s2,
                    where=_where(b2, "multiple_choice", l2),
                    evidence=msg("checks.question_overlap_evidence", screen=s1, block=b1),
                    problem=msg("checks.question_overlap")))
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
            "and wrong verb forms (check every verb letter by letter: a missing or extra letter "
            "makes a different or non-existent word); a word that names the wrong thing for the "
            "context (for example one car part instead of another); unclear or awkward sentences; "
            "and a sentence or phrase that repeats what the same block already said (category "
            "duplicate_text). Do not report singular versus plural address; that is checked elsewhere."
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
            "Your task: check the lesson as a teacher. Each fact should be taught once. Report:\n"
            "- repetition: a screen or a block that repeats content already taught (on this screen "
            "or another one) without adding anything;\n"
            "- question_overlap: two questions (including alternates) on different screens that test "
            "the same decision; and an alternate question that tests a different objective than its "
            "main question (the alternate is shown on a retry, so it must check the same objective "
            "with a new situation);\n"
            "- enrichment_misuse: a did-you-know box that holds a core rule the learner must follow, "
            "a fact that a question tests, or a fact that appears anywhere else in the lesson; "
            "did-you-know is only for an extra fact found nowhere else;\n"
            "- diagram_logic: a decision diagram whose branches are not mutually exclusive, mixes "
            "different kinds of decisions as parallel branches, misses a branch (for example only "
            "'yes'), leaves actions disconnected from the rest of the flow, or contradicts the source."
        ),
    },
}

_FORMATS = {"findings": FINDINGS_FORMAT, "illustrations": ILLUSTRATIONS_FORMAT}
# share of the lesson's questions / illustrations a checklist reply must cover
MIN_COVERAGE = 0.8

Screens = tuple[int, ...] | None  # None = the whole lesson


def _in(screen, screens: Screens) -> bool:
    return not screens or screen in screens


def screens_label(screens: Screens) -> str:
    return ", ".join(str(s) for s in screens) if screens else "all"


def build_check_messages(name: str, lesson_text: str, source_text: str, screens: Screens = None) -> list[BaseMessage]:
    spec = MODEL_CHECKS[name]
    system = (COMMON + " " + spec["prompt"] + _FORMATS[spec["format"]]).replace("CATS", ", ".join(spec["categories"]))
    user = f"The lesson:\n{lesson_text}"
    if screens:
        what = "illustrations" if spec["format"] == "illustrations" else "problems"
        user += (f"\n\nAssess and report only the {what} on screens {screens_label(screens)}; "
                 "the other screens are context.")
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


def _parse_illustrations(data: dict, name: str, expected: int) -> list[Finding]:
    items = _items(data, "illustrations")
    _check_coverage(items, expected, "illustrations")
    return [Finding(check=name, category="illustration_value", screen=_screen(i.get("screen")),
                    where=_text(i.get("where")), evidence=_text(i.get("title")),
                    problem=_text(i.get("reason")), suggestion=_text(i.get("suggestion")))
            for i in items if i.get("adds_value") is False or str(i.get("adds_value")).lower() == "false"]


_PARSERS = {"findings": _parse_plain, "illustrations": _parse_illustrations}


def expected_items(lesson: dict, name: str, screens: Screens = None) -> int:
    if MODEL_CHECKS[name]["format"] == "illustrations":
        return sum(1 for i, s in enumerate(lesson["screens"], 1) if _in(i, screens)
                   for b in s["blocks"] if b["type"] == "illustration")
    return 0


def parse_findings(text: str, name: str, expected: int = 0, screens: Screens = None) -> list[Finding]:
    """The model's reply as findings; ValueError when it is not usable (sent back to the model).
    With `screens`, findings on other screens (or on no screen) are dropped."""
    fmt = MODEL_CHECKS[name]["format"]
    found = _PARSERS[fmt](_json_object(text, fmt), name, expected)
    return [f for f in found if not screens or f.screen in screens]


# ---- single answer: every question on its own, from the question's excerpts only ----------------
SINGLE_ANSWER_PROMPT = (
    "You check the multiple-choice questions of a Hebrew e-learning lesson. For each question "
    "you get only its source excerpts, the situation and the options; you are not told which "
    "option the author marked as correct. A good question presents a situation, asks for a "
    "decision, and has exactly one defensible answer and plausible wrong options.\n"
    "For the question:\n"
    "- kind: \"recall\" when the learner only has to repeat a rule or a fact (the stem already "
    "states the key condition and asks what the rule says, what to turn on, what is true); "
    "\"decision\" when the learner must interpret details of the situation to choose an action. "
    "kind_reason: why.\n"
    "For EVERY option, on its own:\n"
    "- rule: the rule of the excerpts that would make this option the right action, with its "
    "condition (or \"none\").\n"
    "- check: what the situation says about that condition. When the excerpts give numbers "
    "(distances, times, counts, depths, speeds), compute what the situation's details mean and "
    "write the calculation. For example, if the excerpts say shelves stand about 2 m apart and "
    "the worker can barely see one shelf, the visible distance is about 2 m or less.\n"
    "- borderline: true when the computed value is at the rule's threshold or within the "
    "precision of the excerpts' own estimate (an 'about 2 m' estimate cannot separate 'below 2 m' "
    "from 'at least 2 m'), so a learner cannot tell whether the condition holds. If your check "
    "says the value is 'about' the threshold, or 'not clearly' above or below it, borderline is "
    "true.\n"
    "- justified: true when a rule exists and its condition holds, is borderline, or is left "
    "open by the situation; false only when the details clearly exclude it or no rule supports "
    "the option.\n"
    "- worst_reading: the option and the situation read with every vague place, time or amount "
    "word in its least favourable plausible sense, written out (for example, a spot described "
    "as 'after the junction' may be right after it; 'soon' may be too late).\n"
    "- at_risk: true when, under worst_reading, the option would break a rule of the excerpts "
    "(for example a rule that forbids stopping right after a junction).\n"
    "- weak: true when a learner would rule the option out without knowing the material, "
    "because it is absurd, obviously dangerous, careless or unrelated.\n"
    "Write every reason in Hebrew. Reply with only a JSON object: "
    '{"questions": [{"id": "<question id>", "kind": "decision" or "recall", "kind_reason": "...", '
    '"options": [{"letter": "A", "rule": "...", "check": "...", "borderline": true or false, '
    '"justified": true or false, "worst_reading": "...", "at_risk": true or false, '
    '"weak": true or false, "reason": "..."}]}]}, with every question and every option.'
) + SOURCE_DATA_RULE

MODEL_CHECKS["single_answer"] = {
    "categories": ("multiple_correct", "wrong_answer", "inconsistent_scenario", "recall_question", "distractors"),
    "source": True,
    "format": "single_answer",
    "prompt": SINGLE_ANSWER_PROMPT,
}


def _block_chunks(block: dict, chunks: list[dict]) -> list[dict]:
    locs = block.get("sources") or []
    return [c for c in chunks if any(chunk_matches(c["locator"], loc) or c["locator"] == loc for loc in locs)]


def single_answer_items(lesson: dict, chunks: list[dict], screens: Screens = None) -> list[dict]:
    """Each question and alternate with only the excerpts its block cites."""
    items = []
    for s_no, b_no, label, q in questions(lesson):
        if not _in(s_no, screens):
            continue
        block = lesson["screens"][s_no - 1]["blocks"][b_no - 1]
        qid = block.get("id") or f"q{s_no}"
        items.append({
            "id": qid if label == "main" else f"{qid}-alternate",
            "screen": s_no, "where": _where(b_no, "multiple_choice", label),
            "question": q["question"], "options": q["options"], "correct_index": q["correct_index"],
            "excerpts": _block_chunks(block, chunks) or cited_chunks(lesson, chunks),
        })
    return items


def build_single_answer_messages(items: list[dict]) -> list[BaseMessage]:
    parts = []
    for it in items:
        options = "\n".join(f"{chr(ord('A') + k)}. {o}" for k, o in enumerate(it["options"]))
        parts.append(f"### Question {it['id']}\nSource excerpts:\n{render_source(it['excerpts'])}\n"
                     f"Situation and question: {it['question']}\nOptions:\n{options}")
    return [SystemMessage(SINGLE_ANSWER_PROMPT), HumanMessage("\n\n".join(parts))]


def _truthy(value) -> bool:
    return value is True or str(value).strip().lower() == "true"


def parse_single_answer(text: str, items: list[dict]) -> list[Finding]:
    data = _json_object(text, "questions")
    replies = {str(r.get("id", "")).strip(): r for r in _items(data, "questions")}
    missing = [it["id"] for it in items if it["id"] not in replies]
    if len(missing) > (1 - MIN_COVERAGE) * len(items):
        raise ValueError(f"questions {missing} are missing; judge every question and every option")
    out = []
    for it in items:
        reply = replies.get(it["id"])
        if not reply:
            continue
        base = {"check": "single_answer", "screen": it["screen"], "where": it["where"]}
        if str(reply.get("kind", "")).strip().lower() == "recall":
            out.append(Finding(category="recall_question", evidence=it["question"],
                               problem=_text(reply.get("kind_reason")), **base))
        verdicts, risky, weak = {}, {}, []
        for o in reply.get("options") or []:
            if isinstance(o, dict):
                letter = str(o.get("letter", "")).strip().upper()[:1]
                k = ord(letter) - ord("A") if letter else -1
                if 0 <= k < len(it["options"]):
                    reason = " ".join(x for x in (_text(o.get("check")), _text(o.get("reason"))) if x)
                    verdicts[k] = (_truthy(o.get("justified")) or _truthy(o.get("borderline")), reason)
                    if _truthy(o.get("at_risk")):
                        risky[k] = _text(o.get("worst_reading")) or reason
                    if _truthy(o.get("weak")):
                        weak.append((k, _text(o.get("reason"))))
        correct = it["correct_index"]
        others = [k for k, (ok, _) in sorted(verdicts.items()) if ok and k != correct]
        if others:
            out.append(Finding(category="multiple_correct",
                               evidence="; ".join(it["options"][k] for k in others),
                               problem="; ".join(verdicts[k][1] for k in others), **base))
        if correct in verdicts and not verdicts[correct][0]:
            out.append(Finding(category="wrong_answer", evidence=it["options"][correct],
                               problem=verdicts[correct][1], **base))
        elif correct in risky:
            out.append(Finding(category="inconsistent_scenario", evidence=it["options"][correct],
                               problem=risky[correct], **base))
        weak = [(k, r) for k, r in weak if k != correct]
        if weak:
            out.append(Finding(category="distractors", evidence="; ".join(it["options"][k] for k, _ in weak),
                               problem="; ".join(r for _, r in weak if r), **base))
    return out


ModelCall = Callable[..., LLMResult]


def _ask_until_valid(name: str, messages: list[BaseMessage], parse, call: ModelCall) -> tuple[list[Finding], list[LLMResult]]:
    calls: list[LLMResult] = []
    error = ""
    for _ in range(MODEL_ATTEMPTS):
        result = call(messages, timeout=CHECK_TIMEOUT_SECONDS, role="checker")
        calls.append(result)
        try:
            return parse(result.text), calls
        except ValueError as e:
            error = str(e)
            messages = messages + [
                AIMessage(result.text),
                HumanMessage(f"That reply was invalid: {error}. Reply again with only the JSON object."),
            ]
    raise ValueError(f"{name}: no valid reply after {MODEL_ATTEMPTS} attempts: {error}")


def run_model_check(name: str, lesson: dict, lesson_text: str, source_text: str,
                    call: ModelCall = call_llm, screens: Screens = None,
                    chunks: list[dict] | None = None) -> tuple[list[Finding], list[LLMResult]]:
    if MODEL_CHECKS[name]["format"] == "single_answer":
        items = single_answer_items(lesson, chunks or [], screens)
        if not items:
            return [], []
        return _ask_until_valid(name, build_single_answer_messages(items),
                                lambda text: parse_single_answer(text, items), call)
    expected = expected_items(lesson, name, screens)
    if screens and MODEL_CHECKS[name]["format"] == "illustrations" and not expected:
        return [], []
    return _ask_until_valid(name, build_check_messages(name, lesson_text, source_text, screens),
                            lambda text: parse_findings(text, name, expected, screens), call)


# screens per call for the per-question check: shorter replies time out less and focus better
# Smaller batches in balanced, where the extra calls are cheap: a balanced acceptance run with
# 3 screens per call missed question 7 (20.11.2026).
SINGLE_ANSWER_BATCH = {"free": 3, "balanced": 2}


def batch_screens(fmt: str) -> int | None:
    """Screens per call for a batched check format in the active profile (None = not batched)."""
    from agent.llm import active_profile

    if fmt != "single_answer":
        return None
    return SINGLE_ANSWER_BATCH.get(active_profile(), SINGLE_ANSWER_BATCH["free"])


def check_jobs(lesson: dict, model_checks: tuple[str, ...], screens: Screens = None) -> list[tuple[str, Screens]]:
    """(check name, screens) per model call. The per-question check goes in batches of
    screens; the others run once, on the whole lesson or on `screens`."""
    numbers = list(screens) if screens else list(range(1, len(lesson["screens"]) + 1))
    jobs: list[tuple[str, Screens]] = []
    for name in model_checks:
        size = batch_screens(MODEL_CHECKS[name]["format"])
        if size:
            jobs += [(name, tuple(numbers[i:i + size])) for i in range(0, len(numbers), size)]
        else:
            jobs.append((name, tuple(screens) if screens else None))
    return jobs


def job_label(name: str, screens: Screens) -> str:
    return name if not screens else f"{name} {screens_label(screens)}"


def run_checkers(lesson: dict, chunks: list[dict], call: ModelCall = call_llm,
                 model_checks: tuple[str, ...] | None = None, workers: int = 6,
                 screens: Screens = None, code: bool = True,
                 jobs_list: list[tuple[str, Screens]] | None = None) -> CheckReport:
    """Code checks (unless code=False), then the model checks in parallel, on the whole lesson
    or only on `screens` (or exactly `jobs_list`). A failed model check is recorded, not raised."""
    model_checks = tuple(MODEL_CHECKS) if model_checks is None else model_checks
    report = CheckReport()
    if code:
        for check in CODE_CHECKS.values():
            report.findings += check(lesson)
    lesson_text = render_lesson(lesson)
    source_text = render_source(cited_chunks(lesson, chunks))
    planned = jobs_list if jobs_list is not None else check_jobs(lesson, model_checks, screens)
    jobs = {job_label(name, s): (name, s) for name, s in planned}

    def run(label: str) -> None:
        name, job_screens = jobs[label]
        findings, calls = run_model_check(name, lesson, lesson_text, source_text, call, job_screens, chunks)
        report.findings.extend(findings)
        report.calls.extend(calls)
        report.failed.pop(label, None)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {label: pool.submit(run, label) for label in jobs}
        for label, future in futures.items():
            try:
                future.result()
            except Exception as e:  # noqa: BLE001 - one check failing must not stop the others
                report.failed[label] = f"{type(e).__name__}: {e}"
    # A second chance, one at a time: free models often time out only under parallel load.
    for label in list(report.failed):
        if "MissingApiKeyError" not in report.failed[label]:
            try:
                run(label)
            except Exception as e:  # noqa: BLE001
                report.failed[label] = f"{type(e).__name__}: {e}"
    return report


# ---- choosing the checker models ----------------------------------------------------------------
def family(model: str) -> str:
    """The model family, from the OpenRouter id: "google/gemma-4-31b-it:free" -> "google"."""
    return model.split("/", 1)[0].lower()


def checker_models(writer_models: list[str], checkers: list[str]) -> tuple[list[str], list[str]]:
    """(checkers of another family than every writer that wrote, the same-family rest)."""
    writers = {family(m) for m in writer_models}
    other = [m for m in checkers if family(m) not in writers]
    return other, [m for m in checkers if m not in other]

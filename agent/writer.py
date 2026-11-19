"""Stage 3 writer: turn the approved structure into lesson screens (agent/lesson_schema.py).

Screens are written a few at a time (screen_groups: 3-4 per model call), so a whole
lesson stays well under 30 calls even on free models. Each call gets the group's planned
screens, the source excerpts their locators point to (numbered E1, E2, ...), the lesson
type and the writing rules. The model cites excerpt ids; the code turns them into
locators, sets ids, titles and file names, and validates every screen with the schema
models. Errors go back to the model once; after that the next writer model is tried.
"""

import json
import math
import re

from pydantic import ValidationError

from agent.guards import SOURCE_DATA_RULE, chunk_text_for_prompt, wrap_sources
from agent.lesson_schema import (
    COMPLIANCE_MAX_WORDS,
    PALETTES,
    Screen,
    screen_word_count,
    validate_lesson,
)
from agent.messages import msg
from agent.topics import strip_internal_refs

GROUP_MAX = 4
EXCERPT_BUDGET_CHARS = 14_000
MIN_EXCERPT_CHARS = 300
ILLUSTRATION_HEIGHT = 320
EXCERPT_ID = re.compile(r"(?<![A-Za-z0-9])E\d+(?![A-Za-z0-9])")
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


# ---- groups and excerpts ------------------------------------------------------------------
def screen_groups(count: int, max_size: int = GROUP_MAX) -> list[tuple[int, int]]:
    """Balanced 1-based inclusive ranges of at most max_size screens: 9 -> 3+3+3, 10 -> 4+3+3."""
    if count <= 0:
        return []
    groups = math.ceil(count / max_size)
    base, extra = divmod(count, groups)
    ranges, start = [], 1
    for i in range(groups):
        size = base + (1 if i < extra else 0)
        ranges.append((start, start + size - 1))
        start += size
    return ranges


def chunk_matches(chunk_locator: dict, unit_locator: dict) -> bool:
    """Does a chunk belong to a unit locator ({file, slide} / {file, page} / paragraphs)?"""
    if chunk_locator.get("file") != unit_locator.get("file"):
        return False
    for key in ("slide", "page"):
        if key in unit_locator:
            return chunk_locator.get(key) == unit_locator[key]
    if "slide" in chunk_locator or "page" in chunk_locator or "paragraph" not in chunk_locator:
        return False
    start = unit_locator["paragraph"]
    return start <= chunk_locator["paragraph"] <= unit_locator.get("paragraph_end", start)


def group_excerpts(plan_screens: list[dict], chunks: list[dict]) -> list[tuple[str, dict]]:
    """The chunks the group's planned points cite, in order, as (id, chunk) pairs E1, E2, ..."""
    picked: list[dict] = []
    for screen in plan_screens:
        for point in screen["points"]:
            for loc in point["sources"]:
                for chunk in chunks:
                    if chunk_matches(chunk["locator"], loc) and not any(chunk is c for c in picked):
                        picked.append(chunk)
    return [(f"E{i}", c) for i, c in enumerate(picked, 1)]


def _describe(locator: dict) -> str:
    parts = [locator["file"]]
    for key in ("slide", "page", "paragraph"):
        if key in locator:
            parts.append(f"{key} {locator[key]}")
    return ", ".join(parts)


def format_excerpts(excerpts: list[tuple[str, dict]], budget: int = EXCERPT_BUDGET_CHARS) -> str:
    """Excerpts for the prompt, wrapped as source data; long ones are trimmed, none dropped."""
    per = max(MIN_EXCERPT_CHARS, budget // max(1, len(excerpts)))
    lines = []
    for eid, chunk in excerpts:
        text = chunk_text_for_prompt(chunk)
        if len(text) > per:
            text = text[: per - 3].rstrip() + "..."
        lines.append(f"[{eid}] ({_describe(chunk['locator'])}, {chunk.get('kind', 'body')})\n{text}")
    return wrap_sources("\n\n".join(lines))


# ---- prompt ---------------------------------------------------------------------------------
TYPE_NAMES = {"training": "training", "compliance": "compliance"}
TYPE_WRITING = {
    "training": (
        "Training lesson: a warm, direct tone that speaks to the learner. A screen may be rich: "
        "explain why each rule exists, not only what it says, because people follow rules they "
        "understand. You may add one did_you_know block per screen with an interesting fact that "
        "appears in the excerpts but is not the screen's main point."
    ),
    "compliance": (
        "Compliance lesson: a formal tone. Every word that is not in the owner's document is a "
        "risk, so stay close to the document's wording. No did_you_know blocks. The explanatory "
        f"text of a screen (paragraphs, list items, stop-and-think) must total at most "
        f"{COMPLIANCE_MAX_WORDS} words."
    ),
}

WRITE_PROMPT = (
    "You write the screens of a {type_name} e-learning lesson in Hebrew, following the approved "
    "structure exactly: for each planned screen, its content points, its question idea, and "
    "its illustration and decision diagram if planned.\n"
    "{type_writing}\n"
    "Grounding: the lesson must be trustworthy, so every sentence must come from the numbered "
    "source excerpts. List the ids of the excerpts that support each block in its \"sources\" "
    "(for example [\"E2\"]). Do not add facts that are not in the excerpts. The excerpt ids and "
    "file names are internal: never write them inside a text field.\n"
    "Address: speak to the learner in the second person plural everywhere (questions, "
    "feedback, stop-and-think, illustration descriptions), never in the singular; options are "
    "written as actions (infinitive), not in the first person.\n"
    "Teaching: a screen must actually teach its objective, so it should not be thin. Use "
    "paragraphs and lists, but never say the same thing twice: a list must not repeat the "
    "sentences of a paragraph on the screen, and a screen must not repeat another screen. Where predicting helps learning, add a stop_and_think block: a "
    "question the learner answers in their head first, then the explanation.\n"
    "Questions: each screen has exactly one multiple_choice block, built from the screen's "
    "question idea. It checks whether the learner can act correctly, so describe a concrete "
    "situation and ask what to decide or do; a question answered by recalling a number, date, "
    "name or table value tests memory, not use. The answer must follow from what the lesson "
    "taught, never from facts that appear only in the feedback. Give 3-4 plausible options "
    "(in a training lesson at most one may be light or funny). The player shuffles the "
    "options, so no option may refer to another one (no \"all of the above\" or \"none of "
    "the above\"). feedback_correct explains why "
    "the answer is right. feedback_incorrect guides the learner's reasoning without naming, "
    "quoting or hinting at the right option, because the player never reveals the right answer "
    "after a mistake. Add an alternate: a complete different question on the same objective, "
    "with a different situation and its own options, correct_index and feedbacks, used on a retry.\n"
    "Media: add an illustration block only where the plan has one, with a title and a "
    "description of what the learner will see and do (it is built later). Add a "
    "decision_diagram only where the plan has one: 3-5 nodes (kind \"question\" or "
    "\"action\", ids like \"n1\"), edges from -> to with short labels, using only steps found "
    "in the excerpts.\n"
    "{extras}"
    "Reply with only a JSON object and no other text, in this shape:\n"
    "{shape}"
) + SOURCE_DATA_RULE

SHAPE = (
    '{"screens": [{"number": 1, "blocks": [\n'
    ' {"type": "paragraph", "text": "...", "sources": ["E1"]},\n'
    ' {"type": "list", "ordered": false, "items": ["...", "..."], "sources": ["E2"]},\n'
    ' {"type": "stop_and_think", "prediction_question": "...", "explanation": "...", "sources": ["E1"]},\n'
    ' {"type": "did_you_know", "text": "...", "sources": ["E3"]},\n'
    ' {"type": "multiple_choice", "question": "...", "options": ["...", "...", "..."], "correct_index": 1,\n'
    '  "feedback_correct": "...", "feedback_incorrect": "...",\n'
    '  "alternate": {"question": "...", "options": ["...", "...", "..."], "correct_index": 0,\n'
    '                "feedback_correct": "...", "feedback_incorrect": "..."}, "sources": ["E1"]},\n'
    ' {"type": "illustration", "title": "...", "description": "..."},\n'
    ' {"type": "decision_diagram", "title": "...", "nodes": [{"id": "n1", "label": "...", "kind": "question"}],\n'
    '  "edges": [{"from": "n1", "to": "n2", "label": "..."}], "sources": ["E2"]}\n'
    ']}]%s}'
)


def build_messages(plan_screens: list[dict], excerpts: list[tuple[str, dict]], lesson_type: str,
                   first: bool, last: bool) -> list[tuple[str, str]]:
    """(system, user) prompt texts for one group."""
    extras, shape_extra = "", ""
    if first:
        extras += "This is the first group: also give lesson_title, a short Hebrew title for the whole lesson.\n"
        shape_extra += ', "lesson_title": "..."'
    if last:
        extras += ("This is the last group: also give summary, 3-6 short Hebrew points of what the "
                   "whole lesson taught (shown on the closing \"what we learned\" screen).\n")
        shape_extra += ', "summary": ["...", "..."]'
    system = WRITE_PROMPT.format(
        type_name=TYPE_NAMES[lesson_type], type_writing=TYPE_WRITING[lesson_type],
        extras=extras, shape=SHAPE % shape_extra,
    )
    plan = [
        {
            "number": s["number"], "title": s["title"], "topic": s["topic"], "objective": s["objective"],
            "points": [p["text"] for p in s["points"]],
            "question_idea": s["question_idea"],
            "illustration": s["illustration"],
            "decision_diagram": s["decision_diagram"],
        }
        for s in plan_screens
    ]
    user = (
        f"Planned screens (JSON):\n{json.dumps(plan, ensure_ascii=False, indent=1)}\n\n"
        f"Source excerpts:\n\n{format_excerpts(excerpts)}"
    )
    return [("system", system), ("user", user)]


REVISE_EXTRA = (
    "Revision: these screens were already written and a review found the problems listed below. "
    "Rewrite each listed screen so that every problem is fixed, keep what is fine, and do not "
    "introduce new problems. Every screen still follows its plan.\n"
)


def _screen_for_revision(screen: dict, excerpts: list[tuple[str, dict]]) -> dict:
    """A written screen as the writer sees it: locators turned back into excerpt ids."""
    ids = {json.dumps(c["locator"], sort_keys=True): eid for eid, c in excerpts}
    blocks = []
    for b in screen["blocks"]:
        block = {k: v for k, v in b.items() if k not in ("sources", "id", "file", "height", "source_line")}
        if "sources" in b:
            block["sources"] = [ids[k] for k in (json.dumps(loc, sort_keys=True) for loc in b["sources"]) if k in ids]
        blocks.append(block)
    return {"number": int(screen["id"][1:]), "blocks": blocks}


def build_revision_messages(plan_screens: list[dict], excerpts: list[tuple[str, dict]], lesson_type: str,
                            current: list[dict], findings: list[dict]) -> list[tuple[str, str]]:
    """(system, user) for rewriting some screens: the plan, the excerpts, the current version
    of those screens and the review findings to fix (or the content owner's request)."""
    (_, system), (_, user) = build_messages(plan_screens, excerpts, lesson_type, first=False, last=False)
    system = system.replace("Reply with only a JSON object", REVISE_EXTRA + "Reply with only a JSON object", 1)
    problems = [
        {"screen": f.get("screen"), "where": f.get("where", ""), "problem": f.get("problem", ""),
         "quote": f.get("evidence", ""), "suggestion": f.get("suggestion", "")}
        for f in findings
    ]
    user += (
        f"\n\nCurrent version of these screens (JSON):\n"
        f"{json.dumps([_screen_for_revision(s, excerpts) for s in current], ensure_ascii=False, indent=1)}"
        f"\n\nProblems to fix (JSON):\n{json.dumps(problems, ensure_ascii=False, indent=1)}"
    )
    return [("system", system), ("user", user)]


# ---- parsing the model's reply --------------------------------------------------------------------
TEXT_FIELDS = {
    "paragraph": ("text",),
    "did_you_know": ("text",),
    "stop_and_think": ("prediction_question", "explanation"),
    "illustration": ("title", "description"),
    "decision_diagram": ("title",),
}
QUESTION_FIELDS = ("question", "feedback_correct", "feedback_incorrect")


def _pydantic_problems(e: ValidationError) -> list[str]:
    out = []
    for err in e.errors(include_url=False):
        loc = ".".join(str(p) for p in err["loc"] if p not in TEXT_FIELDS and p not in (
            "list", "multiple_choice"))
        out.append(f"{loc}: {err['msg'].removeprefix('Value error, ')}")
    return out


def shuffle_options(question: dict, seed: str) -> dict:
    """Shuffle a question's options in place and move correct_index with them.

    Models tend to put the right answer in the first two places; a learner who notices
    that can guess. The order is seeded by the question id, so the same lesson always
    comes out the same way.
    """
    import random

    options = question.get("options") or []
    index = question.get("correct_index")
    if not isinstance(index, int) or not 0 <= index < len(options):
        return question  # invalid: left for validation to report
    order = list(range(len(options)))
    random.Random(seed).shuffle(order)
    question["options"] = [options[i] for i in order]
    question["correct_index"] = order.index(index)
    return question


def source_line(locators: list[dict]) -> str:
    from agent.nodes import source_label

    return ", ".join(dict.fromkeys(source_label(loc).strip("()") for loc in locators))


def parse_group(text: str, plan_screens: list[dict], excerpts: list[tuple[str, dict]], lesson_type: str,
                first: bool, last: bool) -> dict:
    """Validate a group reply and return {"screens": [schema screens], "lesson_title"?, "summary"?}.
    Raises ValueError with every problem found, phrased for the model to fix."""
    match = _JSON_BLOCK.search(text)
    if not match:
        raise ValueError("no JSON object found in the reply")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"the reply is not valid JSON: {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get("screens"), list):
        raise ValueError('the JSON must be an object with a "screens" list')

    by_id = {eid: chunk for eid, chunk in excerpts}
    files = tuple(dict.fromkeys(c["locator"]["file"] for _, c in excerpts))
    plan_by_number = {s["number"]: s for s in plan_screens}
    written = {s.get("number"): s for s in data["screens"] if isinstance(s, dict)}
    problems: list[str] = []
    missing = [n for n in plan_by_number if n not in written]
    extra = [n for n in written if n not in plan_by_number]
    if missing:
        problems.append(f"screens {missing} are missing; write exactly screens {list(plan_by_number)}")
    if extra:
        problems.append(f"screens {extra} were not requested; write exactly screens {list(plan_by_number)}")

    def clean(value) -> str:
        return strip_internal_refs(str(value or ""), files, EXCERPT_ID)

    def locators(block: dict, where: str) -> list[dict]:
        ids = [str(x).strip() for x in block.get("sources") or []]
        unknown = [x for x in ids if x not in by_id]
        if unknown:
            problems.append(f"{where}: unknown excerpt ids {unknown}; use only {list(by_id)}")
        locs = []
        for x in ids:
            if x in by_id and by_id[x]["locator"] not in locs:
                locs.append(dict(by_id[x]["locator"]))
        return locs

    screens = []
    for number, plan in plan_by_number.items():
        raw = written.get(number)
        if not isinstance(raw, dict):
            continue
        where_screen = f"screen {number}"
        blocks = []
        counts = {"multiple_choice": 0, "illustration": 0, "decision_diagram": 0}
        for i, b in enumerate(raw.get("blocks") or []):
            if not isinstance(b, dict):
                problems.append(f"{where_screen} block {i + 1}: must be an object")
                continue
            kind = b.get("type")
            where = f"{where_screen} block {i + 1} ({kind})"
            if kind in counts:
                counts[kind] += 1
            block = {"type": kind}
            if kind in ("paragraph", "did_you_know", "stop_and_think"):
                for f in TEXT_FIELDS[kind]:
                    block[f] = clean(b.get(f))
                block["sources"] = locators(b, where)
                if kind == "did_you_know":
                    if lesson_type == "compliance":
                        problems.append(f"{where}: a compliance lesson has no did_you_know blocks")
                    block["source_line"] = source_line(block["sources"]) or "-"
            elif kind == "list":
                block["ordered"] = bool(b.get("ordered", False))
                block["items"] = [clean(x) for x in b.get("items") or []]
                block["sources"] = locators(b, where)
            elif kind == "multiple_choice":
                block["id"] = f"q{number}"
                for f in QUESTION_FIELDS:
                    block[f] = clean(b.get(f))
                block["options"] = [clean(o) for o in b.get("options") or []]
                block["correct_index"] = b.get("correct_index")
                alt = b.get("alternate") if isinstance(b.get("alternate"), dict) else {}
                block["alternate"] = {f: clean(alt.get(f)) for f in QUESTION_FIELDS}
                block["alternate"]["options"] = [clean(o) for o in alt.get("options") or []]
                block["alternate"]["correct_index"] = alt.get("correct_index")
                shuffle_options(block, f"q{number}")
                shuffle_options(block["alternate"], f"q{number}-alternate")
                block["sources"] = locators(b, where)
            elif kind == "illustration":
                block.update(title=clean(b.get("title")), description=clean(b.get("description")),
                             file=f"assets/illustrations/s{number}.html", height=ILLUSTRATION_HEIGHT)
            elif kind == "decision_diagram":
                block["title"] = clean(b.get("title"))
                block["nodes"] = [
                    {"id": str(n.get("id", "")).strip().lower(), "label": clean(n.get("label")), "kind": n.get("kind")}
                    for n in b.get("nodes") or [] if isinstance(n, dict)
                ]
                block["edges"] = [
                    {k: v for k, v in {"from": str(e.get("from", "")).strip().lower(),
                                       "to": str(e.get("to", "")).strip().lower(),
                                       "label": clean(e.get("label")) or None}.items() if v is not None}
                    for e in b.get("edges") or [] if isinstance(e, dict)
                ]
                block["sources"] = locators(b, where)
            else:
                problems.append(f"{where}: unknown block type {kind!r}")
                continue
            blocks.append(block)

        if counts["multiple_choice"] != 1:
            problems.append(f"{where_screen}: needs exactly one multiple_choice block, found {counts['multiple_choice']}")
        if bool(plan["illustration"]) != (counts["illustration"] > 0):
            problems.append(f"{where_screen}: the plan {'has' if plan['illustration'] else 'has no'} illustration; "
                            f"{'add' if plan['illustration'] else 'remove'} the illustration block")
        if bool(plan["decision_diagram"]) != (counts["decision_diagram"] > 0):
            problems.append(f"{where_screen}: the plan {'has' if plan['decision_diagram'] else 'has no'} decision "
                            f"diagram; {'add' if plan['decision_diagram'] else 'remove'} the decision_diagram block")

        screen = {"id": f"s{number}", "title": plan["title"], "topic": plan["topic"], "blocks": blocks}
        try:
            model = Screen.model_validate(screen)
            if lesson_type == "compliance":
                words = screen_word_count(model)
                if words > COMPLIANCE_MAX_WORDS:
                    problems.append(f"{where_screen}: {words} words of text; a compliance screen allows at most "
                                    f"{COMPLIANCE_MAX_WORDS}")
        except ValidationError as e:
            problems += [f"{where_screen}: {p}" for p in _pydantic_problems(e)]
        screens.append(screen)

    result: dict = {"screens": screens}
    if first:
        title = clean(data.get("lesson_title"))
        if len(title) < 2:
            problems.append("lesson_title is missing")
        result["lesson_title"] = title
    if last:
        summary = [clean(p) for p in data.get("summary") or [] if clean(p)]
        if not summary:
            problems.append("summary is missing: give 3-6 short points")
        result["summary"] = summary
    if problems:
        raise ValueError("; ".join(problems))
    return result


# ---- assembly ---------------------------------------------------------------------------------
def assemble_lesson(screens: list[dict], lesson_type: str, title: str, summary: list[str]) -> dict:
    """The full lesson.json; raises agent.lesson_schema.LessonValidationError if invalid."""
    from agent.lesson_schema import LessonValidationError

    lesson = {
        "schema_version": 1,
        "title": title or msg("lesson.default_title"),
        "lesson_type": lesson_type,
        "palette": dict(PALETTES[lesson_type]),
        "motif": {"description": "", "svg": ""},
        "screens": sorted(screens, key=lambda s: int(s["id"][1:])),
        "summary": {"points": summary},
        "certificate": True,
    }
    errors = validate_lesson(lesson)
    if errors:
        raise LessonValidationError(errors)
    return lesson

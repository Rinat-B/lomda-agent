"""Lesson structure (stage 2): screens and motifs, validated with Pydantic.

The model replies with JSON that cites source units by id ("U3"); parse_structure()
validates it against the approved topic map and the lesson type, and maps ids back to
locators. Rules enforced in code:
  - every topic of the map gets at least one screen, and each screen names a map topic;
  - 2-4 content points per screen, each citing at least one known unit id;
  - 2-3 motif descriptions;
  - compliance lessons: the content points of a screen total at most 80 words.
"""

import json
import re

from pydantic import BaseModel, Field, ValidationError

from agent.messages import msg_list
from agent.topics import Unit, strip_internal_refs, unit_file_names

TRAINING = "training"
COMPLIANCE = "compliance"
LESSON_TYPES = (TRAINING, COMPLIANCE)
COMPLIANCE_MAX_WORDS = 80
MAX_SCREENS = 24
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def other_type(lesson_type: str) -> str:
    return COMPLIANCE if lesson_type == TRAINING else TRAINING


class PointOut(BaseModel):
    text: str = Field(min_length=2)
    sources: list[str] = Field(min_length=1)


class ScreenOut(BaseModel):
    number: int | None = None
    title: str = Field(min_length=2)
    topic: str = Field(min_length=2)
    objective: str | None = None
    points: list[PointOut] = Field(min_length=2, max_length=4)
    question_idea: str = Field(min_length=5)
    illustration: str | None = None
    decision_diagram: bool = False
    reason: str | None = None  # why the screen is planned this way (used to answer "why?")


class StructureOut(BaseModel):
    screens: list[ScreenOut] = Field(min_length=1, max_length=MAX_SCREENS)
    motifs: list[str] = Field(min_length=2, max_length=3)


def _no_illustration(value: str | None) -> bool:
    if value is None:
        return True
    words = {"", "none", "null", "no", "n/a", "-"} | {w.lower() for w in msg_list("structure.none_words")}
    return value.strip().lower() in words


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def parse_structure(text: str, units: list[Unit], topics: list[dict], lesson_type: str,
                    screens: int | None = None) -> dict:
    """Validate the model's JSON and return the structure with locators. Raises ValueError.
    `screens`: the exact number of screens the owner asked for (when it can be met)."""
    match = _JSON_BLOCK.search(text)
    if not match:
        raise ValueError("no JSON object found in the reply")
    try:
        parsed = StructureOut.model_validate_json(match.group(0))
    except ValidationError as e:
        raise ValueError(f"JSON does not match the schema: {e.errors(include_url=False)}") from e

    if screens and len(parsed.screens) != screens:
        raise ValueError(f"plan exactly {screens} screens (the owner asked for {screens}); you gave {len(parsed.screens)}")
    by_id = {u.id: u for u in units}
    files = unit_file_names(units)

    def clean(text: str) -> str:
        """Display text without unit ids or file references (sources are shown separately)."""
        return strip_internal_refs(text, files)

    objective_of = {t["name"].strip(): t["objective"] for t in topics}
    covered: set[str] = set()
    screens = []
    for number, s in enumerate(parsed.screens, 1):
        topic = clean(s.topic)
        if topic not in objective_of:
            raise ValueError(
                f"screen {number} names topic {topic!r}, which is not in the map; "
                f"use one of: {list(objective_of)}"
            )
        covered.add(topic)
        points = []
        for p in s.points:
            ids = list(dict.fromkeys(x.strip() for x in p.sources))
            unknown = [x for x in ids if x not in by_id]
            if unknown:
                raise ValueError(f"screen {number} cites unknown source ids {unknown}; use only the given ids")
            text = clean(p.text)
            if len(text) < 2:
                raise ValueError(f"screen {number} has a content point with no text besides source references")
            points.append({"text": text, "sources": [dict(by_id[x].locator) for x in ids]})
        if lesson_type == COMPLIANCE:
            words = sum(word_count(p["text"]) for p in points)
            if words > COMPLIANCE_MAX_WORDS:
                raise ValueError(
                    f"screen {number} has {words} words of content points; a compliance screen "
                    f"allows at most {COMPLIANCE_MAX_WORDS}"
                )
        screens.append({
            "number": number,  # renumbered in order
            "title": clean(s.title),
            "topic": topic,
            "objective": objective_of[topic],  # always the approved map's wording
            "points": points,
            "question_idea": clean(s.question_idea),
            "illustration": None if _no_illustration(s.illustration) else clean(s.illustration) or None,
            "decision_diagram": bool(s.decision_diagram),
            "reason": clean(s.reason) if s.reason else None,
        })
    missing = [name for name in objective_of if name not in covered]
    if missing:
        raise ValueError(f"these topics have no screen: {missing}; every objective needs at least one screen")
    motifs = [clean(m) for m in parsed.motifs if clean(m)]
    if len(motifs) < 2:
        raise ValueError("give 2-3 non-empty motif descriptions")
    return {"lesson_type": lesson_type, "screens": screens, "motifs": motifs}


def structure_json(structure: dict, units: list[Unit]) -> str:
    """A previous structure in the model's own format (unit ids), for revision requests."""
    id_of = {json.dumps(u.locator, sort_keys=True): u.id for u in units}

    def ids(locators):
        return [i for i in (id_of.get(json.dumps(loc, sort_keys=True)) for loc in locators) if i]

    screens = [
        {
            "number": s["number"],
            "title": s["title"],
            "topic": s["topic"],
            "points": [{"text": p["text"], "sources": ids(p["sources"])} for p in s["points"]],
            "question_idea": s["question_idea"],
            "illustration": s["illustration"],
            "decision_diagram": s["decision_diagram"],
        }
        for s in structure["screens"]
    ]
    return json.dumps({"screens": screens, "motifs": structure["motifs"]}, ensure_ascii=False, indent=1)

"""The lesson format: Pydantic models matching assets/lesson.schema.json, plus rules that
JSON Schema cannot express, and validate_lesson() which runs both.

Rules checked here (besides the structure):
  - multiple choice: correct_index within the options (also in the alternate), distinct
    options, and the wrong-answer feedback does not reveal the correct option;
  - decision diagram: edges connect existing nodes, and there is a start node;
  - illustration files stay inside assets/illustrations/;
  - ids of screens and questions are unique;
  - compliance lessons: no "did you know?" enrichment, and at most 80 words of text per screen;
  - the palette meets WCAG AA contrast (4.5:1) for every text/background pair the player uses.
"""

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

SCHEMA_PATH = Path(__file__).parent / "assets" / "lesson.schema.json"
COMPLIANCE_MAX_WORDS = 80
AA_RATIO = 4.5

# Default palettes by lesson type (SPEC: training white with one moderate color; compliance
# dark and restrained). All text/background pairs below pass WCAG AA.
PALETTES = {
    "training": {
        "background": "#FFFFFF", "surface": "#F3F6F9", "text": "#1F2933", "muted": "#4B5866",
        "primary": "#1D63A0", "primary_text": "#FFFFFF", "accent": "#256B5C",
        "highlight": "#FFF3BF", "success": "#1B6E3F", "error": "#A9231A", "border": "#C5CED8",
    },
    "compliance": {
        "background": "#14181F", "surface": "#1D232C", "text": "#E7EAEE", "muted": "#AEB6C2",
        "primary": "#86AEDD", "primary_text": "#0F141A", "accent": "#A9BFD3",
        "highlight": "#3B3A26", "success": "#86CFA3", "error": "#F2968D", "border": "#3A4452",
    },
}

# (foreground, background) pairs the player draws text with.
CONTRAST_PAIRS = [
    ("text", "background"), ("text", "surface"), ("muted", "background"), ("muted", "surface"),
    ("primary_text", "primary"), ("primary", "background"), ("primary", "surface"),
    ("accent", "background"), ("accent", "surface"), ("success", "surface"), ("error", "surface"),
    ("text", "highlight"),
]


class LessonValidationError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("invalid lesson:\n" + "\n".join(f"- {e}" for e in errors))


# ---- models -----------------------------------------------------------------------------------
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


Text = Annotated[str, Field(min_length=2)]


class Locator(_Strict):
    file: str = Field(min_length=1)
    slide: int | None = Field(default=None, ge=1)
    page: int | None = Field(default=None, ge=1)
    paragraph: int | None = Field(default=None, ge=1)
    paragraph_end: int | None = Field(default=None, ge=1)
    url: str | None = Field(default=None, pattern=r"^https?://")
    title: str | None = None


Sources = Annotated[list[Locator], Field(min_length=1)]


class Palette(_Strict):
    background: str
    surface: str
    text: str
    muted: str
    primary: str
    primary_text: str
    accent: str
    highlight: str
    success: str
    error: str
    border: str


class Motif(_Strict):
    description: str = ""
    svg: str = ""


class Paragraph(_Strict):
    type: Literal["paragraph"]
    text: Text
    sources: Sources


class ListBlock(_Strict):
    type: Literal["list"]
    ordered: bool = False
    items: Annotated[list[Text], Field(min_length=2)]
    sources: Sources


class DidYouKnow(_Strict):
    type: Literal["did_you_know"]
    text: Text
    source_line: Text
    source_url: str | None = Field(default=None, pattern=r"^https?://")
    sources: Sources


class StopAndThink(_Strict):
    type: Literal["stop_and_think"]
    prediction_question: Text
    explanation: Text
    sources: Sources


def _normalized(text: str) -> str:
    return re.sub(r"\W+", " ", text).strip().lower()


class Question(_Strict):
    question: Text
    options: Annotated[list[Text], Field(min_length=3, max_length=4)]
    correct_index: int = Field(ge=0, le=3)
    feedback_correct: Text
    feedback_incorrect: Text

    @model_validator(mode="after")
    def _answer_rules(self):
        if self.correct_index >= len(self.options):
            raise ValueError(f"correct_index {self.correct_index} is out of range for {len(self.options)} options")
        if len({_normalized(o) for o in self.options}) != len(self.options):
            raise ValueError("options must be distinct")
        correct = _normalized(self.options[self.correct_index])
        if correct and correct in _normalized(self.feedback_incorrect):
            raise ValueError("feedback_incorrect reveals the correct answer; it must stay general")
        return self


class MultipleChoice(Question):
    type: Literal["multiple_choice"]
    id: str = Field(pattern=r"^[a-z0-9_-]+$")
    alternate: Question
    sources: Sources


class Illustration(_Strict):
    type: Literal["illustration"]
    file: str = Field(pattern=r"^assets/illustrations/[A-Za-z0-9_-]+\.html$")
    title: Text
    description: Text
    height: int | None = Field(default=None, ge=120, le=900)


class DiagramNode(_Strict):
    id: str = Field(pattern=r"^[a-z0-9_-]+$")
    label: Text
    kind: Literal["question", "action"]


class DiagramEdge(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    from_: str = Field(alias="from")
    to: str
    label: str | None = None


class DecisionDiagram(_Strict):
    type: Literal["decision_diagram"]
    title: Text
    nodes: Annotated[list[DiagramNode], Field(min_length=3, max_length=5)]
    edges: Annotated[list[DiagramEdge], Field(min_length=2)]
    sources: Sources

    @model_validator(mode="after")
    def _graph_rules(self):
        ids = [n.id for n in self.nodes]
        if len(set(ids)) != len(ids):
            raise ValueError("node ids must be unique")
        for e in self.edges:
            for end in (e.from_, e.to):
                if end not in ids:
                    raise ValueError(f"edge {e.from_!r} -> {e.to!r} uses unknown node {end!r}")
        targets = {e.to for e in self.edges}
        if not any(i not in targets for i in ids):
            raise ValueError("the diagram needs a start node (a node no edge points to)")
        return self


Block = Annotated[
    Union[Paragraph, ListBlock, DidYouKnow, StopAndThink, MultipleChoice, Illustration, DecisionDiagram],
    Field(discriminator="type"),
]


class Screen(_Strict):
    id: str = Field(pattern=r"^[a-z0-9_-]+$")
    title: Text
    topic: str | None = None
    blocks: Annotated[list[Block], Field(min_length=1)]


class Summary(_Strict):
    points: Annotated[list[Text], Field(min_length=1)]


class Lesson(_Strict):
    schema_version: Literal[1]
    title: Annotated[str, Field(min_length=2, max_length=120)]
    lesson_type: Literal["training", "compliance"]
    palette: Palette
    motif: Motif
    passing_score: float | None = Field(default=None, ge=0, le=1)
    screens: Annotated[list[Screen], Field(min_length=1)]
    summary: Summary
    certificate: bool

    @model_validator(mode="after")
    def _lesson_rules(self):
        problems = []
        screen_ids = [s.id for s in self.screens]
        if len(set(screen_ids)) != len(screen_ids):
            problems.append("screen ids must be unique")
        question_ids = [b.id for s in self.screens for b in s.blocks if isinstance(b, MultipleChoice)]
        if len(set(question_ids)) != len(question_ids):
            problems.append("question ids must be unique")
        if self.lesson_type == "compliance":
            for i, s in enumerate(self.screens):
                if any(isinstance(b, DidYouKnow) for b in s.blocks):
                    problems.append(f"screens[{i}] ({s.id}): compliance lessons have no did_you_know blocks")
                words = screen_word_count(s)
                if words > COMPLIANCE_MAX_WORDS:
                    problems.append(
                        f"screens[{i}] ({s.id}): {words} words; a compliance screen allows at most "
                        f"{COMPLIANCE_MAX_WORDS}"
                    )
        problems += contrast_problems(self.palette.model_dump())
        if problems:
            raise ValueError("; ".join(problems))
        return self


def screen_word_count(screen: Screen) -> int:
    """Words of explanatory text on a screen (questions and diagrams excluded)."""
    texts: list[str] = []
    for b in screen.blocks:
        if isinstance(b, (Paragraph, DidYouKnow)):
            texts.append(b.text)
        elif isinstance(b, ListBlock):
            texts += b.items
        elif isinstance(b, StopAndThink):
            texts += [b.prediction_question, b.explanation]
    return sum(len(t.split()) for t in texts)


# ---- contrast -----------------------------------------------------------------------------------
def _luminance(hex_color: str) -> float:
    channels = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(a: str, b: str) -> float:
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def contrast_problems(palette: dict) -> list[str]:
    problems = []
    for fg, bg in CONTRAST_PAIRS:
        try:
            ratio = contrast_ratio(palette[fg], palette[bg])
        except (KeyError, ValueError, TypeError):
            continue  # malformed colors are reported by the schema
        if ratio < AA_RATIO:
            problems.append(f"palette: {fg} on {bg} has contrast {ratio:.2f}:1, below WCAG AA {AA_RATIO}:1")
    return problems


# ---- validation ---------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def lesson_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _path(parts) -> str:
    out = ""
    for p in parts:
        out += f"[{p}]" if isinstance(p, int) else (f".{p}" if out else str(p))
    return out or "(lesson)"


def schema_errors(data) -> list[str]:
    """JSON Schema errors as "path: message" lines, most specific first."""
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(lesson_schema())
    errors = sorted(validator.iter_errors(data), key=lambda e: (len(e.absolute_path), list(map(str, e.absolute_path))))
    lines = []
    for e in errors:
        if e.validator in ("allOf", "if"):  # wrappers around the specific block errors
            continue
        lines.append(f"{_path(e.absolute_path)}: {e.message}")
    return list(dict.fromkeys(lines))


def model_errors(data) -> list[str]:
    """Pydantic errors (structure + cross-field rules) as "path: message" lines."""
    try:
        Lesson.model_validate(data)
    except ValidationError as e:
        lines = []
        for err in e.errors(include_url=False):
            loc = [p for p in err["loc"] if p not in (
                "paragraph", "list", "did_you_know", "stop_and_think", "multiple_choice",
                "illustration", "decision_diagram")]
            message = err["msg"].removeprefix("Value error, ")
            lines.append(f"{_path(loc)}: {message}")
        return list(dict.fromkeys(lines))
    return []


def validate_lesson(data) -> list[str]:
    """All problems with a lesson (empty list = valid). The JSON Schema runs first; the
    Pydantic rules run only on structurally valid data, so each problem is reported once."""
    errors = schema_errors(data)
    return errors or model_errors(data)


def load_lesson(data) -> Lesson:
    errors = validate_lesson(data)
    if errors:
        raise LessonValidationError(errors)
    return Lesson.model_validate(data)

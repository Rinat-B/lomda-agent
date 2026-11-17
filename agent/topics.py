"""Topic map: group source chunks into units, prompt the model, validate its JSON with Pydantic."""

import json
import re
from dataclasses import dataclass

from pydantic import BaseModel, Field, ValidationError, field_validator

MIN_TOPICS = 4
MAX_TOPICS = 8
# Rough character budget for the source text sent to the model (about 8k tokens).
SOURCE_BUDGET_CHARS = 24_000
MIN_UNIT_CHARS = 120
PARAGRAPHS_PER_UNIT = 8  # paragraph-based files without headings are grouped in blocks


# ---- source units -----------------------------------------------------------------------
@dataclass
class Unit:
    """A slide, a page, or a block of paragraphs: what a topic can cite."""

    id: str  # "U1", "U2", ... (short, so the model can cite it reliably)
    locator: dict  # {"file", "slide"} | {"file", "page"} | {"file", "paragraph", "paragraph_end"}
    text: str


def build_units(chunks: list[dict]) -> list[Unit]:
    """Group chunks by file and slide/page; paragraph files by heading (or blocks of 8)."""
    groups: list[tuple[dict, list[str]]] = []
    for chunk in chunks:
        loc = chunk["locator"]
        if "slide" in loc or "page" in loc:
            key = "slide" if "slide" in loc else "page"
            unit_loc = {"file": loc["file"], key: loc[key]}
            if groups and groups[-1][0] == unit_loc:
                groups[-1][1].append(chunk["text"])
            else:
                groups.append((unit_loc, [chunk["text"]]))
            continue
        # Paragraph-based (docx/txt/md): start a new unit at a heading, a new file,
        # or when the current block is full.
        last = groups[-1] if groups else None
        same_file = last is not None and last[0]["file"] == loc["file"] and "paragraph" in last[0]
        if (
            not same_file
            or chunk.get("kind") == "heading"
            or len(last[1]) >= PARAGRAPHS_PER_UNIT
        ):
            groups.append(({"file": loc["file"], "paragraph": loc["paragraph"],
                            "paragraph_end": loc["paragraph"]}, [chunk["text"]]))
        else:
            last[0]["paragraph_end"] = loc["paragraph"]
            last[1].append(chunk["text"])
    units = []
    for i, (loc, texts) in enumerate(groups, 1):
        if loc.get("paragraph_end") == loc.get("paragraph"):
            loc.pop("paragraph_end", None)
        units.append(Unit(f"U{i}", loc, "\n".join(texts)))
    return units


def fit_to_budget(units: list[Unit], budget: int = SOURCE_BUDGET_CHARS) -> list[Unit]:
    """Shorten unit texts so the total fits the budget. Every file keeps an equal share
    and every unit keeps at least MIN_UNIT_CHARS: units are trimmed, never dropped."""
    total = sum(len(u.text) for u in units)
    if total <= budget:
        return units
    files = list(dict.fromkeys(u.locator["file"] for u in units))
    per_file = budget // len(files)
    shortened = []
    for f in files:
        file_units = [u for u in units if u.locator["file"] == f]
        per_unit = max(MIN_UNIT_CHARS, per_file // len(file_units))
        for u in file_units:
            text = u.text if len(u.text) <= per_unit else u.text[: per_unit - 3].rstrip() + "..."
            shortened.append(Unit(u.id, u.locator, text))
    order = {u.id: i for i, u in enumerate(units)}
    return sorted(shortened, key=lambda u: order[u.id])


def describe_unit(unit: Unit) -> str:
    loc = unit.locator
    if "slide" in loc:
        where = f"slide {loc['slide']}"
    elif "page" in loc:
        where = f"page {loc['page']}"
    elif "paragraph_end" in loc:
        where = f"paragraphs {loc['paragraph']}-{loc['paragraph_end']}"
    else:
        where = f"paragraph {loc['paragraph']}"
    return f"{loc['file']}, {where}"


def format_units(units: list[Unit]) -> str:
    return "\n\n".join(f"[{u.id}] ({describe_unit(u)})\n{u.text}" for u in units)


# ---- structured output ------------------------------------------------------------------
class TopicOut(BaseModel):
    name: str = Field(min_length=2)
    objective: str = Field(min_length=5)
    sources: list[str] = Field(min_length=1)


class TopicMapOut(BaseModel):
    topics: list[TopicOut] = Field(min_length=MIN_TOPICS, max_length=MAX_TOPICS)

    @field_validator("topics")
    @classmethod
    def _unique_names(cls, topics: list[TopicOut]) -> list[TopicOut]:
        names = [t.name.strip() for t in topics]
        if len(set(names)) != len(names):
            raise ValueError("topic names must be unique")
        return topics


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_topic_map(text: str, units: list[Unit], objective_prefix: str) -> list[dict]:
    """Validate the model's JSON and map unit ids to locators. Raises ValueError if invalid."""
    match = _JSON_BLOCK.search(text)
    if not match:
        raise ValueError("no JSON object found in the reply")
    try:
        parsed = TopicMapOut.model_validate_json(match.group(0))
    except ValidationError as e:
        raise ValueError(f"JSON does not match the schema: {e.errors(include_url=False)}") from e

    by_id = {u.id: u for u in units}
    topics = []
    for t in parsed.topics:
        ids = list(dict.fromkeys(s.strip() for s in t.sources))
        unknown = [s for s in ids if s not in by_id]
        if unknown:
            raise ValueError(f"topic {t.name!r} cites unknown source ids {unknown}; use only the given ids")
        objective = t.objective.strip()
        if not objective.startswith(objective_prefix):
            raise ValueError(f"objective of topic {t.name!r} must start with {objective_prefix!r}")
        topics.append({
            "name": t.name.strip(),
            "objective": objective,
            "sources": [dict(by_id[s].locator) for s in ids],
        })
    return topics


def topic_map_json(topics: list[dict], units: list[Unit]) -> str:
    """A previous map in the model's own format (unit ids), for revision requests."""
    id_of = {json.dumps(u.locator, sort_keys=True): u.id for u in units}
    out = []
    for t in topics:
        ids = [id_of.get(json.dumps(loc, sort_keys=True)) for loc in t["sources"]]
        out.append({"name": t["name"], "objective": t["objective"], "sources": [i for i in ids if i]})
    return json.dumps({"topics": out}, ensure_ascii=False, indent=1)

"""The lesson's visual motif: a 4-part contract, two candidates from two models, Lucide icons.

The contract (what a model returns; validated with Pydantic):
  1. idea: one Hebrew sentence, what the image says about the lesson;
  2. icons: one or two names from the bundled Lucide subset (agent/assets/lucide/icons.json);
  3. accent: the palette colour of the drawing, "primary" or "accent" (colours come only from the
     lesson type's palette, so contrast and the dark compliance theme hold);
  4. layout: "single" (one icon), "pair" (two side by side) or "badge" (one icon in a circle).

No model writes SVG: the code draws the motif from the real icon paths, and the result still goes
through the SVG cleaner (agent/svg_clean.py), as everything does before it enters lesson.json.
Two candidates are asked for in parallel, each from a different writer model (another family when
the list has one); the content owner picks one at gate 2 (Enter = the first).
"""

import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from agent.messages import msg
from agent.svg_clean import clean_svg

ICONS_PATH = Path(__file__).parent / "assets" / "lucide" / "icons.json"
CANDIDATES = 2
LAYOUTS = ("single", "pair", "badge")

PROMPT = (
    "You design the visual motif of a Hebrew e-learning lesson: a small emblem shown at the top of "
    "every screen and on the certificate. It is built only from icons of this list (Lucide): {icons}.\n"
    "Choose what the emblem says about the lesson: its main subject or its main decision, not "
    "decoration. Use the planned motif ideas below if they fit.\n"
    "Reply with only a JSON object with exactly these four parts:\n"
    '{{"idea": "<one Hebrew sentence: what the emblem says>", "icons": ["<name>"] (one or two names '
    'from the list), "accent": "primary" or "accent", "layout": "single", "pair" (two icons) or "badge" '
    '(one icon in a circle)}}'
)


@lru_cache(maxsize=1)
def icons() -> dict[str, str]:
    return json.loads(ICONS_PATH.read_text(encoding="utf-8"))["icons"]


class MotifContract(BaseModel):
    idea: str = Field(min_length=4, max_length=200)
    icons: list[str] = Field(min_length=1, max_length=2)
    accent: Literal["primary", "accent"] = "primary"
    layout: Literal["single", "pair", "badge"] = "badge"

    @field_validator("icons")
    @classmethod
    def _known(cls, value: list[str]) -> list[str]:
        unknown = [v for v in value if v not in icons()]
        if unknown:
            raise ValueError(f"unknown icons {unknown}; use only names from the list")
        return value


def parse_contract(text: str) -> MotifContract:
    from agent.checkers import _json_object

    contract = MotifContract.model_validate(_json_object(text, "icons"))
    if contract.layout == "pair" and len(contract.icons) < 2:
        contract = contract.model_copy(update={"layout": "badge"})
    if contract.layout != "pair":
        contract = contract.model_copy(update={"icons": contract.icons[:1]})
    return contract


def _icon(name: str, x: float, y: float, size: float, colour: str) -> str:
    scale = size / 24
    return (f'<g transform="translate({x:g} {y:g}) scale({scale:g})" fill="none" stroke="{colour}" '
            f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{icons()[name]}</g>')


def render(contract: MotifContract, palette: dict) -> str:
    """The motif SVG (48 x 48, or 96 x 48 for a pair), cleaned."""
    colour = palette[contract.accent]
    if contract.layout == "pair":
        width, body = 96, _icon(contract.icons[0], 6, 6, 36, colour) + _icon(contract.icons[1], 54, 6, 36, colour)
    elif contract.layout == "badge":
        width = 48
        body = (f'<circle cx="24" cy="24" r="22" fill="{palette["surface"]}" stroke="{colour}" stroke-width="2"/>'
                + _icon(contract.icons[0], 11, 11, 26, colour))
    else:
        width, body = 48, _icon(contract.icons[0], 4, 4, 40, colour)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} 48" width="{width}" height="48" '
           f'role="img" aria-hidden="true">{body}</svg>')
    return clean_svg(svg)[0]


def candidate_models(writers: list[str], count: int = CANDIDATES) -> list[str]:
    """Writer models for the candidates: different families first, then the rest of the list."""
    from agent.checkers import family

    picked: list[str] = []
    for m in writers:
        if family(m) not in {family(p) for p in picked}:
            picked.append(m)
    picked += [m for m in writers if m not in picked]
    return picked[:count]


def messages(title: str, motif_ideas: list[str], lesson_type: str):
    from langchain_core.messages import HumanMessage, SystemMessage

    system = PROMPT.format(icons=", ".join(sorted(icons())))
    ideas = "\n".join(f"- {m}" for m in motif_ideas) or "(none)"
    return [SystemMessage(system), HumanMessage(f"Lesson type: {lesson_type}\nSubject: {title}\n\nPlanned motif ideas:\n{ideas}")]


def candidates(title: str, motif_ideas: list[str], lesson_type: str, palette: dict, call, writers: list[str]) -> tuple[list[dict], dict]:
    """([{"description", "svg", "icons", "layout", "model"}], {model: error}) from parallel calls,
    one per model; an invalid reply gets one retry with the error."""
    from langchain_core.messages import AIMessage, HumanMessage

    def one(model: str) -> dict:
        prompt = messages(title, motif_ideas, lesson_type)
        error = ""
        for _ in range(2):
            result = call(prompt, model)
            try:
                contract = parse_contract(result.text)
            except ValueError as e:
                error = str(e)
                prompt = prompt + [AIMessage(result.text), HumanMessage(f"That reply was invalid: {error}. Reply again with only the JSON object.")]
                continue
            return {"description": contract.idea, "svg": render(contract, palette), "icons": contract.icons,
                    "layout": contract.layout, "accent": contract.accent, "model": result.model}
        raise ValueError(error)

    models = candidate_models(writers)
    out, failed = [], {}
    with ThreadPoolExecutor(max_workers=len(models) or 1) as pool:
        futures = {m: pool.submit(one, m) for m in models}
    for model, future in futures.items():
        try:
            out.append(future.result())
        except Exception as e:  # noqa: BLE001 - the other candidate may still come
            failed[model] = f"{type(e).__name__}: {e}"
    return out, failed


def preview_html(cands: list[dict], palette: dict) -> str:
    """A small local page that shows the candidates side by side (gate 2)."""
    from html import escape

    cards = "".join(
        f'<figure style="margin:0;padding:16px;border:1px solid {palette["border"]};border-radius:10px;'
        f'background:{palette["surface"]}"><div>{c["svg"]}</div><figcaption style="margin-top:8px">'
        f'{i}. {escape(c["description"])}</figcaption></figure>' for i, c in enumerate(cands, 1))
    return (f'<!doctype html><html dir="rtl" lang="he"><head><meta charset="utf-8"><title>{escape(msg("motif.preview_title"))}</title></head>'
            f'<body style="font-family:Segoe UI,Arial,sans-serif;background:{palette["background"]};color:{palette["text"]};'
            f'padding:16px;display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(160px,1fr))">'
            f'{cards}</body></html>')

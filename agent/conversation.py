"""The conversational parts around the lesson flow: the opening message with an estimate, one
clarification question (audience and length) with defaults, and the agent's suggestions for the
next step. None of them skips a gate: the agent proposes and explains, the owner decides.

Settings in models.yaml (`conversation`): seconds per model call per profile (from the measured
runs), the default number of screens, and the threshold below which the material counts as thin.
"""

import math

from agent.messages import msg

DEFAULTS = {"seconds_per_call": {"free": 55, "balanced": 20}, "thin_material_chars": 2500,
            "default_screens_min": 5, "default_screens_max": 12, "wide_research_ideas": 5}


def settings() -> dict:
    from agent.llm import load_config

    return {**DEFAULTS, **(load_config().get("conversation") or {})}


def material_chars(chunks: list[dict]) -> int:
    return sum(len(c.get("text") or "") for c in chunks)


def default_screens(units: int) -> int:
    """About one screen per source unit plus an opening and practice, within the configured range."""
    s = settings()
    return max(s["default_screens_min"], min(s["default_screens_max"], units + 3))


def estimate(screens: int, lesson_type: str, profile: str) -> tuple[int, int]:
    """(about how many model calls, about how many minutes) for a whole lesson, from the measured
    runs: planning and research, writing (3-4 screens a call), illustrations with their checks, the
    first review and two fix rounds."""
    from agent.nodes import illustration_cap

    research = 7 if lesson_type == "training" else 0
    illustrations = min(illustration_cap(), max(1, screens // 3))
    calls = (6 + research + 2 + math.ceil(screens / 4) + illustrations          # up to the review
             + 8 + illustrations + 1                                           # the first review
             + 2 * (math.ceil(screens / 4) + 4))                               # two fix rounds
    seconds = settings()["seconds_per_call"].get(profile, 40)
    return calls, max(1, round(calls * seconds / 60))


def opening_message(screens: int, lesson_type: str, profile: str, units: int) -> str:
    calls, minutes = estimate(screens, lesson_type, profile)
    return msg("conversation.opening", type=msg(f"lesson_types.{lesson_type}"), units=units, screens=screens,
               calls=calls, low=max(1, round(minutes * 0.75)), high=round(minutes * 1.25), profile=profile)


def missing(requirements: dict | None, audience_in_source: str | None) -> list[str]:
    """What the clarification question asks about: the audience (if neither the request nor the
    material names one) and the length (if the request does not give a number of screens)."""
    req = requirements or {}
    out = []
    if not req.get("audience") and not audience_in_source:
        out.append("audience")
    if not req.get("screens") and not req.get("screens_about"):
        out.append("screens")
    return out


def is_thin(chunks: list[dict]) -> bool:
    return material_chars(chunks) < settings()["thin_material_chars"]


def thin_suggestion(lesson_type: str, chars: int) -> dict:
    """The suggestion when the material is thin: training can widen the web research or take
    another file; compliance only another file (its truth is the owner's material alone)."""
    key = "conversation.thin.training" if lesson_type == "training" else "conversation.thin.compliance"
    return {"when": "start", "text": msg(key, chars=chars), "reason": msg("conversation.thin.reason", chars=chars)}


def review_suggestion(open_count: int) -> dict | None:
    if not open_count:
        return None
    return {"when": "review", "text": msg("conversation.review_suggestion", count=open_count),
            "reason": msg("conversation.review_reason", count=open_count)}

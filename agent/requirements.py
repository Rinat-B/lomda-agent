"""The user's own requirements for the lesson, taken from the request and kept in the state.

classify_type already reads the request with the small model; it now also returns:
  - reason: why this lesson type (Hebrew, shown at the type gate and used to answer "why?");
  - requirements: explicit wishes in the request: screens (a number), audience, emphasis, style;
  - audience_in_source: the audience the material itself names, if any.

The requirements go to topic_map, plan_structure and the writer. A requirement that conflicts with
the source or the SPEC (fewer screens than topics, more than MAX_SCREENS) is not silently changed:
the plan uses the nearest possible value and gate 2 says so and offers an alternative.
"""

from agent.structure import MAX_SCREENS
from agent.topics import MIN_TOPICS

KEYS = ("screens", "audience", "emphasis", "style")


def _text(value) -> str | None:
    value = " ".join(str(value or "").split())
    return value if value and value.lower() not in ("null", "none", "-", "n/a") else None


def parse_requirements(data: dict) -> dict:
    """{"screens": int | None, "audience", "emphasis", "style": str | None} from the model's JSON."""
    raw = data.get("requirements") if isinstance(data.get("requirements"), dict) else {}
    out: dict = {k: _text(raw.get(k)) for k in ("audience", "emphasis", "style")}
    try:
        screens = int(str(raw.get("screens")).strip())
        out["screens"] = screens if screens > 0 else None
    except (TypeError, ValueError):
        out["screens"] = None
    return out


def merge(base: dict | None, new: dict | None) -> dict:
    """The requirements with every non-empty value of `new` (an answer to the clarification) on top."""
    out = {k: None for k in KEYS} | dict(base or {})
    for k, v in (new or {}).items():
        if v not in (None, ""):
            out[k] = v
    return out


def prompt_lines(req: dict | None) -> str:
    """The requirements for a model prompt (English labels); empty when there are none."""
    req = req or {}
    lines = []
    if req.get("screens"):
        lines.append(f"- number of screens: {req['screens']}")
    for key, label in (("audience", "target audience"), ("emphasis", "emphasis"), ("style", "style")):
        if req.get(key):
            lines.append(f"- {label}: {req[key]}")
    if not lines:
        return ""
    return ("The content owner's requirements (follow them; they shape the lesson, they never add facts "
            "that are not in the source):\n" + "\n".join(lines))


def screens_target(req: dict | None, topic_count: int) -> tuple[int | None, dict | None]:
    """(the number of screens to plan, a note for gate 2 when the request cannot be met as asked).
    Every topic gets a screen (SPEC), and a lesson has at most MAX_SCREENS screens."""
    asked = (req or {}).get("screens")
    if not asked:
        return None, None
    if asked < topic_count:
        return topic_count, {"requested": asked, "used": topic_count, "why": "topics", "topics": topic_count}
    if asked > MAX_SCREENS:
        return MAX_SCREENS, {"requested": asked, "used": MAX_SCREENS, "why": "max", "topics": topic_count}
    return asked, None


def topics_hint(req: dict | None) -> str:
    """For topic_map: a short lesson needs fewer topics (every topic gets a screen)."""
    asked = (req or {}).get("screens")
    if not asked:
        return ""
    if asked < MIN_TOPICS:
        return (f"The owner asked for {asked} screens, fewer than the minimum of {MIN_TOPICS} topics: plan "
                f"{MIN_TOPICS} topics, the fewest allowed; the reviewer will be told.")
    return f"The owner asked for {asked} screens and every topic gets at least one screen: plan at most {asked} topics."

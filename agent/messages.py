"""Load user-facing strings from agent/assets/messages_he.json by dotted key."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

MESSAGES_PATH = Path(__file__).parent / "assets" / "messages_he.json"


@lru_cache(maxsize=1)
def _load() -> dict[str, Any]:
    with MESSAGES_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def msg_or_none(key: str, **kwargs: Any) -> str | None:
    """Like msg(), but returns None for a missing key."""
    try:
        return msg(key, **kwargs)
    except KeyError:
        return None


def msg(key: str, **kwargs: Any) -> str:
    """Return the message for a dotted key (e.g. "cli.no_question"), formatted with kwargs."""
    node: Any = _load()
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"Missing message key: {key}")
        node = node[part]
    if not isinstance(node, str):
        raise KeyError(f"Message key is not a string: {key}")
    return node.format(**kwargs) if kwargs else node

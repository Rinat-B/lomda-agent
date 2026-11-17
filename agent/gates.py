"""Human-in-the-loop gate answers: parse "[Y/N + comment]" replies.

The accepted words (including the Hebrew-keyboard letters on the Y and N keys) live in
agent/assets/messages_he.json under gate.answers, so no Hebrew is kept in this file.
"""

import re
from dataclasses import dataclass
from typing import Callable

from agent.messages import msg, msg_list

_FIRST_WORD = re.compile(r"^\s*([^\s,:;.\-]+)[\s,:;.\-]*(.*)$", re.DOTALL)


@dataclass(frozen=True)
class GateDecision:
    approved: bool
    feedback: str = ""


def parse_decision(text: str) -> GateDecision | None:
    """Approve: Y/y/yes and their Hebrew equivalents. Change: N/n/no (+ comment).
    Any other non-empty text is a change request with that text as the comment.
    Returns None for empty input (ask again)."""
    text = (text or "").strip()
    if not text:
        return None
    approve = {w.lower() for w in msg_list("gate.answers.approve")}
    reject = {w.lower() for w in msg_list("gate.answers.reject")}
    match = _FIRST_WORD.match(text)
    first, rest = (match.group(1), match.group(2).strip()) if match else (text, "")
    if first.lower() in approve:
        return GateDecision(approved=True, feedback=rest)
    if first.lower() in reject:
        return GateDecision(approved=False, feedback=rest)
    return GateDecision(approved=False, feedback=text)


# ---- topics gate (CLI side) -------------------------------------------------------------
def format_topic_map(topics: list[dict]) -> str:
    """The topic map as Hebrew Markdown: topic, objective, sources."""
    from agent.nodes import source_label

    lines = [msg("gate.topics.title", count=len(topics)), ""]
    for i, t in enumerate(topics, 1):
        sources = ", ".join(dict.fromkeys(source_label(loc) for loc in t["sources"]))
        lines += [
            msg("gate.topics.topic", index=i, name=t["name"]),
            "",
            msg("gate.topics.objective", objective=t["objective"]),
            "",
            msg("gate.topics.sources", sources=sources),
            "",
        ]
    return "\n".join(lines).rstrip()


def ask_topics_gate(topics: list[dict], ask: Callable[[str], str], show: Callable[[str], None],
                    say: Callable[..., None]) -> dict:
    """Show the map and read the reviewer's decision. Returns the interrupt resume value."""
    show(format_topic_map(topics))
    while True:
        decision = parse_decision(ask(msg("gate.topics.ask")))
        if decision is None:
            say(msg("gate.topics.ask_again"), style="yellow")
            continue
        if decision.approved:
            return {"approved": True, "feedback": decision.feedback}
        feedback = decision.feedback
        while not feedback:
            feedback = ask(msg("gate.topics.ask_feedback")).strip()
        say(msg("gate.topics.revising"))
        return {"approved": False, "feedback": feedback}

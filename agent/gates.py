"""Human-in-the-loop gate answers: parse "[Y/N + comment]" replies.

The accepted words (including the Hebrew-keyboard letters on the Y and N keys) live in
agent/assets/messages_he.json under gate.answers, so no Hebrew is kept in this file.
"""

import re
from dataclasses import dataclass
from typing import Callable

from agent.messages import msg, msg_list

_FIRST_WORD = re.compile(r"^\s*([^\s,:;.\-]+)[\s,:;.\-]*(.*)$", re.DOTALL)
EXIT_WORDS = {"/quit", "quit", "exit", "/exit"}


class GateExit(Exception):
    """The user asked to quit while a gate was waiting. The conversation stays paused."""


@dataclass(frozen=True)
class GateDecision:
    approved: bool
    feedback: str = ""
    # True when the reply started with an approve/reject word (Y, N, ...): decided without
    # a model. False for free text, which the gate node classifies with a small model.
    explicit: bool = True


def parse_decision(text: str) -> GateDecision | None:
    """Approve: Y/y/yes and their Hebrew equivalents. Change: N/n/no (+ comment).
    Any other non-empty text is returned as free text (explicit=False) for the model to
    classify; until then it counts as a change request with the whole text as comment.
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
    return GateDecision(approved=False, feedback=text, explicit=False)


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


def format_structure(structure: dict) -> str:
    """The lesson structure as Hebrew Markdown, one block per screen (a list, not a table,
    so the console's Hebrew display stays intact)."""
    from agent.nodes import source_label

    type_label = msg(f"lesson_types.{structure['lesson_type']}")
    lines = [msg("gate.structure.title", type=type_label, count=len(structure["screens"])), ""]
    if structure["lesson_type"] == "compliance":
        from agent.structure import COMPLIANCE_MAX_WORDS

        lines += [msg("gate.structure.word_limit", max_words=COMPLIANCE_MAX_WORDS), ""]
    for s in structure["screens"]:
        lines += [
            msg("gate.structure.screen", number=s["number"], title=s["title"]),
            "",
            msg("gate.structure.topic", topic=s["topic"]),
            "",
            msg("gate.structure.objective", objective=s["objective"]),
            "",
            msg("gate.structure.points"),
            "",
        ]
        for p in s["points"]:
            sources = ", ".join(dict.fromkeys(source_label(loc) for loc in p["sources"]))
            lines.append(msg("gate.structure.point", text=p["text"], sources=sources))
        lines += [
            "",
            msg("gate.structure.question", question=s["question_idea"]),
            "",
            msg("gate.structure.illustration", illustration=s["illustration"] or msg("gate.structure.none")),
            "",
            msg("gate.structure.diagram", value=msg("gate.structure.yes" if s["decision_diagram"] else "gate.structure.no")),
            "",
        ]
    lines += [msg("gate.structure.motifs_title"), ""]
    lines += [msg("gate.structure.motif", index=i, text=m) for i, m in enumerate(structure["motifs"], 1)]
    return "\n".join(lines).rstrip()


def _read(ask: Callable[[str], str], prompt: str) -> str:
    text = ask(prompt)
    if text.strip().lower() in EXIT_WORDS:
        raise GateExit
    return text


def ask_type_gate(request: dict, ask: Callable[[str], str], say: Callable[..., None]) -> dict:
    """Short [Y/N] gate for the lesson type: Y keeps the proposal, N switches type."""
    proposed = msg(f"lesson_types.{request['lesson_type']}")
    while True:
        decision = parse_decision(_read(ask, msg("gate.type.ask", type=proposed)))
        if decision is not None and decision.explicit:
            return {"confirm": decision.approved}
        say(msg("gate.type.ask_again"), style="yellow")


def ask_review_gate(request: dict, ask: Callable[[str], str], show: Callable[[str], None],
                    say: Callable[..., None]) -> dict:
    """Show the reviewed item (or the answer to the last question) and read the reply.

    Returns the interrupt resume value: {"approved", "feedback"} for Y/N answers (decided
    here, no model), or {"text"} for free text (classified by the gate node).
    """
    gate = request["gate"]
    if request.get("answer"):
        show(request["answer"])  # the item is unchanged; it is already on screen
    elif gate == "structure":
        show(format_structure(request["structure"]))
    else:  # "topics"; older saved conversations used the key "topics"
        show(format_topic_map(request.get("topic_map") or request.get("topics")))
    while True:
        text = _read(ask, msg(f"gate.{gate}.ask"))
        decision = parse_decision(text)
        if decision is None:
            say(msg(f"gate.{gate}.ask_again"), style="yellow")
            continue
        if not decision.explicit:
            return {"text": text.strip()}
        if decision.approved:
            return {"approved": True, "feedback": decision.feedback}
        feedback = decision.feedback
        while not feedback:
            feedback = _read(ask, msg(f"gate.{gate}.ask_feedback")).strip()
        say(msg(f"gate.{gate}.revising"))
        return {"approved": False, "feedback": feedback}


def ask_gate(request: dict, ask: Callable[[str], str], show: Callable[[str], None],
             say: Callable[..., None]) -> dict:
    """Answer whichever gate the graph is waiting at."""
    if request.get("gate") == "type":
        return ask_type_gate(request, ask, say)
    return ask_review_gate(request, ask, show, say)

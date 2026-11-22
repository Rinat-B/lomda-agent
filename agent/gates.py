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


OPEN_FINDINGS_SHOWN = 8


def format_review(summary: dict) -> str:
    """Gate 3: the lesson's title, review counts, coverage and the open findings."""
    lines = [
        msg("gate.review.title", title=summary["title"]), "",
        msg("gate.review.summary", type=msg(f"lesson_types.{summary['lesson_type']}"), screens=summary["screens"],
            rounds=summary["rounds"], found=summary["found"], fixed=summary["fixed"], open=len(summary["open"])),
    ]
    cov = summary.get("coverage")
    if cov:
        lines.append(msg("gate.review.coverage", taught=cov["taught"], total=cov["total"], percent=cov["percent"]))
        for point in cov["missing"][:OPEN_FINDINGS_SHOWN]:
            lines.append(msg("gate.review.missing_item", point=point))
    if summary.get("budget_stop"):
        lines.append(msg("gate.review.budget_stop", **summary["budget_stop"]))
    lines += ["", msg("gate.review.open_title"), ""]
    open_ = summary["open"]
    if not open_:
        lines.append(msg("gate.review.no_open"))
    for f in open_[:OPEN_FINDINGS_SHOWN]:
        problem = " ".join(str(f.get("problem") or "").split())
        if f.get("screen"):
            lines.append(msg("gate.review.open_item", screen=f["screen"], problem=problem))
        else:
            lines.append(msg("gate.review.open_lesson_item", problem=problem))
    if len(open_) > OPEN_FINDINGS_SHOWN:
        lines.append(msg("gate.review.open_more", count=len(open_) - OPEN_FINDINGS_SHOWN))
    return "\n".join(lines)


def format_unverified(facts: list[dict]) -> str:
    """Gate 2: web facts without a verified source, numbered, with the warning."""
    lines = [msg("gate.structure.unverified_title"), "", msg("gate.structure.unverified_note"), ""]
    lines += [msg("gate.structure.unverified_item", number=i, fact=f["fact"], reason=f.get("reason") or "-")
              for i, f in enumerate(facts, 1)]
    return "\n".join(lines)


def ask_accepted_facts(facts: list[dict], ask: Callable[[str], str], say: Callable[..., None]) -> list[str]:
    """The unverified facts the owner marks to include: numbers, or Enter for none (after three
    answers that are neither, none)."""
    for _ in range(3):
        text = _read(ask, msg("gate.structure.unverified_ask")).strip()
        if not text:
            return []
        numbers = [n.strip() for n in re.split(r"[,\s]+", text) if n.strip()]
        if all(n.isdigit() and 1 <= int(n) <= len(facts) for n in numbers):
            return [facts[int(n) - 1]["id"] for n in dict.fromkeys(numbers)]
        say(msg("gate.structure.unverified_again"), style="yellow")
    return []


def ask_review_gate(request: dict, ask: Callable[[str], str], show: Callable[[str], None],
                    say: Callable[..., None], preview: Callable[[dict], None] | None = None) -> dict:
    """Show the reviewed item (or the answer to the last question) and read the reply.

    Returns the interrupt resume value: {"approved", "feedback"} for Y/N answers (decided
    here, no model), or {"text"} for free text (classified by the gate node).
    """
    gate = request["gate"]
    if request.get("answer"):
        show(request["answer"])  # the item is unchanged; it is already on screen
    elif gate == "structure":
        show(format_structure(request["structure"]))
        if request.get("unverified"):
            show(format_unverified(request["unverified"]))
    elif gate == "review":
        if preview is not None:
            preview(request["lesson"], request.get("illustrations"))
            say(msg("gate.review.preview"))
        show(format_review(request["summary"]))
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
            reply = {"approved": True, "feedback": decision.feedback}
            if request.get("unverified") and not request.get("answer"):
                reply["accepted_facts"] = ask_accepted_facts(request["unverified"], ask, say)
            return reply
        feedback = decision.feedback
        while not feedback:
            feedback = _read(ask, msg(f"gate.{gate}.ask_feedback")).strip()
        say(msg(f"gate.{gate}.revising"))
        return {"approved": False, "feedback": feedback}


def display_folder(folder: str) -> str:
    """The output folder as the user knows it: relative to the program's folder."""
    from pathlib import Path

    from agent.persistence import output_dir

    path = Path(folder)
    try:
        return str(path.relative_to(output_dir().parent))
    except ValueError:
        return str(path)


def ask_write_files_gate(request: dict, ask: Callable[[str], str], say: Callable[..., None],
                         say_path: Callable[..., None] | None = None) -> dict:
    """[Y/N] before writing the lesson folder: the target, the files, a warning if it exists."""
    path_line = say_path or say
    say(msg("gate.write_files.title"))
    path_line(display_folder(request["folder"]))
    for name in request.get("files", []):
        path_line(msg("gate.write_files.file", name=name))
    if request.get("exists"):
        say(msg("gate.write_files.exists"), style="yellow")
    while True:
        decision = parse_decision(_read(ask, msg("gate.write_files.ask")))
        if decision is not None and decision.explicit:
            return {"confirm": decision.approved}
        say(msg("gate.write_files.ask_again"), style="yellow")


def ask_browser_check_gate(request: dict, ask: Callable[[str], str], say: Callable[..., None],
                           say_path: Callable[..., None] | None = None) -> dict:
    """[Y/N] before running the generated illustrations in the browser (Code-Exec)."""
    path_line = say_path or say
    say(msg("gate.browser_check.title"))
    for name in request.get("files", []):
        path_line(msg("gate.browser_check.file", name=name))
    while True:
        decision = parse_decision(_read(ask, msg("gate.browser_check.ask", count=len(request.get("files", [])))))
        if decision is not None and decision.explicit:
            return {"confirm": decision.approved}
        say(msg("gate.browser_check.ask_again"), style="yellow")


def ask_gate(request: dict, ask: Callable[[str], str], show: Callable[[str], None],
             say: Callable[..., None], preview: Callable[[dict], None] | None = None,
             say_path: Callable[..., None] | None = None) -> dict:
    """Answer whichever gate the graph is waiting at."""
    if request.get("gate") == "type":
        return ask_type_gate(request, ask, say)
    if request.get("gate") in ("write_files", "save"):  # "save": conversations paused before day 6
        return ask_write_files_gate(request, ask, say, say_path)
    if request.get("gate") == "browser_check":
        return ask_browser_check_gate(request, ask, say, say_path)
    return ask_review_gate(request, ask, show, say, preview)

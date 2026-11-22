"""/summary [path]: the conversation's summary as a local text file (after [Y/N]).

What it holds: the date, the source files, every request with what came of it (an answer, a lesson
and where it was saved, or a stop), and the cost by node and by model. All text is Hebrew, from
messages_he.json. The history lives in the session (Session.history) for the current program run.
"""

from datetime import datetime
from pathlib import Path

from agent.messages import msg

ANSWER_CHARS = 400


def default_path(base: Path) -> Path:
    return base / f"summary-{datetime.now():%Y-%m-%d-%H%M}.txt"


def target_path(arg: str, base: Path) -> Path:
    """The file to write: the given path, a file inside a given folder, or output/summary-<time>.txt."""
    raw = arg.strip().strip('"').strip("'").strip()
    if not raw:
        return default_path(base)
    path = Path(raw).expanduser()
    if path.is_dir() or raw.endswith(("/", "\\")):
        return default_path(path)
    return path if path.suffix else path.with_suffix(".txt")


def build_summary(session) -> str:
    lines = [msg("summary.title"), msg("summary.date", date=datetime.now().strftime("%d.%m.%Y %H:%M")), ""]
    lines.append(msg("summary.files_title"))
    lines += [msg("summary.file", name=p.name, path=str(p)) for p in session.files] or [msg("summary.none")]
    lines += ["", msg("summary.turns_title")]
    if not session.history:
        lines.append(msg("summary.none"))
    for i, item in enumerate(session.history, 1):
        kind = item.get("kind")
        if kind == "question":
            lines += ["", msg("summary.question", index=i, text=" ".join(item["text"].split()))]
        elif kind == "answer":
            text = " ".join(item["text"].split())
            lines.append(msg("summary.answer", text=text[:ANSWER_CHARS] + ("..." if len(text) > ANSWER_CHARS else "")))
        elif kind == "lesson":
            lines.append(msg("summary.lesson", title=item.get("title") or "-", path=item.get("path") or "-"))
            if item.get("report"):
                lines.append(msg("summary.report", path=item["report"]))
        elif kind == "stopped":
            lines.append(msg("summary.stopped"))
    totals = session.totals(session.usage)
    lines += ["", msg("summary.cost_title"),
              msg("summary.cost_total", calls=totals.calls, tokens=totals.total_tokens, cost=totals.cost)]
    for key in ("node", "model"):
        rows = session.totals_by(key)
        if rows:
            lines.append(msg(f"summary.by_{key}"))
            lines += [msg("summary.cost_item", name=name, calls=t.calls, tokens=t.total_tokens, cost=t.cost)
                      for name, t in rows.items()]
    return "\n".join(lines) + "\n"

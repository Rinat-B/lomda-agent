"""Local persistence: LangGraph checkpoints + conversation data, in one SQLite file.

data/checkpoints.sqlite lives next to agent.exe (or in the project root when run from
source). It holds:
  - LangGraph's own checkpoint tables (SqliteSaver): where each conversation's graph
    stopped, including a pending gate (interrupt).
  - a "conversations" table: the files added to each conversation, the chunks read from
    them, and the model usage, so /resume works after the program was closed without
    reading the files again.
"""

import json
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from agent.session import Session, SourceDoc, UsageRecord

DB_FILE_NAME = "checkpoints.sqlite"


def data_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "data"
    return Path(__file__).resolve().parent.parent / "data"


def db_path() -> Path:
    return data_dir() / DB_FILE_NAME


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(path), check_same_thread=False)


def open_checkpointer(path: Path | None = None):
    """A SqliteSaver on the conversation database (tables are created on first use)."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    saver = SqliteSaver(_connect(path or db_path()))
    saver.setup()
    return saver


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ConversationStore:
    """Files, chunks and usage per conversation (thread_id)."""

    def __init__(self, path: Path | None = None):
        self.conn = _connect(path or db_path())
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS conversations ("
            " thread_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,"
            " files TEXT NOT NULL, sources TEXT NOT NULL, usage TEXT NOT NULL)"
        )
        self.conn.commit()

    def save(self, session: Session) -> None:
        if not session.files and not session.usage:
            return  # nothing worth keeping yet
        files = json.dumps([str(p) for p in session.files], ensure_ascii=False)
        sources = json.dumps({str(p): asdict(s) for p, s in session.sources.items()}, ensure_ascii=False)
        usage = json.dumps([asdict(u) for u in session.usage], ensure_ascii=False)
        now = _now()
        self.conn.execute(
            "INSERT INTO conversations (thread_id, created_at, updated_at, files, sources, usage)"
            " VALUES (?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(thread_id) DO UPDATE SET updated_at=excluded.updated_at,"
            " files=excluded.files, sources=excluded.sources, usage=excluded.usage",
            (session.thread_id, now, now, files, sources, usage),
        )
        self.conn.commit()

    def load(self, thread_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT files, sources, usage, updated_at FROM conversations WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        if row is None:
            return None
        files, sources, usage, updated_at = row
        return {
            "files": json.loads(files),
            "sources": json.loads(sources),
            "usage": json.loads(usage),
            "updated_at": updated_at,
        }

    def restore(self, thread_id: str, session: Session) -> bool:
        """Replace the session's contents with a saved conversation (and rebuild the index)."""
        data = self.load(thread_id)
        session.reset()
        session.thread_id = thread_id
        if data is None:
            return False
        session.files = [Path(f) for f in data["files"]]
        session.sources = {Path(p): SourceDoc(**s) for p, s in data["sources"].items()}
        session.usage = [UsageRecord(**u) for u in data["usage"]]
        session.index.rebuild(session.all_chunks())
        return True


@dataclass
class PausedConversation:
    thread_id: str
    paused_at: datetime  # local time
    files: list[str]  # file names
    gate: str  # e.g. "topics"


def paused_conversations(graph, store: ConversationStore) -> list[PausedConversation]:
    """Conversations whose graph is waiting at a gate, newest first."""
    threads = {c.config["configurable"]["thread_id"] for c in graph.checkpointer.list(None)}
    paused = []
    for thread_id in threads:
        snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
        if not snapshot.interrupts:
            continue
        request = snapshot.interrupts[0].value
        gate = request.get("gate", "") if isinstance(request, dict) else ""
        when = datetime.fromisoformat(snapshot.created_at).astimezone() if snapshot.created_at else datetime.now()
        data = store.load(thread_id) or {"files": []}
        paused.append(PausedConversation(thread_id, when, [Path(f).name for f in data["files"]], gate))
    return sorted(paused, key=lambda p: p.paused_at, reverse=True)

"""Per-conversation state: added source files and model usage."""

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from agent.llm import LLMResult
from agent.tools import ChunkIndex

SUPPORTED_EXTENSIONS = (".pptx", ".docx", ".pdf", ".txt", ".md")


@dataclass
class UsageRecord:
    node: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost: float

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class UsageTotals:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, r: UsageRecord) -> None:
        self.calls += 1
        self.prompt_tokens += r.prompt_tokens
        self.completion_tokens += r.completion_tokens
        self.cost += r.cost


@dataclass
class SourceDoc:
    """A source file read by the source-reader MCP server."""

    unit_type: str  # "slide", "page" or "paragraph"
    units: int
    chunks: list[dict]  # each: {"text", "kind", "locator": {"file", "slide"/"page", "paragraph"}}


@dataclass
class Session:
    files: list[Path] = field(default_factory=list)
    sources: dict[Path, SourceDoc] = field(default_factory=dict)
    usage: list[UsageRecord] = field(default_factory=list)
    index: ChunkIndex = field(default_factory=ChunkIndex)
    # LangGraph checkpointer thread for this conversation; a new one per /new.
    thread_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    topic_map: list[dict] | None = None  # the approved topic map
    lesson_type: str | None = None  # training | compliance, once confirmed
    structure: dict | None = None  # the approved lesson structure
    run_start: int = 0  # index in usage where the current run began (report.md counts from here)

    def reset(self) -> None:
        self.files.clear()
        self.sources.clear()
        self.usage.clear()
        self.index.reset()
        self.thread_id = uuid.uuid4().hex
        self.topic_map = None
        self.lesson_type = None
        self.structure = None
        self.run_start = 0

    def all_chunks(self) -> list[dict]:
        return [chunk for source in self.sources.values() for chunk in source.chunks]

    def has_file(self, path: Path) -> bool:
        return path.resolve() in {f.resolve() for f in self.files}

    def add_file(self, path: Path, source: SourceDoc | None = None) -> None:
        resolved = path.resolve()
        self.files.append(resolved)
        if source is not None:
            self.sources[resolved] = source
            self.index.rebuild(self.all_chunks())

    def record_usage(self, node: str, result: LLMResult) -> None:
        self.usage.append(
            UsageRecord(node, result.model, result.prompt_tokens, result.completion_tokens, result.cost)
        )

    @staticmethod
    def totals(records: list[UsageRecord]) -> UsageTotals:
        t = UsageTotals()
        for r in records:
            t.add(r)
        return t

    def totals_by(self, key: str) -> dict[str, UsageTotals]:
        """Totals grouped by "node" or "model", in first-seen order."""
        groups: dict[str, UsageTotals] = defaultdict(UsageTotals)
        for r in self.usage:
            groups[getattr(r, key)].add(r)
        return dict(groups)

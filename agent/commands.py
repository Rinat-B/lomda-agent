"""CLI commands (help, /add, /files, /cost, /new, /quit) and file-path detection."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from agent.console import say, say_path, show_markdown, visual_prompt
from agent.errors import report_error
from agent.messages import msg, msg_or_none
from agent.session import SUPPORTED_EXTENSIONS, Session, SourceDoc, UsageTotals

SOURCE_READER = "source-reader"
READ_ERROR_CODES = {"not_found", "not_a_file", "unsupported", "encrypted", "corrupt", "empty"}

HELP_WORDS = {"help", "?", "/help"}
QUIT_WORDS = {"/quit", "quit", "exit", "/exit"}

_EXT = "|".join(e.lstrip(".") for e in SUPPORTED_EXTENSIONS)
# A quoted path (may contain spaces) or an unquoted token, ending in a supported extension.
_PATH_IN_TEXT = re.compile(
    rf"\"(?P<dq>[^\"]+\.(?:{_EXT}))\"|'(?P<sq>[^']+\.(?:{_EXT}))'|(?P<bare>[^\s\"'<>|,;()]+\.(?:{_EXT}))(?!\w)",
    re.IGNORECASE,
)
_DRIVE = re.compile(r"[A-Za-z]:[\\/]")
_HEBREW_PREFIX = re.compile(f"^[{chr(0x05D0)}-{chr(0x05EA)}]{{1,3}}-")


@dataclass
class TurnResult:
    quit: bool = False
    question: str | None = None


def format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _path_candidates(token: str) -> list[str]:
    """Ways to read a token as a path: as is, without a short Hebrew prefix + hyphen, from a drive letter."""
    candidates = [token]
    prefix = _HEBREW_PREFIX.match(token)
    if prefix:
        candidates.append(token[prefix.end():])
    drive = _DRIVE.search(token)
    if drive and drive.start() > 0:
        candidates.append(token[drive.start():])
    return candidates


def _looks_like_path(token: str) -> bool:
    return "/" in token or "\\" in token


class CommandHandler:
    def __init__(
        self,
        session: Session,
        out: Callable[..., None] = say,
        out_markdown: Callable[[str], None] = show_markdown,
        out_path: Callable[[str], None] = say_path,
        mcp=None,
        ask_secret: Callable[[str], str] | None = None,
    ):
        self.session = session
        self.out = out
        self.out_markdown = out_markdown
        # File paths are printed on their own left-to-right line, never inside a Hebrew sentence.
        self.out_path = out_path
        self.mcp = mcp  # agent.mcp_client.MCPManager (or a test double)
        self.ask_secret = ask_secret

    # ---- dispatch -------------------------------------------------------------------
    def handle(self, line: str) -> TurnResult:
        text = line.strip()
        lower = text.lower()
        if not text:
            self.out(msg("cli.no_question"), style="yellow")
            return TurnResult()
        if lower in HELP_WORDS:
            self.help()
            return TurnResult()
        if lower in QUIT_WORDS:
            return TurnResult(quit=True)
        command, _, arg = text.partition(" ")
        command = command.lower()
        if command == "/add":
            self.add(arg)
            return TurnResult()
        if command == "/files":
            self.files()
            return TurnResult()
        if command == "/cost":
            self.cost()
            return TurnResult()
        if command == "/new":
            self.new()
            return TurnResult()
        if command == "/mcp":
            self.mcp_command(arg)
            return TurnResult()
        if text.startswith("/"):
            self.out(msg("commands.unknown", command=command), style="yellow")
            return TurnResult()
        return self.message(text)

    # ---- commands -------------------------------------------------------------------
    def help(self) -> None:
        self.out_markdown(msg("help.text", extensions=", ".join(SUPPORTED_EXTENSIONS)))

    def add(self, arg: str) -> bool:
        raw = arg.strip().strip('"').strip("'").strip()
        if not raw:
            self.out(msg("commands.add.usage"), style="yellow")
            return False
        return self._add_path(Path(raw).expanduser())

    def _add_path(self, path: Path) -> bool:
        if not path.exists():
            self.out(msg("commands.add.not_found"), style="yellow")
            self.out_path(str(path))
            return False
        if not path.is_file():
            self.out(msg("commands.add.not_a_file"), style="yellow")
            self.out_path(str(path))
            return False
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            self.out(
                msg(
                    "commands.add.unsupported",
                    name=path.name,
                    extensions=", ".join(SUPPORTED_EXTENSIONS),
                ),
                style="yellow",
            )
            return False
        if self.session.has_file(path):
            self.out(msg("commands.add.duplicate", name=path.name))
            return False
        source = self._read_source(path)
        if source is None:
            return False
        self.session.add_file(path, source)
        self.out(msg("commands.add.added", name=path.name, count=len(self.session.files)), style="green")
        self.out(self._read_summary(source))
        return True

    def _read_source(self, path: Path) -> SourceDoc | None:
        """Read the file through the source-reader MCP tool. None (with a message) on failure."""
        if self.mcp is None:
            self.out(msg("commands.read.no_mcp"), style="yellow")
            return None
        self.out(msg("commands.read.reading", name=path.name))
        try:
            result = self.mcp.call(SOURCE_READER, "read_source", {"path": str(path)})
        except Exception as e:
            report_error(e, node=SOURCE_READER)
            return None
        if not isinstance(result, dict) or not result.get("ok"):
            code = result.get("error", "unknown") if isinstance(result, dict) else "unknown"
            key = f"commands.read.errors.{code}"
            self.out(msg(key if code in READ_ERROR_CODES else "commands.read.errors.unknown", name=path.name),
                     style="yellow")
            return None
        return SourceDoc(unit_type=result["unit_type"], units=result["units"], chunks=result["chunks"])

    @staticmethod
    def _read_summary(source: SourceDoc) -> str:
        return msg(
            "commands.read.summary",
            units=source.units,
            unit_label=msg(f"commands.read.units.{source.unit_type}"),
            chunks=len(source.chunks),
        )

    def files(self) -> None:
        if not self.session.files:
            self.out(msg("commands.files.empty"))
            return
        self.out(msg("commands.files.header", count=len(self.session.files)))
        for i, path in enumerate(self.session.files, 1):
            size = format_size(path.stat().st_size) if path.exists() else msg("commands.files.missing")
            kind = path.suffix.lstrip(".").upper()
            self.out(msg("commands.files.item", index=i, name=path.name, kind=kind, size=size))
            source = self.session.sources.get(path)
            if source is not None:
                self.out("   " + self._read_summary(source))
            self.out_path("   " + str(path))

    # ---- /mcp -----------------------------------------------------------------------
    def mcp_command(self, arg: str) -> None:
        if self.mcp is None:
            self.out(msg("commands.read.no_mcp"), style="yellow")
            return
        sub, _, rest = arg.strip().partition(" ")
        if not sub:
            self.mcp_list()
        elif sub.lower() == "auth":
            self.mcp_auth(rest.strip())
        else:
            self.out(msg("commands.mcp.usage"), style="yellow")

    def mcp_list(self) -> None:
        self.mcp.ensure_all()
        self.out(msg("commands.mcp.header", count=len(self.mcp.states)))
        for name, state in self.mcp.states.items():
            if state.connected:
                status = msg("commands.mcp.status_connected")
            else:
                status = msg("commands.mcp.status_failed")
            auth = ""
            if self.mcp.needs_auth(name):
                auth = msg("commands.mcp.auth_set" if name in self.mcp.credentials else "commands.mcp.auth_missing")
            self.out(msg("commands.mcp.server", name=name, status=status, auth=auth))
            for tool in state.tools:
                description = msg_or_none(f"commands.mcp.tool_descriptions.{tool}") or ""
                self.out(msg("commands.mcp.tool", tool=tool, description=description))
            if not state.connected and state.error:
                self.out(msg("commands.mcp.error_hint"), style="yellow")

    def mcp_auth(self, name: str) -> None:
        if not name:
            self.out(msg("commands.mcp.auth_usage"), style="yellow")
            return
        if name not in self.mcp.specs:
            self.out(msg("commands.mcp.unknown_server", name=name, servers=", ".join(self.mcp.specs)), style="yellow")
            return
        ask = self.ask_secret
        if ask is None:
            from agent.config import read_secret

            ask = read_secret
        try:
            secret = ask(visual_prompt(msg("commands.mcp.auth_prompt", name=name))).strip()
        except (EOFError, KeyboardInterrupt):
            secret = ""
        if not secret:
            self.out(msg("commands.mcp.auth_cancelled"), style="yellow")
            return
        try:
            self.mcp.set_credentials(name, secret)
        except Exception as e:
            report_error(e, node=name)
            return
        self.out(msg("commands.mcp.auth_saved", name=name), style="green")
        if not self.mcp.needs_auth(name):
            self.out(msg("commands.mcp.auth_not_required", name=name))

    def cost(self) -> None:
        if not self.session.usage:
            self.out(msg("commands.cost.empty"))
            return
        self.out(self._totals_line("commands.cost.total", self.session.totals(self.session.usage)))
        for key in ("node", "model"):
            self.out(msg(f"commands.cost.by_{key}"))
            for name, totals in self.session.totals_by(key).items():
                self.out(self._totals_line("commands.cost.item", totals, name=name))

    @staticmethod
    def _totals_line(key: str, t: UsageTotals, **extra) -> str:
        return msg(
            key,
            calls=t.calls,
            prompt_tokens=t.prompt_tokens,
            completion_tokens=t.completion_tokens,
            total_tokens=t.total_tokens,
            cost=t.cost,
            **extra,
        )

    def new(self) -> None:
        self.session.reset()
        self.out(msg("commands.new.done"), style="green")

    # ---- plain messages -------------------------------------------------------------
    def message(self, text: str) -> TurnResult:
        """Add any file paths found in the text; return the text as a question.

        If the message is nothing but paths, no question is sent to the model.
        """
        remaining = text
        for match in _PATH_IN_TEXT.finditer(text):
            token = match.group("dq") or match.group("sq") or match.group("bare")
            quoted = match.group("bare") is None
            existing = next((Path(c) for c in _path_candidates(token) if Path(c).is_file()), None)
            if existing is not None:
                self._add_path(existing)
            elif quoted or _looks_like_path(token):
                self.out(msg("commands.add.detected_not_found"), style="yellow")
                self.out_path(_path_candidates(token)[-1])
            else:
                continue  # a bare name like "README.md" that is not a local file: leave it as text
            remaining = remaining.replace(match.group(0), " ")

        if not re.sub(r"[\W_]+", "", remaining):
            return TurnResult()
        return TurnResult(question=text)

"""Central error handling: classify failures, print one [ERROR] line, log the traceback."""

import logging
import sys
import traceback
from pathlib import Path

from agent.console import log, show_traceback
from agent.messages import msg

LOG_FILE_NAME = "agent.log"
_logger = logging.getLogger("lomda_agent")
_debug = False


class MissingApiKeyError(RuntimeError):
    """OPENROUTER_API_KEY is not set."""


class AllModelsFailedError(RuntimeError):
    """Every model in models.yaml failed. Keeps each model's exception."""

    def __init__(self, failures: list[tuple[str, BaseException]]):
        self.failures = failures
        last = failures[-1][1] if failures else None
        super().__init__(f"all models failed; last error: {last!r}")


class NodeError(RuntimeError):
    """An error raised inside a graph node, tagged with the node name."""

    def __init__(self, node: str, cause: BaseException):
        self.node = node
        self.cause = cause
        super().__init__(f"{node}: {cause!r}")


def log_dir() -> Path:
    """logs/ next to the exe when frozen, otherwise in the project root."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "logs"
    return Path(__file__).resolve().parent.parent / "logs"


def setup_logging(debug: bool = False) -> Path:
    """Send error tracebacks to logs/agent.log. With debug, also show them on screen."""
    global _debug
    _debug = debug
    path = log_dir() / LOG_FILE_NAME
    if not _logger.handlers:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        _logger.addHandler(handler)
        _logger.setLevel(logging.INFO)
        _logger.propagate = False
    return path


def _chain(exc: BaseException):
    """The exception and its causes/contexts, outermost first."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def status_code(exc: BaseException) -> int | None:
    """HTTP status code from an API error or any exception in its cause chain."""
    for e in _chain(exc):
        code = getattr(e, "status_code", None)
        if code is None:
            code = getattr(getattr(e, "response", None), "status_code", None)
        if isinstance(code, int):
            return code
    return None


def is_connection_error(exc: BaseException) -> bool:
    """True for network failures (no internet, DNS, refused, timeout)."""
    import httpx
    import openai

    return any(
        isinstance(e, (openai.APIConnectionError, httpx.TransportError, ConnectionError))
        for e in _chain(exc)
    )


def classify(exc: BaseException) -> str:
    """Map an exception to a message kind in messages_he.json under "errors"."""
    if isinstance(exc, NodeError):
        exc = exc.cause
    if any(isinstance(e, MissingApiKeyError) for e in _chain(exc)):
        return "missing_key"
    if isinstance(exc, AllModelsFailedError) and exc.failures:
        errors = [e for _, e in exc.failures]
        if all(is_connection_error(e) for e in errors):
            return "offline"
        if all(status_code(e) == 429 for e in errors):
            return "rate_limited"
        exc = errors[-1]
    code = status_code(exc)
    if code == 401:
        return "unauthorized"
    if code == 402:
        return "no_credit"
    if code == 429:
        return "rate_limited"
    if is_connection_error(exc):
        return "offline"
    return "unknown"


def format_error(node: str, exc: BaseException) -> str:
    kind = classify(exc)
    return msg(
        "errors.format",
        node=node,
        what=msg(f"errors.{kind}.what", detail=_short(exc)),
        todo=msg(f"errors.{kind}.todo", log_file=str(log_dir() / LOG_FILE_NAME)),
    )


def _short(exc: BaseException) -> str:
    root = exc.cause if isinstance(exc, NodeError) else exc
    code = status_code(root)
    name = type(root).__name__
    return f"{name} {code}" if code else name


def report_error(exc: BaseException, node: str | None = None) -> str:
    """Print one [ERROR] line, write the full traceback to the log file. Never raises."""
    if node is None:
        node = exc.node if isinstance(exc, NodeError) else "cli"
    line = format_error(node, exc)
    tb = "".join(traceback.format_exception(exc))
    try:
        _logger.error("%s\n%s", line, tb)
    except Exception:
        pass
    log(line)
    if _debug:
        show_traceback(tb)
    return line

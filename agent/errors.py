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


def is_timeout(exc: BaseException) -> bool:
    """True when the request timed out (the server is slow, not necessarily unreachable)."""
    import httpx
    import openai

    return any(isinstance(e, (openai.APITimeoutError, httpx.TimeoutException, TimeoutError)) for e in _chain(exc))


def is_connection_error(exc: BaseException) -> bool:
    """True for network failures (no internet, DNS, refused). Timeouts are not counted:
    in openai a timeout is a subclass of APIConnectionError, but it means "slow"."""
    import httpx
    import openai

    if is_timeout(exc):
        return False
    return any(
        isinstance(e, (openai.APIConnectionError, httpx.TransportError, ConnectionError))
        for e in _chain(exc)
    )


def _is_temporary(exc: BaseException) -> bool:
    """Busy or unavailable model: 429, 5xx, or a timeout."""
    code = status_code(exc)
    return is_timeout(exc) or code == 429 or (code is not None and 500 <= code < 600)


# OpenRouter's daily cap on free-model requests (per account, resets at 00:00 UTC).
_DAILY_QUOTA_MARKS = ("free-models-per-day", "openrouter_free_tier_daily")


def _find_key(obj, key: str):
    """Case-insensitive search for a key in nested dicts/lists."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() == key.lower():
                return v
            found = _find_key(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_key(v, key)
            if found is not None:
                return found
    return None


def _as_int(value) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def daily_quota_info(exc: BaseException) -> dict | None:
    """If the error is OpenRouter's free-models-per-day limit: {"limit", "reset_ms"} (either
    may be None). Looks through causes, NodeError and AllModelsFailedError failures."""
    candidates = [exc]
    if isinstance(exc, NodeError):
        candidates.append(exc.cause)
    for c in list(candidates):
        if isinstance(c, AllModelsFailedError):
            candidates += [e for _, e in c.failures]
    found = False
    for candidate in candidates:
        for e in _chain(candidate):
            body = getattr(e, "body", None)
            text = f"{e} {body!r}"
            if not any(mark in text for mark in _DAILY_QUOTA_MARKS):
                continue
            found = True  # wrappers (NodeError) repeat the message but carry no data: keep looking
            headers = getattr(getattr(e, "response", None), "headers", None) or {}
            limit = _as_int(headers.get("x-ratelimit-limit")) or _as_int(_find_key(body, "X-RateLimit-Limit"))
            reset = _as_int(headers.get("x-ratelimit-reset")) or _as_int(_find_key(body, "X-RateLimit-Reset"))
            if limit or reset:
                return {"limit": limit, "reset_ms": reset}
    return {"limit": None, "reset_ms": None} if found else None


def quota_reset_text(reset_ms: int | None, now=None) -> str:
    """When the daily quota resets, in local time: today / tomorrow / a date."""
    from datetime import datetime, timedelta, timezone

    if not reset_ms:
        return msg("errors.daily_quota.reset_unknown")
    when = datetime.fromtimestamp(reset_ms / 1000, tz=timezone.utc).astimezone()
    today = (now or datetime.now().astimezone()).date()
    time_text = when.strftime("%H:%M")
    if when.date() == today:
        return msg("errors.daily_quota.reset_today", time=time_text)
    if when.date() == today + timedelta(days=1):
        return msg("errors.daily_quota.reset_tomorrow", time=time_text)
    return msg("errors.daily_quota.reset_date", date=when.strftime("%d.%m.%Y"), time=time_text)


def classify(exc: BaseException) -> str:
    """Map an exception to a message kind in messages_he.json under "errors"."""
    if isinstance(exc, NodeError):
        exc = exc.cause
    if any(isinstance(e, MissingApiKeyError) for e in _chain(exc)):
        return "missing_key"
    if daily_quota_info(exc) is not None:
        return "daily_quota"  # before rate_limited: a 429 that will not clear in a minute
    if isinstance(exc, AllModelsFailedError) and exc.failures:
        errors = [e for _, e in exc.failures]
        if all(is_connection_error(e) for e in errors):
            return "offline"
        if all(status_code(e) == 429 for e in errors):
            return "rate_limited"
        if all(_is_temporary(e) for e in errors):
            return "models_unavailable"  # a mix of busy (429), down (5xx) and slow (timeout)
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


DEFAULT_FREE_DAILY_LIMIT = 50


def format_error(node: str, exc: BaseException) -> str:
    kind = classify(exc)
    details = {"detail": _short(exc), "log_file": str(log_dir() / LOG_FILE_NAME)}
    if kind == "daily_quota":
        info = daily_quota_info(exc) or {}
        details["limit"] = info.get("limit") or DEFAULT_FREE_DAILY_LIMIT
        details["reset"] = quota_reset_text(info.get("reset_ms"))
    return msg(
        "errors.format",
        node=node,
        what=msg(f"errors.{kind}.what", **details),
        todo=msg(f"errors.{kind}.todo", **details),
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

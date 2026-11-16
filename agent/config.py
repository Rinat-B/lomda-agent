"""Environment loading for both source runs and the frozen (PyInstaller) executable."""

import getpass
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from agent.messages import msg


def is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def configure_stdio() -> None:
    """Use UTF-8 for stdin/stdout/stderr so Hebrew works in pipes and in the frozen exe.

    The frozen exe ignores PYTHONIOENCODING, and redirected streams default to the
    Windows ANSI code page, which cannot round-trip Hebrew text.
    """
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def load_env() -> None:
    """Load .env next to the executable when frozen, otherwise from the current directory."""
    if is_frozen():
        load_dotenv(Path(sys.executable).resolve().parent / ".env")
    else:
        load_dotenv()


def ensure_openrouter_key() -> bool:
    """Make sure OPENROUTER_API_KEY is set; ask for it with hidden input if missing.

    The key is kept in this process's environment only and is never written to disk.
    """
    if os.getenv("OPENROUTER_API_KEY"):
        return True
    try:
        key = getpass.getpass(msg("cli.ask_api_key")).strip()
    except (EOFError, KeyboardInterrupt):
        return False
    if not key:
        return False
    os.environ["OPENROUTER_API_KEY"] = key
    return True

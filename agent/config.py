"""Environment loading for both source runs and the frozen (PyInstaller) executable."""

import getpass
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from agent.console import say, visual_prompt
from agent.messages import msg

MAX_KEY_ATTEMPTS = 3
# API keys are printable ASCII without spaces. Anything else (e.g. a raw Ctrl+V
# character, "\x16") makes OpenRouter answer every request with HTTP 400.
_VALID_KEY = re.compile(r"[\x21-\x7e]+")


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


def _windows_clipboard_text() -> str:
    """Return the Windows clipboard as text, or "" if unavailable."""
    import ctypes
    from ctypes import wintypes

    cf_unicodetext = 13
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32.GetClipboardData.restype = wintypes.HANDLE
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = wintypes.LPVOID
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]

    if not user32.OpenClipboard(None):
        return ""
    try:
        handle = user32.GetClipboardData(cf_unicodetext)
        if not handle:
            return ""
        ptr = kernel32.GlobalLock(handle)
        if not ptr:
            return ""
        try:
            return ctypes.wstring_at(ptr)
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()


def read_secret(prompt: str) -> str:
    """Hidden input. On a Windows console, Ctrl+V pastes and Backspace works.

    getpass on Windows reads raw keys, so Ctrl+V arrives as "\\x16" instead of pasting.
    """
    if sys.platform != "win32" or not sys.stdin.isatty():
        return getpass.getpass(prompt)

    import msvcrt

    for ch in prompt:
        msvcrt.putwch(ch)
    chars: list[str] = []
    while True:
        ch = msvcrt.getwch()
        if ch in ("\r", "\n"):
            break
        if ch == "\x03":
            raise KeyboardInterrupt
        if ch == "\x08":
            if chars:
                chars.pop()
        elif ch == "\x16":
            chars.extend(_windows_clipboard_text())
        elif ch in ("\x00", "\xe0"):
            msvcrt.getwch()  # arrow/function keys send a second code; ignore both
        else:
            chars.append(ch)
    msvcrt.putwch("\r")
    msvcrt.putwch("\n")
    return "".join(chars)


def is_valid_key(key: str) -> bool:
    return bool(_VALID_KEY.fullmatch(key))


def ensure_openrouter_key() -> bool:
    """Make sure OPENROUTER_API_KEY is set; ask for it with hidden input if missing.

    The key is kept in this process's environment only and is never written to disk.
    """
    if os.getenv("OPENROUTER_API_KEY"):
        return True
    for _ in range(MAX_KEY_ATTEMPTS):
        try:
            key = read_secret(visual_prompt(msg("cli.ask_api_key"))).strip()
        except (EOFError, KeyboardInterrupt):
            return False
        if not key:
            return False
        if is_valid_key(key):
            os.environ["OPENROUTER_API_KEY"] = key
            return True
        say(msg("cli.invalid_api_key"), style="yellow")
    return False

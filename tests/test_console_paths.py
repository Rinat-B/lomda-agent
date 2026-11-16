"""Path lines are displayed left-to-right even when they contain Hebrew folder names."""

import io

from rich.console import Console

import agent.console as console_mod

ALEF = chr(0x05D0)
SHIN_LAMED_WAW_MEM = "".join(map(chr, [0x05E9, 0x05DC, 0x05D5, 0x05DD]))  # a Hebrew word


def render(monkeypatch, fn, text):
    monkeypatch.setenv("HEBREW_BIDI", "on")
    buf = io.StringIO()
    monkeypatch.setattr(console_mod, "console", Console(file=buf, width=120, color_system=None))
    fn(text)
    return buf.getvalue().rstrip()


def test_path_keeps_segment_order(monkeypatch):
    path = rf"C:\Users\{ALEF}\AppData\Local\lesson.docx"
    assert render(monkeypatch, console_mod.say_path, path) == path


def test_hebrew_folder_name_reads_correctly_inside_ltr_path(monkeypatch):
    path = rf"C:\docs\{SHIN_LAMED_WAW_MEM}\a.pdf"
    shown = render(monkeypatch, console_mod.say_path, path)
    # Segments stay left-to-right; only the Hebrew word is reversed for display.
    assert shown == rf"C:\docs\{SHIN_LAMED_WAW_MEM[::-1]}\a.pdf"


def _redraw(monkeypatch, typed: str) -> str:
    """Run ask() on a fake terminal and return the redrawn line (after the control codes)."""
    monkeypatch.setenv("HEBREW_BIDI", "on")
    buf = io.StringIO()
    fake = Console(file=buf, width=120, color_system=None, force_terminal=True, legacy_windows=False)
    monkeypatch.setattr(fake, "input", lambda *a, **k: typed)
    monkeypatch.setattr(console_mod, "console", fake)
    assert console_mod.ask(SHIN_LAMED_WAW_MEM + ": ") == typed
    return buf.getvalue().rsplit("\x1b[2K", 1)[-1].rstrip()


def test_command_input_is_redrawn_left_to_right(monkeypatch):
    typed = rf"/add C:\Users\{ALEF}\docs\{SHIN_LAMED_WAW_MEM}\a.docx"
    shown = _redraw(monkeypatch, typed)
    assert shown.startswith(SHIN_LAMED_WAW_MEM[::-1] + ": /add C:\\Users\\")
    assert shown.endswith(rf"\docs\{SHIN_LAMED_WAW_MEM[::-1]}\a.docx")


def test_hebrew_question_is_still_redrawn_right_to_left(monkeypatch):
    shown = _redraw(monkeypatch, SHIN_LAMED_WAW_MEM + " abc").strip()
    assert shown == "abc " + SHIN_LAMED_WAW_MEM[::-1] + " :" + SHIN_LAMED_WAW_MEM[::-1]


def test_regular_hebrew_lines_unchanged(monkeypatch):
    """say() still uses the automatic RTL direction for lines with Hebrew."""
    line = SHIN_LAMED_WAW_MEM + " abc"
    shown = render(monkeypatch, console_mod.say, line).strip()
    assert shown == "abc " + SHIN_LAMED_WAW_MEM[::-1]

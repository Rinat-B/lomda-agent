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


def test_regular_hebrew_lines_unchanged(monkeypatch):
    """say() still uses the automatic RTL direction for lines with Hebrew."""
    line = SHIN_LAMED_WAW_MEM + " abc"
    shown = render(monkeypatch, console_mod.say, line).strip()
    assert shown == "abc " + SHIN_LAMED_WAW_MEM[::-1]

"""Terminal output: rich rendering plus optional Hebrew bidi reordering per wrapped line.

Most terminals (notably the Windows console) draw every line left-to-right, so Hebrew
appears reversed. When HEBREW_BIDI is enabled, every renderable is first laid out by
rich at the terminal width, and only then is each physical line reordered for display
with the Unicode Bidirectional Algorithm (python-bidi). Reordering after wrapping keeps
a long paragraph from swapping text between lines.

HEBREW_BIDI (in .env): "on", "off", or "auto" (default; on for Windows only).
"""

import os
import re
import sys

from bidi import algorithm as bidi_alg
from rich.cells import cell_len
from rich.console import Console, RenderableType
from rich.control import Control
from rich.markdown import Markdown
from rich.segment import ControlType, Segment
from rich.style import Style
from rich.text import Text

console = Console()

# Hebrew block U+0590-U+05FF and Hebrew presentation forms U+FB1D-U+FB4F.
_HEBREW = re.compile(f"[{chr(0x0590)}-{chr(0x05FF)}{chr(0xFB1D)}-{chr(0xFB4F)}]")
_ON_VALUES = {"on", "true", "1", "yes"}
_OFF_VALUES = {"off", "false", "0", "no"}


def bidi_enabled() -> bool:
    mode = os.getenv("HEBREW_BIDI", "auto").strip().lower()
    if mode in _ON_VALUES:
        return True
    if mode in _OFF_VALUES:
        return False
    return sys.platform == "win32"


def _base_dir(text: str) -> str:
    """Line direction: RTL for any line containing Hebrew, LTR otherwise.

    An RTL base also keeps brackets correct: bidi.algorithm does not implement the
    bracket-pair rule (N0), and in LTR lines mixed with Hebrew it can show "(" as ")".
    """
    return "R" if _HEBREW.search(text) else "L"


def _visual_order(text: str, base_dir: str) -> list[tuple[str, int]]:
    """Run the bidi algorithm and return (display char, original index) in visual order.

    Mirrors bidi.algorithm.get_display, but tags each char with its source index so
    per-character styles can follow it.
    """
    storage = bidi_alg.get_empty_storage()
    storage["base_level"] = bidi_alg.PARAGRAPH_LEVELS[base_dir]
    storage["base_dir"] = base_dir
    bidi_alg.get_embedding_levels(text, storage)
    for i, ch in enumerate(storage["chars"]):
        ch["idx"] = i
    bidi_alg.explicit_embed_and_overrides(storage, False)
    bidi_alg.resolve_weak_types(storage, False)
    bidi_alg.resolve_neutral_types(storage, False)
    bidi_alg.resolve_implicit_levels(storage, False)
    bidi_alg.reorder_resolved_levels(storage, False)
    bidi_alg.apply_mirroring(storage, False)
    return [(ch["ch"], ch["idx"]) for ch in storage["chars"]]


def _reorder_line(segments: list[Segment]) -> Text:
    """Turn one rendered line into a styled Text in visual (display) order."""
    chars: list[str] = []
    styles: list[Style | None] = []
    for seg in segments:
        if seg.control:
            continue
        chars.extend(seg.text)
        styles.extend([seg.style] * len(seg.text))
    line = "".join(chars)

    if _HEBREW.search(line):
        ordered = [(ch, styles[idx]) for ch, idx in _visual_order(line, _base_dir(line))]
    else:
        ordered = list(zip(chars, styles))

    # Merge consecutive chars with the same style into one run.
    out = Text(no_wrap=True, end="")
    run, run_style = "", None
    for ch, style in ordered:
        if run and style != run_style:
            out.append(run, run_style)
            run = ""
        run, run_style = run + ch, style
    if run:
        out.append(run, run_style)
    return out


def show(renderable: RenderableType, style: str | None = None) -> None:
    """Print any rich renderable, reordering Hebrew per wrapped line when enabled."""
    if not bidi_enabled():
        console.print(renderable, style=style)
        return
    # One column narrower than the terminal, so a full line never triggers an auto-wrap.
    options = console.options.update_width(max(console.width - 1, 20))
    lines = console.render_lines(renderable, options, style=Style.parse(style) if style else None, pad=True)
    for line in lines:
        console.print(_reorder_line(line), soft_wrap=True)


def say(text: str, style: str | None = None) -> None:
    """Print a plain message (no markup), e.g. a string from messages_he.json."""
    show(Text(text), style=style)


def log(line: str) -> None:
    """Print a log line such as [INFO] or [COST]."""
    say(line)


def show_markdown(text: str) -> None:
    """Render model output as formatted Markdown."""
    show(Markdown(text))


def visual_prompt(prompt: str) -> str:
    """Reorder a single-line input prompt for display, keeping trailing spaces at the end."""
    if not (bidi_enabled() and _HEBREW.search(prompt)):
        return prompt
    stripped = prompt.rstrip()
    trailing = prompt[len(stripped):]
    return "".join(ch for ch, _ in _visual_order(stripped, _base_dir(stripped))) + trailing


def ask(prompt: str, style: str | None = None) -> str:
    """Read a line of input after a (possibly Hebrew) single-line prompt.

    The terminal echoes typed text left-to-right, which we cannot control while typing.
    After Enter, if the answer has Hebrew, the echoed line(s) are erased and redrawn
    in display order so the screen history reads correctly.
    """
    shown_prompt = visual_prompt(prompt)
    answer = console.input(Text(shown_prompt, style=style or ""))
    if bidi_enabled() and console.is_terminal and _HEBREW.search(answer):
        used = cell_len(shown_prompt) + cell_len(answer)
        rows = max(1, -(-used // console.width))  # ceil division
        for _ in range(rows):
            console.control(
                Control.move(0, -1),
                Control.move_to_column(0),
                Control((ControlType.ERASE_IN_LINE, 2)),
            )
        show(Text.assemble((prompt.rstrip() + " ", style or ""), answer))
    return answer

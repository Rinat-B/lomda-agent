"""Export a lesson to a self-contained folder that opens from file:// in any browser.

    out_dir/
      index.html            the fixed player page
      lesson.js             window.LESSON = {...}   (not JSON + fetch: fetch is blocked on file://)
      assets/player.css, assets/player.js, assets/strings.he.js
      assets/illustrations/*.html   copied from the lesson's source folder

The lesson is validated first (agent.lesson_schema); nothing is written for an invalid
lesson, and no file is written outside out_dir.

    python -m agent.lesson_export <lesson.json> <out_dir>
"""

import json
import shutil
import sys
from pathlib import Path

from agent.lesson_schema import LessonValidationError, validate_lesson

PLAYER_DIR = Path(__file__).parent / "assets" / "player"
PLAYER_ASSETS = ("player.css", "player.js", "strings.he.js")


def lesson_js(lesson: dict) -> str:
    """The lesson as a script. "</" is escaped so the data cannot close a <script> tag."""
    data = json.dumps(lesson, ensure_ascii=False, indent=1).replace("</", "<\\/")
    return f"window.LESSON = {data};\n"


def _inside(path: Path, folder: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(folder.resolve()):
        raise ValueError(f"refusing to use a path outside {folder}: {path}")
    return resolved


def illustration_files(lesson: dict) -> list[str]:
    return [
        block["file"]
        for screen in lesson.get("screens", [])
        for block in screen.get("blocks", [])
        if block.get("type") == "illustration"
    ]


def planned_files(lesson: dict) -> list[str]:
    """The files export_lesson writes, relative to the lesson folder."""
    return ["index.html", "lesson.js", *(f"assets/{name}" for name in PLAYER_ASSETS), *illustration_files(lesson)]


PLACEHOLDER_TEMPLATE = PLAYER_DIR / "illustration_placeholder.html"


def placeholder_html(title: str, description: str) -> str:
    """A stand-in page for an illustration that is not built yet: its title and description."""
    from html import escape

    page = PLACEHOLDER_TEMPLATE.read_text(encoding="utf-8")
    return page.replace("__TITLE__", escape(title)).replace("__DESCRIPTION__", escape(description))


def index_html(title: str) -> str:
    """The player page with the lesson's title in the browser tab (also before the script runs)."""
    from html import escape

    page = (PLAYER_DIR / "index.html").read_text(encoding="utf-8")
    return page.replace("<title>Lesson</title>", f"<title>{escape(title)}</title>", 1)


def export_lesson(lesson: dict, out_dir: Path, source_dir: Path | None = None, placeholders: bool = False,
                  illustrations: dict[str, str] | None = None) -> Path:
    """Write the lesson folder; returns the path of index.html.
    `illustrations` maps an illustration file to its built HTML page; otherwise the file is copied
    from `source_dir` (assets/illustrations/*.html), and with placeholders=True an illustration
    with neither gets a page with its title and description instead."""
    errors = validate_lesson(lesson)
    if errors:
        raise LessonValidationError(errors)

    out_dir = Path(out_dir)
    built = illustrations or {}
    planned = []  # (built page or None, source file or None, destination, block)
    blocks = {
        b["file"]: b for s in lesson["screens"] for b in s["blocks"] if b.get("type") == "illustration"
    }
    for rel in illustration_files(lesson):
        if rel in built:
            planned.append((built[rel], None, out_dir / rel, blocks[rel]))
            continue
        src = None
        if source_dir is not None:
            src = _inside(Path(source_dir) / rel, Path(source_dir))
        if src is None or not src.is_file():
            if not placeholders:
                if source_dir is None:
                    raise FileNotFoundError(f"illustration {rel} needs a source_dir to copy it from")
                raise FileNotFoundError(f"illustration file not found: {src}")
            src = None
        planned.append((None, src, out_dir / rel, blocks[rel]))

    out_dir.mkdir(parents=True, exist_ok=True)
    assets = out_dir / "assets"
    assets.mkdir(exist_ok=True)
    (out_dir / "index.html").write_text(index_html(lesson["title"]), encoding="utf-8")
    for name in PLAYER_ASSETS:
        shutil.copyfile(PLAYER_DIR / name, assets / name)
    for page, src, dest, block in planned:
        dest = _inside(dest, out_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if page is not None:
            dest.write_text(page, encoding="utf-8")
        elif src is None:
            dest.write_text(placeholder_html(block["title"], block["description"]), encoding="utf-8")
        else:
            shutil.copyfile(src, dest)
    (out_dir / "lesson.js").write_text(lesson_js(lesson), encoding="utf-8")
    return out_dir / "index.html"


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 2:
        print("usage: python -m agent.lesson_export <lesson.json> <out_dir>")
        return 2
    lesson_path, out_dir = Path(args[0]), Path(args[1])
    lesson = json.loads(lesson_path.read_text(encoding="utf-8"))
    try:
        index = export_lesson(lesson, out_dir, source_dir=lesson_path.parent)
    except LessonValidationError as e:
        print(e)
        return 1
    print(f"exported: {index.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

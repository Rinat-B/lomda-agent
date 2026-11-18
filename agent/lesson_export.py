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


def export_lesson(lesson: dict, out_dir: Path, source_dir: Path | None = None) -> Path:
    """Write the lesson folder; returns the path of index.html.
    `source_dir` holds the lesson's assets/illustrations/*.html (the illustration files)."""
    errors = validate_lesson(lesson)
    if errors:
        raise LessonValidationError(errors)

    out_dir = Path(out_dir)
    illustrations = []
    for rel in illustration_files(lesson):
        if source_dir is None:
            raise FileNotFoundError(f"illustration {rel} needs a source_dir to copy it from")
        src = _inside(Path(source_dir) / rel, Path(source_dir))
        if not src.is_file():
            raise FileNotFoundError(f"illustration file not found: {src}")
        illustrations.append((src, _inside(out_dir / rel, out_dir) if out_dir.exists() else out_dir / rel))

    out_dir.mkdir(parents=True, exist_ok=True)
    assets = out_dir / "assets"
    assets.mkdir(exist_ok=True)
    shutil.copyfile(PLAYER_DIR / "index.html", out_dir / "index.html")
    for name in PLAYER_ASSETS:
        shutil.copyfile(PLAYER_DIR / name, assets / name)
    for src, dest in illustrations:
        dest = _inside(dest, out_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
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

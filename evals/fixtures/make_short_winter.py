"""Build evals/fixtures/short_winter.pptx from short_winter.json.

A small, fixed Hebrew source (5 slides with speaker notes) for development and evals:
braking distances, an if/then process for skids, and visibility.
Run from the project root:  .venv\\Scripts\\python evals\\fixtures\\make_short_winter.py
The content lives in the JSON file, so this script holds no Hebrew text.
"""

import json
from pathlib import Path

from pptx import Presentation
from pptx.enum.text import PP_ALIGN

HERE = Path(__file__).resolve().parent
CONTENT = HERE / "short_winter.json"
OUTPUT = HERE / "short_winter.pptx"


def _rtl(paragraph) -> None:
    """Mark a paragraph right-to-left and right-aligned (Hebrew)."""
    paragraph.alignment = PP_ALIGN.RIGHT
    paragraph._p.get_or_add_pPr().set("rtl", "1")


def build(content_path: Path = CONTENT, output_path: Path = OUTPUT) -> Path:
    slides = json.loads(content_path.read_text(encoding="utf-8"))["slides"]
    prs = Presentation()
    for i, data in enumerate(slides):
        if i == 0:
            slide = prs.slides.add_slide(prs.slide_layouts[0])  # title slide
            slide.placeholders[1].text = data["subtitle"]
            _rtl(slide.placeholders[1].text_frame.paragraphs[0])
        else:
            slide = prs.slides.add_slide(prs.slide_layouts[1])  # title and content
            body = slide.placeholders[1].text_frame
            body.text = data["bullets"][0]
            for bullet in data["bullets"][1:]:
                body.add_paragraph().text = bullet
            for paragraph in body.paragraphs:
                _rtl(paragraph)
        slide.shapes.title.text = data["title"]
        _rtl(slide.shapes.title.text_frame.paragraphs[0])
        slide.notes_slide.notes_text_frame.text = data["notes"]
    prs.save(output_path)
    return output_path


if __name__ == "__main__":
    print("written:", build())

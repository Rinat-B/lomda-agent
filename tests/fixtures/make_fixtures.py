"""Generate the small sample files used by the tests.

Run from the project root:  .venv\\Scripts\\python tests\\fixtures\\make_fixtures.py
The generated files are committed; re-run only to change them.
sample_he.txt (Hebrew text) is a hand-written data file and is not generated here.
"""

from io import BytesIO
from pathlib import Path

HERE = Path(__file__).resolve().parent


def make_pptx(path: Path) -> None:
    from pptx import Presentation
    from pptx.util import Inches

    prs = Presentation()
    # 1: title slide
    s = prs.slides.add_slide(prs.slide_layouts[0])
    s.shapes.title.text = "Winter Driving Basics"
    s.placeholders[1].text = "Driver training | Fleet department"
    # 2: bullets + speaker notes
    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = "Braking distance"
    body = s.placeholders[1].text_frame
    body.text = "Dry road at 80 km/h: about 55 m"
    for line in ("Wet road at 80 km/h: about 80 m", "Icy road: up to 10 times longer"):
        body.add_paragraph().text = line
    s.notes_slide.notes_text_frame.text = "Keep a gap of at least 4 seconds on wet roads."
    # 3: table + grouped text boxes, no notes
    s = prs.slides.add_slide(prs.slide_layouts[5])
    s.shapes.title.text = "Tire pressure"
    table = s.shapes.add_table(3, 2, Inches(1), Inches(2), Inches(6), Inches(1.5)).table
    for r, row in enumerate([("Vehicle", "PSI"), ("Van", "50"), ("Truck", "110")]):
        for c, value in enumerate(row):
            table.cell(r, c).text = value
    group = s.shapes.add_group_shape()
    group.shapes.add_textbox(Inches(1), Inches(4), Inches(3), Inches(0.5)).text_frame.text = "Check monthly"
    group.shapes.add_textbox(Inches(4), Inches(4), Inches(3), Inches(0.5)).text_frame.text = "Check when cold"
    prs.save(path)


def make_docx(path: Path) -> None:
    import docx

    d = docx.Document()
    d.add_heading("Warehouse Safety Procedure", level=1)
    d.add_paragraph("This procedure applies to all warehouse staff and visitors.")
    d.add_paragraph("Forklift loads above 2.5 tons require a spotter.")
    d.add_paragraph("")  # empty paragraph: must be skipped
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "Role", "Equipment"
    t.cell(1, 0).text, t.cell(1, 1).text = "Picker", "Safety shoes"
    d.add_heading("Emergency drill", level=2)
    d.add_paragraph("Stop work and walk to the assembly point.")
    d.save(path)


def _pdf(pages: list[list[str]]) -> bytes:
    """A minimal PDF (Helvetica, one text line per entry) without extra dependencies."""
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    pages_id = len(objects) + 1 + 2 * len(pages)  # reserved: after all page + content objects
    page_ids = []
    for lines in pages:
        ops = ["BT", "/F1 11 Tf", "14 TL", "72 760 Td"]
        for line in lines:
            safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.append(f"({safe}) Tj T*")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1")
        content = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        page_ids.append(
            add(
                b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 612 792] "
                b"/Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>" % (pages_id, font, content)
            )
        )
    kids = b" ".join(b"%d 0 R" % i for i in page_ids)
    assert add(b"<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, len(page_ids))) == pages_id
    catalog = add(b"<< /Type /Catalog /Pages %d 0 R >>" % pages_id)

    out = BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, 1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n" % i + body + b"\nendobj\n")
    xref = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1))
    for off in offsets:
        out.write(b"%010d 00000 n \n" % off)
    out.write(b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, catalog, xref))
    return out.getvalue()


def make_pdf(path: Path) -> None:
    path.write_bytes(
        _pdf(
            [
                [
                    "Code of Ethics",
                    "1. Conflicts of interest",
                    "1.1 Employees must report any personal interest that may affect a",
                    "business decision, in writing, within 7 days.",
                    "1.2 Gifts above 250 NIS must be declined.",
                ],
                [
                    "2. Reporting violations",
                    "2.1 Report to the compliance officer within 48 hours.",
                ],
            ]
        )
    )


def make_headers_pdf(path: Path) -> None:
    """3 pages with a running header and "Page N of 3" footer; one paragraph crosses a page."""
    header = "Logist Ltd | Code of Ethics | v3.0"
    path.write_bytes(
        _pdf(
            [
                [header, "1. Gifts", "1.1 Gifts up to 250 NIS are allowed, provided that", "Page 1 of 3"],
                [header, "they are reported to the manager within 7 days.", "1.2 Cash gifts are forbidden.",
                 "Page 2 of 3"],
                [header, "2. Reporting", "2.1 Report violations within 48 hours.", "Page 3 of 3"],
            ]
        )
    )


def make_headers_docx(path: Path) -> None:
    """Real header/footer parts, plus a body line repeated at the top of every page."""
    import docx
    from docx.enum.text import WD_BREAK

    d = docx.Document()
    section = d.sections[0]
    section.header.paragraphs[0].text = "Logist Ltd | Warehouse Safety"
    section.footer.paragraphs[0].text = "Internal document"
    for page, body in enumerate(["Helmets are required in aisles.", "Forklifts stop at crossings.",
                                 "Exits must stay clear."], 1):
        p = d.add_paragraph()
        if page > 1:
            p.add_run().add_break(WD_BREAK.PAGE)
        p.add_run("Procedure B-104 (typed into the body)")
        d.add_paragraph(body)
    d.add_paragraph("Internal document")  # a copy of the footer text in the body
    d.save(path)


def make_encrypted_pdf(src: Path, path: Path) -> None:
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter()
    for page in PdfReader(str(src)).pages:
        writer.add_page(page)
    writer.encrypt(user_password="secret", algorithm="AES-128")
    with path.open("wb") as f:
        writer.write(f)


def main() -> None:
    make_pptx(HERE / "sample.pptx")
    make_docx(HERE / "sample.docx")
    make_pdf(HERE / "sample.pdf")
    make_encrypted_pdf(HERE / "sample.pdf", HERE / "encrypted.pdf")
    make_headers_pdf(HERE / "headers.pdf")
    make_headers_docx(HERE / "headers.docx")
    # A password-protected Office file is an OLE container, not a zip.
    (HERE / "encrypted.docx").write_bytes(bytes.fromhex("D0CF11E0A1B11AE1") + b"\0" * 504)
    (HERE / "corrupt.pptx").write_bytes(b"this is not a zip file")
    (HERE / "sample.txt").write_text(
        "First paragraph about safety.\nIt continues on a second line.\n\nSecond paragraph.\n\n\nThird.\n",
        encoding="utf-8",
    )
    (HERE / "sample.md").write_text("# Title\n\nIntro paragraph.\n\n## Section\n\nBody text.\n", encoding="utf-8")
    (HERE / "empty.txt").write_text("   \n\n", encoding="utf-8")
    print("fixtures written to", HERE)


if __name__ == "__main__":
    main()

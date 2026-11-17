"""MCP server "source-reader": reads lesson source files into located text chunks.

Tool: read_source(path) -> {"ok": true, "file", "kind", "units", "unit_type", "chunks": [...]}
                         or {"ok": false, "error": <code>, "detail": <English text>}

Each chunk: {"text", "kind", "locator": {"file", "slide" | "page" (optional), "paragraph"}}
  - pptx: one unit per slide; chunks per text shape and table, plus the speaker notes
  - docx: paragraphs and tables in body order
  - pdf:  one unit per page; paragraphs split on blank lines (else on lines)
  - txt, md: paragraphs split on blank lines

Error codes: not_found, not_a_file, unsupported, encrypted, corrupt, empty.
Runs over stdio:  python -m agent.mcp_servers.source_reader   (or agent.exe --mcp source-reader)
Nothing may be printed to stdout here: stdout carries the MCP protocol.
"""

import re
from pathlib import Path
from typing import Any

SERVER_NAME = "source-reader"
SUPPORTED = (".pptx", ".docx", ".pdf", ".txt", ".md")
_OLE_SIGNATURE = bytes.fromhex("D0CF11E0A1B11AE1")  # encrypted Office files are OLE containers
_BLANK_LINES = re.compile(r"\n\s*\n")


class SourceError(Exception):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _chunk(text: str, kind: str, file: str, paragraph: int, **where: int) -> dict:
    return {"text": text, "kind": kind, "locator": {"file": file, **where, "paragraph": paragraph}}


def _clean(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.replace("\r", "")).strip()


_NEW_BLOCK = re.compile(r"^(\d+(\.\d+)*\.?\s|[-*•]\s)")  # "3.", "3.2 ", bullets
_SENTENCE_END = re.compile(r"[.!?:]$")


def _merge_lines(lines: list[str]) -> list[str]:
    """Rebuild paragraphs from wrapped lines (PDF text has no blank lines).

    A new paragraph starts at a numbered section or bullet, or after a line that
    ends a sentence; other lines continue the current paragraph.
    """
    paragraphs: list[str] = []
    for line in (_clean(l) for l in lines):
        if not line:
            continue
        if paragraphs and not _NEW_BLOCK.match(line) and not _SENTENCE_END.search(paragraphs[-1]):
            paragraphs[-1] = f"{paragraphs[-1]} {line}"
        else:
            paragraphs.append(line)
    return paragraphs


def _paragraphs(text: str, merge_wrapped: bool = False) -> list[str]:
    """Split on blank lines. Without blank lines: merge wrapped lines (PDF) or use lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = [_clean(p) for p in _BLANK_LINES.split(text)]
    parts = [p for p in parts if p]
    if len(parts) <= 1 and "\n" in text.strip():
        lines = text.split("\n")
        parts = _merge_lines(lines) if merge_wrapped else [p for p in map(_clean, lines) if p]
    return parts


def _edge_key(line: str) -> str:
    """Compare header/footer lines ignoring numbers ("Page 1 of 4" == "Page 2 of 4")."""
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line)).strip()


def repeated_edge_lines(pages: list[list[str]], edge: int = 2) -> list[set[int]]:
    """Header/footer detection: per page, the indexes of lines to drop.

    A line is a header/footer when its number-insensitive form appears among the first
    or last `edge` lines of more than half of the pages (needs at least 2 pages). Only
    lines at the very top or bottom of a page are dropped.
    """
    drops: list[set[int]] = [set() for _ in pages]
    if len(pages) < 2:
        return drops
    counts: dict[str, int] = {}
    for lines in pages:
        for key in {_edge_key(l) for l in lines[:edge] + lines[-edge:] if l.strip()}:
            counts[key] = counts.get(key, 0) + 1
    repeated = {key for key, n in counts.items() if key and n > len(pages) / 2}
    for lines, drop in zip(pages, drops):
        i = 0
        while i < min(edge, len(lines)) and _edge_key(lines[i]) in repeated:
            drop.add(i)
            i += 1
        j = len(lines) - 1
        while j >= max(i, len(lines) - edge) and _edge_key(lines[j]) in repeated:
            drop.add(j)
            j -= 1
    return drops


def strip_repeated_edges(pages: list[list[str]], edge: int = 2) -> list[list[str]]:
    """The pages without their header/footer lines (see repeated_edge_lines)."""
    drops = repeated_edge_lines(pages, edge)
    return [[l for i, l in enumerate(lines) if i not in drop] for lines, drop in zip(pages, drops)]


def _table_text(rows: list[list[str]]) -> str:
    return "\n".join(" | ".join(_clean(cell) for cell in row) for row in rows)


# ---- formats ---------------------------------------------------------------------------
def _shape_chunks(shapes, out: list[tuple[str, str]]) -> None:
    """Collect (kind, text) from slide shapes, recursing into groups."""
    from pptx.enum.shapes import MSO_SHAPE_TYPE, PP_PLACEHOLDER

    for shape in shapes:
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            _shape_chunks(shape.shapes, out)
            continue
        if getattr(shape, "has_table", False) and shape.has_table:
            rows = [[cell.text for cell in row.cells] for row in shape.table.rows]
            if any(any(c.strip() for c in r) for r in rows):
                out.append(("table", _table_text(rows)))
            continue
        if not getattr(shape, "has_text_frame", False) or not shape.has_text_frame:
            continue
        lines = [_clean(p.text) for p in shape.text_frame.paragraphs]
        text = "\n".join(line for line in lines if line)
        if not text:
            continue
        kind = "body"
        if shape.is_placeholder and shape.placeholder_format.type in (
            PP_PLACEHOLDER.TITLE,
            PP_PLACEHOLDER.CENTER_TITLE,
        ):
            kind = "title"
        out.append((kind, text))


def read_pptx(path: Path) -> tuple[int, list[dict]]:
    from pptx import Presentation

    prs = Presentation(str(path))
    chunks: list[dict] = []
    for slide_no, slide in enumerate(prs.slides, 1):
        found: list[tuple[str, str]] = []
        _shape_chunks(slide.shapes, found)
        if slide.has_notes_slide:
            notes = _clean(slide.notes_slide.notes_text_frame.text if slide.notes_slide.notes_text_frame else "")
            if notes:
                found.append(("notes", notes))
        for i, (kind, text) in enumerate(found, 1):
            chunks.append(_chunk(text, kind, path.name, i, slide=slide_no))
    return len(prs.slides), chunks


_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _starts_new_page(block) -> bool:
    """A paragraph that begins after a page break (explicit or as last rendered by Word)."""
    for el in block.iter():
        if el.tag == f"{_W}lastRenderedPageBreak":
            return True
        if el.tag == f"{_W}br" and el.get(f"{_W}type") == "page":
            return True
        if el.tag == f"{_W}pageBreakBefore" and el.get(f"{_W}val", "true") not in ("0", "false"):
            return True
    return False


def read_docx(path: Path) -> tuple[int, list[dict]]:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(str(path))
    # Real headers/footers live in separate parts and are never read. Body lines that
    # copy them, or that repeat at the top/bottom of most pages, are dropped as well.
    header_footer = {
        _edge_key(_clean(p.text))
        for section in document.sections
        for part in (section.header, section.footer, section.first_page_header, section.first_page_footer)
        for p in part.paragraphs
        if p.text.strip()
    }

    pages: list[list[tuple[str, str]]] = [[]]  # per page: (kind, text)
    for block in document.element.body.iterchildren():
        tag = block.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = Paragraph(block, document)
            if _starts_new_page(block) and pages[-1]:
                pages.append([])
            text = _clean(para.text)
            if text and _edge_key(text) not in header_footer:
                style = (para.style.name if para.style is not None else "") or ""
                kind = "heading" if style.lower().startswith(("heading", "title")) else "body"
                pages[-1].append((kind, text))
        elif tag == "tbl":
            table = Table(block, document)
            rows = [[cell.text for cell in row.cells] for row in table.rows]
            if any(any(c.strip() for c in r) for r in rows):
                pages[-1].append(("table", _table_text(rows)))

    drops = repeated_edge_lines([[text for _, text in page] for page in pages])
    chunks: list[dict] = []
    for page, drop in zip(pages, drops):
        for i, (kind, text) in enumerate(page):
            if i in drop and kind != "table":  # tables are never headers/footers
                continue
            chunks.append(_chunk(text, kind, path.name, len(chunks) + 1))
    return len(chunks), chunks


def read_pdf(path: Path) -> tuple[int, list[dict]]:
    from pypdf import PdfReader
    from pypdf.errors import FileNotDecryptedError, PdfReadError

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                if not reader.decrypt(""):
                    raise SourceError("encrypted", "PDF is password protected")
            except (FileNotDecryptedError, NotImplementedError) as e:
                raise SourceError("encrypted", str(e)) from e
        pages = list(reader.pages)
        raw_lines = [(page.extract_text() or "").replace("\r", "").split("\n") for page in pages]
        page_lines = strip_repeated_edges([[l for l in lines if l.strip()] for lines in raw_lines])
        chunks: list[dict] = []
        for page_no, lines in enumerate(page_lines, 1):
            paragraphs = _paragraphs("\n".join(lines), merge_wrapped=True)
            # A paragraph cut by the page break continues on this page: join it to the
            # previous chunk (which keeps its starting page in the locator).
            if (
                paragraphs
                and chunks
                and not _SENTENCE_END.search(chunks[-1]["text"])
                and not _NEW_BLOCK.match(paragraphs[0])
            ):
                chunks[-1]["text"] += " " + paragraphs.pop(0)
            for i, text in enumerate(paragraphs, 1):
                chunks.append(_chunk(text, "body", path.name, i, page=page_no))
    except FileNotDecryptedError as e:
        raise SourceError("encrypted", str(e)) from e
    except PdfReadError as e:
        raise SourceError("corrupt", str(e)) from e
    return len(pages), chunks


def read_text(path: Path) -> tuple[int, list[dict]]:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1255"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise SourceError("corrupt", "text is not UTF-8 or Windows-1255")
    parts = _paragraphs(text)
    kind = "body"
    chunks = []
    for i, p in enumerate(parts, 1):
        kind = "heading" if path.suffix.lower() == ".md" and p.startswith("#") else "body"
        chunks.append(_chunk(p, kind, path.name, i))
    return len(chunks), chunks


_READERS = {
    ".pptx": ("slide", read_pptx),
    ".docx": ("paragraph", read_docx),
    ".pdf": ("page", read_pdf),
    ".txt": ("paragraph", read_text),
    ".md": ("paragraph", read_text),
}


def read_source_file(path_str: str) -> dict[str, Any]:
    """Read a source file into chunks. Never raises: errors come back as {"ok": false, ...}."""
    try:
        path = Path(path_str).expanduser()
        if not path.exists():
            raise SourceError("not_found", str(path))
        if not path.is_file():
            raise SourceError("not_a_file", str(path))
        suffix = path.suffix.lower()
        if suffix not in _READERS:
            raise SourceError("unsupported", suffix or "(no extension)")
        if suffix in (".pptx", ".docx"):
            with path.open("rb") as f:
                if f.read(8) == _OLE_SIGNATURE:
                    raise SourceError("encrypted", "Office file is password protected")
        unit_type, reader = _READERS[suffix]
        try:
            units, chunks = reader(path)
        except SourceError:
            raise
        except Exception as e:  # malformed zip/xml/pdf and friends
            raise SourceError("corrupt", f"{type(e).__name__}: {e}") from e
        if not chunks:
            raise SourceError("empty", "no text found")
        return {
            "ok": True,
            "file": str(path.resolve()),
            "kind": suffix.lstrip("."),
            "unit_type": unit_type,
            "units": units,
            "chunks": chunks,
        }
    except SourceError as e:
        return {"ok": False, "error": e.code, "detail": e.detail, "file": path_str}
    except Exception as e:  # last resort: the server must keep running
        return {"ok": False, "error": "corrupt", "detail": f"{type(e).__name__}: {e}", "file": path_str}


# ---- MCP server ------------------------------------------------------------------------
def build_server():
    from mcp.server.fastmcp import FastMCP

    # Quiet logging: the server's stderr shares the user's console.
    server = FastMCP(SERVER_NAME, log_level="ERROR")

    @server.tool()
    def read_source(path: str) -> dict:
        """Read a lesson source file (pptx, docx, pdf, txt, md) into text chunks.

        Each chunk has its text, a kind (title/body/notes/table/heading), and a locator
        (file, slide or page, paragraph). On failure returns ok=false with an error code.
        """
        return read_source_file(path)

    return server


def main() -> None:
    build_server().run("stdio")


if __name__ == "__main__":
    main()

"""The source-reader parsing logic (called in-process, no MCP transport)."""

from pathlib import Path

import pytest

from agent.mcp_servers.source_reader import _merge_lines, read_source_file

FIXTURES = Path(__file__).parent / "fixtures"


def read(name: str) -> dict:
    result = read_source_file(str(FIXTURES / name))
    assert result["ok"], result
    return result


def texts(result: dict) -> list[str]:
    return [c["text"] for c in result["chunks"]]


def test_pptx_slides_notes_table_and_groups():
    r = read("sample.pptx")
    assert (r["unit_type"], r["units"], len(r["chunks"])) == ("slide", 3, 9)
    by_slide = {}
    for c in r["chunks"]:
        by_slide.setdefault(c["locator"]["slide"], []).append(c)
    assert [c["kind"] for c in by_slide[1]] == ["title", "body"]
    assert [c["kind"] for c in by_slide[2]] == ["title", "body", "notes"]
    notes = by_slide[2][2]
    assert notes["text"] == "Keep a gap of at least 4 seconds on wet roads."
    assert notes["locator"] == {"file": "sample.pptx", "slide": 2, "paragraph": 3}
    table = next(c for c in by_slide[3] if c["kind"] == "table")
    assert table["text"] == "Vehicle | PSI\nVan | 50\nTruck | 110"
    assert {"Check monthly", "Check when cold"} <= set(texts(r))  # inside a group shape


def test_docx_paragraphs_tables_and_headings_in_order():
    r = read("sample.docx")
    assert texts(r) == [
        "Warehouse Safety Procedure",
        "This procedure applies to all warehouse staff and visitors.",
        "Forklift loads above 2.5 tons require a spotter.",
        "Role | Equipment\nPicker | Safety shoes",
        "Emergency drill",
        "Stop work and walk to the assembly point.",
    ]
    assert [c["kind"] for c in r["chunks"]] == ["heading", "body", "body", "table", "heading", "body"]
    assert r["chunks"][3]["locator"] == {"file": "sample.docx", "paragraph": 4}


def test_pdf_pages_and_merged_paragraphs():
    r = read("sample.pdf")
    assert (r["unit_type"], r["units"]) == ("page", 2)
    page1 = [c for c in r["chunks"] if c["locator"]["page"] == 1]
    assert page1[2]["text"] == (
        "1.1 Employees must report any personal interest that may affect a "
        "business decision, in writing, within 7 days."
    )
    assert page1[2]["locator"] == {"file": "sample.pdf", "page": 1, "paragraph": 3}
    page2 = [c["text"] for c in r["chunks"] if c["locator"]["page"] == 2]
    assert page2 == ["2. Reporting violations", "2.1 Report to the compliance officer within 48 hours."]


def test_txt_paragraphs_split_on_blank_lines():
    r = read("sample.txt")
    assert texts(r) == [
        "First paragraph about safety.\nIt continues on a second line.",
        "Second paragraph.",
        "Third.",
    ]
    assert r["chunks"][1]["locator"] == {"file": "sample.txt", "paragraph": 2}


def test_md_headings():
    r = read("sample.md")
    assert [c["kind"] for c in r["chunks"]] == ["heading", "body", "heading", "body"]


def test_hebrew_text_file():
    r = read("sample_he.txt")
    assert r["units"] == 3
    assert "forklift" in r["chunks"][1]["text"] and "2.5" in r["chunks"][1]["text"]


@pytest.mark.parametrize(
    "name, code",
    [
        ("corrupt.pptx", "corrupt"),
        ("encrypted.pdf", "encrypted"),
        ("encrypted.docx", "encrypted"),
        ("empty.txt", "empty"),
        ("does_not_exist.pdf", "not_found"),
    ],
)
def test_errors_are_returned_not_raised(name, code):
    r = read_source_file(str(FIXTURES / name))
    assert r["ok"] is False and r["error"] == code


def test_unsupported_and_directory(tmp_path):
    (tmp_path / "a.xlsx").write_text("x")
    assert read_source_file(str(tmp_path / "a.xlsx"))["error"] == "unsupported"
    assert read_source_file(str(tmp_path))["error"] == "not_a_file"


def test_pdf_headers_footers_removed_and_cross_page_paragraph_joined():
    r = read("headers.pdf")
    all_text = "\n".join(texts(r))
    assert "Logist Ltd" not in all_text and "Page " not in all_text
    assert texts(r) == [
        "1. Gifts",
        "1.1 Gifts up to 250 NIS are allowed, provided that they are reported to the manager within 7 days.",
        "1.2 Cash gifts are forbidden.",
        "2. Reporting",
        "2.1 Report violations within 48 hours.",
    ]
    assert r["chunks"][1]["locator"] == {"file": "headers.pdf", "page": 1, "paragraph": 2}


def test_docx_header_footer_and_repeated_body_lines_removed():
    r = read("headers.docx")
    assert texts(r) == [
        "Helmets are required in aisles.",
        "Forklifts stop at crossings.",
        "Exits must stay clear.",
    ]


def test_repeated_edges_needs_most_pages():
    from agent.mcp_servers.source_reader import strip_repeated_edges

    pages = [["Header", "a"], ["Header", "b"], ["c", "d"]]
    assert strip_repeated_edges(pages) == [["a"], ["b"], ["c", "d"]]  # 2 of 3 pages: removed
    pages = [["Title", "a"], ["b", "c"], ["d", "e"]]
    assert strip_repeated_edges(pages) == pages  # only once: kept
    assert strip_repeated_edges([["Only page", "x"]]) == [["Only page", "x"]]


def test_merge_lines_rules():
    lines = ["1. Title", "1.1 first line of", "a wrapped paragraph.", "Next sentence starts here", "and ends."]
    assert _merge_lines(lines) == [
        "1. Title",
        "1.1 first line of a wrapped paragraph.",
        "Next sentence starts here and ends.",
    ]

import shutil
from pathlib import Path

import pytest

from agent.commands import CommandHandler, format_size
from agent.messages import msg
from tests.conftest import make_result


FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_FOR = {".pptx": "sample.pptx", ".docx": "sample.docx", ".pdf": "sample.pdf", ".txt": "sample.txt", ".md": "sample.md"}


def copy_sample(dest: Path) -> Path:
    """Copy a real sample file of the same type to dest (so /add can read it)."""
    shutil.copy(FIXTURES / SAMPLE_FOR[dest.suffix.lower()], dest)
    return dest


@pytest.fixture
def handler(session, fake_mcp):
    return CommandHandler(session, mcp=fake_mcp)


@pytest.fixture
def lesson(tmp_path) -> Path:
    return copy_sample(tmp_path / "lesson.docx")


# ---- help / quit / unknown -------------------------------------------------------------
@pytest.mark.parametrize("word", ["help", "?", "HELP", "/help"])
def test_help_shows_guide(handler, output, word):
    result = handler.handle(word)
    assert not result.quit and result.question is None
    out = output()
    for command in ("/add", "/files", "/cost", "/new", "/quit"):
        assert command in out


@pytest.mark.parametrize("word", ["/quit", "quit", "exit", "EXIT"])
def test_quit_words(handler, word):
    assert handler.handle(word).quit


def test_unknown_command_keeps_running(handler, output):
    result = handler.handle("/bogus")
    assert not result.quit and result.question is None
    assert msg("commands.unknown", command="/bogus") in output()


def test_empty_line(handler, output):
    assert handler.handle("   ").question is None
    assert msg("cli.no_question") in output()


# ---- /add ------------------------------------------------------------------------------
def test_add_existing_supported_file(handler, session, lesson, output):
    handler.handle(f"/add {lesson}")
    assert session.files == [lesson.resolve()]
    assert msg("commands.add.added", name="lesson.docx", count=1) in output()


def test_add_quoted_path_with_spaces(handler, session, tmp_path):
    p = copy_sample(tmp_path / "my slides.pptx")
    handler.handle(f'/add "{p}"')
    assert session.files == [p.resolve()]


@pytest.mark.parametrize("ext", [".pptx", ".docx", ".pdf", ".txt", ".md", ".PDF"])
def test_add_all_supported_types(handler, session, tmp_path, ext):
    p = copy_sample(tmp_path / f"f{ext}")
    handler.handle(f"/add {p}")
    assert len(session.files) == 1


def test_add_missing_file(handler, session, tmp_path, output):
    missing = tmp_path / "nope.pdf"
    result = handler.handle(f"/add {missing}")
    assert session.files == [] and not result.quit
    assert msg("commands.add.not_found") in output() and str(missing) in output()


def test_add_unsupported_type(handler, session, tmp_path, output):
    p = tmp_path / "data.xlsx"
    p.write_text("x")
    handler.handle(f"/add {p}")
    assert session.files == []
    assert "data.xlsx" in output() and ".pptx" in output()


def test_add_directory(handler, session, tmp_path, output):
    handler.handle(f"/add {tmp_path}")
    assert session.files == []
    assert msg("commands.add.not_a_file") in output() and str(tmp_path) in output()


def test_add_without_argument(handler, output):
    handler.handle("/add")
    assert msg("commands.add.usage") in output()


def test_add_duplicate(handler, session, lesson, output):
    handler.handle(f"/add {lesson}")
    handler.handle(f"/add {lesson}")
    assert len(session.files) == 1
    assert msg("commands.add.duplicate", name="lesson.docx") in output()


# ---- path detection in regular messages ------------------------------------------------
def test_message_with_path_adds_file_and_keeps_question(handler, session, lesson):
    result = handler.handle(f"Build a lesson from {lesson} please")
    assert session.files == [lesson.resolve()]
    assert result.question == f"Build a lesson from {lesson} please"


def test_message_with_hebrew_prefix_on_path(handler, session, lesson):
    # Hebrew text built from code points (no Hebrew literals in .py files):
    # "BNH LWMDH " ("build a lesson") followed by the prefix "M-" ("from") glued to the path.
    words = "".join(map(chr, [0x05D1, 0x05E0, 0x05D4, 0x20, 0x05DC, 0x05D5, 0x05DE, 0x05D3, 0x05D4, 0x20]))
    prefix = chr(0x05DE) + "-"
    text = words + prefix + str(lesson)
    result = handler.handle(text)
    assert session.files == [lesson.resolve()]
    assert result.question == text


def test_message_with_quoted_path(handler, session, tmp_path):
    p = copy_sample(tmp_path / "a b.pdf")
    handler.handle(f'summarize "{p}"')
    assert session.files == [p.resolve()]


def test_message_only_path_sends_no_question(handler, session, lesson):
    result = handler.handle(str(lesson))
    assert session.files == [lesson.resolve()]
    assert result.question is None


def test_message_with_missing_path_warns_and_continues(handler, session, tmp_path, output):
    missing = tmp_path / "gone.pptx"
    result = handler.handle(f"use {missing} for the lesson")
    assert session.files == []
    assert result.question is not None
    assert msg("commands.add.detected_not_found") in output() and str(missing) in output()


def test_bare_filename_that_does_not_exist_is_just_text(handler, session, output):
    result = handler.handle("what should README.md contain?")
    assert session.files == [] and result.question
    assert output() == ""


# ---- /files ----------------------------------------------------------------------------
def test_files_empty(handler, output):
    handler.handle("/files")
    assert msg("commands.files.empty") in output()


def test_files_lists_added(handler, lesson, output):
    handler.handle(f"/add {lesson}")
    handler.handle("/files")
    out = output()
    assert msg("commands.files.header", count=1) in out
    assert f"lesson.docx (DOCX, {format_size(lesson.stat().st_size)})" in out
    units_label = msg("commands.read.units.paragraph")
    assert msg("commands.read.summary", units=6, unit_label=units_label, chunks=6) in out


# ---- /add reads through the source-reader tool -------------------------------------------
def test_add_reads_file_and_reports_counts(handler, session, tmp_path, fake_mcp, output):
    p = copy_sample(tmp_path / "deck.pptx")
    handler.handle(f"/add {p}")
    assert fake_mcp.calls == [("source-reader", "read_source", {"path": str(p)})]
    source = session.sources[p.resolve()]
    assert (source.unit_type, source.units, len(source.chunks)) == ("slide", 3, 9)
    label = msg("commands.read.units.slide")
    assert msg("commands.read.summary", units=3, unit_label=label, chunks=9) in output()


@pytest.mark.parametrize(
    "fixture, code",
    [("corrupt.pptx", "corrupt"), ("encrypted.pdf", "encrypted"), ("encrypted.docx", "encrypted"),
     ("empty.txt", "empty")],
)
def test_add_unreadable_file_is_not_added_and_session_continues(handler, session, fixture, code, output):
    result = handler.handle(f"/add {FIXTURES / fixture}")
    assert session.files == [] and session.sources == {} and not result.quit
    assert msg(f"commands.read.errors.{code}", name=fixture) in output()
    assert handler.handle("/files").quit is False


def test_add_without_mcp_does_not_add(session, lesson, output):
    CommandHandler(session).handle(f"/add {lesson}")
    assert session.files == []
    assert msg("commands.read.no_mcp") in output()


def test_new_clears_read_sources(handler, session, lesson):
    handler.handle(f"/add {lesson}")
    assert session.sources
    handler.handle("/new")
    assert session.sources == {}


# ---- /mcp ------------------------------------------------------------------------------
def test_mcp_lists_servers_and_tools(handler, output):
    handler.handle("/mcp")
    out = output()
    assert msg("commands.mcp.header", count=1) in out
    assert "source-reader" in out and msg("commands.mcp.status_connected") in out
    assert "read_source" in out


def test_mcp_auth_saves_secret_for_session_only(session, fake_mcp, output):
    prompts = []
    handler = CommandHandler(session, mcp=fake_mcp, ask_secret=lambda p: (prompts.append(p), " s3cret ")[1])
    handler.handle("/mcp auth source-reader")
    assert fake_mcp.credentials == {"source-reader": "s3cret"}
    assert len(prompts) == 1
    out = output()
    assert msg("commands.mcp.auth_saved", name="source-reader") in out
    assert "s3cret" not in out  # never echoed


def test_mcp_auth_unknown_server(handler, fake_mcp, output):
    handler.handle("/mcp auth nope")
    assert fake_mcp.credentials == {}
    assert msg("commands.mcp.unknown_server", name="nope", servers="source-reader") in output()


def test_mcp_auth_empty_input_saves_nothing(session, fake_mcp, output):
    CommandHandler(session, mcp=fake_mcp, ask_secret=lambda p: "  ").handle("/mcp auth source-reader")
    assert fake_mcp.credentials == {}
    assert msg("commands.mcp.auth_cancelled") in output()


def test_mcp_auth_without_name(handler, output):
    handler.handle("/mcp auth")
    assert msg("commands.mcp.auth_usage") in output()


def test_format_size():
    assert format_size(10) == "10 B"
    assert format_size(2048) == "2.0 KB"
    assert format_size(5 * 1024 * 1024) == "5.0 MB"


# ---- /cost -----------------------------------------------------------------------------
def test_cost_empty(handler, output):
    handler.handle("/cost")
    assert msg("commands.cost.empty") in output()


def test_cost_by_node_and_model(handler, session, output):
    session.record_usage("scope_check", make_result("a/free", 10, 1, 0.001))
    session.record_usage("answer", make_result("a/free", 20, 30, 0.002))
    session.record_usage("scope_check", make_result("b/free", 5, 1, 0.0))
    handler.handle("/cost")
    out = output()

    def item(name, calls, total, cost):
        return msg("commands.cost.item", name=name, calls=calls, total_tokens=total, cost=cost)

    total = msg(
        "commands.cost.total", calls=3, prompt_tokens=35, completion_tokens=32, total_tokens=67, cost=0.003
    )
    assert total in out
    assert out.index(msg("commands.cost.by_node")) < out.index(msg("commands.cost.by_model"))
    assert item("scope_check", 2, 17, 0.001) in out
    assert item("answer", 1, 50, 0.002) in out
    assert item("a/free", 2, 61, 0.003) in out
    assert item("b/free", 1, 6, 0.0) in out


# ---- /new ------------------------------------------------------------------------------
def test_new_resets_files_and_cost(handler, session, lesson, output):
    handler.handle(f"/add {lesson}")
    session.record_usage("answer", make_result())
    handler.handle("/new")
    assert session.files == [] and session.usage == []
    assert msg("commands.new.done") in output()

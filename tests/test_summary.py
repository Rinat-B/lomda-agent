"""/summary [path]: the conversation's summary to a local text file, after [Y/N]."""

from pathlib import Path

from agent.commands import CommandHandler
from agent.messages import msg
from agent.session import Session, SourceDoc, UsageRecord
from agent.summary import build_summary, target_path


def session_with_history() -> Session:
    s = Session()
    s.add_file(Path("deck.pptx"), SourceDoc("slide", 2, [{"text": "x", "kind": "body",
                                                           "locator": {"file": "deck.pptx", "slide": 1}}]))
    s.history += [{"kind": "question", "text": "מה מרחק העצירה?"}, {"kind": "answer", "text": "28 מ׳ ביבש."},
                  {"kind": "question", "text": "בנה לומדה"},
                  {"kind": "lesson", "title": "נהיגה בחורף", "path": "output/deck/index.html", "report": "output/deck/report.md"}]
    s.usage.append(UsageRecord("write_lesson", "m/free", 100, 50, 0.0))
    return s


def handler(session, answers, outputs):
    replies = iter(answers)
    return CommandHandler(session, out=lambda text, **k: outputs.append(text), ask_line=lambda p: next(replies))


def test_the_summary_lists_files_requests_lessons_and_cost():
    text = build_summary(session_with_history())
    assert msg("summary.question", index=1, text="מה מרחק העצירה?") in text
    assert msg("summary.lesson", title="נהיגה בחורף", path="output/deck/index.html") in text
    assert "deck.pptx" in text and msg("summary.by_node") in text and "write_lesson" in text


def test_y_writes_the_file_and_n_does_not(tmp_path, monkeypatch):
    import agent.persistence as persistence

    monkeypatch.setattr(persistence, "output_dir", lambda: tmp_path)
    outputs: list[str] = []
    target = tmp_path / "my-summary.txt"
    assert not handler(session_with_history(), ["N"], outputs).summary(str(target))
    assert not target.exists() and msg("commands.summary.not_saved") in outputs
    assert handler(session_with_history(), ["Y"], outputs).summary(str(target))
    assert "נהיגה בחורף" in target.read_text(encoding="utf-8")
    outputs.clear()
    handler(session_with_history(), ["N"], outputs).summary(str(target))
    assert msg("commands.summary.exists") in outputs  # a warning before overwriting


def test_the_default_target_and_a_folder_target(tmp_path):
    assert target_path("", tmp_path).parent == tmp_path and target_path("", tmp_path).name.startswith("summary-")
    assert target_path(str(tmp_path), tmp_path / "x").parent == tmp_path
    assert target_path(str(tmp_path / "notes"), tmp_path).suffix == ".txt"


def test_an_empty_conversation_has_nothing_to_summarise():
    outputs: list[str] = []
    assert not handler(Session(), [], outputs).summary("")
    assert outputs == [msg("commands.summary.empty")]


def test_the_command_is_dispatched_and_in_help(tmp_path, monkeypatch):
    import agent.persistence as persistence

    monkeypatch.setattr(persistence, "output_dir", lambda: tmp_path)
    outputs: list[str] = []
    handler(session_with_history(), ["Y"], outputs).handle("/summary")
    assert list(tmp_path.glob("summary-*.txt"))
    assert "/summary" in msg("help.text", extensions="")

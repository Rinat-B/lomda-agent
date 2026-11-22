"""Memory and transparency: a change to the saved lesson in the next message, the conversation
history kept with the conversation (so /summary is complete after /resume)."""

from pathlib import Path

import agent.nodes as nodes
from agent import cli
from agent.commands import CommandHandler
from agent.graph import build_graph
from agent.persistence import ConversationStore, open_checkpointer
from agent.session import Session, SourceDoc
from agent.summary import build_summary
from tests.test_lesson_flow import FakeModel, deck_chunks
from tests.test_navigation import script


def setup(monkeypatch, tmp_path, graph=None):
    import agent.persistence as persistence

    session = Session()
    session.add_file(Path("deck.pptx"), SourceDoc("slide", 2, deck_chunks(2)))
    model = FakeModel()
    monkeypatch.setattr(nodes, "call_llm", model)
    monkeypatch.setattr(cli, "open_preview", lambda path: None)
    monkeypatch.setattr(persistence, "output_dir", lambda: tmp_path / "output")
    return session, model, graph or build_graph()


def test_a_change_after_saving_rewrites_only_that_screen_and_saves_a_new_version(monkeypatch, tmp_path, output):
    session, model, graph = setup(monkeypatch, tmp_path)
    cli.run_question("build", session, graph, ask_fn=script({}, save="Y"))
    first = graph.get_state(cli.graph_config(session)).values
    assert first["saved_path"] and session.last_run
    writes_before = model.kinds.count("write")
    topic_maps = model.kinds.count("topic_map")

    cli.run_question("במסך 2 תחליף את השאלה", session, graph, ask_fn=script({}, save="Y"))
    second = graph.get_state(cli.graph_config(session)).values
    assert model.kinds.count("topic_map") == topic_maps  # nothing planned again
    assert model.kinds.count("write") == writes_before + 1  # one rewrite, for screen 2
    revision = model.prompts_of("write")[-1]
    assert "במסך 2 תחליף את השאלה" in revision
    assert second["edit_of"] == first["saved_path"]
    assert Path(second["saved_path"]).parent != Path(first["saved_path"]).parent  # a new version folder
    assert Path(first["saved_path"]).is_file()  # the old one stays
    assert [s["id"] for s in second["lesson"]["screens"]] == [s["id"] for s in first["lesson"]["screens"]]
    owner = [f for f in second["findings"] if f["check"] == "owner"]
    assert [f["screen"] for f in owner] == [2]


def test_a_new_request_is_a_new_lesson_not_an_edit(monkeypatch, tmp_path, output):
    session, model, graph = setup(monkeypatch, tmp_path)
    cli.run_question("build", session, graph, ask_fn=script({}, save="Y"))
    cli.run_question("build another lesson", session, graph, ask_fn=script({}, save="N"))
    assert model.kinds.count("topic_map") == 2


def test_history_is_kept_with_the_conversation_and_summary_is_complete_after_resume(monkeypatch, tmp_path, output):
    db = tmp_path / "c.sqlite"
    session, model, graph = setup(monkeypatch, tmp_path, build_graph(open_checkpointer(db)))
    store = ConversationStore(db)
    cli.run_question("build", session, graph, ask_fn=script({}, save="Y"), store=store)
    cli.run_question("בנה עוד אחת", session, graph, ask_fn=script({"topics": ["exit"]}), store=store)
    # a new program: /resume, then /summary
    session2 = Session()
    graph2 = build_graph(open_checkpointer(db))
    store2 = ConversationStore(db)
    assert CommandHandler(session2, store=store2, graph=graph2).handle("/resume 1").resume
    text = build_summary(session2)
    assert "build" in text and "בנה עוד אחת" in text
    assert any(h["kind"] == "lesson" for h in session2.history)


def test_session_keeps_only_the_last_run():
    s = Session()
    assert not hasattr(s, "topic_map") and not hasattr(s, "structure") and s.last_run is None

"""Pausing at a gate, "closing" the program, and /resume with SqliteSaver. Fake model only."""

from pathlib import Path

import pytest

import agent.nodes as nodes
from agent.cli import run_graph, run_question
from agent.commands import CommandHandler
from agent.graph import build_graph
from agent.messages import msg
from agent.persistence import ConversationStore, data_dir, open_checkpointer, paused_conversations
from agent.session import Session, SourceDoc
from tests.test_lesson_flow import FakeModel, deck_chunks, rendered, scripted

QUESTION = "build me a lesson from the deck"


@pytest.fixture
def model(monkeypatch):
    fake = FakeModel()
    monkeypatch.setattr(nodes, "call_llm", fake)
    return fake


@pytest.fixture
def db(tmp_path) -> Path:
    return tmp_path / "data" / "checkpoints.sqlite"


def new_session_with_deck() -> Session:
    session = Session()
    session.add_file(Path("deck.pptx"), SourceDoc("slide", 2, deck_chunks(2)))
    return session


def answers(*items):
    """Scripted gate answers; the save gate (after writing) answers N."""
    return scripted(list(items))


def open_program(db: Path):
    """Everything a fresh start of agent.exe creates (the "after reopening" state)."""
    graph = build_graph(open_checkpointer(db))
    return graph, ConversationStore(db)


# ---- [INFO] for the gate is printed once per visit ------------------------------------------
def test_gate_info_line_printed_once_on_approve(model, output):
    run_question(QUESTION, new_session_with_deck(), build_graph(), ask_fn=answers("Y", "Y", "Y"))
    assert output().count("[INFO] Entering Node: gate_topics") == 1
    assert output().count("[INFO] Entering Node: topic_map") == 1


def test_gate_info_line_once_per_visit_after_a_change(model, output):
    run_question(QUESTION, new_session_with_deck(), build_graph(), ask_fn=answers("Y", "N add tires", "Y", "Y"))
    out = output()
    assert out.count("[INFO] Entering Node: topic_map") == 2
    assert out.count("[INFO] Entering Node: gate_topics") == 2  # two visits, no re-entry lines


# ---- pause, close, resume -------------------------------------------------------------------
def test_pause_close_resume_approve(model, db, output):
    # First program run: build, then quit at the gate.
    graph, store = open_program(db)
    session = new_session_with_deck()
    store.save(session)
    keep_running = run_question(QUESTION, session, graph, ask_fn=answers("Y", "exit"), store=store)
    assert keep_running is False
    assert msg("gate.paused") in output()
    thread = session.thread_id

    # "Close" and reopen: brand-new objects on the same database file.
    graph2, store2 = open_program(db)
    paused = paused_conversations(graph2, store2)
    assert [(p.thread_id, p.files, p.gate) for p in paused] == [(thread, ["deck.pptx"], "topics")]

    session2 = Session()
    handler = CommandHandler(session2, store=store2, graph=graph2, ask_line=answers("1"))
    result = handler.handle("/resume")
    assert result.resume is True
    assert session2.thread_id == thread
    assert [p.name for p in session2.files] == ["deck.pptx"]
    assert len(session2.index) == len(deck_chunks(2))  # chunks restored, no /add needed
    out = output()
    assert msg("gate.names.topics") in out and "deck.pptx" in out

    prompts_before = len(model.topic_map_prompts())
    assert run_graph(session2, graph2, None, ask_fn=answers("Y", "Y"), store=store2) is True
    assert len(model.topic_map_prompts()) == prompts_before  # the saved map is shown, not rebuilt
    assert rendered("gate.topics.title", count=4) in output()
    assert msg("gate.topics.approved") in output()
    assert session2.last_run['topic_map'] is not None and len(session2.last_run['topic_map']) == 4
    assert paused_conversations(graph2, store2) == []


def test_resume_can_request_changes_after_restart(model, db, output):
    graph, store = open_program(db)
    session = new_session_with_deck()
    store.save(session)
    run_question(QUESTION, session, graph, ask_fn=answers("Y", "exit"), store=store)

    graph2, store2 = open_program(db)
    session2 = Session()
    CommandHandler(session2, store=store2, graph=graph2).handle("/resume 1")
    run_graph(session2, graph2, None, ask_fn=answers("N add tires", "Y", "Y"), store=store2)
    assert "apply this feedback: add tires" in model.topic_map_prompts()[-1]
    assert [t["name"] for t in session2.last_run['topic_map']][-1] == "TIRES"
    assert output().count("[INFO] Entering Node: gate_topics") == 2  # first run + the revised map


def test_eof_at_gate_also_pauses(model, db, output):
    graph, store = open_program(db)

    def eof(prompt):
        raise EOFError

    assert run_question(QUESTION, new_session_with_deck(), graph, ask_fn=eof, store=store) is False
    assert len(paused_conversations(graph, store)) == 1


# ---- /resume edge cases -------------------------------------------------------------------
def test_resume_with_nothing_paused(db, output):
    graph, store = open_program(db)
    result = CommandHandler(Session(), store=store, graph=graph).handle("/resume")
    assert result.resume is False and msg("resume.none") in output()


@pytest.mark.parametrize("choice, key", [("", "resume.cancelled"), ("7", "resume.invalid"), ("x", "resume.invalid")])
def test_resume_cancel_and_invalid_choice(model, db, output, choice, key):
    graph, store = open_program(db)
    run_question(QUESTION, new_session_with_deck(), graph, ask_fn=answers("exit"), store=store)
    session = Session()
    result = CommandHandler(session, store=store, graph=graph, ask_line=answers(choice)).handle("/resume")
    assert result.resume is False and session.files == []
    assert msg(key, choice=choice) in output()


def test_new_conversation_keeps_paused_ones(model, db):
    graph, store = open_program(db)
    session = new_session_with_deck()
    run_question(QUESTION, session, graph, ask_fn=answers("exit"), store=store)
    CommandHandler(session, store=store, graph=graph).handle("/new")
    assert len(paused_conversations(graph, store)) == 1


# ---- store details --------------------------------------------------------------------------
def test_store_skips_empty_sessions(db):
    store = ConversationStore(db)
    store.save(Session())
    assert store.conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 0


def test_store_round_trips_usage(db):
    from tests.conftest import make_result

    store = ConversationStore(db)
    session = new_session_with_deck()
    session.record_usage("topic_map", make_result("m/free", 100, 50, 0.01))
    store.save(session)
    restored = Session()
    assert store.restore(session.thread_id, restored)
    assert restored.usage == session.usage


def test_data_dir_is_next_to_the_project_when_not_frozen():
    assert data_dir() == Path(__file__).resolve().parent.parent / "data"


# ---- resume from the type gate and from the structure gate ------------------------------
def pause_then_reopen(model, db, before_exit: list[str]):
    """Run until `exit` (after the given answers), then reopen and /resume the conversation."""
    graph, store = open_program(db)
    session = new_session_with_deck()
    store.save(session)
    run_question(QUESTION, session, graph, ask_fn=answers(*before_exit, "exit"), store=store)
    graph2, store2 = open_program(db)
    session2 = Session()
    assert CommandHandler(session2, store=store2, graph=graph2).handle("/resume 1").resume is True
    return graph2, store2, session2


def test_resume_from_the_type_gate(model, db, output):
    graph, store = open_program(db)
    session = new_session_with_deck()
    run_question(QUESTION, session, graph, ask_fn=answers("exit"), store=store)
    [paused] = paused_conversations(graph, store)
    assert paused.gate == "type"

    graph2, store2, session2 = pause_then_reopen(model, db, [])
    # N at the type gate switches the proposed training type to compliance.
    run_graph(session2, graph2, None, ask_fn=answers("N", "Y", "Y"), store=store2)
    assert session2.last_run['lesson_type'] == "compliance"
    assert session2.last_run['structure']["lesson_type"] == "compliance"
    assert msg("gate.type.chosen", type=msg("lesson_types.compliance")) in output()


def test_resume_from_the_structure_gate(model, db, output):
    graph2, store2, session2 = pause_then_reopen(model, db, ["Y", "Y"])  # type, topics
    out = output()
    assert msg("gate.names.structure") in out
    plans_before = len(model.prompts_of("plan_structure"))
    run_graph(session2, graph2, None, ask_fn=answers("Y"), store=store2)
    assert len(model.prompts_of("plan_structure")) == plans_before  # the saved structure is shown
    assert msg("gate.structure.approved") in output()
    assert session2.last_run['structure'] is not None and len(session2.last_run['structure']["screens"]) == 4
    assert paused_conversations(graph2, store2) == []

"""Three scripted conversations end to end with a fake model (no cost), each checking the final state.
Run by evals/conversation_eval.py too; the results table is in evals/results/README.md.

1. "Build a lesson of 5 screens", a question in the middle, then back to the topics with a change.
2. Cancel at a gate, close the program, /resume and finish.
3. A change after saving ("on screen 2 replace the question"): only that screen, a new version.
"""

from pathlib import Path

import pytest

import agent.nodes as nodes
from agent import cli
from agent.commands import CommandHandler
from agent.graph import build_graph
from agent.messages import msg
from agent.persistence import ConversationStore, open_checkpointer, paused_conversations
from agent.session import Session, SourceDoc
from agent.summary import build_summary
from tests.test_lesson_flow import FakeModel, deck_chunks
from tests.test_navigation import gate_answers, script


@pytest.fixture
def world(monkeypatch, tmp_path):
    import agent.persistence as persistence

    model = FakeModel()
    monkeypatch.setattr(nodes, "call_llm", model)
    monkeypatch.setattr(cli, "open_preview", lambda path: None)
    monkeypatch.setattr(persistence, "output_dir", lambda: tmp_path / "output")
    return model, tmp_path


def new_session() -> Session:
    session = Session()
    session.add_file(Path("deck.pptx"), SourceDoc("slide", 2, deck_chunks(2)))
    return session


def state_of(graph, session) -> dict:
    return graph.get_state(cli.graph_config(session)).values


def test_scenario_1_five_screens_a_question_and_back_to_the_topics(world, output):
    model, tmp = world
    session, graph = new_session(), build_graph()
    cli.run_question("בנה לומדה של 5 מסכים לנהגים חדשים", session, graph, ask_fn=script({
        "structure": ["למה בחרת במבנה הזה?", "Y"],
        "review": ["רגע, תוריד את הנושא השלישי", "Y"],
        "back": ["Y"],
    }, save="Y"))
    state = state_of(graph, session)
    assert len(state["lesson"]["screens"]) == 5 and state["requirements"]["screens"] == 5
    assert state["requirements"]["audience"] == "נהגים חדשים"
    assert any("FAKE ANSWER" in a for a in gate_answers(graph, session))  # the question was answered
    assert model.kinds.count("topic_map") == 2 and model.kinds.count("research_ideas") == 1
    assert "remove the third topic" in model.prompts_of("topic_map")[1]
    assert state["saved_path"] and Path(state["saved_path"]).is_file()


def test_scenario_2_cancel_then_resume_in_a_new_program(world, output):
    model, tmp = world
    db = tmp / "conv.sqlite"
    session, graph, store = new_session(), build_graph(open_checkpointer(db)), ConversationStore(db)
    store.save(session)
    cli.run_question("build", session, graph, ask_fn=script({"topics": ["עזוב, לא צריך עכשיו"], "cancel": ["Y"]}),
                     store=store)
    assert msg("gate.cancel.done") in output()
    graph2, store2, session2 = build_graph(open_checkpointer(db)), ConversationStore(db), Session()
    [paused] = paused_conversations(graph2, store2)
    assert paused.gate == "topics"
    assert CommandHandler(session2, store=store2, graph=graph2).handle("/resume 1").resume
    cli.run_graph(session2, graph2, None, ask_fn=script({}, save="Y"), store=store2)
    state = state_of(graph2, session2)
    assert state["lesson"] and state["saved_path"]
    assert model.kinds.count("topic_map") == 1  # nothing was redone
    assert "build" in build_summary(session2)  # the history came back with /resume


def test_scenario_3_a_change_after_saving(world, output):
    model, tmp = world
    session, graph = new_session(), build_graph()
    cli.run_question("build", session, graph, ask_fn=script({}, save="Y"))
    first = state_of(graph, session)
    cli.run_question("במסך 2 תחליף את השאלה", session, graph, ask_fn=script({}, save="Y"))
    second = state_of(graph, session)
    assert second["edit_of"] == first["saved_path"]
    assert Path(second["saved_path"]).parent.name.endswith("-2")
    changed = [n for n, (a, b) in enumerate(zip(first["lesson"]["screens"], second["lesson"]["screens"]), 1) if a != b]
    assert changed in ([2], [])  # only screen 2 may change (the fake writer may return the same text)
    assert [f["screen"] for f in second["findings"] if f["check"] == "owner"] == [2]

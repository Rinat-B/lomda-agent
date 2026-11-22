"""The conversation at the gates: intents (question, status, cancel, back), / commands at a gate,
and going back to an earlier gate through the checkpoint history (time travel). Fake models."""

from pathlib import Path

import pytest

import agent.nodes as nodes
from agent import cli, gates
from agent.graph import build_graph
from agent.messages import msg
from agent.navigation import find_gate_snapshot, rerun_estimate, status_text
from agent.session import Session, SourceDoc, UsageRecord
from tests.test_lesson_flow import FakeModel, deck_chunks


@pytest.fixture
def setup(monkeypatch, tmp_path):
    import agent.persistence as persistence

    session = Session()
    session.add_file(Path("deck.pptx"), SourceDoc("slide", 2, deck_chunks(2)))
    model = FakeModel()
    monkeypatch.setattr(nodes, "call_llm", model)
    monkeypatch.setattr(cli, "open_preview", lambda path: None)
    monkeypatch.setattr(persistence, "output_dir", lambda: tmp_path / "output")
    return session, model, build_graph()


GATE_PROMPTS = {
    "topics": lambda: msg("gate.topics.ask"),
    "structure": lambda: msg("gate.structure.ask"),
    "review": lambda: msg("gate.review.ask"),
    "back": lambda: msg("gate.back.ask"),
    "cancel": lambda: msg("gate.cancel.ask"),
}


def script(per_gate: dict[str, list[str]], log: list | None = None, save: str = "N"):
    """Answers by gate prompt; a gate with no more scripted answers gets Y. write_files: `save`,
    browser check N, the motif and anything else: Enter."""
    queues = {k: list(v) for k, v in per_gate.items()}

    def ask(prompt: str) -> str:
        if log is not None:
            log.append(prompt)
        keys = {name: text() for name, text in GATE_PROMPTS.items()}
        keys["type"] = None
        for name, text in keys.items():
            is_type = name == "type" and prompt in (msg("gate.type.ask", type=msg("lesson_types.training")),
                                                    msg("gate.type.ask", type=msg("lesson_types.compliance")))
            if prompt == text or is_type:
                q = queues.get(name) or []
                return q.pop(0) if q else "Y"
        if prompt == msg("gate.write_files.ask"):
            return save
        if prompt.startswith(msg("gate.browser_check.ask", count=0).split("0")[0]):
            return "N"
        return ""

    return ask


def gate_answers(graph, session) -> list[str]:
    """Every answer a gate showed in this thread (from the checkpoint history; the printed text is
    reordered for the Hebrew console)."""
    return [snap.values["gate_answer"] for snap in graph.get_state_history(cli.graph_config(session))
            if snap.values.get("gate_answer")]


def values(graph, session):
    return graph.get_state(cli.graph_config(session)).values


def runs(model, kind):
    return model.kinds.count(kind)


# ---- intents --------------------------------------------------------------------------------------
def test_a_question_is_answered_with_the_run_state_and_the_gate_stays(setup, output):
    session, model, graph = setup
    cli.run_question("build", session, graph, ask_fn=script({"topics": ["למה בחרת את הנושא הזה?"]}))
    out = output()
    assert "FAKE ANSWER | with run state" in out
    assert msg("gate.still_here.topics") in out
    assert runs(model, "topic_map") == 1  # the map did not change


def test_status_says_where_we_are_what_is_done_and_the_cost(setup, output):
    session, model, graph = setup
    cli.run_question("build", session, graph, ask_fn=script({"structure": ["איפה אנחנו?"]}))
    answers = gate_answers(graph, session)
    assert any(msg("status.where", stage=msg("status.stage.structure"), number=3, total=6) in a for a in answers)
    assert any(msg("status.done.topics", count=len(values(graph, session)["topic_map"])) in a for a in answers)
    assert any(msg("gate.still_here.structure") in a for a in answers)
    assert "answer" not in model.kinds  # status is code, no model call


def test_status_text_counts_calls_and_cost():
    records = [UsageRecord("topic_map", "m", 1, 1, 0.01), UsageRecord("plan_structure", "m", 1, 1, 0.02)]
    text = status_text({"lesson_type": "training", "topic_map": [{}] * 3}, "structure", records)
    assert msg("status.cost", calls=2, cost=0.03) in text


def test_cancel_asks_then_pauses_at_the_gate_for_resume(setup, output):
    session, model, graph = setup
    cli.run_question("build", session, graph, ask_fn=script({"topics": ["עזוב, לא צריך"], "cancel": ["Y"]}))
    assert msg("gate.cancel.done") in output()
    snapshot = graph.get_state(cli.graph_config(session))
    assert snapshot.next == ("gate_topics",) and snapshot.interrupts
    assert not snapshot.interrupts[0].value.get("confirm")  # /resume shows the gate, not the question
    cli.run_graph(session, graph, None, ask_fn=script({}))  # /resume
    assert values(graph, session)["lesson"]


def test_cancel_declined_keeps_going(setup, output):
    session, model, graph = setup
    cli.run_question("build", session, graph, ask_fn=script({"topics": ["עזוב"], "cancel": ["N"]}))
    assert msg("gate.cancel.declined") in output() and values(graph, session)["lesson"]


def test_a_slash_command_at_a_gate_runs_and_the_gate_asks_again(setup, output):
    session, model, graph = setup
    done = []
    gates.set_command_hook(lambda line: done.append(line))
    inner = script({})
    sent: list[str] = []

    def ask(prompt: str) -> str:
        if not sent and prompt == msg("gate.topics.ask"):
            sent.append("/cost")
            return "/cost"
        return inner(prompt)

    try:
        cli.run_question("build", session, graph, ask_fn=ask)
    finally:
        gates.set_command_hook(None)
    assert done == ["/cost"]
    assert "classify" not in model.kinds  # never sent to the classifier
    assert values(graph, session)["lesson"]


# ---- going back (time travel) ---------------------------------------------------------------------
def test_back_from_gate_3_to_gate_1_with_a_change(setup, output):
    session, model, graph = setup
    cli.run_question("בנה לומדה של 5 מסכים", session, graph,
                     ask_fn=script({"review": ["רגע, תוריד את הנושא השלישי", "Y"], "back": ["Y"]}))
    state = values(graph, session)
    out = output()
    assert msg("gate.back.title.topics") in out and msg("gate.back.going.topics") in out
    assert runs(model, "classify_type") == 1 and runs(model, "research_ideas") == 1  # not again
    assert runs(model, "topic_map") == 2 and runs(model, "plan_structure") == 2
    assert "remove the third topic" in model.prompts_of("topic_map")[1]  # the change reached the map
    assert state["lesson_type"] == "training" and state["requirements"]["screens"] == 5  # kept
    assert state["lesson"] and len(state["lesson"]["screens"]) == len(state["structure"]["screens"])


def test_back_from_gate_3_to_gate_2_without_a_change(setup, output):
    session, model, graph = setup
    cli.run_question("build", session, graph, ask_fn=script({"review": ["תחזור למבנה", "Y"], "back": ["Y"]}))
    assert msg("gate.back.going.structure") in output()
    assert runs(model, "topic_map") == 1  # the map stays
    assert runs(model, "plan_structure") == 1  # gate 2 is shown again with the same structure
    writes = runs(model, "write")
    assert writes >= 2 and values(graph, session)["lesson"]  # the lesson is written again after gate 2


def test_back_declined_stays_at_the_gate(setup, output):
    session, model, graph = setup
    cli.run_question("build", session, graph, ask_fn=script({"review": ["תחזור לנושאים", "Y"], "back": ["N"]}))
    assert msg("gate.back.declined") in output() and runs(model, "topic_map") == 1


def test_no_gate_before_the_topics():
    from agent.navigation import ALLOWED

    assert ALLOWED["topics"] == () and set(ALLOWED["review"]) == {"topics", "structure"}


def test_the_estimate_counts_the_steps_that_run_again():
    records = [UsageRecord(n, "m", 1, 1, 0.0) for n in
               ("classify_type", "research", "topic_map", "plan_structure", "motif", "motif", "write_lesson", "evaluator")]
    steps, calls = rerun_estimate("topics", records)
    assert calls == 6 and "research" not in steps
    assert rerun_estimate("structure", records)[1] == 5


def test_find_gate_snapshot_returns_none_without_a_visit():
    class G:
        def get_state_history(self, config):
            return iter(())

    assert find_gate_snapshot(G(), {}, "gate_topics") is None

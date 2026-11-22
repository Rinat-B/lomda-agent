"""The run's spending cap (MAX_COST_USD): a [Y/N] gate before a paid node that would cross it.
Y continues with a raised cap; N with a whole lesson finishes in order (lesson and report.md
saved, open findings marked); N before the lesson pauses the run, and /resume continues it,
also from a new process (the SQLite checkpoint). Every model call is faked with a fixed cost."""

import pytest

import agent.nodes as nodes
from agent import cli, cost_cap
from agent.gates import CostPause, ask_cost_gate
from agent.graph import build_graph
from agent.messages import msg
from agent.persistence import open_checkpointer
from tests.test_review_flow import ReviewModel, deck_session, out_dir  # noqa: F401 - fixtures

CALL_COST = 0.1


class PricedModel(ReviewModel):
    """The review model of tests/test_review_flow.py; every call costs CALL_COST."""

    def __call__(self, messages, **kwargs):
        result = super().__call__(messages, **kwargs)
        result.cost = CALL_COST
        return result


def scripted(cost_answers: list[str], save: str = "Y"):
    """Y at the type, topics, structure and review gates; the cost gate from `cost_answers`."""
    cost_asked: list[str] = []
    answers = iter(cost_answers)

    def ask(prompt: str) -> str:
        if prompt == msg("cost_cap.ask"):
            cost_asked.append(prompt)
            return next(answers)
        if prompt == msg("gate.write_files.ask"):
            return save
        if prompt.startswith(msg("gate.browser_check.ask", count=0).split("0")[0]):
            return "N"
        return "Y"

    ask.cost_asked = cost_asked  # type: ignore[attr-defined]
    return ask


def run(monkeypatch, session, ask, cap: float, graph=None):
    monkeypatch.setenv("MAX_COST_USD", str(cap))
    model = PricedModel()
    monkeypatch.setattr(nodes, "call_llm", model)
    graph = graph or build_graph()
    cli.run_question("build the lesson", session, graph, ask_fn=ask)
    return graph, model


def state_of(graph, session) -> dict:
    return graph.get_state(cli.graph_config(session)).values


# ---- the cap and the estimate ----------------------------------------------------------------------
def test_max_cost_usd_in_env_wins_over_the_default(monkeypatch):
    monkeypatch.delenv("MAX_COST_USD", raising=False)
    assert cost_cap.default_cap() == 1.5  # models.yaml, as in the SPEC
    monkeypatch.setenv("MAX_COST_USD", "0.8")
    assert cost_cap.default_cap() == 0.8
    monkeypatch.setenv("MAX_COST_USD", "not a number")
    assert cost_cap.default_cap() == 1.5
    assert cost_cap.run_cap({"cost_cap": 2.25}) == 2.25  # a Y raised it for this run


def test_the_estimate_uses_this_nodes_calls_or_the_runs_mean():
    from agent.session import UsageRecord

    records = [UsageRecord("write_lesson", "m", 1, 1, 0.3), UsageRecord("evaluator", "m", 1, 1, 0.1),
               UsageRecord("evaluator", "m", 1, 1, 0.1)]
    node, rest = cost_cap.estimate(records, "evaluator", 4, 10)
    assert node == pytest.approx(0.4) and rest == pytest.approx(1.0)
    node, _ = cost_cap.estimate(records, "fix_screens", 2, 0)  # no fix call yet: the run's mean
    assert node == pytest.approx(2 * 0.5 / 3)
    assert not cost_cap.crosses(0.5, 0.0, 0.4)  # a free step never asks
    assert cost_cap.raised_cap(1.2, 0.5, 1.5) == pytest.approx(1.71)


def test_the_gate_shows_the_cost_so_far_and_the_estimate(output):
    request = cost_cap.gate_request("evaluator", 1.2345, 0.4, 0.9, 1.5, True)
    shown: list[str] = []
    assert ask_cost_gate(request, lambda p: "Y", lambda text, **k: shown.append(text)) == {"confirm": True}
    assert msg("cost_cap.spent", spent=1.2345, node="evaluator", estimate=0.4) in shown
    assert msg("cost_cap.rest", rest=0.9) in shown
    assert ask_cost_gate(request, lambda p: "N", lambda *a, **k: None) == {"confirm": False}
    with pytest.raises(CostPause):  # no lesson yet: N pauses the run
        ask_cost_gate({**request, "lesson_ready": False}, lambda p: "N", lambda *a, **k: None)


# ---- in the graph ----------------------------------------------------------------------------------
def test_y_raises_the_cap_for_this_run_and_the_lesson_is_finished(monkeypatch, deck_session, output, out_dir):
    ask = scripted(["Y", "Y", "Y", "Y"])
    # 0.75: the research estimate (8 calls) already crosses it, then the cap is raised as needed
    graph, model = run(monkeypatch, deck_session, ask, cap=0.75)
    state = state_of(graph, deck_session)
    assert ask.cost_asked  # the gate was asked
    assert state["cost_cap"] > 0.75 and not state.get("cost_stop")
    assert len(model.revisions) == 1  # the fix round still ran
    assert [f["status"] for f in state["findings"]] == ["fixed"]
    assert (out_dir / "deck" / "index.html").is_file()
    assert msg("cost_cap.raised", cap=state["cost_cap"]) in output()


def test_n_keeps_the_whole_lesson_and_the_report_says_the_run_stopped(monkeypatch, deck_session, output, out_dir):
    ask = scripted(["N"])
    # 1.25: research, planning (with two motif candidates), writing and the illustrations fit; the first full check does not
    graph, model = run(monkeypatch, deck_session, ask, cap=1.25)
    state = state_of(graph, deck_session)
    assert len(ask.cost_asked) == 1
    stop = state["cost_stop"]
    assert stop["cap"] == 1.25 and stop["node"] == "evaluator"
    assert model.revisions == []  # the fix rounds were skipped
    assert state["lesson"] and len(state["lesson"]["screens"]) == len(state["structure"]["screens"])
    folder = out_dir / "deck"
    assert (folder / "index.html").is_file() and (folder / "lesson.json").is_file()
    report = (folder / "report.md").read_text(encoding="utf-8")
    open_ = [f for f in state["findings"] if f["status"] == "open"]
    assert msg("report.cost_stop", node=stop["node"], spent=stop["spent"], cap=stop["cap"], open=len(open_)) in report
    assert all(f["status"] == "open" for f in state["findings"])
    if open_:
        assert open_[0]["problem"] in report
    assert sum(r.cost for r in deck_session.usage) <= 1.25  # nothing paid after the stop
    from agent.gates import format_review

    assert msg("cost_cap.gate_line", **stop) in format_review(nodes.review_summary(state))  # gate 3 says it too
    assert "$1.1000" in output()


def test_n_before_the_lesson_pauses_and_resume_continues_from_a_new_process(monkeypatch, deck_session, output,
                                                                             out_dir, tmp_path):
    db = tmp_path / "checkpoints.sqlite"
    ask = scripted(["N"])
    # 0.25: the research step (about 8 calls) would cross it, long before the lesson is written
    graph, model = run(monkeypatch, deck_session, ask, cap=0.25, graph=build_graph(open_checkpointer(db)))
    state = state_of(graph, deck_session)
    assert state.get("lesson") is None  # stopped before the lesson was written
    snapshot = graph.get_state(cli.graph_config(deck_session))
    assert snapshot.interrupts and snapshot.interrupts[0].value["gate"] == "cost_cap"
    assert msg("cost_cap.paused") in output()
    calls_before = len(deck_session.usage)

    # a new process: a new graph on the same checkpoint file, then /resume answers Y
    again = build_graph(open_checkpointer(db))
    from agent.persistence import ConversationStore, paused_conversations

    store = ConversationStore(tmp_path / "store.sqlite")
    store.save(deck_session)
    assert [p.gate for p in paused_conversations(again, store)] == ["cost_cap"]
    assert cli.run_graph(deck_session, again, None, ask_fn=scripted(["Y", "Y", "Y", "Y", "Y"]))
    final = state_of(again, deck_session)
    assert final["lesson"] and final["saved_path"]
    assert len(deck_session.usage) > calls_before  # it continued, with no node run twice before the gate


def test_a_run_stopped_inside_a_node_is_listed_and_continues(monkeypatch, deck_session, output, out_dir, tmp_path):
    """Ctrl+C or a closed window inside a node: the last node's checkpoint is kept."""
    db = tmp_path / "checkpoints.sqlite"
    monkeypatch.setenv("MAX_COST_USD", "100")
    model = PricedModel()
    calls = {"n": 0}

    def dies_once_in_the_evaluator(messages, **kwargs):
        if "proofread" in messages[0].content and calls["n"] == 0:
            calls["n"] += 1
            raise KeyboardInterrupt
        return model(messages, **kwargs)

    monkeypatch.setattr(nodes, "call_llm", dies_once_in_the_evaluator)
    graph = build_graph(open_checkpointer(db))
    cli.run_question("build the lesson", deck_session, graph, ask_fn=scripted([]))
    snapshot = graph.get_state(cli.graph_config(deck_session))
    assert snapshot.next and not snapshot.interrupts  # stopped inside a node
    from agent.persistence import ConversationStore, paused_conversations

    store = ConversationStore(tmp_path / "store.sqlite")
    store.save(deck_session)
    again = build_graph(open_checkpointer(db))
    assert [p.gate for p in paused_conversations(again, store)] == ["stopped"]
    assert cli.run_graph(deck_session, again, None, ask_fn=scripted([]))
    assert state_of(again, deck_session)["saved_path"]

"""The conversation around the flow: the opening message with an estimate, one clarification
question (audience, length) with defaults, the agent's suggestions (thin material, open findings)
recorded in report.md, and the reasons shown at the gates. Fake models."""

from pathlib import Path

import pytest

import agent.nodes as nodes
from agent import cli
from agent.conversation import default_screens, estimate, missing, opening_message
from agent.graph import build_graph
from agent.messages import msg
from agent.session import Session, SourceDoc
from tests.test_lesson_flow import FakeModel, deck_chunks
from tests.test_navigation import script


def make(monkeypatch, tmp_path, chunks=None):
    import agent.persistence as persistence

    session = Session()
    session.add_file(Path("deck.pptx"), SourceDoc("slide", 2, chunks or deck_chunks(2)))
    model = FakeModel()
    monkeypatch.setattr(nodes, "call_llm", model)
    monkeypatch.setattr(cli, "open_preview", lambda path: None)
    monkeypatch.setattr(persistence, "output_dir", lambda: tmp_path / "output")
    return session, model, build_graph()


def asked_prompts(per_gate=None, clarify_answer=""):
    log: list[str] = []
    inner = script(per_gate or {}, save="Y")

    def ask(prompt: str) -> str:
        log.append(prompt)
        if prompt == msg("conversation.ask"):
            return clarify_answer
        return inner(prompt)

    return ask, log


# ---- units ---------------------------------------------------------------------------------------------
def test_what_is_missing():
    assert missing({}, None) == ["audience", "screens"]
    assert missing({"audience": "נהגים"}, None) == ["screens"]
    assert missing({"screens": 5}, "נהגי החברה") == []


def test_the_estimate_grows_with_the_lesson_and_depends_on_the_profile():
    small, big = estimate(5, "training", "free"), estimate(12, "training", "free")
    assert big[0] > small[0] and small[1] > estimate(5, "training", "balanced")[1]
    assert estimate(5, "compliance", "free")[0] < small[0]  # no web research
    assert 5 <= default_screens(1) and default_screens(50) <= 12
    text = opening_message(8, "training", "free", 5)
    assert "3" in text and msg("lesson_types.training") in text


# ---- in the graph --------------------------------------------------------------------------------------
def test_one_clarification_with_defaults_when_audience_and_length_are_missing(monkeypatch, tmp_path, output):
    session, model, graph = make(monkeypatch, tmp_path)
    ask, log = asked_prompts()
    cli.run_question("build", session, graph, ask_fn=ask)
    assert log.count(msg("conversation.ask")) == 1
    state = graph.get_state(cli.graph_config(session)).values
    assert state["requirements"]["audience"] == msg("conversation.default_audience")
    assert state["requirements"]["screens_about"] == default_screens(2)
    assert "about" in model.prompts_of("plan_structure")[0]


def test_no_clarification_when_the_request_says_it_all(monkeypatch, tmp_path, output):
    import agent.conversation as conversation

    monkeypatch.setattr(conversation, "is_thin", lambda chunks: False)  # no suggestion either
    session, model, graph = make(monkeypatch, tmp_path)
    ask, log = asked_prompts()
    cli.run_question("בנה לומדה של 5 מסכים לנהגים חדשים", session, graph, ask_fn=ask)
    assert msg("conversation.ask") not in log


def test_an_answer_sets_the_requirements(monkeypatch, tmp_path, output):
    session, model, graph = make(monkeypatch, tmp_path)
    ask, log = asked_prompts(clarify_answer="מנהלי משמרת, 6 מסכים")
    cli.run_question("build", session, graph, ask_fn=ask)
    state = graph.get_state(cli.graph_config(session)).values
    assert state["requirements"]["screens"] == 6 and len(state["lesson"]["screens"]) == 6
    assert "clarify" in model.kinds  # the small model read the answer


def test_the_opening_message_says_the_stages_and_the_estimate(monkeypatch, tmp_path, output):
    session, model, graph = make(monkeypatch, tmp_path)
    ask, _ = asked_prompts()
    cli.run_question("build", session, graph, ask_fn=ask)
    calls, minutes = estimate(default_screens(2), "training", "free")
    assert str(calls) in output()


def test_thin_material_gets_a_suggestion_recorded_in_the_report(monkeypatch, tmp_path, output):
    session, model, graph = make(monkeypatch, tmp_path)  # the fake deck is a few hundred characters
    ask, log = asked_prompts(clarify_answer="תרחיבי חיפוש")
    cli.run_question("build", session, graph, ask_fn=ask)
    state = graph.get_state(cli.graph_config(session)).values
    start = [s for s in state["suggestions"] if s["when"] == "start"]
    assert start and start[0]["reason"]
    report = Path(state["report_path"]).read_text(encoding="utf-8")
    assert msg("report.suggestions_title") in report


def test_the_type_reason_and_topic_reasons_are_shown(monkeypatch, tmp_path, output):
    session, model, graph = make(monkeypatch, tmp_path)
    ask, _ = asked_prompts()
    cli.run_question("build", session, graph, ask_fn=ask)
    state = graph.get_state(cli.graph_config(session)).values
    assert state["reasons"]["type"]


@pytest.mark.parametrize("open_count, expected", [(0, None), (2, "review")])
def test_open_findings_get_a_suggestion(open_count, expected):
    from agent.conversation import review_suggestion

    s = review_suggestion(open_count)
    assert (s or {}).get("when") == expected

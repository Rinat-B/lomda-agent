"""The owner's explicit requirements (screens, audience, emphasis, style): taken from the request,
kept in the state, and honoured by the plan and the writer; a conflict is said at gate 2."""

import json
from pathlib import Path

import agent.nodes as nodes
from agent import cli
from agent.gates import format_screens_note
from agent.graph import build_graph
from agent.messages import msg
from agent.requirements import merge, parse_requirements, prompt_lines, screens_target
from agent.session import Session, SourceDoc
from tests.test_lesson_flow import FakeModel, deck_chunks, scripted


def test_parse_and_merge():
    req = parse_requirements({"requirements": {"screens": "5", "audience": "נהגים חדשים", "style": "null"}})
    assert req == {"audience": "נהגים חדשים", "emphasis": None, "style": None, "screens": 5}
    assert merge(req, {"emphasis": "בטיחות", "screens": None})["emphasis"] == "בטיחות"
    assert "number of screens: 5" in prompt_lines(req) and "target audience" in prompt_lines(req)
    assert prompt_lines({}) == ""


def test_the_screen_count_is_met_or_explained():
    assert screens_target({"screens": 5}, 4) == (5, None)
    target, note = screens_target({"screens": 2}, 6)
    assert target == 6 and note["requested"] == 2 and note["why"] == "topics"
    assert msg("gate.structure.screens_note_topics", requested=2, used=6, topics=6) == format_screens_note(note)
    assert screens_target({"screens": 99}, 4)[1]["why"] == "max"
    assert screens_target({}, 4) == (None, None)


def run(monkeypatch, request: str, slides: int = 2):
    import agent.persistence as persistence

    session = Session()
    session.add_file(Path("deck.pptx"), SourceDoc("slide", slides, deck_chunks(slides)))
    model = FakeModel()
    monkeypatch.setattr(nodes, "call_llm", model)
    monkeypatch.setattr(cli, "open_preview", lambda path: None)
    monkeypatch.setattr(persistence, "output_dir", lambda: Path("NO_OUTPUT"))
    graph = build_graph()
    cli.run_question(request, session, graph, ask_fn=scripted(["Y", "Y", "Y"]))
    return graph.get_state(cli.graph_config(session)).values, model


def test_build_a_lesson_of_5_screens_gives_5_screens(monkeypatch, output):
    state, model = run(monkeypatch, "בנה לומדה של 5 מסכים לנהגים חדשים")
    assert state["requirements"]["screens"] == 5 and state["requirements"]["audience"] == "נהגים חדשים"
    assert len(state["structure"]["screens"]) == 5 and len(state["lesson"]["screens"]) == 5
    plan_prompt = model.prompts_of("plan_structure")[0]
    assert "Plan exactly 5 screens" in plan_prompt and "target audience: נהגים חדשים" in plan_prompt
    assert all("target audience: נהגים חדשים" in u for u in model.prompts_of("write"))
    assert state["reasons"]["type"] == "סיבה מזויפת"


def test_fewer_screens_than_topics_is_said_at_gate_2(monkeypatch, output):
    state, _ = run(monkeypatch, "בנה לומדה של 2 מסכים")
    note = state["screens_note"]
    assert note["requested"] == 2 and note["used"] == len(state["topic_map"])
    assert "ביקשת 2 מסכים" in output()

"""Lesson building: intent classification, topic map, and the [Y/N] approval gate.

All model calls are faked; the real graph runs with its in-memory checkpointer.
"""

import json
from pathlib import Path

import pytest

import agent.nodes as nodes
from agent.cli import run_question
from agent.gates import format_topic_map, parse_decision
from agent.graph import build_graph
from agent.messages import msg
from agent.session import Session, SourceDoc
from agent.tools import ChunkIndex
from agent.topics import MIN_UNIT_CHARS, build_units, fit_to_budget, parse_topic_map, topic_map_json
from tests.conftest import make_result

DATA = json.loads((Path(__file__).parent / "fixtures" / "retrieval_he.json").read_text(encoding="utf-8"))
PREFIX = msg("topic_map.objective_prefix")


def rendered(key: str, **kw) -> str:
    """A message as rich prints it from Markdown (without the leading # marks)."""
    return msg(key, **kw).lstrip("# ")

# Hebrew words from code points (no Hebrew literals in .py files).
KEN = chr(0x05DB) + chr(0x05DF)  # "yes"
LO = chr(0x05DC) + chr(0x05D0)  # "no"
TET = chr(0x05D8)  # the Y key on a Hebrew keyboard
MEM = chr(0x05DE)  # the N key on a Hebrew keyboard
MA = chr(0x05DE) + chr(0x05D4)  # "what": starts with MEM but is a different word


# ---- gate answers -------------------------------------------------------------------------
@pytest.mark.parametrize("text", ["Y", "y", "yes", KEN, TET, f"  {TET}  "])
def test_approve_answers(text):
    assert parse_decision(text).approved is True


@pytest.mark.parametrize(
    "text, feedback",
    [("N add tires", "add tires"), ("n: add tires", "add tires"), (f"{LO} add tires", "add tires"),
     (f"{MEM} add tires", "add tires"), ("N", ""), (MEM, "")],
)
def test_change_answers(text, feedback):
    decision = parse_decision(text)
    assert decision.approved is False and decision.feedback == feedback


@pytest.mark.parametrize("text", ["please add a topic on tires", f"{MA} about tires?"])
def test_free_text_is_a_change_request(text):
    decision = parse_decision(text)
    assert decision.approved is False and decision.feedback == text


def test_empty_answer_asks_again():
    assert parse_decision("   ") is None


# ---- intent -------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "reply, intent",
    [("BUILD_LESSON", nodes.INTENT_BUILD), ("QUESTION", nodes.INTENT_QUESTION),
     ("OUT_OF_SCOPE", nodes.INTENT_OUT), ("unclear", nodes.INTENT_QUESTION)],
)
def test_classify_intent(reply, intent):
    assert nodes.classify_intent(reply) == intent


def test_build_lesson_without_files_asks_for_add(monkeypatch):
    monkeypatch.setattr(nodes, "call_llm", lambda m: make_result(text="BUILD_LESSON"))
    result = nodes.scope_check({"question": "build"}, {"configurable": {"index": ChunkIndex()}})
    assert result == {"intent": nodes.INTENT_NEEDS_FILES, "answer": msg("scope.need_files")}
    assert nodes.route_after_scope_check(result) == "end"


# ---- units and budget -----------------------------------------------------------------------
def deck_chunks(slides: int, text: str = "slide text") -> list[dict]:
    return [{"text": f"{text} {s}", "kind": "body", "locator": {"file": "deck.pptx", "slide": s, "paragraph": p}}
            for s in range(1, slides + 1) for p in (1, 2)]


def test_units_group_by_slide_and_by_heading():
    chunks = deck_chunks(2) + [
        {"text": "Heading A", "kind": "heading", "locator": {"file": "doc.docx", "paragraph": 1}},
        {"text": "a1", "kind": "body", "locator": {"file": "doc.docx", "paragraph": 2}},
        {"text": "Heading B", "kind": "heading", "locator": {"file": "doc.docx", "paragraph": 3}},
    ]
    units = build_units(chunks)
    assert [u.locator for u in units] == [
        {"file": "deck.pptx", "slide": 1},
        {"file": "deck.pptx", "slide": 2},
        {"file": "doc.docx", "paragraph": 1, "paragraph_end": 2},
        {"file": "doc.docx", "paragraph": 3},
    ]
    assert [u.id for u in units] == ["U1", "U2", "U3", "U4"]


def test_budget_shortens_but_keeps_every_file_and_unit():
    chunks = deck_chunks(40, "x" * 2000) + [
        {"text": "small file text", "kind": "body", "locator": {"file": "small.txt", "paragraph": 1}}
    ]
    units = build_units(chunks)
    fitted = fit_to_budget(units, budget=10_000)
    assert [u.id for u in fitted] == [u.id for u in units]  # nothing dropped
    assert {u.locator["file"] for u in fitted} == {"deck.pptx", "small.txt"}
    assert all(len(u.text) <= max(MIN_UNIT_CHARS, 5_000 // 40) for u in fitted if u.locator["file"] == "deck.pptx")
    assert fitted[-1].text == "small file text"


# ---- structured output ----------------------------------------------------------------------
def topics_json(n: int, extra_name: str | None = None, ids=("U1",)) -> str:
    topics = [{"name": f"Topic {i}", "objective": f"{PREFIX}x {i}", "sources": list(ids)} for i in range(1, n + 1)]
    if extra_name:
        topics.append({"name": extra_name, "objective": f"{PREFIX}y", "sources": list(ids)})
    return json.dumps({"topics": topics}, ensure_ascii=False)


def test_parse_topic_map_maps_ids_to_locators():
    units = build_units(deck_chunks(2))
    topics = parse_topic_map("Sure! " + topics_json(4, ids=("U1", "U2", "U2")), units, PREFIX)
    assert len(topics) == 4
    assert topics[0]["sources"] == [{"file": "deck.pptx", "slide": 1}, {"file": "deck.pptx", "slide": 2}]


@pytest.mark.parametrize(
    "reply",
    [topics_json(3), topics_json(9), topics_json(4, ids=("U99",)), "no json here",
     json.dumps({"topics": [{"name": f"T{i}", "objective": "wrong start", "sources": ["U1"]} for i in range(4)]})],
    ids=["too few", "too many", "unknown id", "no json", "objective prefix"],
)
def test_parse_topic_map_rejects_invalid(reply):
    with pytest.raises(ValueError):
        parse_topic_map(reply, build_units(deck_chunks(2)), PREFIX)


def test_previous_map_round_trips_to_unit_ids():
    units = build_units(deck_chunks(2))
    topics = parse_topic_map(topics_json(4, ids=("U2",)), units, PREFIX)
    assert json.loads(topic_map_json(topics, units))["topics"][0]["sources"] == ["U2"]


# ---- the whole flow -------------------------------------------------------------------------
class FakeModel:
    """Gatekeeper says BUILD_LESSON; the topic mapper answers 4 topics, then 5 after feedback.
    Optionally the first topic-map reply is invalid, to exercise the retry."""

    def __init__(self, invalid_first: bool = False):
        self.prompts: list[tuple[str, str]] = []
        self.timeouts: list = []
        self.invalid_first = invalid_first

    def __call__(self, messages, **kwargs):
        self.timeouts.append(kwargs.get("timeout"))
        system, user = messages[0].content, messages[-1].content
        self.prompts.append((system, user))
        if system.startswith("You are a gatekeeper"):
            return make_result(text="BUILD_LESSON")
        if self.invalid_first:
            self.invalid_first = False
            return make_result(text="not json")
        full_user = messages[1].content
        if "Reviewer asked" in full_user or "reviewer asked" in full_user:
            return make_result(text=topics_json(4, extra_name="TIRES"))
        return make_result(text=topics_json(4))

    def topic_map_prompts(self):
        return [u for s, u in self.prompts if s.startswith("You are an instructional designer")]


@pytest.fixture
def session_with_deck():
    session = Session()
    session.add_file(Path("deck.pptx"), SourceDoc("slide", 2, deck_chunks(2)))
    return session


def run_flow(monkeypatch, session, answers: list[str], model: FakeModel):
    monkeypatch.setattr(nodes, "call_llm", model)
    asked = iter(answers)
    run_question("build me a lesson from the deck", session, build_graph(), ask_fn=lambda prompt: next(asked))


def test_approve_right_away(monkeypatch, session_with_deck, output):
    model = FakeModel()
    run_flow(monkeypatch, session_with_deck, ["Y"], model)
    out = output()
    assert msg("gate.topics.approved") in out
    assert len(session_with_deck.topic_map) == 4
    assert "[INFO] Entering Node: topic_map" in out and "[INFO] Entering Node: gate_topics" in out
    assert rendered("gate.topics.title", count=4) in out


def test_change_request_goes_back_to_topic_map_then_approve(monkeypatch, session_with_deck, output):
    model = FakeModel()
    run_flow(monkeypatch, session_with_deck, ["N add a topic on tires", "Y"], model)
    prompts = model.topic_map_prompts()
    assert len(prompts) == 2
    assert "apply this feedback: add a topic on tires" in prompts[1]
    assert "Your previous topic map" in prompts[1]
    assert [t["name"] for t in session_with_deck.topic_map][-1] == "TIRES"
    out = output()
    assert rendered("gate.topics.title", count=5) in out and msg("gate.topics.approved") in out


def test_hebrew_keyboard_letters(monkeypatch, session_with_deck, output):
    model = FakeModel()
    run_flow(monkeypatch, session_with_deck, [f"{MEM} add a topic on tires", TET], model)
    assert len(model.topic_map_prompts()) == 2
    assert session_with_deck.topic_map is not None and len(session_with_deck.topic_map) == 5


def test_n_without_comment_asks_what_to_change(monkeypatch, session_with_deck, output):
    model = FakeModel()
    run_flow(monkeypatch, session_with_deck, ["N", "add tires", "y"], model)
    assert "apply this feedback: add tires" in model.topic_map_prompts()[1]


def test_free_text_and_empty_answers(monkeypatch, session_with_deck, output):
    model = FakeModel()
    run_flow(monkeypatch, session_with_deck, ["", "please add tires", KEN], model)
    assert msg("gate.topics.ask_again") in output()
    assert "apply this feedback: please add tires" in model.topic_map_prompts()[1]


def test_invalid_model_json_is_retried(monkeypatch, session_with_deck, output):
    model = FakeModel(invalid_first=True)
    run_flow(monkeypatch, session_with_deck, ["Y"], model)
    assert len(model.topic_map_prompts()) == 2  # retry inside topic_map
    assert len(session_with_deck.topic_map) == 4


def test_a_new_question_after_the_gate_starts_fresh(monkeypatch, session_with_deck, output):
    model = FakeModel()
    run_flow(monkeypatch, session_with_deck, ["N add tires", "Y"], model)
    run_flow(monkeypatch, session_with_deck, ["Y"], model)
    # The second run starts from a fresh state: no leftover feedback or previous map.
    assert "Your previous topic map" not in model.topic_map_prompts()[-1]


def test_topic_map_uses_the_long_timeout(monkeypatch, session_with_deck, output):
    from agent.llm import LONG_REQUEST_TIMEOUT_SECONDS

    model = FakeModel()
    run_flow(monkeypatch, session_with_deck, ["Y"], model)
    assert model.timeouts == [None, LONG_REQUEST_TIMEOUT_SECONDS]  # scope_check default, topic_map long


def test_no_edge_bypasses_the_gate():
    edges = {(e.source, e.target) for e in build_graph().get_graph().edges}
    assert {t for s, t in edges if s == "topic_map"} == {"gate_topics"}
    assert {t for s, t in edges if s == "gate_topics"} == {"topic_map", "__end__"}
    assert not any(t == "__end__" and s == "topic_map" for s, t in edges)


def test_format_topic_map_shows_topic_objective_sources():
    topics = parse_topic_map(topics_json(4), build_units(deck_chunks(2)), PREFIX)
    text = format_topic_map(topics)
    assert msg("gate.topics.topic", index=1, name="Topic 1") in text
    assert msg("gate.topics.objective", objective=f"{PREFIX}x 1") in text
    assert msg("rag.source.slide", file="deck.pptx", number=1) in text

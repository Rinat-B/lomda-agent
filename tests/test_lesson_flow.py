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
    monkeypatch.setattr(nodes, "call_llm", lambda m, **kw: make_result(text="BUILD_LESSON"))
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
def structure_json_reply(user: str, extra: bool = False, ids=("U1",)) -> str:
    """A valid plan_structure reply: one screen per topic of the approved map in the prompt."""
    start = user.index("Approved topic map (JSON):\n") + len("Approved topic map (JSON):\n")
    topics = json.loads(user[start:user.index("\n", start)])["topics"]
    screens = [
        {"number": i, "title": f"Screen {i}", "topic": t["name"],
         "points": [{"text": f"point {i}a", "sources": list(ids)}, {"text": f"point {i}b", "sources": list(ids)}],
         "question_idea": f"decide something {i}", "illustration": None if i % 2 else f"illustration {i}",
         "decision_diagram": i == 1}
        for i, t in enumerate(topics, 1)
    ]
    if extra:
        screens.append({**screens[0], "number": len(screens) + 1, "title": "EXTRA"})
    import re as _re

    wanted = _re.search(r"Plan exactly (\d+) screens", user)
    while wanted and len(screens) < int(wanted.group(1)):  # a second screen for a topic, as asked
        screens.append({**screens[len(screens) % len(topics)], "number": len(screens) + 1,
                        "title": f"Screen {len(screens) + 1}"})
    return json.dumps({"screens": screens, "motifs": ["motif one", "motif two"]}, ensure_ascii=False)


FAKE_MOTIF = '{"idea": "Winter road", "icons": ["car", "route"], "accent": "primary", "layout": "pair"}'
EMPTY_LOGIC_TABLE = ('{"source_rules": [], "decisions": [], "numbers": [], "choices": [], "cause_effect": [], '
                     '"feedback": [], "situation": {"name_missing": ""}}')


def prompt_kind(system: str) -> str:
    if system.startswith("You are a gatekeeper"):
        return "scope"
    if system.startswith("Decide which kind"):
        return "classify_type"
    if system.startswith("You are an instructional designer. Plan the screens"):
        return "plan_structure"
    if system.startswith("You are an instructional designer planning"):
        return "topic_map"
    if system.startswith("You interpret"):
        return "classify"
    if system.startswith("You help a reviewer"):
        return "answer"
    if system.startswith("You write the screens"):
        return "write"
    if system.startswith("You check the multiple-choice questions"):
        return "single_answer"
    if system.startswith("You review a Hebrew e-learning lesson"):
        return "check"
    if system.startswith("You list the key points"):
        return "key_points"
    if system.startswith("You build one interactive illustration"):
        return "illustration"
    if system.startswith("The interactive illustration you built failed"):
        return "illustration_fix"
    if system.startswith("You check the decision diagrams"):
        return "check"
    if system.startswith("You design the visual motif"):
        return "motif"
    if system.startswith("You check the LOGIC of one interactive illustration"):
        return "illustration_logic"
    if system.startswith("The interactive illustration below teaches"):
        return "illustration_logic_fix"
    if system.startswith("You design an interactive illustration"):
        return "illustration_design"
    if system.startswith("You plan web research"):
        return "research_ideas"
    if system.startswith("You compare web findings"):
        return "research_gaps"
    return "other"


FAKE_ILLUSTRATION = (
    '<!doctype html><html dir="rtl" lang="he"><body><button id="b">go</button><p id="out">0</p>'
    '<script>document.getElementById("b").onclick = () => {document.getElementById("out").textContent = "1"};'
    '</script></body></html>'
)


def checker_reply(kind: str, user: str) -> str:
    """A valid, empty-handed reply for every checker prompt (no findings)."""
    import re

    if kind == "single_answer":
        ids = re.findall(r"### Question (\S+)", user)
        return json.dumps({"questions": [{"id": i, "options": []} for i in ids]})
    if kind == "key_points":  # two points the fake lesson teaches, one it does not
        return json.dumps({"points": [{"text": "Screen 1 explains the rule.", "source": "deck.pptx, slide 1"},
                                      {"text": "Screen 2 explains the rule.", "source": "deck.pptx, slide 1"},
                                      {"text": "Tyres need 3 mm of tread.", "source": "deck.pptx, slide 2"}]})
    return json.dumps({"findings": [], "diagrams": [],
                       "illustrations": [{"adds_value": True}] * user.count(", illustration] Title")})


def writer_reply(system: str, user: str) -> str:
    """A valid write_lesson reply for the planned screens and excerpts in the prompt."""
    import re

    start = user.index("Planned screens (JSON):\n") + len("Planned screens (JSON):\n")
    plan = json.loads(user[start:user.index("\n\nSource excerpts:")])
    eid = re.findall(r"\[(E\d+)\]", user)[0]
    screens = []
    for s in plan:
        n = s["number"]
        blocks = [
            {"type": "paragraph", "text": f"Screen {n} explains the rule.", "sources": [eid]},
            {"type": "multiple_choice", "question": f"Situation {n}: what do you do?",
             "options": [f"act {n} a", f"act {n} b", f"act {n} c"], "correct_index": 1,
             "feedback_correct": "Right, because of the rule.", "feedback_incorrect": "Think about the rule again.",
             "alternate": {"question": f"Other situation {n}: what now?",
                           "options": [f"do x{n}", f"do y{n}", f"do z{n}"],
                           "correct_index": 0, "feedback_correct": "Yes.", "feedback_incorrect": "Not quite."},
             "sources": [eid]},
        ]
        if s["illustration"]:
            blocks.append({"type": "illustration", "title": f"Illustration {n}", "description": "The learner moves a slider."})
        if s["decision_diagram"]:
            blocks.append({"type": "decision_diagram", "title": "Decide",
                           "nodes": [{"id": "n1", "label": "Is it wet?", "kind": "question"},
                                     {"id": "n2", "label": "Slow down", "kind": "action"},
                                     {"id": "n3", "label": "Keep going", "kind": "action"}],
                           "edges": [{"from": "n1", "to": "n2", "label": "yes"}, {"from": "n1", "to": "n3", "label": "no"}],
                           "sources": [eid]})
        screens.append({"number": n, "blocks": blocks})
    reply: dict = {"screens": screens}
    if "This is the first group" in system:
        reply["lesson_title"] = "Fake lesson"
    if "This is the last group" in system:
        reply["summary"] = ["We learned one thing.", "And another."]
    return json.dumps(reply)


def scripted(answers, save: str = "N", review: str = "Y"):
    """ask_fn for tests: answers the gates in order; gate 3 gets `review` (default Y) and
    the write_files [Y/N] gets `save` (default N)."""
    asked = iter(answers)

    def ask(prompt: str) -> str:
        if prompt == msg("gate.write_files.ask"):
            return save
        if prompt == msg("gate.review.ask"):
            return review
        if prompt.startswith(msg("gate.browser_check.ask", count=0).split("0")[0]):
            return "N"  # unit tests do not start a browser
        if prompt.startswith(msg("gate.structure.motif_ask", count=0).split("0")[0]):
            return ""  # the first motif candidate
        return next(asked)

    return ask


class FakeModel:
    """Fake models for the whole lesson flow:
    gatekeeper -> BUILD_LESSON; classify_type -> `lesson_type`; topic map -> 4 topics (5 after
    feedback); structure -> one screen per topic (one more after feedback); gate classifier ->
    CHANGE with the reviewer's text. Optionally the first topic-map reply is invalid."""

    def __init__(self, invalid_first: bool = False, lesson_type: str = "training"):
        self.prompts: list[tuple[str, str]] = []
        self.kinds: list[str] = []
        self.timeouts: list = []
        self.invalid_first = invalid_first
        self.lesson_type = lesson_type

    def __call__(self, messages, **kwargs):
        self.timeouts.append(kwargs.get("timeout"))
        system, user = messages[0].content, messages[-1].content
        kind = prompt_kind(system)
        self.prompts.append((system, user))
        self.kinds.append(kind)
        first_user = messages[1].content
        revising = "reviewer asked" in first_user.lower()
        if kind == "scope":
            return make_result(text="BUILD_LESSON")
        if kind == "classify_type":
            import re as _re

            request = user.split("User request:", 1)[-1]
            screens = _re.search(r"(\d+) מסכים", request)
            audience = "נהגים חדשים" if "נהגים חדשים" in request else None
            return make_result(text=json.dumps({"type": self.lesson_type, "reason": "סיבה מזויפת",
                                                "requirements": {"screens": int(screens.group(1)) if screens else None,
                                                                 "audience": audience}}, ensure_ascii=False))
        if kind == "plan_structure":
            return make_result(text=structure_json_reply(first_user, extra=revising))
        if kind == "classify":
            reply = user.split("Reviewer reply: ", 1)[-1]
            return make_result(text=json.dumps({"decision": "CHANGE", "change": reply}))
        if kind == "topic_map":
            if self.invalid_first:
                self.invalid_first = False
                return make_result(text="not json")
            return make_result(text=topics_json(4, extra_name="TIRES" if revising else None))
        if kind == "write":
            return make_result(text=writer_reply(system, first_user))
        if kind in ("check", "single_answer", "key_points"):
            return make_result(text=checker_reply(kind, first_user))
        if kind in ("illustration", "illustration_fix", "illustration_logic_fix"):
            return make_result(text=FAKE_ILLUSTRATION)
        if kind == "illustration_logic":
            return make_result(text=EMPTY_LOGIC_TABLE)
        if kind == "motif":
            return make_result(text=FAKE_MOTIF, model=kwargs.get("models", ["m/free"])[0])
        if kind == "illustration_design":
            return make_result(text='{"title": "New idea", "description": "The learner drags a marker and sees the distance."}')
        if kind == "research_ideas":
            return make_result(text='{"ideas": []}')
        if kind == "research_gaps":
            return make_result(text='{"findings": [], "ideas": []}')
        raise AssertionError(f"unexpected prompt: {system[:70]}")

    def prompts_of(self, kind: str) -> list[str]:
        return [u for (s, u), k in zip(self.prompts, self.kinds) if k == kind]

    def topic_map_prompts(self):
        return self.prompts_of("topic_map")


@pytest.fixture
def session_with_deck():
    session = Session()
    session.add_file(Path("deck.pptx"), SourceDoc("slide", 2, deck_chunks(2)))
    return session


def run_flow(monkeypatch, session, answers: list[str], model: FakeModel):
    """Answers in order: the type gate, the topics gate, then the structure gate (the save
    gate answers N unless the test says otherwise)."""
    monkeypatch.setattr(nodes, "call_llm", model)
    run_question("build me a lesson from the deck", session, build_graph(), ask_fn=scripted(answers))


def test_approve_right_away(monkeypatch, session_with_deck, output):
    model = FakeModel()
    run_flow(monkeypatch, session_with_deck, ["Y", "Y", "Y"], model)
    out = output()
    assert msg("gate.topics.approved") in out and msg("gate.structure.approved") in out
    assert len(session_with_deck.topic_map) == 4
    assert len(session_with_deck.structure["screens"]) == 4
    for node in ("classify_type", "gate_type", "topic_map", "gate_topics", "plan_structure", "gate_structure"):
        assert f"[INFO] Entering Node: {node}" in out
    assert rendered("gate.topics.title", count=4) in out


def test_change_request_goes_back_to_topic_map_then_approve(monkeypatch, session_with_deck, output):
    model = FakeModel()
    run_flow(monkeypatch, session_with_deck, ["Y", "N add a topic on tires", "Y", "Y"], model)
    prompts = model.topic_map_prompts()
    assert len(prompts) == 2
    assert "apply this feedback: add a topic on tires" in prompts[1]
    assert "Your previous topic map" in prompts[1]
    assert [t["name"] for t in session_with_deck.topic_map][-1] == "TIRES"
    out = output()
    assert rendered("gate.topics.title", count=5) in out and msg("gate.topics.approved") in out


def test_hebrew_keyboard_letters(monkeypatch, session_with_deck, output):
    model = FakeModel()
    run_flow(monkeypatch, session_with_deck, [TET, f"{MEM} add a topic on tires", TET, TET], model)
    assert len(model.topic_map_prompts()) == 2
    assert session_with_deck.topic_map is not None and len(session_with_deck.topic_map) == 5


def test_n_without_comment_asks_what_to_change(monkeypatch, session_with_deck, output):
    model = FakeModel()
    run_flow(monkeypatch, session_with_deck, ["Y", "N", "add tires", "y", "Y"], model)
    assert "apply this feedback: add tires" in model.topic_map_prompts()[1]


def test_free_text_and_empty_answers(monkeypatch, session_with_deck, output):
    model = FakeModel()
    run_flow(monkeypatch, session_with_deck, ["Y", "", "please add tires", KEN, "Y"], model)
    assert msg("gate.topics.ask_again") in output()
    assert "apply this feedback: please add tires" in model.topic_map_prompts()[1]


def test_invalid_model_json_is_retried(monkeypatch, session_with_deck, output):
    model = FakeModel(invalid_first=True)
    run_flow(monkeypatch, session_with_deck, ["Y", "Y", "Y"], model)
    assert len(model.topic_map_prompts()) == 2  # retry inside topic_map
    assert len(session_with_deck.topic_map) == 4


def test_a_new_question_after_the_gate_starts_fresh(monkeypatch, session_with_deck, output):
    model = FakeModel()
    run_flow(monkeypatch, session_with_deck, ["Y", "N add tires", "Y", "Y"], model)
    run_flow(monkeypatch, session_with_deck, ["Y", "Y", "Y"], model)
    # The second run starts from a fresh state: no leftover feedback or previous map.
    assert "Your previous topic map" not in model.topic_map_prompts()[-1]


def test_long_generations_use_the_long_timeout(monkeypatch, session_with_deck, output):
    from agent.llm import LONG_REQUEST_TIMEOUT_SECONDS

    model = FakeModel()
    run_flow(monkeypatch, session_with_deck, ["Y", "Y", "Y"], model)
    timeouts = dict(zip(model.kinds, model.timeouts))
    assert timeouts["scope"] is None and timeouts["classify_type"] is None  # short calls: default
    assert timeouts["topic_map"] == LONG_REQUEST_TIMEOUT_SECONDS
    assert timeouts["plan_structure"] == LONG_REQUEST_TIMEOUT_SECONDS


def test_no_edge_bypasses_a_gate():
    edges = {(e.source, e.target) for e in build_graph().get_graph().edges}
    successors = lambda node: {t for s, t in edges if s == node}  # noqa: E731
    assert successors("classify_type") == {"gate_type"}
    assert successors("topic_map") == {"gate_topics"}
    assert successors("plan_structure") == {"gate_structure"}
    # A gate goes back (change), to itself (question answered), or on (approved).
    assert successors("gate_topics") == {"topic_map", "gate_topics", "plan_structure"}
    assert successors("gate_structure") == {"plan_structure", "gate_structure", "write_lesson"}
    assert successors("gate_type") == {"research", "topic_map"}  # research: training only
    assert successors("research") == {"topic_map"}
    assert successors("write_lesson") == {"write_lesson", "build_illustrations"}
    assert successors("build_illustrations") == {"browser_check"}
    assert successors("browser_check") == {"assemble_lesson"}
    assert successors("assemble_lesson") == {"evaluator"}
    assert successors("evaluator") == {"fix_screens", "gate_review"}
    assert successors("fix_screens") == {"assemble_lesson"}
    assert successors("gate_review") == {"gate_review", "fix_screens", "write_files"}
    assert successors("write_files") == {"__end__"}


def test_format_topic_map_shows_topic_objective_sources():
    topics = parse_topic_map(topics_json(4), build_units(deck_chunks(2)), PREFIX)
    text = format_topic_map(topics)
    assert msg("gate.topics.topic", index=1, name="Topic 1") in text
    assert msg("gate.topics.objective", objective=f"{PREFIX}x 1") in text
    assert msg("rag.source.slide", file="deck.pptx", number=1) in text

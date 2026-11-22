"""Stage 3: write_lesson (groups, retries, model fallback), assembly, and the save gate.
All model calls are faked."""

import json
from pathlib import Path

import pytest

import agent.cli as cli
import agent.nodes as nodes
from agent.errors import AllModelsFailedError, NodeError
from agent.graph import build_graph
from agent.lesson_schema import validate_lesson
from agent.messages import msg
from agent.session import Session, SourceDoc
from agent.writer import (
    assemble_lesson,
    build_messages,
    chunk_matches,
    group_excerpts,
    parse_group,
    screen_groups,
)
from tests.conftest import make_result
from tests.test_lesson_flow import FakeModel, deck_chunks, scripted, structure_json_reply, writer_reply


# ---- groups ----------------------------------------------------------------------------------
@pytest.mark.parametrize("count", range(1, 25))
def test_groups_cover_every_screen_once_with_at_most_four(count):
    groups = screen_groups(count)
    numbers = [n for a, b in groups for n in range(a, b + 1)]
    assert numbers == list(range(1, count + 1))
    sizes = [b - a + 1 for a, b in groups]
    assert max(sizes) <= 4 and max(sizes) - min(sizes) <= 1  # balanced
    if count >= 3:
        assert min(sizes) >= 2


@pytest.mark.parametrize("count, expected", [(9, [(1, 3), (4, 6), (7, 9)]), (10, [(1, 4), (5, 7), (8, 10)]),
                                             (16, [(1, 4), (5, 8), (9, 12), (13, 16)]), (4, [(1, 4)])])
def test_group_examples(count, expected):
    assert screen_groups(count) == expected


def test_a_long_lesson_stays_far_below_30_calls():
    # scope + type + topic map + structure + one call per group (no retries)
    assert 4 + len(screen_groups(24)) < 30


# ---- excerpts --------------------------------------------------------------------------------
def test_chunk_matches_slide_page_and_paragraph_ranges():
    assert chunk_matches({"file": "a.pptx", "slide": 2, "paragraph": 3}, {"file": "a.pptx", "slide": 2})
    assert not chunk_matches({"file": "a.pptx", "slide": 3, "paragraph": 1}, {"file": "a.pptx", "slide": 2})
    assert chunk_matches({"file": "b.pdf", "page": 1, "paragraph": 9}, {"file": "b.pdf", "page": 1})
    unit = {"file": "c.docx", "paragraph": 4, "paragraph_end": 6}
    assert chunk_matches({"file": "c.docx", "paragraph": 5}, unit)
    assert not chunk_matches({"file": "c.docx", "paragraph": 7}, unit)
    assert not chunk_matches({"file": "other.docx", "paragraph": 5}, unit)


def plan_from(user_json: str) -> list[dict]:
    from agent.structure import parse_structure
    from agent.topics import build_units, parse_topic_map
    from tests.test_lesson_flow import PREFIX, topics_json

    units = build_units(deck_chunks(2))
    topics = parse_topic_map(topics_json(4, ids=("U1", "U2")), units, PREFIX)
    user = "Approved topic map (JSON):\n" + json.dumps({"topics": [{"name": t["name"]} for t in topics]}) + "\n"
    return parse_structure(structure_json_reply(user, ids=("U1", "U2")), units, topics, "training")["screens"]


@pytest.fixture
def plan():
    return plan_from("")


def test_group_excerpts_are_the_cited_chunks_numbered_once(plan):
    excerpts = group_excerpts(plan, deck_chunks(2))
    assert [eid for eid, _ in excerpts] == ["E1", "E2", "E3", "E4"]  # 2 slides x 2 chunks, no repeats
    assert {c["locator"]["slide"] for _, c in excerpts} == {1, 2}


def test_messages_carry_type_rules_extras_and_wrapped_sources(plan):
    excerpts = group_excerpts(plan, deck_chunks(2))
    (_, system), (_, user) = build_messages(plan, excerpts, "compliance", first=True, last=False)
    assert "Compliance lesson" in system and "80 words" in system and "No did_you_know" in system
    assert "lesson_title" in system and "summary" not in system.split("Reply with")[0].split("Media:")[1]
    assert "<source_material>" in user and "[E1]" in user
    (_, system), _ = build_messages(plan, excerpts, "training", first=False, last=True)
    assert "This is the last group" in system and "Training lesson" in system


# ---- parsing a group ---------------------------------------------------------------------------
def reply_for(plan, excerpts, lesson_type="training", first=True, last=True, change=None) -> str:
    (_, system), (_, user) = build_messages(plan, excerpts, lesson_type, first, last)
    data = json.loads(writer_reply(system, user))
    if change:
        change(data)
    return json.dumps(data)


def test_valid_group_becomes_schema_screens(plan):
    excerpts = group_excerpts(plan, deck_chunks(2))
    result = parse_group(reply_for(plan, excerpts), plan, excerpts, "training", True, True)
    screens = result["screens"]
    assert [s["id"] for s in screens] == ["s1", "s2", "s3", "s4"]
    assert screens[0]["title"] == plan[0]["title"] and screens[0]["topic"] == plan[0]["topic"]
    mc = next(b for b in screens[0]["blocks"] if b["type"] == "multiple_choice")
    assert mc["id"] == "q1" and mc["sources"] == [{"file": "deck.pptx", "slide": 1, "paragraph": 1}]
    illustration = next(b for b in screens[1]["blocks"] if b["type"] == "illustration")
    assert illustration["file"] == "assets/illustrations/s2.html"
    assert result["lesson_title"] == "Fake lesson" and result["summary"]


def test_excerpt_ids_never_reach_the_text(plan):
    excerpts = group_excerpts(plan, deck_chunks(2))
    text = reply_for(plan, excerpts, change=lambda d: d["screens"][0]["blocks"][0].update(
        text="Brake early (E1, deck.pptx) see [E2]."))
    result = parse_group(text, plan, excerpts, "training", True, True)
    assert result["screens"][0]["blocks"][0]["text"] == "Brake early see."


def test_did_you_know_gets_a_source_line_from_its_locators(plan):
    excerpts = group_excerpts(plan, deck_chunks(2))
    dyk = {"type": "did_you_know", "text": "An interesting fact.", "sources": ["E3"]}
    text = reply_for(plan, excerpts, change=lambda d: d["screens"][0]["blocks"].insert(1, dyk))
    block = parse_group(text, plan, excerpts, "training", True, True)["screens"][0]["blocks"][1]
    assert block["source_line"] == msg("rag.source.slide", file="deck.pptx", number=2).strip("()")


BAD = [
    ("missing screen", lambda d: d["screens"].pop(), "are missing"),
    ("extra screen", lambda d: d["screens"].append({**d["screens"][0], "number": 99}), "were not requested"),
    ("unknown excerpt", lambda d: d["screens"][0]["blocks"][0].update(sources=["E77"]), "unknown excerpt ids"),
    ("two questions", lambda d: d["screens"][0]["blocks"].append(d["screens"][0]["blocks"][1]), "exactly one multiple_choice"),
    ("missing planned illustration", lambda d: d["screens"][1]["blocks"].pop(), "add the illustration block"),
    ("unplanned diagram", lambda d: d["screens"][1]["blocks"].append(d["screens"][0]["blocks"][-1]), "remove the decision_diagram"),
    ("feedback reveals the answer", lambda d: d["screens"][0]["blocks"][1].update(
        feedback_incorrect="The answer is act 1 b"), "reveals the correct answer"),
    ("no sources", lambda d: d["screens"][0]["blocks"][0].update(sources=[]), "sources"),
    ("no title", lambda d: d.pop("lesson_title"), "lesson_title is missing"),
    ("no summary", lambda d: d.pop("summary"), "summary is missing"),
    ("not json", None, "no JSON object"),
]


@pytest.mark.parametrize("change, expected", [(c, e) for _, c, e in BAD], ids=[n for n, _, _ in BAD])
def test_invalid_groups_are_rejected_with_fixable_messages(plan, change, expected):
    excerpts = group_excerpts(plan, deck_chunks(2))
    text = "not json" if change is None else reply_for(plan, excerpts, change=change)
    with pytest.raises(ValueError, match=expected):
        parse_group(text, plan, excerpts, "training", True, True)


def test_compliance_rejects_did_you_know_and_long_screens(plan):
    excerpts = group_excerpts(plan, deck_chunks(2))
    dyk = {"type": "did_you_know", "text": "A fact.", "sources": ["E1"]}
    with pytest.raises(ValueError, match="no did_you_know"):
        parse_group(reply_for(plan, excerpts, "compliance", change=lambda d: d["screens"][0]["blocks"].append(dyk)),
                    plan, excerpts, "compliance", True, True)
    long = lambda d: d["screens"][0]["blocks"][0].update(text="word " * 90)  # noqa: E731
    with pytest.raises(ValueError, match="at most 80"):
        parse_group(reply_for(plan, excerpts, "compliance", change=long), plan, excerpts, "compliance", True, True)


# ---- answer positions ---------------------------------------------------------------------------
def test_shuffle_keeps_the_right_answer_and_is_deterministic():
    from agent.writer import shuffle_options

    def question():
        return {"options": ["right", "w1", "w2", "w3"], "correct_index": 0}

    a, b = shuffle_options(question(), "q1"), shuffle_options(question(), "q1")
    assert a == b  # same seed, same order
    assert a["options"][a["correct_index"]] == "right"
    assert sorted(a["options"]) == sorted(question()["options"])


def test_shuffle_spreads_the_right_answer_over_all_positions():
    from agent.writer import shuffle_options

    positions = {shuffle_options({"options": ["r", "a", "b", "c"], "correct_index": 0}, f"q{i}")["correct_index"]
                 for i in range(1, 25)}
    assert positions == {0, 1, 2, 3}  # a model that always answers "first" no longer shows


def test_parsed_questions_are_shuffled_with_their_index(plan):
    excerpts = group_excerpts(plan, deck_chunks(2))
    screens = parse_group(reply_for(plan, excerpts), plan, excerpts, "training", True, True)["screens"]
    for screen in screens:
        mc = next(b for b in screen["blocks"] if b["type"] == "multiple_choice")
        n = screen["id"][1:]
        assert mc["options"][mc["correct_index"]] == f"act {n} b"  # the fake's right answer
        assert mc["alternate"]["options"][mc["alternate"]["correct_index"]] == f"do x{n}"


# ---- retry and model fallback ----------------------------------------------------------------
class ScriptedWriter:
    """call_llm stand-in: replies per model from a queue; records (model) per call."""

    def __init__(self, replies: dict):
        self.replies = {m: list(r) for m, r in replies.items()}
        self.calls: list[str] = []

    def __call__(self, messages, **kwargs):
        model = kwargs["models"][0]
        self.calls.append(model)
        reply = self.replies[model].pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return make_result(model=model, text=reply)


@pytest.fixture
def two_writers(monkeypatch):
    import agent.llm as llm

    monkeypatch.setattr(llm, "load_models", lambda role="writer": ["w1:free", "w2:free"])


def _parse_ok(text):
    if text != "GOOD":
        raise ValueError("bad screens")
    return {"ok": True}


def test_invalid_reply_is_returned_to_the_same_model_once(monkeypatch, two_writers, output):
    writer = ScriptedWriter({"w1:free": ["bad", "GOOD"]})
    monkeypatch.setattr(nodes, "call_llm", writer)
    assert nodes._write_group([], {}, _parse_ok)[0] == {"ok": True}
    assert writer.calls == ["w1:free", "w1:free"]


def test_still_invalid_after_the_retry_moves_to_the_next_model(monkeypatch, two_writers, output):
    writer = ScriptedWriter({"w1:free": ["bad", "bad again"], "w2:free": ["GOOD"]})
    monkeypatch.setattr(nodes, "call_llm", writer)
    assert nodes._write_group([], {}, _parse_ok)[0] == {"ok": True}
    assert writer.calls == ["w1:free", "w1:free", "w2:free"]
    assert msg("llm.invalid_output", model="w1:free", next_model="w2:free") in output()


def test_a_failing_model_is_skipped(monkeypatch, two_writers, output):
    writer = ScriptedWriter({"w1:free": [AllModelsFailedError([("w1:free", TimeoutError())])], "w2:free": ["GOOD"]})
    monkeypatch.setattr(nodes, "call_llm", writer)
    assert nodes._write_group([], {}, _parse_ok)[0] == {"ok": True}
    assert writer.calls == ["w1:free", "w2:free"]


def test_no_valid_writer_raises_a_node_error(monkeypatch, two_writers, output):
    writer = ScriptedWriter({"w1:free": ["bad", "bad"], "w2:free": ["bad", "bad"]})
    monkeypatch.setattr(nodes, "call_llm", writer)
    with pytest.raises(NodeError, match="no writer model produced valid screens"):
        nodes._write_group([], {}, _parse_ok)


# ---- assembly -----------------------------------------------------------------------------------
def test_assembled_lesson_is_valid_and_complete(plan):
    excerpts = group_excerpts(plan, deck_chunks(2))
    result = parse_group(reply_for(plan, excerpts), plan, excerpts, "training", True, True)
    lesson = assemble_lesson(result["screens"], "training", result["lesson_title"], result["summary"])
    assert validate_lesson(lesson) == []
    assert lesson["palette"]["background"] == "#FFFFFF" and lesson["motif"] == {"description": "", "svg": ""}
    assert lesson["certificate"] is True and lesson["summary"]["points"] == result["summary"]


# ---- the whole flow: write in groups, assemble, save ----------------------------------------------
class EightScreenModel(FakeModel):
    """FakeModel whose structure has two screens per topic: 8 screens -> groups of 4 and 4."""

    def __call__(self, messages, **kwargs):
        from tests.test_lesson_flow import prompt_kind

        if prompt_kind(messages[0].content) == "plan_structure":
            self.kinds.append("plan_structure")
            self.prompts.append((messages[0].content, messages[-1].content))
            data = json.loads(structure_json_reply(messages[1].content))
            doubled = []
            for s in data["screens"]:
                doubled += [s, {**s, "title": s["title"] + " b", "illustration": None, "decision_diagram": False}]
            for i, s in enumerate(doubled, 1):
                s["number"] = i
            data["screens"] = doubled
            return make_result(text=json.dumps(data))
        return super().__call__(messages, **kwargs)


@pytest.fixture
def deck_session():
    session = Session()
    session.add_file(Path("deck.pptx"), SourceDoc("slide", 2, deck_chunks(2)))
    return session


@pytest.fixture
def no_browser(monkeypatch, tmp_path):
    import agent.persistence as persistence

    opened = []
    monkeypatch.setattr(cli, "open_preview", opened.append)
    monkeypatch.setattr(persistence, "output_dir", lambda: tmp_path / "output")
    return opened


def test_lesson_is_written_in_groups_assembled_and_saved(monkeypatch, deck_session, output, no_browser, tmp_path):
    model = EightScreenModel()
    monkeypatch.setattr(nodes, "call_llm", model)
    cli.run_question("build the lesson", deck_session, build_graph(), ask_fn=scripted(["Y", "Y", "Y"], save="Y"))
    out = output()
    assert model.kinds.count("write") == 2  # 8 screens, one call per group
    assert "[INFO] Entering Node: write_lesson (screens 1-4)" in out
    assert "[INFO] Entering Node: write_lesson (screens 5-8)" in out
    assert "[INFO] Entering Node: assemble_lesson" in out and "[INFO] Entering Node: write_files" in out
    assert len(model.kinds) < 30
    saved = Path(no_browser[0])
    assert saved == tmp_path / "output" / "deck" / "index.html" and saved.is_file()
    lesson_js = (saved.parent / "lesson.js").read_text(encoding="utf-8")
    lesson = json.loads(lesson_js[len("window.LESSON = "):].rstrip().rstrip(";"))
    assert validate_lesson(lesson) == [] and len(lesson["screens"]) == 8
    built = saved.parent / "assets" / "illustrations" / "s3.html"
    assert built.is_file() and "<button" in built.read_text(encoding="utf-8")  # the built illustration
    assert msg("lesson.files_saved") in out
    assert (saved.parent / "lesson.json").is_file()
    report = (saved.parent / "report.md").read_text(encoding="utf-8")
    assert lesson["title"] in report and msg("report.open_title") in report
    for node in ("evaluator", "gate_review"):
        assert f"[INFO] Entering Node: {node}" in out


def test_n_at_the_save_gate_writes_nothing(monkeypatch, deck_session, output, no_browser, tmp_path):
    monkeypatch.setattr(nodes, "call_llm", FakeModel())
    cli.run_question("build the lesson", deck_session, build_graph(), ask_fn=scripted(["Y", "Y", "Y"], save="N"))
    assert no_browser == [] and not (tmp_path / "output").exists()
    assert msg("lesson.not_saved") in output()


def test_an_existing_output_folder_is_never_overwritten(tmp_path):
    from agent.persistence import unique_output_folder

    base = tmp_path / "output"
    (base / "short_winter").mkdir(parents=True)
    assert unique_output_folder("short_winter.pptx", base) == base / "short_winter-2"
    assert unique_output_folder("C:/x/my file.pdf", base) == base / "my_file"


def test_did_you_know_shows_the_document_title_not_the_file_name():
    from agent.writer import source_line, source_titles

    chunks = [{"text": "נהיגה בטוחה בחורף", "locator": {"file": "short_winter.pptx", "slide": 1, "paragraph": 1}},
              {"text": "x" * 300, "locator": {"file": "long_first.docx", "paragraph": 1}},
              {"text": "web", "locator": {"file": "gov.il", "url": "https://gov.il/a", "title": "t"}}]
    titles = source_titles(chunks)
    assert titles == {"short_winter.pptx": "נהיגה בטוחה בחורף", "long_first.docx": "long_first"}
    line = source_line([{"file": "short_winter.pptx", "slide": 1}], titles)
    assert line.startswith("נהיגה בטוחה בחורף") and "short_winter.pptx" not in line


def test_the_writer_is_told_did_you_know_is_never_administrative():
    from agent.writer import TYPE_WRITING

    assert "never an administrative detail" in TYPE_WRITING["training"]

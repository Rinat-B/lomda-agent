"""Stage 3 review in the graph: evaluator -> fix_screens (only the screens with findings, at
most 2 rounds) -> gate 3 -> write_files with report.md. All models are faked."""

import json
import re
from pathlib import Path

import pytest

import agent.nodes as nodes
from agent import cli
from agent.evaluation import (
    MAX_FIX_ROUNDS,
    coverage,
    fix_groups,
    owner_findings,
    reconcile_code_findings,
)
from agent.gates import ask_write_files_gate, format_review
from agent.graph import build_graph
from agent.messages import msg
from agent.report import build_report
from agent.session import Session, SourceDoc
from tests.conftest import make_result
from tests.test_lesson_flow import FakeModel, deck_chunks, prompt_kind

BAD_WORD = "bad word on screen 2"


class ReviewModel(FakeModel):
    """The proofreader reports one problem on screen 2 in the first check; the verifier
    resolves it unless `never_fixed`."""

    def __init__(self, never_fixed: bool = False, **kw):
        super().__init__(**kw)
        self.never_fixed = never_fixed
        self.revisions: list[str] = []  # user prompts of fix_screens calls

    def __call__(self, messages, **kwargs):
        system, user = messages[0].content, messages[1].content
        kind = prompt_kind(system)
        if kind == "write" and "Problems to fix" in user:
            self.revisions.append(user)
        if kind == "check" and "proofread" in system and self.kinds.count("check_language") == 0:
            self.kinds.append("check_language")
            self.prompts.append((system, user))
            return make_result(text=json.dumps({"findings": [
                {"screen": 2, "where": "block 1 (paragraph)", "category": "language",
                 "evidence": "Screen 2 explains", "problem": BAD_WORD, "suggestion": "fix it"}]}))
        if kind == "verify":
            self.kinds.append("verify")
            self.prompts.append((system, user))
            ids = re.findall(r'"id": "(F\d+)"', user)
            return make_result(text=json.dumps({"resolved": [{"id": i, "resolved": not self.never_fixed, "reason": "ok"}
                                                             for i in ids], "new_findings": []}))
        return super().__call__(messages, **kwargs)


REVIEW_ASKED: list[str] = []


def answers(gates: list[str], reviews: list[str], save: str = "N"):
    """Type, topics and structure gates from `gates`; gate 3 answers from `reviews`."""
    asked, reviewed = iter(gates), iter(reviews)
    REVIEW_ASKED.clear()

    def ask(prompt: str) -> str:
        if prompt == msg("gate.write_files.ask"):
            return save
        if prompt == msg("gate.review.ask"):
            REVIEW_ASKED.append(prompt)
            return next(reviewed)
        return next(asked)

    return ask


@pytest.fixture
def deck_session():
    session = Session()
    session.add_file(Path("deck.pptx"), SourceDoc("slide", 2, deck_chunks(2)))
    return session


@pytest.fixture
def out_dir(monkeypatch, tmp_path):
    import agent.persistence as persistence

    monkeypatch.setattr(cli, "open_preview", lambda path: None)
    monkeypatch.setattr(persistence, "output_dir", lambda: tmp_path / "output")
    return tmp_path / "output"


def run(monkeypatch, session, model, reviews=("Y",), save="N"):
    monkeypatch.setattr(nodes, "call_llm", model)
    graph = build_graph()
    cli.run_question("build the lesson", session, graph, ask_fn=answers(["Y", "Y", "Y"], list(reviews), save))
    return graph


def final_state(graph, session):
    return graph.get_state(cli.graph_config(session)).values


def revised_numbers(user: str) -> list[int]:
    start = user.index("Planned screens (JSON):\n") + len("Planned screens (JSON):\n")
    return [s["number"] for s in json.loads(user[start:user.index("\n\nSource excerpts:")])]


# ---- the loop in the graph ------------------------------------------------------------------------
def test_a_finding_is_fixed_on_its_screen_only_and_verified(monkeypatch, deck_session, output, out_dir, no_browser_preview):
    model = ReviewModel()
    graph = run(monkeypatch, deck_session, model)
    assert len(model.revisions) == 1 and revised_numbers(model.revisions[0]) == [2]
    assert BAD_WORD in model.revisions[0]  # the finding is sent to the writer
    assert model.kinds.count("verify") == 1
    state = final_state(graph, deck_session)
    assert state["review_round"] == 1
    assert [(f["problem"], f["status"]) for f in state["findings"]] == [(BAD_WORD, "fixed")]
    out = output()
    for node in ("evaluator", "fix_screens", "gate_review", "write_files"):
        assert f"[INFO] Entering Node: {node}" in out
    assert "[INFO] Entering Node: fix_screens (round 1 of 2)" in out
    assert "[INFO] Entering Node: evaluator (verify screens 2)" in out
    assert len(no_browser_preview) == 1  # gate 3 opened the preview once
    assert len(model.kinds) < 30


def test_fixing_stops_after_two_rounds_and_the_finding_stays_open(monkeypatch, deck_session, output, out_dir):
    model = ReviewModel(never_fixed=True)
    graph = run(monkeypatch, deck_session, model, save="Y")
    assert len(model.revisions) == MAX_FIX_ROUNDS
    state = final_state(graph, deck_session)
    assert state["review_round"] == MAX_FIX_ROUNDS
    open_ = [f for f in state["findings"] if f["status"] == "open"]
    assert [f["problem"] for f in open_] == [BAD_WORD]
    assert BAD_WORD in output()  # gate 3 lists the open finding
    report = (out_dir / "deck" / "report.md").read_text(encoding="utf-8")
    assert BAD_WORD in report and msg("report.rounds", rounds=2, max=2) in report


def test_owner_change_at_gate_3_rewrites_the_named_screen(monkeypatch, deck_session, output, out_dir):
    model = ReviewModel()
    run(monkeypatch, deck_session, model, reviews=["N make screen 3 shorter", "Y"])
    assert len(model.revisions) == 2  # the automatic fix of screen 2, then the owner's screen 3
    assert revised_numbers(model.revisions[1]) == [3]
    assert "make screen 3 shorter" in model.revisions[1]
    assert len(REVIEW_ASKED) == 2  # gate 3 again after the change


def test_no_findings_goes_straight_to_gate_3(monkeypatch, deck_session, output, out_dir):
    model = FakeModel()
    graph = run(monkeypatch, deck_session, model)
    assert "fix_screens" not in output()
    assert final_state(graph, deck_session)["findings"] == []
    assert msg("gate.review.no_open") in output()


# ---- choosing checkers of another family ------------------------------------------------------------
def test_checkers_of_another_family_and_the_same_family_fallback(monkeypatch, output):
    from agent.errors import AllModelsFailedError

    monkeypatch.setattr(nodes, "load_models", lambda role: ["google/c1:free", "nvidia/c2:free"], raising=False)
    import agent.llm as llm

    monkeypatch.setattr(llm, "load_models", lambda role="writer": ["google/c1:free", "nvidia/c2:free"])
    used: list[list[str]] = []

    def fake_call(messages, **kw):
        used.append(kw["models"])
        if kw["models"] == ["google/c1:free"]:  # the only other-family checker is down
            raise AllModelsFailedError([("google/c1:free", RuntimeError("429"))])
        return make_result(model=kw["models"][0])

    monkeypatch.setattr(nodes, "call_llm", fake_call)
    review: dict = {}
    call = nodes._checker_call({}, {"writer_models": ["nvidia/w1:free"]}, review)
    assert call([], timeout=10).model == "nvidia/c2:free"
    assert used == [["google/c1:free"], ["nvidia/c2:free"]]  # another family first, then the fallback
    assert review["same_family"] == ["nvidia"]
    assert msg("evaluator.same_family", writer_families="nvidia", models="nvidia/c2:free") in output()

    used.clear()
    review = {}
    call = nodes._checker_call({}, {"writer_models": ["baidu/w:free"]}, review)
    call([], timeout=10)
    assert used[0] == ["google/c1:free", "nvidia/c2:free"] and "same_family" not in review


# ---- bookkeeping -----------------------------------------------------------------------------------
def test_fix_groups_and_owner_findings():
    assert fix_groups([5, 2, 2, 9, 1, 3, 7]) == [[1, 2, 3, 5], [7, 9]]
    added = owner_findings("shorten screens 2 and 4", 6, [], 1)
    assert [f["screen"] for f in added] == [2, 4] and all(f["status"] == "open" for f in added)
    assert [f["screen"] for f in owner_findings("make it friendlier", 3, [], 0)] == [1, 2, 3]


def test_code_findings_are_reconciled_by_content():
    lesson = json.loads((Path(__file__).parent / "fixtures" / "known_bad_lesson.json").read_text(encoding="utf-8"))
    from agent.checkers import CODE_CHECKS
    from agent.evaluation import finding_dict

    old = [finding_dict(f, f"F{i}", 0) for i, f in enumerate((f for c in CODE_CHECKS.values() for f in c(lesson)), 1)]
    old.append({**old[0], "id": "F99", "evidence": "gone now"})  # no longer found
    new = reconcile_code_findings(old, lesson, 1)
    assert next(f for f in new if f["id"] == "F99")["status"] == "fixed"
    assert all(f["status"] == "open" for f in new if f["id"] != "F99")
    assert len(new) == len(old)  # nothing duplicated


def test_coverage_counts_planned_points_the_lesson_cites():
    structure = {"screens": [{"points": [{"sources": [{"file": "a.pptx", "slide": 1}]},
                                         {"sources": [{"file": "a.pptx", "slide": 2}]}]}]}
    lesson = {"screens": [{"blocks": [{"type": "paragraph", "sources": [{"file": "a.pptx", "slide": 1, "paragraph": 2}]}]}]}
    assert coverage(structure, lesson) == (1, 2)


def test_report_lists_open_and_fixed_findings_and_the_same_family_note():
    lesson = json.loads((Path(__file__).parent / "fixtures" / "known_bad_lesson.json").read_text(encoding="utf-8"))
    from agent.session import UsageRecord

    state = {
        "lesson": lesson, "review_round": 1, "writer_models": ["nvidia/w:free"],
        "review": {"same_family": ["nvidia"], "failed": {"pedagogy": "timeout"}},
        "findings": [
            {"id": "F1", "check": "language", "category": "language", "screen": 3, "where": "block 5",
             "evidence": "quote", "problem": "open problem", "suggestion": "do this", "status": "open"},
            {"id": "F2", "check": "address", "category": "address", "screen": None, "where": "",
             "evidence": "", "problem": "lesson-wide", "suggestion": "", "status": "open"},
            {"id": "F3", "check": "grounding", "category": "ungrounded", "screen": 1, "where": "",
             "evidence": "", "problem": "fixed problem", "suggestion": "", "status": "fixed", "fixed_round": 1},
        ],
    }
    usage = [UsageRecord("write_lesson", "nvidia/w:free", 10, 5, 0.0), UsageRecord("evaluator", "google/c:free", 7, 3, 0.01)]
    text = build_report(state, usage, nodes.source_label, elapsed_minutes=12)
    assert msg("report.counts", found=3, fixed=1, open=2) in text
    assert "open problem" in text and "do this" in text and "lesson-wide" in text and "fixed problem" in text
    assert msg("report.same_family", families="nvidia") in text
    assert "pedagogy" in text and "google/c:free" in text
    assert msg("report.elapsed", minutes=12) in text
    assert msg("report.cost", calls=2, tokens="25", cost=0.01) in text


def test_gate_3_text_and_write_files_list(output):
    summary = {"title": "T", "lesson_type": "training", "screens": 8, "rounds": 2, "found": 12, "fixed": 9,
               "open": [{"screen": 4, "problem": "p" * 3}] * 10, "covered": 9, "total": 10}
    text = format_review(summary)
    assert msg("gate.review.open_more", count=2) in text and "90%" in text
    said: list[str] = []
    reply = ask_write_files_gate({"folder": "C:/x/output/deck", "files": ["index.html", "report.md"], "exists": True},
                                 lambda prompt: "Y", lambda text, style=None: said.append(text))
    assert reply == {"confirm": True}
    assert msg("gate.write_files.exists") in said and msg("gate.write_files.file", name="report.md") in said


def test_report_shows_locations_in_hebrew_and_quotes_without_prompt_labels():
    from agent.report import quote_text, where_text

    alt = where_text("block 5 (multiple_choice), alternate")
    assert alt == msg("report.block", number=5, type=msg("report.block_types.multiple_choice")) + ", " + msg("report.alternate")
    assert where_text("block 2 (list), main") == msg("report.block", number=2, type=msg("report.block_types.list"))
    assert where_text("") == "-"
    assert quote_text("[screen 2, block 1, paragraph] text one [screen 2, block 2, list] - text two") == "text one - text two"
    assert quote_text("[screen 3, block 5, multiple_choice] alternate question (shown on retry): what now?") == "what now?"

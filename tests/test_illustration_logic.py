"""The illustration logic checker: the model's table -> findings in code, and the fix loop:
a logic finding rebuilds the page, and the same check decides whether it is fixed."""

import json

import agent.nodes as nodes
from agent import cli
from agent.graph import build_graph
from agent.illustration_logic import CHECK, findings_from_table, logic_files, page_for_prompt
from agent.messages import msg
from tests.conftest import make_result
from tests.test_lesson_flow import EMPTY_LOGIC_TABLE, FAKE_ILLUSTRATION, FakeModel, prompt_kind
from tests.test_review_flow import deck_session, out_dir  # noqa: F401 - fixtures


def table(**rows) -> dict:
    data = json.loads(EMPTY_LOGIC_TABLE)
    data.update(rows)
    return data


def test_each_kind_of_row_becomes_its_category():
    data = table(
        decisions=[{"code_condition": "count >= 2", "code_action": "להמשיך בלי שינוי", "source_rule": "none",
                    "same": False, "why": "אין כלל כזה"},
                   {"code_condition": "count == 1", "code_action": "להאט", "source_rule": "פחות משני עמודים", "same": True}],
        numbers=[{"text": "13.9 מ׳", "in_source": False, "condition_missing": ""},
                 {"text": "28 מ׳", "in_source": True, "condition_missing": "50 קמ״ש"},
                 {"text": "40 מ׳", "in_source": True, "condition_missing": ""}],
        choices=[{"combination": "משאית + יבש", "selectable": True, "source_has_data": False},
                 {"combination": "רכב קל + יבש", "selectable": True, "source_has_data": True},
                 {"combination": "משאית + יבש (מושבת)", "selectable": False, "source_has_data": False}],
        cause_effect=[{"cause": "יותר ערפל", "effect_in_page": "פחות עמודים", "correct": True}],
        feedback=[{"case": "טעות", "text": "הפעולה המתאימה: להאט", "correct": True, "drops_caution": False,
                   "reveals_answer": True}],
        situation={"name_missing": "האחור מחליק"},
    )
    found = {f.category: f for f in findings_from_table(data, 6, "illustration assets/illustrations/s6.html")}
    assert set(found) == {"illustration_rule", "illustration_number", "illustration_choice",
                          "illustration_feedback", "illustration_situation"}
    assert "13.9" in found["illustration_number"].problem and "50 קמ״ש" in found["illustration_number"].problem
    assert found["illustration_choice"].evidence == "משאית + יבש"  # the disabled option is not a finding
    assert msg("illustration_logic.feedback_reveals") in found["illustration_feedback"].problem
    assert all(f.check == CHECK and f.screen == 6 for f in found.values())


def test_a_clean_table_gives_no_finding_and_strings_count_as_booleans():
    assert findings_from_table(json.loads(EMPTY_LOGIC_TABLE), 1, "x") == []
    data = table(cause_effect=[{"cause": "ערפל", "effect_in_page": "יותר עמודים", "correct": "false"}])
    assert [f.category for f in findings_from_table(data, 1, "x")] == ["illustration_cause"]


def test_the_prompt_page_drops_css_and_drawing_data():
    page = '<style>body{color:red}</style><path d="' + "M 1 2 " * 20 + '"/><script>if (n >= 2) go()</script>'
    text = page_for_prompt(page)
    assert "color:red" not in text and "M 1 2" not in text and "if (n >= 2)" in text


def test_findings_are_grouped_by_page():
    lesson = {"screens": [{"id": "s2", "blocks": [{"type": "illustration", "file": "assets/illustrations/s2.html"}]}]}
    findings = [{"check": CHECK, "status": "open", "screen": 2, "where": "illustration assets/illustrations/s2.html"},
                {"check": CHECK, "status": "fixed", "screen": 2, "where": "illustration assets/illustrations/s2.html"},
                {"check": "language", "status": "open", "screen": 2, "where": "block 1"}]
    assert list(logic_files(findings, lesson)) == ["assets/illustrations/s2.html"]
    assert len(logic_files(findings, lesson)["assets/illustrations/s2.html"]) == 1


# ---- in the graph ----------------------------------------------------------------------------------
REVEALS = json.dumps(table(feedback=[{"case": "טעות", "text": "התשובה הנכונה: להאט", "correct": True,
                                      "drops_caution": False, "reveals_answer": True}]), ensure_ascii=False)


class LogicModel(FakeModel):
    """The logic check finds a revealing feedback; after the page is rebuilt it is gone (unless never)."""

    def __init__(self, never_fixed: bool = False, **kw):
        super().__init__(**kw)
        self.never_fixed = never_fixed
        self.logic_checks = 0
        self.rebuilds: list[str] = []

    def __call__(self, messages, **kwargs):
        kind = prompt_kind(messages[0].content)
        if kind == "illustration_logic":
            self.kinds.append(kind)
            self.logic_checks += 1
            rebuilt = "<!-- rebuilt -->" in messages[1].content
            return make_result(text=REVEALS if not rebuilt or self.never_fixed else EMPTY_LOGIC_TABLE)
        if kind == "illustration_logic_fix":
            self.kinds.append(kind)
            self.rebuilds.append(messages[0].content)
            return make_result(text=FAKE_ILLUSTRATION.replace("</body>", "<!-- rebuilt --></body>"))
        return super().__call__(messages, **kwargs)


def ask(prompt: str) -> str:
    if prompt == msg("gate.write_files.ask"):
        return "N"
    if prompt.startswith(msg("gate.browser_check.ask", count=0).split("0")[0]):
        return "N"  # no Edge in this test: the rebuilt page is not run
    return "Y"


def run(monkeypatch, session, model):
    monkeypatch.setattr(nodes, "call_llm", model)
    graph = build_graph()
    cli.run_question("build the lesson", session, graph, ask_fn=ask)
    return graph.get_state(cli.graph_config(session)).values


def test_a_logic_finding_rebuilds_the_page_and_the_same_check_verifies_it(monkeypatch, deck_session, output, out_dir):
    model = LogicModel()
    state = run(monkeypatch, deck_session, model)
    logic = [f for f in state["findings"] if f["check"] == CHECK]
    if not logic:  # the fake plan has no illustration in this deck
        import pytest

        pytest.skip("no illustration was planned")
    pages = len(state["illustrations"])
    assert pages >= 1 and [f["status"] for f in logic] == ["fixed"] * pages  # one finding per page
    assert len(model.rebuilds) == pages and msg("illustration_logic.feedback_reveals") in model.rebuilds[0]
    for f in logic:
        assert "<!-- rebuilt -->" in state["illustrations"][f["where"].removeprefix("illustration ")]
    assert model.logic_checks == 2 * pages  # the first check, then the re-check of each rebuilt page only


def test_a_logic_finding_still_reported_stays_open(monkeypatch, deck_session, output, out_dir):
    model = LogicModel(never_fixed=True)
    state = run(monkeypatch, deck_session, model)
    logic = [f for f in state["findings"] if f["check"] == CHECK]
    if not logic:
        import pytest

        pytest.skip("no illustration was planned")
    assert {f["status"] for f in logic} == {"open"}
    assert len(model.rebuilds) == 2 * len(state["illustrations"])  # one rebuild per page per round, two rounds


def test_the_report_names_the_illustration_in_hebrew():
    from agent.report import where_text

    assert where_text("illustration assets/illustrations/s6.html") == msg(
        "report.where_illustration", file="assets/illustrations/s6.html")


def test_every_graph_node_that_calls_a_model_has_a_hebrew_name_in_the_report():
    from agent.messages import msg_or_none

    for node in ("scope_check", "classify_type", "research", "topic_map", "plan_structure", "motif", "write_lesson",
                 "build_illustrations", "browser_check", "key_points", "evaluator", "fix_screens"):
        assert msg_or_none(f"report.nodes.{node}"), node

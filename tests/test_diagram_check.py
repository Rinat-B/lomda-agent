"""The decision-diagram checker both ways against the source: a source action with no leaf, and a
node that does not come from the source, are findings; the call happens only with a diagram."""

import json

from agent import diagram_check
from agent.checkers import check_jobs, run_checkers
from agent.messages import msg
from tests.conftest import make_result


def lesson_with_diagram() -> dict:
    diagram = {"type": "decision_diagram", "title": "ראות", "sources": [{"file": "w.pptx", "slide": 4}],
               "nodes": [{"id": "q1", "label": "רואים פחות משני עמודים?", "kind": "question"},
                         {"id": "a1", "label": "מאטים ל-50 קמ״ש", "kind": "action"},
                         {"id": "a2", "label": "מאיצים", "kind": "action"}],
               "edges": [{"from": "q1", "to": "a1", "label": "כן"}, {"from": "q1", "to": "a2", "label": "לא"}]}
    return {"title": "ל", "screens": [{"id": "s1", "title": "א", "blocks": [{"type": "paragraph", "text": "x", "sources": []}]},
                        {"id": "s2", "title": "ב", "blocks": [{"type": "paragraph", "text": "y", "sources": []}, diagram]}]}


REPLY = json.dumps({"diagrams": [{"screen": 2,
                                  "source_actions": [{"action": "מאטים ל-50", "quote": "מאטים ל-50 קמ״ש", "leaf": "a1"},
                                                     {"action": "עוצרים בצד", "quote": "עוצרים בצד", "leaf": "none"}],
                                  "nodes": [{"id": "q1", "from_source": True}, {"id": "a1", "from_source": True},
                                            {"id": "a2", "from_source": False, "reason": "אין במקור"}]}]},
                   ensure_ascii=False)


def test_both_directions_become_findings():
    found = diagram_check.parse(REPLY, lesson_with_diagram())
    problems = [f.problem for f in found]
    assert msg("diagram_check.missing_leaf", action="עוצרים בצד") in problems
    assert msg("diagram_check.not_in_source", label="מאיצים", reason="אין במקור") in problems
    assert all(f.screen == 2 and f.category == "diagram_logic" and f.where == "block 2 (decision_diagram)" for f in found)


def test_no_diagram_no_call():
    lesson = lesson_with_diagram()
    lesson["screens"][1]["blocks"].pop()
    assert all(name != "diagram_source" for name, _ in check_jobs(lesson, ("diagram_source",)))
    assert ("diagram_source", None) in check_jobs(lesson_with_diagram(), ("diagram_source",))


def test_the_check_runs_with_the_cited_excerpts():
    prompts = []

    def call(messages, timeout=None, role=None):
        prompts.append(messages)
        return make_result(text=REPLY)

    chunks = [{"text": "מאטים ל-50 קמ״ש; עוצרים בצד", "locator": {"file": "w.pptx", "slide": 4}},
              {"text": "unrelated", "locator": {"file": "w.pptx", "slide": 9}}]
    report = run_checkers(lesson_with_diagram(), chunks, call=call, code=False, jobs_list=[("diagram_source", None)])
    assert len(report.findings) == 2 and "עוצרים בצד" in prompts[0][1].content and "unrelated" not in prompts[0][1].content

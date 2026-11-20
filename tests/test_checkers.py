"""Lesson checkers without a model: the code checks on the known-bad lesson, reply parsing,
retries and batching with a fake model, and the matching used by the acceptance eval."""

import json
import re
from pathlib import Path

import pytest

from agent import checkers
from agent.checkers import (
    CODE_CHECKS,
    Finding,
    check_address,
    check_answer_length,
    check_diagrams,
    check_jobs,
    check_question_overlap,
    check_repetition,
    parse_findings,
    render_lesson,
    run_checkers,
    run_model_check,
)
from evals.checker_eval import finding_matches, load_issues, match_issues
from tests.conftest import make_result

ROOT = Path(__file__).resolve().parent.parent
BAD = json.loads((ROOT / "tests" / "fixtures" / "known_bad_lesson.json").read_text(encoding="utf-8"))
SAMPLE = json.loads((ROOT / "examples" / "sample_lesson" / "lesson.json").read_text(encoding="utf-8"))


def at(findings, screen, category=None):
    return [f for f in findings if f.screen == screen and (category is None or f.category == category)]


# ---- code checks ------------------------------------------------------------------------------
def test_answer_length_flags_a_correct_option_much_longer_than_the_distractors():
    findings = check_answer_length(BAD)
    assert at(findings, 3) and at(findings, 6) and at(findings, 8)
    assert not at(findings, 4)  # screen 4: the options are of similar length


def test_answer_length_ignores_balanced_options():
    lesson = {"screens": [{"blocks": [{"type": "multiple_choice", "question": "q", "correct_index": 0,
                                       "options": ["aaaa bbbb", "cccc dddd", "eeee ffff"]}]}]}
    assert check_answer_length(lesson) == []


def test_diagrams_need_two_branches_per_question_and_branches_that_lead_somewhere_different():
    findings = check_diagrams(BAD)
    assert {f.screen for f in findings} == {4, 6, 7}
    assert "n4" in at(findings, 4)[0].evidence  # both ABS branches go to the same node
    assert all(f.category == "diagram_logic" for f in findings)


def test_diagram_with_an_unreached_node():
    lesson = {"screens": [{"blocks": [{"type": "decision_diagram", "title": "t", "nodes": [
        {"id": "a", "label": "A?", "kind": "question"}, {"id": "b", "label": "B", "kind": "action"},
        {"id": "c", "label": "C", "kind": "action"}, {"id": "d", "label": "D", "kind": "action"}],
        "edges": [{"from": "a", "to": "b"}, {"from": "a", "to": "c"}]}]}]}
    findings = check_diagrams(lesson)
    assert len(findings) == 1 and "d" in findings[0].evidence


def test_address_mix_of_singular_and_plural_is_reported():
    findings = check_address(BAD)
    assert any(f.screen is None for f in findings)
    assert at(findings, 1, "address")


def test_address_consistent_lesson_has_no_finding():
    lesson = {"screens": [{"blocks": [{"type": "paragraph", "text": "קחו בחשבון"}]}]}
    assert check_address(lesson) == []


def test_repetition_finds_the_repeated_list_and_the_repeated_phrase():
    findings = check_repetition(BAD)
    assert at(findings, 5, "repetition")  # screen 5 repeats the list of screen 4
    assert at(findings, 1, "duplicate_text")  # "three short questions" twice in one paragraph


def test_question_overlap_finds_the_alternate_that_repeats_another_screen():
    findings = check_question_overlap(BAD)
    assert (5, "block 6 (multiple_choice), alternate") in [(f.screen, f.where) for f in findings]


def test_code_checks_on_the_sample_lesson_stay_quiet():
    findings = [f for check in CODE_CHECKS.values() for f in check(SAMPLE)]
    assert {f.category for f in findings} <= {"answer_length"}


def test_render_lesson_labels_screens_and_blocks_and_marks_the_answer():
    text = render_lesson(BAD)
    assert "[screen 3, block 5, multiple_choice] alternate question" in text
    assert "(marked correct)" in text and "node n1 (question)" in text
    assert "sources" not in text and "s1" not in text.split()


# ---- parsing model replies ---------------------------------------------------------------------
def test_parse_plain_findings_maps_unknown_categories():
    reply = 'ok {"findings": [{"screen": "3", "where": "block 5", "category": "weird", "evidence": "x",' \
            ' "problem": "p"}, {"screen": 2, "category": "language", "evidence": "y"}]}'
    found = parse_findings(reply, "language")
    assert [(f.screen, f.category) for f in found] == [(3, "other"), (2, "language")]
    single = parse_findings('{"findings": [{"screen": 1, "category": "made_up"}]}', "grounding")
    assert single[0].category == "ungrounded"


def test_parse_ignores_text_and_extra_json_after_the_reply():
    reply = 'Here: {"findings": [{"screen": 1, "category": "language"}]}\nNote {"x": 1} and {not json'
    assert len(parse_findings(reply, "language")) == 1
    assert len(parse_findings('{"a": {"b": 1}} {"findings": []}', "language")) == 0


@pytest.mark.parametrize("reply", ["no json here", '{"findings": "none"}', "{broken", '{"other": []}'])
def test_parse_rejects_unusable_replies(reply):
    with pytest.raises(ValueError):
        parse_findings(reply, "language")


def test_illustration_checklist_reports_only_the_ones_without_value():
    reply = json.dumps({"illustrations": [
        {"screen": 1, "title": "t1", "adds_value": False, "reason": "r"},
        {"screen": 2, "title": "t2", "adds_value": True, "reason": "ok"}]})
    found = parse_findings(reply, "illustrations", expected=2)
    assert [(f.screen, f.category, f.evidence) for f in found] == [(1, "illustration_value", "t1")]


# ---- running with a fake model ----------------------------------------------------------------
def test_invalid_reply_is_sent_back_once_then_accepted():
    replies = iter(["not json", '{"findings": [{"screen": 1, "category": "language", "evidence": "x"}]}'])
    seen = []

    def call(messages, timeout, role):
        seen.append((len(messages), role))
        return make_result(text=next(replies))

    found, calls = run_model_check("language", BAD, render_lesson(BAD), "", call)
    assert len(found) == 1 and len(calls) == 2
    assert seen == [(2, "checker"), (4, "checker")]  # the bad reply and the correction were added


def test_question_check_runs_in_screen_batches_and_on_given_screens():
    jobs = check_jobs(BAD, tuple(checkers.MODEL_CHECKS))
    assert [j for j in jobs if j[0] == "single_answer"] == [
        ("single_answer", (1, 2, 3)), ("single_answer", (4, 5, 6)), ("single_answer", (7, 8))]
    assert ("language", None) in jobs
    again = check_jobs(BAD, ("language", "single_answer"), screens=(2, 5))
    assert again == [("language", (2, 5)), ("single_answer", (2, 5))]


def test_a_check_on_some_screens_says_so_and_drops_other_screens():
    text = checkers.build_check_messages("language", "lesson", "", screens=(2, 5))[1].content
    assert "only the problems on screens 2, 5" in text
    reply = '{"findings": [{"screen": 2, "category": "language"}, {"screen": 3, "category": "language"}]}'
    assert [f.screen for f in parse_findings(reply, "language", screens=(2, 5))] == [2]


def test_a_failing_model_check_is_recorded_and_the_rest_still_run():
    def call(messages, timeout, role):
        system = messages[0].content
        if "proofread" in system:
            raise RuntimeError("down")
        if system.startswith("You check the multiple-choice questions"):  # single answer
            ids = re.findall(r"### Question (\S+)", messages[1].content)
            return make_result(text=json.dumps({"questions": [{"id": i, "options": []} for i in ids]}))
        if '"illustrations"' in system:
            return make_result(text='{"illustrations": [' + ",".join(['{"adds_value": true}'] * 8) + "]}")
        return make_result(text='{"findings": []}')

    report = run_checkers(BAD, [], call=call)
    assert set(report.failed) == {"language"}
    assert len(report.calls) == 6  # grounding, illustrations, pedagogy, 3 question batches
    assert {f.check for f in report.findings} == set(CODE_CHECKS)


# ---- single answer -------------------------------------------------------------------------------
def test_single_answer_items_carry_only_the_questions_excerpts():
    chunks = [{"text": "fog rule", "locator": {"file": "short_winter.pptx", "slide": 4, "paragraph": 2}},
              {"text": "tyres", "locator": {"file": "short_winter.pptx", "slide": 2, "paragraph": 3}}]
    items = checkers.single_answer_items(BAD, chunks, (6, 6))
    assert [i["id"] for i in items] == ["q6", "q6-alternate"]
    assert [c["text"] for c in items[0]["excerpts"]] == ["fog rule"]
    text = checkers.build_single_answer_messages(items)[1].content
    assert "fog rule" in text and "tyres" not in text
    assert "marked" not in text  # the checker is not told which option is correct


def _single_reply(qid, justified):
    return {"id": qid, "options": [{"letter": chr(65 + k), "justified": ok, "reason": f"r{k}"}
                                   for k, ok in enumerate(justified)]}


def test_single_answer_reports_a_second_justified_option_and_an_unjustified_answer():
    items = checkers.single_answer_items(BAD, [], (6, 6))
    main, alt = items
    just_main = [True] * len(main["options"])  # every option holds: more than one correct
    just_alt = [False] * len(alt["options"])  # the marked answer does not hold
    reply = json.dumps({"questions": [_single_reply("q6", just_main), _single_reply("q6-alternate", just_alt)]})
    found = checkers.parse_single_answer(reply, items)
    assert [(f.category, f.where) for f in found] == [
        ("multiple_correct", "block 4 (multiple_choice), main"),
        ("wrong_answer", "block 4 (multiple_choice), alternate")]
    assert found[0].problem.count(";") == len(main["options"]) - 2  # reasons of the other options


def test_single_answer_needs_every_question():
    items = checkers.single_answer_items(BAD, [], (1, 3))
    with pytest.raises(ValueError, match="missing"):
        checkers.parse_single_answer('{"questions": []}', items)


def test_checker_family_differs_from_the_writer():
    checkers_list = ["google/gemma-4-31b-it:free", "nvidia/nemotron-3.5-super-vl:free"]
    other, same = checkers.checker_models(["nvidia/nemotron-3.5-super-vl:free"], checkers_list)
    assert other == ["google/gemma-4-31b-it:free"] and same == ["nvidia/nemotron-3.5-super-vl:free"]
    other, same = checkers.checker_models(["baidu/ernie-5.5-flash-preview:free"], checkers_list)
    assert other == checkers_list and same == []


def test_source_for_the_checkers_is_the_cited_chunks():
    chunks = [{"text": "cited", "locator": {"file": "short_winter.pptx", "slide": 2, "paragraph": 2}},
              {"text": "other file", "locator": {"file": "x.pptx", "slide": 1, "paragraph": 1}}]
    picked = checkers.cited_chunks(BAD, chunks)
    assert [c["text"] for c in picked] == ["cited"]
    assert "<source_material>" in checkers.render_source(picked)


# ---- matching for the acceptance eval ------------------------------------------------------------
def test_issue_matching_needs_screen_category_and_keyword():
    issue = {"id": "X", "screens": [3], "categories": ["language"], "keywords": ["אבלע"]}
    hit = Finding(check="language", category="language", screen=3, evidence="אבלע בלחיצות")
    assert finding_matches(issue, hit)
    assert not finding_matches(issue, hit.model_copy(update={"screen": 4}))
    assert not finding_matches(issue, hit.model_copy(update={"category": "distractors"}))
    assert not finding_matches(issue, hit.model_copy(update={"evidence": "other"}))
    assert finding_matches({"id": "Y", "screens": None, "categories": ["address"]},
                           Finding(check="address", category="address"))


def test_the_code_checks_alone_already_catch_the_structural_issues():
    issues = load_issues()
    findings = [f for check in CODE_CHECKS.values() for f in check(BAD)]
    matched = match_issues(issues, findings)
    caught = {i for i, hits in matched.items() if hits}
    assert {"I01", "I06", "I09", "I13", "I14", "I15", "I16"} <= caught


def test_single_answer_also_reports_recall_questions_and_weak_options():
    items = checkers.single_answer_items(BAD, [], (6, 6))
    main = items[0]
    correct = main["correct_index"]
    options = [{"letter": chr(65 + k), "justified": k == correct, "weak": k != correct and k % 2 == 0,
                "reason": f"r{k}"} for k in range(len(main["options"]))]
    reply = json.dumps({"questions": [{"id": "q6", "kind": "recall", "kind_reason": "repeats the rule",
                                       "options": options},
                                      {"id": "q6-alternate", "kind": "decision", "options": []}]})
    found = checkers.parse_single_answer(reply, items)
    kinds = [f.category for f in found]
    assert kinds == ["recall_question", "distractors"]
    weak = [main["options"][k] for k in range(len(main["options"])) if k != correct and k % 2 == 0]
    assert found[1].evidence == "; ".join(weak)


def test_single_answer_batches_are_smaller_in_balanced(monkeypatch):
    monkeypatch.setenv("PROFILE", "balanced")
    jobs = check_jobs(BAD, ("single_answer",))
    assert jobs == [("single_answer", (1, 2)), ("single_answer", (3, 4)), ("single_answer", (5, 6)),
                    ("single_answer", (7, 8))]
    monkeypatch.setenv("PROFILE", "free")
    assert len(check_jobs(BAD, ("single_answer",))) == 3

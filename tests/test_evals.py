"""The eval tools without real models: run_agent (whole graph, scripted gates), the judge,
and the cost helpers."""

import json
from pathlib import Path

import pytest

import agent.nodes as nodes
from evals import judge
from tests.conftest import make_result
from tests.test_lesson_flow import FakeModel

ROOT = Path(__file__).resolve().parent.parent
BAD = json.loads((ROOT / "tests" / "fixtures" / "known_bad_lesson.json").read_text(encoding="utf-8"))


# ---- judge ---------------------------------------------------------------------------------------
def _scores(**over):
    data = {c: {"score": 4, "reason": f"{c} ok"} for c in judge.CRITERIA}
    data.update(over)
    return json.dumps(data)


def test_judge_scores_every_criterion_with_one_fixed_model():
    seen = []

    def call(messages, timeout, models):
        seen.append(models)
        return make_result(model=models[0], text="Here: " + _scores(format={"score": 2, "reason": "two answers"}))

    scores, results = judge.judge_lesson(BAD, [], call=call, model="google/judge")
    assert seen == [["google/judge"]]
    assert scores["format"] == {"score": 2, "reason": "two answers"}
    assert scores["average"] == 3.5 and scores["model"] == "google/judge"
    assert len(results) == 1


def test_judge_retries_an_invalid_reply_once():
    replies = iter(['{"relevance": {"score": 9}}', _scores()])
    scores, results = judge.judge_lesson(BAD, [], call=lambda m, timeout, models: make_result(text=next(replies)),
                                         model="m")
    assert len(results) == 2 and scores["average"] == 4


def test_judge_gives_up_after_two_invalid_replies():
    with pytest.raises(ValueError, match="judge"):
        judge.judge_lesson(BAD, [], call=lambda m, timeout, models: make_result(text="no"), model="m")


def test_judge_prompt_has_rubric_source_and_lesson_without_marking_ids():
    chunks = [{"text": "rule text", "locator": {"file": "a.pptx", "slide": 1, "paragraph": 1}}]
    system, user = judge.build_messages(BAD, chunks, "training", schema_valid=False)
    assert "Factual accuracy" in system.content and "<source_material>" in user.content
    assert "rule text" in user.content and "INVALID" in user.content and "Screen 1" in user.content


def test_judge_node_reads_the_state_and_records_usage():
    recorded = []
    config = {"configurable": {"judge_call": lambda m, timeout, models: make_result(text=_scores()),
                               "judge_model": "m", "usage_recorder": lambda node, r: recorded.append(node)}}
    out = judge.judge_node({"lesson": BAD, "lesson_type": "training"}, config)
    assert out["judge"]["average"] == 4 and recorded == ["judge"]


def test_default_judge_is_the_balanced_checker():
    from agent.llm import load_config

    assert judge.default_judge_model() == load_config()["profiles"]["balanced"]["checker"][0]


# ---- run_agent -----------------------------------------------------------------------------------
def test_run_agent_runs_the_whole_graph_without_a_keyboard(monkeypatch, tmp_path):
    from evals import run_agent

    model = FakeModel()
    monkeypatch.setattr(nodes, "call_llm", model)
    out = tmp_path / "run"
    summary = run_agent.run_agent(experiment="unit", out_dir=out, writer=["x/writer"])
    assert summary["error"] is None
    assert summary["screens"] >= 1 and (out / "lesson.json").is_file() and (out / "report.md").is_file()
    saved = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert saved["models"]["writer"] == ["x/writer"]
    assert saved["gate_answers"] == ["Y", "Y", "Y", "Y"]  # type, topics, structure, gate 3
    assert saved["calls"] == len(model.kinds) and "write_lesson" in saved["usage_by_node"]
    assert "evaluator" in saved["seconds_by_node"] and saved["coverage"]["total"] == 3
    assert not (ROOT / "output" / "short_winter-99").exists()


def test_run_agent_uses_the_scripted_answers(monkeypatch, tmp_path):
    from evals import run_agent

    model = FakeModel()
    monkeypatch.setattr(nodes, "call_llm", model)
    summary = run_agent.run_agent(answers=["Y", "N add a topic on tires", "Y", "Y", "Y"], out_dir=None)
    assert summary["gate_answers"][:2] == ["Y", "N add a topic on tires"]
    assert len(model.topic_map_prompts()) == 2


# ---- costs ---------------------------------------------------------------------------------------
def test_theoretical_cost_uses_the_paid_model_of_the_same_role():
    from evals import costs

    sonnet, flash = costs.equivalent_model("writer"), costs.equivalent_model("checker")
    prices = {sonnet: {"prompt": 2e-6, "completion": 1e-5}, flash: {"prompt": 1e-6, "completion": 4e-6}}
    writer_call = {"node": "write_lesson", "model": "x/free-writer:free", "prompt_tokens": 1000,
                   "completion_tokens": 100, "cost": 0}
    checker_call = {**writer_call, "node": "evaluator"}
    paid_call = {**writer_call, "model": sonnet, "cost": 0.5}
    assert costs.record_costs(writer_call, prices) == (0.0, 1000 * 2e-6 + 100 * 1e-5)
    assert costs.record_costs(checker_call, prices) == (0.0, 1000 * 1e-6 + 100 * 4e-6)
    assert costs.record_costs(paid_call, prices) == (0.5, 0.5)  # a paid model: theoretical = actual
    assert costs.total_costs([writer_call, paid_call], prices) == (0.5, round(0.5 + 0.003, 6))

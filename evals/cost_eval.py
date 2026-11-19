"""Eval 2: cost per task, on a free model and a paid one.

Five common tasks: topic map, structure plan, writing a group of 4 screens, Hebrew check of a
lesson, and a whole short lesson. To save money the agent tasks reuse the runs of
compare_models.py (evals/results/runs/compare-<writer>-<n>/run.json): the free row is the
first free writer's run, the paid row Sonnet's. Only the Hebrew check runs here, once on the
free checker and once on the paid one (Gemini Flash), on the same lesson.

Actual cost is OpenRouter's usage.cost; theoretical cost of a free call is its tokens at the
price of the balanced model of the same role (evals/costs.py, prices in prices.json).

    .venv\\Scripts\\python -m evals.cost_eval [--free-run compare-ernie-5.5-flash-preview-free-1]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "evals" / "results"
RUNS = RESULTS / "runs"
CSV_PATH = RESULTS / "cost_eval.csv"
SUMMARY_PATH = RESULTS / "cost_summary.md"
FIELDS = ["task", "model", "node", "prompt_tokens", "completion_tokens", "actual_cost_usd", "theoretical_cost_usd",
          "is_free"]
TASK_NODES = {"topic_map": ["topic_map"], "structure_plan": ["plan_structure"], "write_4_screens": ["write_lesson"]}
HEBREW_LESSON = ROOT / "tests" / "fixtures" / "known_bad_lesson.json"


def load_run(name: str) -> dict:
    return json.loads((RUNS / name / "run.json").read_text(encoding="utf-8"))


def pick_run(prefix: str) -> str | None:
    for path in sorted(RUNS.glob(f"{prefix}*/run.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        if run.get("screens") and not run.get("error"):
            return path.parent.name
    return None


def rows_for(task: str, label: str, records: list[dict], prices: dict, free: bool) -> list[dict]:
    from evals.costs import record_costs

    rows = []
    for r in records:
        actual, theoretical = record_costs(r, prices)
        rows.append({"task": task, "model": r["model"], "node": r["node"], "prompt_tokens": r["prompt_tokens"],
                     "completion_tokens": r["completion_tokens"], "actual_cost_usd": round(actual, 6),
                     "theoretical_cost_usd": round(theoretical, 6), "is_free": r["model"].endswith(":free")})
    rows.append({"task": task, "model": label, "node": "TOTAL",
                 "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
                 "completion_tokens": sum(r["completion_tokens"] for r in rows),
                 "actual_cost_usd": round(sum(r["actual_cost_usd"] for r in rows), 6),
                 "theoretical_cost_usd": round(sum(r["theoretical_cost_usd"] for r in rows), 6), "is_free": free})
    return rows


def agent_task_rows(run_name: str, free: bool, prices: dict) -> list[dict]:
    run = load_run(run_name)
    writer = run["models"]["writer"][0]
    records = run["usage_records"]
    rows = []
    for task, task_nodes in TASK_NODES.items():
        picked = [r for r in records if r["node"] in task_nodes]
        if task == "write_4_screens":
            picked = picked[:1]  # the first group: 4 screens in one call
        rows += rows_for(task, writer, picked, prices, free)
    rows += rows_for("full_lesson", f"{writer} (whole run)", records, prices, free)
    return rows


def hebrew_check_rows(prices: dict) -> list[dict]:
    """The language checker on the same lesson: the free checker list, then Gemini Flash."""
    from agent.checkers import render_lesson, run_model_check
    from agent.llm import call_llm, load_config
    from evals.budget import can_spend

    lesson = json.loads(HEBREW_LESSON.read_text(encoding="utf-8"))
    config = load_config()["profiles"]
    rows = []
    for label, models, free in ((config["free"]["checker"][0], config["free"]["checker"], True),
                                (config["balanced"]["checker"][0], config["balanced"]["checker"][:1], False)):
        if not free and not can_spend(0.1):
            rows.append({"task": "hebrew_check", "model": label, "node": "TOTAL", "is_free": free,
                         "actual_cost_usd": "", "theoretical_cost_usd": "", "prompt_tokens": "", "completion_tokens": ""})
            continue

        def call(messages, timeout=None, role="checker", models=models):
            return call_llm(messages, timeout=timeout or 240, models=models)

        _, results = run_model_check("language", lesson, render_lesson(lesson), "", call)
        records = [{"node": "evaluator", "model": r.model, "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens, "cost": r.cost} for r in results]
        rows += rows_for("hebrew_check", label, records, prices, free)
    return rows


def write_summary(rows: list[dict], free_run: str, paid_run: str | None) -> None:
    totals = [r for r in rows if r["node"] == "TOTAL"]
    lines = ["# Cost per task", "",
             f"Free row: run `{free_run}`; paid row: run `{paid_run or 'none (skipped)'}` (from compare_models.py). "
             "Theoretical cost of a free call = its tokens at the price of the balanced model of the same role "
             "(prices in `prices.json`).", "",
             "| task | model | prompt tokens | completion tokens | actual USD | theoretical USD |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in totals:
        lines.append(f"| {r['task']} | {r['model']} | {r['prompt_tokens']} | {r['completion_tokens']} | "
                     f"{r['actual_cost_usd']} | {r['theoretical_cost_usd']} |")
    full = {r["is_free"]: r for r in totals if r["task"] == "full_lesson"}
    by_node: dict[str, float] = {}
    for r in rows:
        if r["task"] == "full_lesson" and r["node"] != "TOTAL":
            by_node[r["node"]] = by_node.get(r["node"], 0.0) + float(r["theoretical_cost_usd"] or 0)
    lines += ["", "## A whole lesson per profile", ""]
    if True in full:
        lines.append(f"- **free profile**: ${full[True]['actual_cost_usd']} actual; at balanced prices it would be "
                     f"${full[True]['theoretical_cost_usd']}.")
    if False in full:
        lines.append(f"- **Sonnet writer** (checkers and small model on the free lists, as in compare_models): "
                     f"${full[False]['actual_cost_usd']} actual; with the balanced checkers and small model it "
                     f"would be about ${full[False]['theoretical_cost_usd']}.")
    if by_node:
        top = sorted(by_node.items(), key=lambda kv: -kv[1])[:3]
        lines += ["", "Most expensive steps of a whole lesson (theoretical, both runs together): "
                  + ", ".join(f"{n} ${v:.4f}" for n, v in top) + "."]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    from agent.config import configure_stdio
    from evals.costs import load_prices

    parser = argparse.ArgumentParser()
    parser.add_argument("--free-run")
    parser.add_argument("--paid-run")
    args = parser.parse_args(argv)
    configure_stdio()
    load_dotenv(ROOT / ".env")
    prices = load_prices()
    free_run = args.free_run or pick_run("compare-ernie")
    paid_run = args.paid_run or pick_run("compare-claude-sonnet")
    if not free_run:
        print("no successful free run in evals/results/runs; run compare_models.py first")
        return 1
    rows = agent_task_rows(free_run, True, prices)
    if paid_run:
        rows += agent_task_rows(paid_run, False, prices)
    rows += hebrew_check_rows(prices)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    write_summary(rows, free_run, paid_run)
    print(SUMMARY_PATH.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

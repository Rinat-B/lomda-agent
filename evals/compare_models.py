"""Eval 1: compare writer models on the same task.

The task: a whole lesson from evals/fixtures/short_winter.pptx with the same gate answers
(Y everywhere). Only the writer changes; the checkers and small models are the free
profile's lists, and one fixed judge (evals/judge.py) scores every lesson by
evals/rubric.md. Each run is a separate process (evals/run_agent.py) with a time limit.
A baseline row asks Sonnet for the lesson in one prompt, without the agent.

Coverage is measured against one fixed list of the source's key points (extracted once,
saved in evals/results/key_points_short_winter.json), so every row uses the same yardstick.

    .venv\\Scripts\\python -m evals.compare_models            # everything
    .venv\\Scripts\\python -m evals.compare_models --runs 1 --writers baidu/ernie-5.5-flash-preview:free

Paid runs are skipped when evals/budget.py says the night's budget would be exceeded.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "evals" / "results"
RUNS_DIR = RESULTS / "runs"
CSV_PATH = RESULTS / "compare_models.csv"
SUMMARY_PATH = RESULTS / "compare_models_summary.md"
KEY_POINTS_PATH = RESULTS / "key_points_short_winter.json"
SOURCE = ROOT / "evals" / "fixtures" / "short_winter.pptx"
SCHEMA_PATH = ROOT / "agent" / "assets" / "lesson.schema.json"
RUN_TIMEOUT_SECONDS = 45 * 60
SONNET = "anthropic/claude-sonnet-5.5"
DEFAULT_WRITERS = ["baidu/ernie-5.5-flash-preview:free", "nvidia/nemotron-3.5-super-vl:free", SONNET]
PAID_RUN_ESTIMATE_USD = 1.0  # until a paid run is measured
JUDGE_ESTIMATE_USD = 0.05
BASELINE_ESTIMATE_USD = 0.3
FIELDS = ["model", "run", "is_free", "duration_sec", "calls", "screens", "relevance", "accuracy", "completeness",
          "format", "judge_avg", "schema_valid", "coverage_pct", "open_findings", "cost_usd",
          "theoretical_cost_usd", "notes"]


def log(text: str) -> None:
    print(time.strftime("%H:%M"), text, flush=True)


def slug(model: str) -> str:
    return model.split("/")[-1].replace(":", "-")


def free_lists() -> tuple[list[str], list[str]]:
    from agent.llm import load_config

    free = load_config()["profiles"]["free"]
    return free["checker"], free["small"]


def source_chunks() -> list[dict]:
    from agent.mcp_servers.source_reader import read_source_file

    return read_source_file(str(SOURCE))["chunks"]


def fixed_key_points(chunks: list[dict]) -> list[dict]:
    """The source's key points, extracted once by the free small model and reused."""
    if KEY_POINTS_PATH.is_file():
        return json.loads(KEY_POINTS_PATH.read_text(encoding="utf-8"))["points"]
    from agent.llm import set_role_models
    from agent.nodes import extract_key_points

    set_role_models("small", free_lists()[1])
    points = extract_key_points(chunks, None) or []
    set_role_models("small", None)
    KEY_POINTS_PATH.write_text(json.dumps({"source": SOURCE.name, "points": points}, ensure_ascii=False, indent=1),
                               encoding="utf-8")
    return points


def run_one(writer: str, run: int) -> dict:
    """One agent run in its own process; returns run.json (or an error summary)."""
    checker, small = free_lists()
    out = RUNS_DIR / f"compare-{slug(writer)}-{run}"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "evals.run_agent", "--experiment", f"compare-{slug(writer)}-{run}",
           "--profile", "free", "--writer", writer, "--checker", ",".join(checker), "--small", ",".join(small),
           "--out", str(out)]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "HEBREW_BIDI": "off"}
    started = time.monotonic()
    with (out / "console.txt").open("w", encoding="utf-8") as console:
        try:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=console, stderr=subprocess.STDOUT,
                           timeout=RUN_TIMEOUT_SECONDS, check=False)
        except subprocess.TimeoutExpired:
            return {"error": f"timeout after {RUN_TIMEOUT_SECONDS // 60} min", "duration_sec": time.monotonic() - started}
    path = out / "run.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"error": "no run.json"}


def judge(lesson: dict | None, chunks: list[dict], schema_valid: bool) -> tuple[dict | None, float, str]:
    """(scores, judge cost, note)."""
    from evals.budget import can_spend
    from evals.judge import judge_lesson

    if not lesson:
        return None, 0.0, "no lesson to judge"
    if not can_spend(JUDGE_ESTIMATE_USD):
        return None, 0.0, "judge skipped: budget"
    try:
        scores, results = judge_lesson(lesson, chunks, lesson_type=lesson.get("lesson_type"), schema_valid=schema_valid)
    except Exception as e:  # noqa: BLE001
        return None, 0.0, f"judge failed: {type(e).__name__}"
    return scores, sum(r.cost for r in results), ""


def schema_ok(lesson: dict | None) -> bool:
    from agent.lesson_schema import validate_lesson

    try:
        return bool(lesson) and not validate_lesson(lesson)
    except Exception:  # noqa: BLE001
        return False


def coverage_pct(lesson: dict | None, points: list[dict]) -> float | None:
    from agent.evaluation import key_point_coverage

    if not lesson or not points:
        return None
    try:
        taught, missing = key_point_coverage(points, lesson)
    except Exception:  # noqa: BLE001 - a malformed baseline lesson
        return None
    return round(100 * len(taught) / max(1, len(taught) + len(missing)), 1)


def row_for(model: str, run: int, summary: dict, scores: dict | None, valid: bool, coverage, notes: list[str]) -> dict:
    from evals.costs import is_free, load_prices, total_costs

    actual, theoretical = total_costs(summary.get("usage_records") or [], load_prices())
    s = scores or {}
    return {
        "model": model, "run": run, "is_free": is_free(model), "duration_sec": round(summary.get("duration_sec") or 0),
        "calls": summary.get("calls"), "screens": summary.get("screens"),
        **{c: (s.get(c) or {}).get("score") for c in ("relevance", "accuracy", "completeness", "format")},
        "judge_avg": s.get("average"), "schema_valid": valid, "coverage_pct": coverage,
        "open_findings": len(summary.get("open_findings") or []) if "open_findings" in summary else "",
        "cost_usd": actual, "theoretical_cost_usd": theoretical, "notes": "; ".join(n for n in notes if n),
    }


def append_row(row: dict) -> None:
    new = not CSV_PATH.is_file()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


BASELINE_PROMPT = (
    "You are an instructional designer. Write a complete Hebrew e-learning lesson from the "
    "source material, for the user's request, as one JSON object that is valid against the "
    "JSON Schema below (lesson.json of a lesson player). Every learner-facing text is in Hebrew. "
    "Use the source's facts only. Reply with only the JSON object.\n\nJSON Schema:\n"
)


def run_baseline(chunks: list[dict], request: str) -> tuple[dict, dict | None]:
    """One prompt to Sonnet, no agent: (summary like run.json, lesson or None)."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from agent.checkers import _json_object, render_source
    from agent.guards import SOURCE_DATA_RULE
    from agent.llm import call_llm

    started = time.monotonic()
    messages = [SystemMessage(BASELINE_PROMPT + SCHEMA_PATH.read_text(encoding="utf-8") + SOURCE_DATA_RULE),
                HumanMessage(f"Source material:\n{render_source(chunks)}\n\nUser request: {request}")]
    summary: dict = {"error": None}
    lesson = None
    try:
        result = call_llm(messages, timeout=600, models=[SONNET])
        summary["usage_records"] = [{"node": "baseline", "model": result.model, "prompt_tokens": result.prompt_tokens,
                                     "completion_tokens": result.completion_tokens, "cost": result.cost}]
        summary["calls"] = 1
        try:
            lesson = _json_object(result.text, "screens")
        except ValueError as e:
            summary["error"] = f"no lesson JSON: {e}"
    except Exception as e:  # noqa: BLE001
        summary["error"] = f"{type(e).__name__}: {e}"
    summary["duration_sec"] = time.monotonic() - started
    summary["screens"] = len(lesson.get("screens") or []) if isinstance(lesson, dict) else 0
    out = RUNS_DIR / "baseline_chat"
    out.mkdir(parents=True, exist_ok=True)
    if lesson:
        (out / "lesson.json").write_text(json.dumps(lesson, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "run.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary, lesson


def write_summary() -> None:
    """The mean per model, from the CSV (the conclusion paragraph is written by hand)."""
    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    models = list(dict.fromkeys(r["model"] for r in rows))
    cols = ["duration_sec", "calls", "screens", "relevance", "accuracy", "completeness", "format", "judge_avg",
            "coverage_pct", "open_findings", "cost_usd", "theoretical_cost_usd"]

    def mean(values):
        nums = [float(v) for v in values if v not in ("", None, "None")]
        return f"{sum(nums) / len(nums):.2f}" if nums else "-"

    lines = ["| model | runs | " + " | ".join(cols) + " | schema_valid |", "|" + " --- |" * (len(cols) + 3)]
    for m in models:
        mine = [r for r in rows if r["model"] == m]
        valid = sum(r["schema_valid"] == "True" for r in mine)
        lines.append(f"| {m} | {len(mine)} | " + " | ".join(mean(r[c] for r in mine) for c in cols)
                     + f" | {valid}/{len(mine)} |")
    table = "\n".join(lines)
    text = SUMMARY_PATH.read_text(encoding="utf-8") if SUMMARY_PATH.is_file() else ""
    start, end = "<!-- table -->", "<!-- /table -->"
    block = f"{start}\n{table}\n{end}"
    if start in text and end in text:
        text = text[:text.index(start)] + block + text[text.index(end) + len(end):]
    else:
        text = "# Writer model comparison\n\nMeans per model (see compare_models.csv for every run).\n\n" + block + "\n"
    SUMMARY_PATH.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    from agent.config import configure_stdio
    from evals.budget import can_spend, status
    from evals.costs import equivalent_model, fetch_prices
    from evals.judge import default_judge_model
    from evals.run_agent import default_request

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--writers", default=",".join(DEFAULT_WRITERS))
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args(argv)
    configure_stdio()
    load_dotenv(ROOT / ".env")
    writers = [w.strip() for w in args.writers.split(",") if w.strip()]
    checker, small = free_lists()
    fetch_prices(sorted(set(writers + checker + small + [default_judge_model(), SONNET]
                            + [equivalent_model(r) for r in ("writer", "checker", "small")])))
    chunks = source_chunks()
    points = fixed_key_points(chunks)
    log(f"key points: {len(points)}; judge: {default_judge_model()}; checkers: {checker}; small: {small}")
    paid_estimate = PAID_RUN_ESTIMATE_USD
    if not args.baseline_only:
        for run in range(1, args.runs + 1):
            for writer in writers:
                notes = []
                paid = not writer.endswith(":free")
                if paid and not can_spend(paid_estimate + JUDGE_ESTIMATE_USD):
                    log(f"skip {writer} run {run}: budget {status()}")
                    append_row({"model": writer, "run": run, "is_free": False, "notes": "skipped: night budget"})
                    continue
                log(f"run {run}: {writer}")
                summary = run_one(writer, run)
                if summary.get("error"):
                    notes.append(f"error: {summary['error'][:120]}")
                failed = (summary.get("review") or {}).get("failed") or {}
                if failed:
                    notes.append(f"{len(failed)} checks failed (429)")
                lesson = None
                lesson_path = RUNS_DIR / f"compare-{slug(writer)}-{run}" / "lesson.json"
                if lesson_path.is_file():
                    lesson = json.loads(lesson_path.read_text(encoding="utf-8"))
                valid = schema_ok(lesson)
                scores, judge_cost, note = judge(lesson, chunks, valid)
                notes.append(note)
                row = row_for(writer, run, summary, scores, valid, coverage_pct(lesson, points), notes)
                append_row(row)
                if paid and row["cost_usd"]:
                    paid_estimate = max(paid_estimate * 0.5, float(row["cost_usd"]) * 1.1)
                log(f"done {writer} run {run}: {json.dumps(row, ensure_ascii=False)} (judge ${judge_cost:.4f})")
                write_summary()
    if not args.no_baseline:
        if can_spend(BASELINE_ESTIMATE_USD + JUDGE_ESTIMATE_USD):
            log("baseline_chat")
            summary, lesson = run_baseline(chunks, default_request(SOURCE))
            valid = schema_ok(lesson)
            scores, _, note = judge(lesson, chunks, valid) if isinstance(lesson, dict) and lesson.get("screens") else (None, 0, "no lesson")
            row = row_for("baseline_chat", 1, summary, scores, valid, coverage_pct(lesson, points),
                          [f"one prompt to {SONNET}, no agent", note, summary.get("error") or ""])
            row["open_findings"] = ""
            append_row(row)
            log(f"done baseline: {json.dumps(row, ensure_ascii=False)}")
            write_summary()
        else:
            log(f"skip baseline: budget {status()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

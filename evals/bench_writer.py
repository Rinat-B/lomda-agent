"""Benchmark writer models on one group of 4 screens (the real write_lesson prompt).

Every model gets the same plan (evals/fixtures/short_winter_plan.json), the same excerpts
from short_winter.pptx and the same prompt as write_lesson. Measured per run: time, and
whether the first reply passes parse_group (valid JSON + schema + plan rules) with no
retry. Results go to evals/results/writer_bench.csv and a summary table is printed.

    .venv\\Scripts\\python evals\\bench_writer.py [--models a:free b:free] [--runs 2]

By default it tries every ":free" model OpenRouter lists, except very small ones.
"""

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from agent.llm import LONG_REQUEST_TIMEOUT_SECONDS, _client  # noqa: E402
from agent.mcp_servers.source_reader import read_source_file  # noqa: E402
from agent.writer import build_messages, group_excerpts, parse_group  # noqa: E402

FIXTURES = ROOT / "evals" / "fixtures"
RESULTS = ROOT / "evals" / "results" / "writer_bench.csv"
TOO_SMALL = ("1.7b", "nano")


def free_models() -> list[str]:
    import httpx

    ids = [m["id"] for m in httpx.get("https://openrouter.ai/api/v1/models", timeout=30).json()["data"]]
    return sorted(m for m in ids if m.endswith(":free") and not any(s in m for s in TOO_SMALL))


def run_once(model: str, messages, plan, excerpts, lesson_type: str, timeout: float) -> dict:
    start = time.monotonic()
    row = {"model": model, "seconds": None, "valid": False, "error": "", "prompt_tokens": 0, "completion_tokens": 0}
    try:
        response = _client(model, timeout).invoke(messages)
        row["seconds"] = round(time.monotonic() - start, 1)
        usage = response.response_metadata.get("token_usage") or {}
        row["prompt_tokens"] = usage.get("prompt_tokens", 0) or 0
        row["completion_tokens"] = usage.get("completion_tokens", 0) or 0
        try:
            parse_group(str(response.content), plan, excerpts, lesson_type, first=True, last=True)
            row["valid"] = True
        except ValueError as e:
            row["error"] = f"invalid: {str(e)[:160]}"
    except Exception as e:  # API error or timeout
        row["seconds"] = round(time.monotonic() - start, 1)
        row["error"] = f"{type(e).__name__}: {str(e)[:120]}"
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=LONG_REQUEST_TIMEOUT_SECONDS)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")

    plan_data = json.loads((FIXTURES / "short_winter_plan.json").read_text(encoding="utf-8"))
    plan, lesson_type = plan_data["screens"], plan_data["lesson_type"]
    source = read_source_file(str(FIXTURES / "short_winter.pptx"))
    excerpts = group_excerpts(plan, source["chunks"])
    (_, system), (_, user) = build_messages(plan, excerpts, lesson_type, first=True, last=True)
    messages = [SystemMessage(system), HumanMessage(user)]

    models = args.models or free_models()
    print(f"{len(models)} models x {args.runs} runs, timeout {args.timeout:.0f}s "
          "(models in parallel, each model's runs one after another)...")

    def runs_of(model: str) -> list[dict]:
        # Sequential per model: free providers answer parallel requests to one model with 429.
        out = []
        for r in range(1, args.runs + 1):
            row = run_once(model, messages, plan, excerpts, lesson_type, args.timeout)
            row["run"] = r
            out.append(row)
        return out

    with ThreadPoolExecutor(max_workers=min(16, len(models))) as pool:
        rows = [row for rows_of_model in pool.map(runs_of, models) for row in rows_of_model]

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "run", "seconds", "valid", "prompt_tokens", "completion_tokens", "error"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["model"], r["run"])))

    summary = []
    for m in models:
        mine = [r for r in rows if r["model"] == m]
        valid = [r for r in mine if r["valid"]]
        avg = round(sum(r["seconds"] for r in valid) / len(valid), 1) if valid else None
        summary.append((m, len(valid), len(mine), avg, "; ".join(sorted({r["error"][:60] for r in mine if r["error"]}))))
    summary.sort(key=lambda s: (-s[1], s[3] if s[3] is not None else 1e9))
    print(f"\n{'model':48} {'valid':>6} {'avg s':>7}  errors")
    for m, ok, total, avg, errors in summary:
        print(f"{m:48} {ok:>3}/{total:<2} {avg if avg is not None else '-':>7}  {errors}")
    print(f"\nwritten: {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

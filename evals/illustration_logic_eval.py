"""Acceptance of the illustration logic checker on the five day-9 illustrations.

    .venv\\Scripts\\python -m evals.illustration_logic_eval            # the active profile's checkers
    .venv\\Scripts\\python -m evals.illustration_logic_eval --runs 2

Each page in tests/fixtures/illustrations/ is checked against its slides of
evals/fixtures/short_winter.pptx. tests/fixtures/illustration_logic_expected.json lists, per page,
the categories the checker must find (must) and the ones that would be false alarms (must_not).
Writes evals/results/illustration_logic_<profile>.txt with every finding and the model's table.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGES = ROOT / "tests" / "fixtures" / "illustrations"
EXPECTED = ROOT / "tests" / "fixtures" / "illustration_logic_expected.json"
SOURCE = ROOT / "evals" / "fixtures" / "short_winter.pptx"


def source_chunks() -> list[dict]:
    from agent.mcp_servers.source_reader import read_source_file

    return read_source_file(str(SOURCE))["chunks"]


RETRY_WAITS = (60, 180, 300)  # free models answer 429 in bursts


def checker_call(usage: list):
    from agent.llm import call_llm, reset_unavailable_models

    def call(messages, timeout=None):
        for wait in (*RETRY_WAITS, None):
            try:
                result = call_llm(messages, role="checker", timeout=timeout or 180)
                usage.append(result)
                return result
            except Exception:  # noqa: BLE001 - wait and try the list again
                if wait is None:
                    raise
                time.sleep(wait)
                reset_unavailable_models()  # a timeout skipped a model; give it another chance

    return call


def _caught(must: dict, findings) -> bool:
    """A finding of the category, with one of the keywords in its problem or evidence."""
    words = must.get("keywords") or []
    return any(f.category == must["category"] and (not words or any(w in f.problem + " " + f.evidence for w in words))
               for f in findings)


def evaluate(runs: int = 1, call=None) -> dict:
    """{"pages": {name: [run results]}, "passed", "total", "cost", "seconds"}."""
    from agent.illustration_logic import check_page

    expected = {k: v for k, v in json.loads(EXPECTED.read_text(encoding="utf-8")).items() if not k.startswith("_")}
    chunks = source_chunks()
    usage: list = []
    call = call or checker_call(usage)
    started = time.monotonic()
    out: dict = {"pages": {}, "passed": 0, "total": 0}
    for name, spec in expected.items():
        html = (PAGES / f"{name}.html").read_text(encoding="utf-8")
        excerpts = [c for c in chunks if c["locator"].get("slide") in spec["slides"]]
        for _ in range(runs):
            try:
                findings, table, _ = check_page(html, spec["description"], spec["title"], excerpts, call, 1,
                                                f"illustration {name}")
                error = None
            except Exception as e:  # noqa: BLE001 - an eval records the failure
                findings, table, error = [], None, f"{type(e).__name__}: {e}"
            found = {f.category for f in findings}
            missed = [m["category"] for m in spec["must"] if not _caught(m, findings)]
            false_alarms = [c for c in spec["must_not"] if c in found]
            ok = not missed and not false_alarms and error is None
            out["pages"].setdefault(name, []).append({"ok": ok, "missed": missed, "false_alarms": false_alarms,
                                                      "error": error, "findings": [f.model_dump() for f in findings],
                                                      "table": table})
            out["passed"] += ok
            out["total"] += 1
    out["cost"] = round(sum(r.cost for r in usage), 6)
    out["models"] = sorted({r.model for r in usage})
    out["seconds"] = round(time.monotonic() - started, 1)
    out["findings"] = sum(len(r["findings"]) for runs_ in out["pages"].values() for r in runs_)
    return out


def write_report(result: dict, profile: str) -> Path:
    lines = [f"illustration logic checker, profile {profile}, models {', '.join(result['models'])}",
             f"passed {result['passed']}/{result['total']}, findings {result['findings']}, cost ${result['cost']}, {result['seconds']} s", ""]
    for name, runs in result["pages"].items():
        for i, r in enumerate(runs, 1):
            status = "OK" if r["ok"] else f"FAIL missed={r['missed']} false_alarms={r['false_alarms']} error={r['error']}"
            lines.append(f"== {name} run {i}: {status}")
            for f in r["findings"]:
                lines.append(f"  - {f['category']}: {f['problem']} [{f['evidence'][:120]}]")
            lines.append("  table: " + json.dumps(r["table"], ensure_ascii=False)[:4000])
            lines.append("")
    path = ROOT / "evals" / "results" / f"illustration_logic_{profile}.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    from dotenv import load_dotenv

    from agent.config import configure_stdio
    from agent.llm import active_profile

    configure_stdio()
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()
    result = evaluate(args.runs)
    path = write_report(result, active_profile())
    print(f"passed {result['passed']}/{result['total']}, ${result['cost']}, {result['seconds']} s -> {path}")
    for name, runs in result["pages"].items():
        for r in runs:
            print(name, "OK" if r["ok"] else f"missed={r['missed']} false_alarms={r['false_alarms']} error={r['error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

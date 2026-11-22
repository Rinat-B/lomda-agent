r"""The agent's conversation: how well the gate classifier reads Hebrew replies, and three scripted
conversations end to end.

    .venv\Scripts\python -m evals.conversation_eval            # the active profile (free by default)
    .venv\Scripts\python -m evals.conversation_eval --no-model # only the scripted conversations

1. Intents: every case of evals/fixtures/conversation_intents.json (about 30 Hebrew replies at gates 1-3,
   some vague) goes through the real classifier (nodes.classify_gate_reply, the small model), and is
   compared with the expected intent. Accuracy per intent and overall -> evals/results/conversation_eval.csv
   and conversation_eval_summary.md.
2. Scenarios: tests/test_conversation_scenarios.py (a fake model, no cost): 5 screens with a question
   and going back to the topics; cancel and /resume; a change after saving. Pass or fail per scenario.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "evals" / "fixtures" / "conversation_intents.json"
RESULTS = ROOT / "evals" / "results"
RETRY_WAITS = (60, 180)


def classify_all(cases: dict) -> list[dict]:
    from agent.llm import reset_unavailable_models
    from agent.nodes import classify_gate_reply

    rows = []
    for case in cases["cases"]:
        started, got, error = time.monotonic(), None, None
        for wait in (*RETRY_WAITS, None):
            try:
                decision, change = classify_gate_reply(case["text"], case["gate"], cases["items"][case["gate"]], None)
                got = decision
                break
            except Exception as e:  # noqa: BLE001 - wait and try again, then record
                error = f"{type(e).__name__}: {str(e)[:120]}"
                if wait is None:
                    break
                time.sleep(wait)
                reset_unavailable_models()
        rows.append({"gate": case["gate"], "text": case["text"], "expected": case["expected"], "got": got or "",
                     "correct": got == case["expected"], "error": error if got is None else "",
                     "seconds": round(time.monotonic() - started, 1)})
    return rows


def intent_of(label: str) -> str:
    return label.split(":", 1)[0] if label else label


def accuracy(rows: list[dict]) -> dict:
    per: dict = defaultdict(lambda: [0, 0])
    for r in rows:
        per[intent_of(r["expected"])][1] += 1
        per[intent_of(r["expected"])][0] += r["correct"]
    return {k: (ok, n) for k, (ok, n) in sorted(per.items())}


def scenarios() -> list[tuple[str, bool]]:
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rA", "tests/test_conversation_scenarios.py"],
                            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = []
    for line in result.stdout.splitlines():
        if line.startswith(("PASSED", "FAILED")) and "::" in line:
            out.append((line.split("::", 1)[1].split(" ")[0], line.startswith("PASSED")))
    return out


def write(rows: list[dict], runs: list[tuple[str, bool]], profile: str, models: list[str]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    if rows:
        with (RESULTS / "conversation_eval.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [f"# Conversation eval ({profile})", ""]
    if rows:
        acc = accuracy(rows)
        total_ok = sum(ok for ok, _ in acc.values())
        lines += [f"Gate classifier ({', '.join(models)}): {total_ok}/{len(rows)} correct "
                  f"({round(100 * total_ok / len(rows))}%).", "", "| intent | correct |", "| --- | --- |"]
        lines += [f"| {k} | {ok}/{n} |" for k, (ok, n) in acc.items()]
        wrong = [r for r in rows if not r["correct"]]
        if wrong:
            lines += ["", "Misread:", ""] + [f"- ({r['gate']}) \"{r['text']}\": expected {r['expected']}, got "
                                             f"{r['got'] or 'no answer'} {r['error']}".rstrip() for r in wrong]
    lines += ["", "| scripted conversation (fake model) | result |", "| --- | --- |"]
    lines += [f"| {name} | {'passed' if ok else 'FAILED'} |" for name, ok in runs]
    path = RESULTS / "conversation_eval_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    from dotenv import load_dotenv

    from agent.config import configure_stdio
    from agent.llm import active_profile, load_models

    configure_stdio()
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-model", action="store_true")
    args = parser.parse_args()
    rows = [] if args.no_model else classify_all(json.loads(CASES.read_text(encoding="utf-8")))
    runs = scenarios()
    path = write(rows, runs, active_profile(), load_models("small"))
    print(path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

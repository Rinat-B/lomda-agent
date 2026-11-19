"""Eval 3: staying on task and not leaking.

Each message of evals/fixtures/scope_questions.json goes through the real entry node,
scope_check, with short_winter.pptx in the session (as after /add); the two injection files
go through /add (the CommandHandler, so injection_guard runs as for a user). Both profiles.

    .venv\\Scripts\\python -m evals.scope_eval [--profiles free,balanced] [--tag before]

evals/results/scope_eval.csv: question, category, expected, profile, response, refused,
correct, then per profile refusal_rate (out of scope), false_refusal_rate (controls) and
injection_caught.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "evals" / "fixtures"
QUESTIONS_PATH = FIXTURES / "scope_questions.json"
SOURCE = FIXTURES / "short_winter.pptx"
RESULTS = ROOT / "evals" / "results"
FIELDS = ["question", "category", "expected", "profile", "response", "refused", "correct"]


def quiet(*_a, **_k) -> None:
    return None


def session_with_source():
    from agent.mcp_servers.source_reader import read_source_file
    from agent.session import Session, SourceDoc

    result = read_source_file(str(SOURCE))
    session = Session()
    session.add_file(SOURCE, SourceDoc(result["unit_type"], result["units"], result["chunks"]))
    return session


BACKOFF_SECONDS = (30, 90, 240)  # on 429 (all models rate limited): wait and try again


def ask_scope(question: str, session) -> tuple[str, str]:
    """(intent, response text) from scope_check, as the graph's first node runs it."""
    import time

    from agent.cli import graph_config
    from agent.errors import AllModelsFailedError, NodeError
    from agent.llm import reset_unavailable_models
    from agent.nodes import scope_check

    for wait in (*BACKOFF_SECONDS, None):
        try:
            update = scope_check({"question": question}, graph_config(session))
            return update.get("intent") or "", " ".join(str(update.get("answer") or "").split())[:200]
        except NodeError as e:
            if wait is None or not isinstance(e.cause, AllModelsFailedError):
                raise
            time.sleep(wait)
            reset_unavailable_models()
    raise RuntimeError("unreachable")


def add_with_guard(path: Path) -> int:
    """The guard step of /add (CommandHandler._guard_source, the code a user's /add runs after
    reading the file); returns how many chunks injection_guard flagged."""
    from agent.commands import CommandHandler
    from agent.mcp_servers.source_reader import read_source_file
    from agent.session import Session, SourceDoc

    result = read_source_file(str(path))
    source = SourceDoc(result["unit_type"], result["units"], result["chunks"])
    CommandHandler(Session(), out=quiet, out_markdown=quiet, out_path=quiet)._guard_source(path, source)
    return sum(1 for c in source.chunks if c.get("flagged"))


def run_profile(profile: str, data: dict) -> list[dict]:
    from agent.llm import reset_model_state

    os.environ["PROFILE"] = profile
    reset_model_state()
    session = session_with_source()
    rows = []
    for q in data["questions"]:
        try:
            intent, response = ask_scope(q["text"], session)
        except Exception as e:  # noqa: BLE001 - a failed call is a wrong answer, recorded
            intent, response = "error", f"{type(e).__name__}: {e}"[:200]
        refused = intent == "out_of_scope"
        correct = refused if q["expected"] == "refuse" else intent in ("question", "build_lesson", "needs_files")
        rows.append({"question": q["text"], "category": q["category"], "expected": q["expected"], "profile": profile,
                     "response": f"{intent}: {response}" if response else intent, "refused": refused,
                     "correct": correct})
    for name in data["injection_files"]:
        try:
            flagged = add_with_guard(FIXTURES / name)
            response = f"{flagged} chunk(s) flagged"
        except Exception as e:  # noqa: BLE001
            flagged, response = 0, f"{type(e).__name__}: {e}"[:200]
        rows.append({"question": name, "category": "injection_file", "expected": "flag", "profile": profile,
                     "response": response, "refused": flagged > 0, "correct": flagged > 0})
    return rows


def rates(rows: list[dict], profile: str) -> dict:
    mine = [r for r in rows if r["profile"] == profile]
    out_q = [r for r in mine if r["expected"] == "refuse"]
    ctrl = [r for r in mine if r["expected"] == "allow"]
    inj = [r for r in mine if r["expected"] == "flag"]
    return {
        "refusal_rate": round(sum(r["refused"] for r in out_q) / len(out_q), 3) if out_q else None,
        "false_refusal_rate": round(sum(r["refused"] for r in ctrl) / len(ctrl), 3) if ctrl else None,
        "injection_caught": f"{sum(r['correct'] for r in inj)}/{len(inj)}",
    }


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    from agent.config import configure_stdio

    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", default="free,balanced")
    parser.add_argument("--tag", default="", help="suffix for the CSV name, e.g. before")
    args = parser.parse_args(argv)
    configure_stdio()
    load_dotenv(ROOT / ".env")
    data = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for profile in [p.strip() for p in args.profiles.split(",") if p.strip()]:
        rows += run_profile(profile, data)
    path = RESULTS / f"scope_eval{('_' + args.tag) if args.tag else ''}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
        for profile in dict.fromkeys(r["profile"] for r in rows):
            for name, value in rates(rows, profile).items():
                w.writerow({"question": name, "profile": profile, "response": value})
    summary = {p: rates(rows, p) for p in dict.fromkeys(r["profile"] for r in rows)}
    print(json.dumps(summary))
    for r in rows:
        if not r["correct"]:
            print("WRONG", r["profile"], r["category"], r["response"][:120], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

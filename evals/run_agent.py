"""Run the whole agent graph without a keyboard: source file -> lesson.json.

Gate answers come from a script (one answer per line; default: Y at every gate). The
write_files [Y/N] is always answered N: the run writes its own files (lesson.json, run.json,
report.md) to the output folder instead of output/. Profile and the writer / checker / small
models can be set from outside. Every run is tagged in LangSmith with the experiment name.

    .venv\\Scripts\\python -m evals.run_agent --experiment try1
    .venv\\Scripts\\python -m evals.run_agent --writer anthropic/claude-sonnet-5.5 --out evals/results/runs/x

Returns (and saves in run.json): the final state summary, usage and cost per node and per
model, time per node, open findings and coverage.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "evals" / "fixtures"
DEFAULT_SOURCE = FIXTURES / "short_winter.pptx"
REQUESTS_PATH = FIXTURES / "requests.json"  # the user's request per fixture (Hebrew, so not in .py)
MAX_GATE_VISITS = 20  # a script that never approves must not loop forever


def default_request(source: Path) -> str:
    requests = json.loads(REQUESTS_PATH.read_text(encoding="utf-8"))
    return requests.get(source.stem) or requests["default"]


def read_answers(path: Path | None) -> list[str]:
    if path is None:
        return []
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _scripted(answers: list[str]):
    """ask() for the gates: the next scripted answer, then Y; write_files always N."""
    from agent.messages import msg

    queue = list(answers)
    asked: list[str] = []

    browser_prefix = msg("gate.browser_check.ask", count=0).split("0")[0]

    def ask(prompt: str) -> str:
        if prompt == msg("gate.write_files.ask"):
            return "N"
        if prompt.startswith(browser_prefix):
            return "Y"  # evals run the browser check
        answer = queue.pop(0) if queue else "Y"
        asked.append(answer)
        return answer

    ask.asked = asked  # type: ignore[attr-defined]
    return ask


def load_session(source: Path):
    """A session with the source read and guarded like /add does (injection_guard)."""
    from agent.guards import scan_chunks
    from agent.llm import call_llm
    from agent.mcp_servers.source_reader import read_source_file
    from agent.session import Session, SourceDoc

    result = read_source_file(str(source))
    if not result.get("ok"):
        raise RuntimeError(f"cannot read {source}: {result}")
    session = Session()
    doc = SourceDoc(unit_type=result["unit_type"], units=result["units"], chunks=result["chunks"])

    def small_model(messages):
        r = call_llm(messages, role="small")
        session.record_usage("injection_guard", r)
        return r

    flagged = scan_chunks(doc.chunks, small_model)
    session.add_file(source, doc)
    return session, flagged


def usage_table(records, key: str) -> dict:
    out: dict = defaultdict(lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0})
    for r in records:
        row = out[getattr(r, key)]
        row["calls"] += 1
        row["prompt_tokens"] += r.prompt_tokens
        row["completion_tokens"] += r.completion_tokens
        row["cost_usd"] = round(row["cost_usd"] + r.cost, 6)
    return dict(out)


def configure_models(profile: str | None, writer: list[str] | None, checker: list[str] | None,
                     small: list[str] | None) -> None:
    from agent import llm

    if profile:
        os.environ["PROFILE"] = profile
    llm.reset_model_state()
    llm.set_role_models("writer", writer)
    llm.set_role_models("checker", checker)
    llm.set_role_models("small", small)


def run_agent(source: Path = DEFAULT_SOURCE, request: str | None = None, answers: list[str] | None = None,
              profile: str | None = None, writer: list[str] | None = None, checker: list[str] | None = None,
              small: list[str] | None = None, experiment: str = "manual", out_dir: Path | None = None) -> dict:
    """Run the graph to the end (or an error) and return the run summary."""
    from langgraph.types import Command

    from agent.cli import FRESH_RUN, graph_config
    from agent.evaluation import coverage_summary
    from agent.gates import ask_gate
    from agent.graph import build_graph
    from agent.llm import active_profile, load_models
    from agent.nodes import source_label
    from agent.report import build_report

    configure_models(profile, writer, checker, small)
    started = time.monotonic()
    summary: dict = {"experiment": experiment, "source": source.name, "profile": active_profile(),
                     "models": {r: load_models(r) for r in ("writer", "checker", "small")}, "error": None}
    session, flagged = load_session(source)
    summary["flagged_chunks"] = len(flagged)
    graph = build_graph()
    config = graph_config(session)
    config.update({"run_name": f"eval:{experiment}", "tags": ["eval", experiment],
                   "metadata": {"experiment": experiment, "writer": summary["models"]["writer"][0],
                                "profile": summary["profile"]}})
    session.run_start = 0
    ask = _scripted(answers or [])
    node_seconds: dict[str, float] = defaultdict(float)
    node_runs: dict[str, int] = defaultdict(int)
    starts: dict[str, tuple[str, float]] = {}

    def stream(payload) -> None:
        for task in graph.stream(payload, config, stream_mode="tasks"):
            if "input" in task:
                starts[task["id"]] = (task["name"], time.monotonic())
            elif task["id"] in starts:
                name, t0 = starts.pop(task["id"])
                node_seconds[name] += time.monotonic() - t0
                node_runs[name] += 1

    def quiet(*_a, **_k) -> None:
        return None

    try:
        stream({**FRESH_RUN, "question": request or default_request(source)})
        for _ in range(MAX_GATE_VISITS):
            snapshot = graph.get_state(config)
            if not snapshot.interrupts:
                break
            resume = ask_gate(snapshot.interrupts[0].value, ask, quiet, quiet, preview=None)
            stream(Command(resume=resume))
        else:
            summary["error"] = f"stopped after {MAX_GATE_VISITS} gate visits"
    except Exception as e:  # noqa: BLE001 - an eval records the failure and goes on
        summary["error"] = f"{type(e).__name__}: {e}"

    values = graph.get_state(config).values
    lesson = values.get("lesson")
    findings = values.get("findings") or []
    records = list(session.usage)
    summary.update({
        "duration_sec": round(time.monotonic() - started, 1),
        "gate_answers": ask.asked,  # type: ignore[attr-defined]
        "screens": len(lesson["screens"]) if lesson else 0,
        "lesson_title": lesson["title"] if lesson else None,
        "review_round": values.get("review_round") or 0,
        "findings_total": len(findings),
        "findings_fixed": sum(f["status"] == "fixed" for f in findings),
        "open_findings": [f for f in findings if f["status"] == "open"],
        "coverage": coverage_summary(values.get("key_points"), lesson) if lesson else None,
        "review": values.get("review"),
        "writer_models": values.get("writer_models"),
        "calls": len(records),
        "cost_usd": round(sum(r.cost for r in records), 6),
        "usage_by_node": usage_table(records, "node"),
        "usage_by_model": usage_table(records, "model"),
        "seconds_by_node": {k: round(v, 1) for k, v in node_seconds.items()},
        "runs_by_node": dict(node_runs),
        "usage_records": [vars(r) for r in records],
    })
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        if lesson:
            (out_dir / "lesson.json").write_text(json.dumps(lesson, ensure_ascii=False, indent=1), encoding="utf-8")
            report = build_report(dict(values), records, source_label, round(summary["duration_sec"] / 60))
            (out_dir / "report.md").write_text(report, encoding="utf-8")
        (out_dir / "run.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    summary["lesson"] = lesson
    summary["key_points"] = values.get("key_points")
    return summary


def _models(value: str | None) -> list[str] | None:
    return [m.strip() for m in value.split(",") if m.strip()] if value else None


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    from agent.config import configure_stdio
    from agent.errors import setup_logging
    from agent.tracing import flush_traces, setup_tracing

    parser = argparse.ArgumentParser(description="Run the lesson agent without a keyboard")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--request", help="the user's request (default: evals/fixtures/requests.json)")
    parser.add_argument("--answers", type=Path, help="gate answers, one per line (default: Y at every gate)")
    parser.add_argument("--profile", choices=("free", "balanced"))
    parser.add_argument("--writer", help="writer model id(s), comma separated")
    parser.add_argument("--checker", help="checker model id(s), comma separated")
    parser.add_argument("--small", help="small model id(s), comma separated")
    parser.add_argument("--experiment", default="manual")
    parser.add_argument("--out", type=Path, help="folder for lesson.json, run.json and report.md")
    args = parser.parse_args(argv)

    configure_stdio()
    load_dotenv(ROOT / ".env")
    setup_logging()
    setup_tracing()
    out = args.out or ROOT / "evals" / "results" / "runs" / args.experiment
    summary = run_agent(args.source, args.request, read_answers(args.answers), args.profile,
                        _models(args.writer), _models(args.checker), _models(args.small), args.experiment, out)
    flush_traces()
    keys = ("experiment", "error", "duration_sec", "screens", "calls", "cost_usd", "review_round",
            "findings_total", "findings_fixed")
    print(json.dumps({k: summary[k] for k in keys} | {"open_findings": len(summary["open_findings"]),
                                                       "coverage": summary["coverage"] and summary["coverage"]["percent"]},
                     ensure_ascii=False))
    return 0 if not summary["error"] else 1


if __name__ == "__main__":
    sys.exit(main())

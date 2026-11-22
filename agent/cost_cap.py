"""The run's spending cap (MAX_COST_USD): a [Y/N] gate before a paid node that would cross it.

The cap comes from MAX_COST_USD in .env, else `max_cost_usd` in models.yaml (1.5). Before a
paid node starts, its calls are estimated (how many calls x the mean cost of a call so far in
this run). If what was spent plus that estimate crosses the cap, the graph stops at a gate
that shows the cost so far and an estimate for the rest of the run:

- Y raises the cap for this run only (to the cost so far plus the estimate for the rest).
- N with a whole lesson written: an orderly finish. The remaining fix rounds and paid steps
  are skipped, the lesson is kept as it is, gate 3 and write_files follow, and report.md says
  that the run stopped at the cap and lists the open findings.
- N before the lesson is written: the conversation stays paused at this gate (the checkpoint
  is saved after every node), and /resume asks again.

The check runs at the start of the node, before any call, so a node that resumes after the
gate repeats no paid call.
"""

import math
import os

from agent.messages import msg

DEFAULT_MAX_COST_USD = 1.5
COST_GATE = "cost_cap"
# Paid nodes that can still produce a whole lesson after an N (the orderly finish).
LESSON_READY_NODES = ("build_illustrations", "evaluator", "fix_screens")


def default_cap() -> float:
    """MAX_COST_USD from the environment (.env) wins over models.yaml."""
    raw = (os.getenv("MAX_COST_USD") or "").strip()
    if raw:
        try:
            value = float(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    from agent.llm import load_config

    return float(load_config().get("max_cost_usd") or DEFAULT_MAX_COST_USD)


def run_cap(state: dict) -> float:
    return float(state.get("cost_cap") or default_cap())


def per_call_cost(records: list, node: str) -> float:
    """Mean cost of one call: this node's earlier calls in the run if any, else all calls."""
    same = [r for r in records if r.node == node]
    pool = same or list(records)
    return sum(r.cost for r in pool) / len(pool) if pool else 0.0


def estimate(records: list, node: str, node_calls: int, run_calls_left: int) -> tuple[float, float]:
    """(this node's estimate, the estimate for the rest of the run) in USD."""
    each = per_call_cost(records, node)
    return each * node_calls, each * max(node_calls, run_calls_left)


def crosses(spent: float, node_estimate: float, cap: float) -> bool:
    return node_estimate > 0 and spent + node_estimate > cap


def raised_cap(spent: float, rest: float, cap: float) -> float:
    """The cap after Y: the cost so far plus the estimate for the rest, rounded up to a cent."""
    return max(cap, math.ceil((spent + rest) * 100) / 100 + 0.01)


def gate_request(node: str, spent: float, node_estimate: float, rest: float, cap: float, lesson_ready: bool) -> dict:
    return {"gate": COST_GATE, "node": node, "spent": round(spent, 4), "estimate": round(node_estimate, 4),
            "rest": round(rest, 4), "cap": cap, "lesson_ready": lesson_ready}


def stop_note(request: dict) -> dict:
    """What the state keeps after N (report.md and gate 3 show it)."""
    return {"node": request["node"], "spent": request["spent"], "cap": request["cap"]}


def stop_line(stop: dict) -> str:
    return msg("cost_cap.stopped", node=stop["node"], spent=stop["spent"], cap=stop["cap"])

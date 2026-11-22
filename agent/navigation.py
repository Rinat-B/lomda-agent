"""Moving around the lesson flow from a gate: status, going back to an earlier gate, cancelling.

Going back (time travel): the owner, at gate 2 or 3, asks to return to the topics or the structure
("back to the topics", "wait, remove the third topic"). The gate node only proposes it: it says what
will run again and about how many calls, and the CLI asks [Y/N]. On Y, go_back() finds in the
thread's checkpoint history (graph.get_state_history) the last time the graph stood at that gate,
and forks from there with graph.update_state:
  - with a change ("remove the third topic"): as if the owner had answered that gate with the change,
    so the producing node runs again with it (topic_map or plan_structure);
  - without a change: as if the producing node had just finished, so the gate is shown again.
Everything decided before that gate stays as it was (lesson type, research and web facts, the
owner's requirements); everything after it (the map or structure, writing, illustrations, review) is
not in that older state and is made again by the normal flow. No edge skips a gate.
"""

from agent.messages import msg

# gate -> (gate node, the node that produces what the gate shows)
GATES = {"topics": ("gate_topics", "topic_map"), "structure": ("gate_structure", "plan_structure")}
# what runs again after going back to a gate (in order), for the estimate shown before [Y/N]
RERUN = {
    "topics": ["topic_map", "plan_structure", "write_lesson", "build_illustrations", "browser_check",
               "evaluator", "fix_screens"],
    "structure": ["plan_structure", "write_lesson", "build_illustrations", "browser_check", "evaluator",
                  "fix_screens"],
}
# the usage-record node names that belong to each re-run step (motif and key points run inside them)
_COUNTED = {"plan_structure": ("plan_structure", "motif"), "evaluator": ("evaluator", "key_points")}
# the gates one can go back to from each gate
ALLOWED = {"topics": (), "structure": ("topics",), "review": ("topics", "structure")}
NOT_RERUN = ("classify_type", "research")  # the type and the web facts stay


class GoBack(Exception):
    """The owner confirmed going back to an earlier gate; the CLI does the time travel."""

    def __init__(self, target: str, change: str = ""):
        super().__init__(target)
        self.target = target
        self.change = change


def rerun_estimate(target: str, records: list) -> tuple[list[str], int]:
    """(the steps that run again, about how many model calls: as many as they took in this run)."""
    steps = RERUN[target]
    names = {n for step in steps for n in _COUNTED.get(step, (step,))}
    return steps, sum(1 for r in records if r.node in names)


def back_proposal(target: str, change: str, records: list) -> dict:
    steps, calls = rerun_estimate(target, records)
    return {"kind": "back", "target": target, "change": change, "steps": steps, "calls": calls}


def describe_proposal(p: dict) -> str:
    steps = ", ".join(msg(f"report.nodes.{s}") for s in p["steps"])
    kept = ", ".join(msg(f"report.nodes.{s}") for s in NOT_RERUN)
    lines = [msg(f"gate.back.title.{p['target']}")]
    if p.get("change"):
        lines.append(msg("gate.back.change", change=p["change"]))
    lines += [msg("gate.back.rerun", steps=steps), msg("gate.back.kept", kept=kept),
              msg("gate.back.calls", calls=p["calls"])]
    return "\n".join(lines)


def find_gate_snapshot(graph, config: dict, gate_node: str):
    """The latest checkpoint (in this branch of the thread's history) where the graph waited at gate_node."""
    for snapshot in graph.get_state_history(config):
        if tuple(snapshot.next) == (gate_node,):
            return snapshot
    return None


def go_back(graph, config: dict, target: str, change: str = "") -> dict:
    """Fork the thread from the last visit to the target gate (see the module doc). Returns the new config."""
    gate_node, producer = GATES[target]
    snapshot = find_gate_snapshot(graph, config, gate_node)
    if snapshot is None:
        raise ValueError(f"no earlier visit to {gate_node} in this conversation")
    if change:
        forked = graph.update_state(snapshot.config, {"feedback": change, "approved": False, "gate_answer": None,
                                                      "pending": None}, as_node=gate_node)
    else:
        forked = graph.update_state(snapshot.config, {"gate_answer": None, "pending": None}, as_node=producer)
    # the run's own settings (the session's index, the usage recorder) with the new checkpoint
    return {**config, "configurable": {**config.get("configurable", {}), **forked["configurable"]}}


# ---- status ---------------------------------------------------------------------------------------
STAGES = ("type", "topics", "structure", "writing", "review", "saving")


def status_text(state: dict, gate: str, records: list) -> str:
    """Where the run is, what is done and what is left, calls and cost so far (code, no model)."""
    stage = {"type": "type", "topics": "topics", "structure": "structure", "review": "review"}.get(gate, "writing")
    done: list[str] = []
    if state.get("lesson_type"):
        done.append(msg("status.done.type", type=msg(f"lesson_types.{state['lesson_type']}")))
    if state.get("research"):
        facts = state.get("web_facts") or []
        done.append(msg("status.done.research", searches=state["research"].get("searches", 0),
                        verified=sum(f.get("status") == "verified" for f in facts)))
    if state.get("topic_map") and stage != "topics":
        done.append(msg("status.done.topics", count=len(state["topic_map"])))
    if state.get("structure") and stage not in ("topics", "structure"):
        done.append(msg("status.done.structure", count=len(state["structure"]["screens"])))
    if state.get("lesson"):
        findings = state.get("findings") or []
        done.append(msg("status.done.review", rounds=state.get("review_round") or 0, found=len(findings),
                        fixed=sum(f["status"] == "fixed" for f in findings),
                        open=sum(f["status"] == "open" for f in findings)))
    index = STAGES.index(stage)
    left = [msg(f"status.stage.{s}") for s in STAGES[index + 1:]]
    lines = [msg("status.where", stage=msg(f"status.stage.{stage}"), number=index + 1, total=len(STAGES))]
    lines += [msg("status.done_title")] + [f"- {d}" for d in done or [msg("status.nothing_yet")]]
    lines += [msg("status.left", stages=", ".join(left) or "-"),
              msg("status.cost", calls=len(records), cost=sum(r.cost for r in records))]
    return "\n".join(lines)

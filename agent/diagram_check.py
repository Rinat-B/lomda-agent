"""The decision-diagram checker, both ways against the source (SPEC: "every word from the source, and
every action in the source is a leaf").

One model call for all the lesson's diagrams, with the excerpts their screens cite. The model fills
a table and the code decides (like the single-answer and illustration logic checkers):

- source -> diagram: each action the source gives for the diagram's decision, and the leaf (action
  node) that holds it, or "none": an action of the source missing from the diagram is a finding;
- diagram -> source: each node's label, and whether its condition or action and its words come
  from the source (a paraphrase that keeps the meaning is fine; a new condition, threshold or
  action is not): a node not from the source is a finding;
- a leaf that is a question, or a question node that is a leaf, is left to the code check
  (check_diagrams), which already tests the structure.

Findings use the category diagram_logic, check "diagram_source", and enter the fix loop.
"""

from agent.checkers import Finding, _json_object, cited_chunks, render_source
from agent.guards import SOURCE_DATA_RULE
from agent.messages import msg

CHECK = "diagram_source"

PROMPT = (
    "You check the decision diagrams of a Hebrew e-learning lesson against the source excerpts, in "
    "both directions. A diagram has question nodes and action nodes (the leaves) joined by edges.\n"
    "For EACH diagram fill:\n"
    "1. source_actions: every action the excerpts give for the decision this diagram teaches (the "
    "'then' part of each if-then rule on that decision), with a short exact quote, and leaf: the id "
    "of the action node that says it, or \"none\" if no leaf says it. Only actions of THIS decision, "
    "not other rules in the excerpts.\n"
    "2. nodes: every node with from_source: true when its condition or action, and its key words, "
    "come from the excerpts (a paraphrase that keeps the meaning is fine); false when it adds a "
    "condition, a threshold, an action or a fact the excerpts do not give. reason: why, in Hebrew.\n"
    "Reply with only this JSON object:\n"
    '{"diagrams": [{"screen": 1, "source_actions": [{"action": "...", "quote": "...", "leaf": "a1"}], '
    '"nodes": [{"id": "q1", "from_source": true, "reason": "..."}]}]}\n\n'
    + SOURCE_DATA_RULE
)


def diagrams(lesson: dict) -> list[tuple[int, int, dict]]:
    """(screen number, block number, block) for every decision diagram."""
    return [(s_no, b_no, b) for s_no, s in enumerate(lesson["screens"], 1)
            for b_no, b in enumerate(s["blocks"], 1) if b["type"] == "decision_diagram"]


def messages(lesson: dict, chunks: list[dict]):
    from langchain_core.messages import HumanMessage, SystemMessage

    found = diagrams(lesson)
    screens = {"screens": [lesson["screens"][s_no - 1] for s_no, _, _ in found]}
    parts = []
    for s_no, _, b in found:
        parts.append(f"Diagram on screen {s_no}: {b['title']}")
        parts += [f"  node {n['id']} ({n['kind']}): {n['label']}" for n in b["nodes"]]
        parts += [f"  edge {e['from']} -> {e['to']}" + (f" [{e['label']}]" if e.get("label") else "") for e in b["edges"]]
    user = f"Source excerpts:\n{render_source(cited_chunks(screens, chunks))}\n\nThe diagrams:\n" + "\n".join(parts)
    return [SystemMessage(PROMPT), HumanMessage(user)]


def _false(value) -> bool:
    return value is False or (isinstance(value, str) and value.strip().lower() in ("false", "no"))


def parse(text: str, lesson: dict) -> list[Finding]:
    data = _json_object(text, "diagrams")
    by_screen = {s_no: (b_no, b) for s_no, b_no, b in diagrams(lesson)}
    out: list[Finding] = []
    for d in data.get("diagrams") or []:
        if not isinstance(d, dict):
            continue
        try:
            screen = int(d.get("screen"))
        except (TypeError, ValueError):
            continue
        if screen not in by_screen:
            continue
        b_no, block = by_screen[screen]
        where = f"block {b_no} (decision_diagram)"
        labels = {n["id"]: n["label"] for n in block["nodes"]}
        leaves = {n["id"] for n in block["nodes"] if n["kind"] == "action"}
        for a in d.get("source_actions") or []:
            if isinstance(a, dict) and str(a.get("leaf") or "none").strip() not in leaves:
                out.append(Finding(check=CHECK, category="diagram_logic", screen=screen, where=where,
                                   evidence=str(a.get("quote") or a.get("action") or "")[:300],
                                   problem=msg("diagram_check.missing_leaf", action=a.get("action", ""))))
        for n in d.get("nodes") or []:
            if isinstance(n, dict) and _false(n.get("from_source")) and n.get("id") in labels:
                out.append(Finding(check=CHECK, category="diagram_logic", screen=screen, where=where,
                                   evidence=labels[n["id"]],
                                   problem=msg("diagram_check.not_in_source", label=labels[n["id"]],
                                               reason=n.get("reason", ""))))
    return out

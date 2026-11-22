"""LangGraph wiring.

    START -> scope_check -+-> answer -> END                                  (question)
                          +-> END                                            (out of scope / no files)
                          +-> classify_type -> gate_type -> [research, training only] -> topic_map -> gate_topics
                                                                ^           |  ^  |
                                                     change ----+           |  +--+ question
                                                                            v (approved)
                                                    gate_structure <- plan_structure
                                                     |  ^  |              ^
                                        question ----+--+  +-- change ----+
                                                     v (approved)
                               write_lesson (one group of 3-4 screens per run; loops until done)
                                                     v
                               build_illustrations -> browser_check [Y/N] (Code-Exec)
                                                     v
                         +--------------------> assemble_lesson -> evaluator (all checkers / verify fixes)
                         |                                            |  open findings, round < 2
                    fix_screens <-------------------------------------+
                         ^                                            v (no findings, or 2 rounds done)
                         +------------ change (owner) --------- gate_review (gate 3, preview) --+ question
                                                                      v (approved)           <--+
                                                         write_files [Y/N] (+ report.md) -> END

Each producing node continues only to its gate: no edge skips a gate.

The paid nodes are wrapped by cost_guarded: before one starts, a [Y/N] gate if its estimated
calls would cross the run's MAX_COST_USD (agent/cost_cap.py).
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    AgentState,
    answer,
    assemble_lesson_node,
    browser_check,
    build_illustrations,
    clarify,
    classify_type,
    cost_guarded,
    evaluator,
    fix_screens,
    gate_review,
    gate_structure,
    gate_topics,
    gate_type,
    plan_structure,
    research,
    route_after_gate_structure,
    route_after_evaluator,
    route_after_gate_review,
    route_after_gate_topics,
    route_after_clarify,
    route_after_gate_type,
    route_after_scope_check,
    route_after_write,
    scope_check,
    topic_map,
    write_files,
    write_lesson,
)


def build_graph(checkpointer=None):
    """Compile the graph. A checkpointer is required for interrupt(); defaults to in-memory.
    Each conversation uses its own thread_id (see Session.thread_id)."""
    graph = StateGraph(AgentState)
    graph.add_node("scope_check", scope_check)
    graph.add_node("answer", answer)
    graph.add_node("classify_type", classify_type)
    graph.add_node("gate_type", gate_type)
    graph.add_node("clarify", clarify)
    graph.add_node("research", cost_guarded("research", research))
    graph.add_node("topic_map", cost_guarded("topic_map", topic_map))
    graph.add_node("gate_topics", gate_topics)
    graph.add_node("plan_structure", cost_guarded("plan_structure", plan_structure))
    graph.add_node("gate_structure", gate_structure)
    graph.add_node("write_lesson", cost_guarded("write_lesson", write_lesson))
    graph.add_node("build_illustrations", cost_guarded("build_illustrations", build_illustrations))
    graph.add_node("browser_check", browser_check)
    graph.add_node("assemble_lesson", assemble_lesson_node)
    graph.add_node("evaluator", cost_guarded("evaluator", evaluator))
    graph.add_node("fix_screens", cost_guarded("fix_screens", fix_screens))
    graph.add_node("gate_review", gate_review)
    graph.add_node("write_files", write_files)

    graph.add_edge(START, "scope_check")
    graph.add_conditional_edges(
        "scope_check", route_after_scope_check,
        {"answer": "answer", "classify_type": "classify_type", "end": END},
    )
    graph.add_edge("answer", END)
    graph.add_edge("classify_type", "gate_type")
    graph.add_conditional_edges("gate_type", route_after_gate_type, {"gate_type": "gate_type", "clarify": "clarify"})
    graph.add_conditional_edges("clarify", route_after_clarify, {"research": "research", "topic_map": "topic_map"})
    graph.add_edge("research", "topic_map")
    graph.add_edge("topic_map", "gate_topics")
    graph.add_conditional_edges(
        "gate_topics", route_after_gate_topics,
        {"topic_map": "topic_map", "gate_topics": "gate_topics", "plan_structure": "plan_structure"},
    )
    graph.add_edge("plan_structure", "gate_structure")
    graph.add_conditional_edges(
        "gate_structure", route_after_gate_structure,
        {"plan_structure": "plan_structure", "gate_structure": "gate_structure", "write_lesson": "write_lesson"},
    )
    graph.add_conditional_edges(
        "write_lesson", route_after_write,
        {"write_lesson": "write_lesson", "build_illustrations": "build_illustrations"},
    )
    graph.add_edge("build_illustrations", "browser_check")
    graph.add_edge("browser_check", "assemble_lesson")
    graph.add_edge("assemble_lesson", "evaluator")
    graph.add_conditional_edges(
        "evaluator", route_after_evaluator, {"fix_screens": "fix_screens", "gate_review": "gate_review"},
    )
    graph.add_edge("fix_screens", "assemble_lesson")
    graph.add_conditional_edges(
        "gate_review", route_after_gate_review,
        {"gate_review": "gate_review", "fix_screens": "fix_screens", "write_files": "write_files"},
    )
    graph.add_edge("write_files", END)
    return graph.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())

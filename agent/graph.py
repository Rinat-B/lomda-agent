"""LangGraph wiring.

    START -> scope_check -+-> answer -> END                                  (question)
                          +-> END                                            (out of scope / no files)
                          +-> classify_type -> gate_type -> topic_map -> gate_topics
                                                                ^           |  ^  |
                                                     change ----+           |  +--+ question
                                                                            v (approved)
                                              END <- gate_structure <- plan_structure
                                                       |  ^  |              ^
                                          question ----+--+  +-- change ----+

Each producing node continues only to its gate: no edge skips a gate.
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    AgentState,
    answer,
    classify_type,
    gate_structure,
    gate_topics,
    gate_type,
    plan_structure,
    route_after_gate_structure,
    route_after_gate_topics,
    route_after_scope_check,
    scope_check,
    topic_map,
)


def build_graph(checkpointer=None):
    """Compile the graph. A checkpointer is required for interrupt(); defaults to in-memory.
    Each conversation uses its own thread_id (see Session.thread_id)."""
    graph = StateGraph(AgentState)
    graph.add_node("scope_check", scope_check)
    graph.add_node("answer", answer)
    graph.add_node("classify_type", classify_type)
    graph.add_node("gate_type", gate_type)
    graph.add_node("topic_map", topic_map)
    graph.add_node("gate_topics", gate_topics)
    graph.add_node("plan_structure", plan_structure)
    graph.add_node("gate_structure", gate_structure)

    graph.add_edge(START, "scope_check")
    graph.add_conditional_edges(
        "scope_check", route_after_scope_check,
        {"answer": "answer", "classify_type": "classify_type", "end": END},
    )
    graph.add_edge("answer", END)
    graph.add_edge("classify_type", "gate_type")
    graph.add_edge("gate_type", "topic_map")
    graph.add_edge("topic_map", "gate_topics")
    graph.add_conditional_edges(
        "gate_topics", route_after_gate_topics,
        {"topic_map": "topic_map", "gate_topics": "gate_topics", "plan_structure": "plan_structure"},
    )
    graph.add_edge("plan_structure", "gate_structure")
    graph.add_conditional_edges(
        "gate_structure", route_after_gate_structure,
        {"plan_structure": "plan_structure", "gate_structure": "gate_structure", "end": END},
    )
    return graph.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())

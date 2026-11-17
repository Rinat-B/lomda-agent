"""LangGraph wiring.

    START -> scope_check -+-> answer ----------------------------> END   (question)
                          +-> topic_map -> gate_topics -+-> END          (approved)
                          |       ^                     |
                          |       +---- change request -+
                          +-> END                         (out of scope / no files)

topic_map can only continue to gate_topics: there is no edge around the gate.
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    AgentState,
    answer,
    gate_topics,
    route_after_gate,
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
    graph.add_node("topic_map", topic_map)
    graph.add_node("gate_topics", gate_topics)

    graph.add_edge(START, "scope_check")
    graph.add_conditional_edges(
        "scope_check", route_after_scope_check, {"answer": "answer", "topic_map": "topic_map", "end": END}
    )
    graph.add_edge("answer", END)
    graph.add_edge("topic_map", "gate_topics")
    graph.add_conditional_edges("gate_topics", route_after_gate, {"topic_map": "topic_map", "end": END})
    return graph.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())

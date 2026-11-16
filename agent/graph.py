"""LangGraph wiring: scope_check -> (answer | END)."""

from langgraph.graph import END, START, StateGraph

from agent.nodes import AgentState, answer, route_after_scope_check, scope_check


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("scope_check", scope_check)
    graph.add_node("answer", answer)
    graph.add_edge(START, "scope_check")
    graph.add_conditional_edges("scope_check", route_after_scope_check, {"answer": "answer", "end": END})
    graph.add_edge("answer", END)
    return graph.compile()

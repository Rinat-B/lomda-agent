"""Graph state and node logic."""

from typing import Callable, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from agent.console import log
from agent.errors import NodeError
from agent.llm import LLMResult, call_llm
from agent.messages import msg

SCOPE_CHECK_PROMPT = (
    "You are a gatekeeper for an assistant that helps build e-learning lessons. "
    "Decide whether the user's question is related to creating e-learning lessons, "
    "instructional design, teaching, or learning in general. "
    "Reply with exactly one word: IN_SCOPE or OUT_OF_SCOPE. Do not explain."
)

ANSWER_PROMPT = (
    "You are a helpful assistant for building e-learning lessons and instructional design. "
    "Always answer in Hebrew, clearly and concisely."
)

# Called as recorder(node_name, result) after every successful model call.
UsageRecorder = Callable[[str, LLMResult], None]


class AgentState(TypedDict, total=False):
    question: str
    in_scope: bool
    answer: str


def _enter(name: str) -> None:
    log(f"[INFO] Entering Node: {name}")


def _call(node: str, messages: list, config: RunnableConfig | None) -> LLMResult:
    """Call the model for a node: tag errors with the node name and record usage."""
    try:
        result = call_llm(messages)
    except Exception as e:
        raise NodeError(node, e) from e
    recorder: UsageRecorder | None = ((config or {}).get("configurable") or {}).get("usage_recorder")
    if recorder:
        recorder(node, result)
    return result


def scope_check(state: AgentState, config: RunnableConfig) -> dict:
    _enter("scope_check")
    result = _call("scope_check", [SystemMessage(SCOPE_CHECK_PROMPT), HumanMessage(state["question"])], config)
    # Check OUT_OF_SCOPE first because "IN_SCOPE" is a substring of it.
    # If the reply is unclear, allow the question rather than refuse a valid one.
    in_scope = "OUT_OF_SCOPE" not in result.text.upper()
    update: dict = {"in_scope": in_scope}
    if not in_scope:
        update["answer"] = msg("graph.out_of_scope")
    return update


def answer(state: AgentState, config: RunnableConfig) -> dict:
    _enter("answer")
    result = _call("answer", [SystemMessage(ANSWER_PROMPT), HumanMessage(state["question"])], config)
    return {"answer": result.text}


def route_after_scope_check(state: AgentState) -> str:
    return "answer" if state["in_scope"] else "end"

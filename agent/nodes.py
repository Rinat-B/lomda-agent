"""Graph state and node logic."""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage

from agent.console import log
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


class AgentState(TypedDict, total=False):
    question: str
    in_scope: bool
    answer: str
    models_used: Annotated[list[str], operator.add]
    prompt_tokens: Annotated[int, operator.add]
    completion_tokens: Annotated[int, operator.add]
    cost: Annotated[float, operator.add]


def _usage(result: LLMResult) -> dict:
    return {
        "models_used": [result.model],
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "cost": result.cost,
    }


def _enter(name: str) -> None:
    log(f"[INFO] Entering Node: {name}")


def scope_check(state: AgentState) -> dict:
    _enter("scope_check")
    result = call_llm([SystemMessage(SCOPE_CHECK_PROMPT), HumanMessage(state["question"])])
    # Check OUT_OF_SCOPE first because "IN_SCOPE" is a substring of it.
    # If the reply is unclear, allow the question rather than refuse a valid one.
    in_scope = "OUT_OF_SCOPE" not in result.text.upper()
    update = {"in_scope": in_scope, **_usage(result)}
    if not in_scope:
        update["answer"] = msg("graph.out_of_scope")
    return update


def answer(state: AgentState) -> dict:
    _enter("answer")
    result = call_llm([SystemMessage(ANSWER_PROMPT), HumanMessage(state["question"])])
    return {"answer": result.text, **_usage(result)}


def route_after_scope_check(state: AgentState) -> str:
    return "answer" if state["in_scope"] else "end"

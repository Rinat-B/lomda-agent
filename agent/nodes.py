"""Graph state and node logic."""

import re
from typing import Callable, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from agent.console import log
from agent.errors import NodeError
from agent.llm import LLMResult, call_llm
from agent.messages import msg
from agent.tools import ChunkIndex, make_retrieve_tool

RETRIEVE_K = 5
FALLBACK_SOURCES = 2  # if the model cites nothing, show the top retrieved chunks
NOT_FOUND = "NOT_FOUND"

SCOPE_CHECK_PROMPT = (
    "You are a gatekeeper for an assistant that helps build e-learning lessons. "
    "Decide whether the user's question is related to creating e-learning lessons, "
    "instructional design, teaching, or learning in general. "
    "Reply with exactly one word: IN_SCOPE or OUT_OF_SCOPE. Do not explain."
)

SCOPE_CHECK_WITH_SOURCES = (
    " The user has added source files for a lesson (for example training slides, "
    "procedures or policies). Any question about the content of such material is IN_SCOPE."
)

ANSWER_PROMPT = (
    "You are a helpful assistant for building e-learning lessons and instructional design. "
    "Always answer in Hebrew, clearly and concisely."
)

RAG_PROMPT = (
    "You answer questions about the user's lesson source files, using ONLY the numbered "
    "excerpts provided. Answer in Hebrew, clearly and concisely. After each fact you use, "
    "cite its excerpt number in square brackets, for example [2]. Do not use outside "
    "knowledge and do not guess. If the excerpts do not contain the answer, reply with "
    f"exactly {NOT_FOUND} and nothing else."
)

# Called as recorder(node_name, result) after every successful model call.
UsageRecorder = Callable[[str, LLMResult], None]

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


class AgentState(TypedDict, total=False):
    question: str
    in_scope: bool
    answer: str
    sources: list[dict]  # locators of the chunks the answer is based on


def _enter(name: str) -> None:
    log(f"[INFO] Entering Node: {name}")


def _configurable(config: RunnableConfig | None) -> dict:
    return (config or {}).get("configurable") or {}


def _index(config: RunnableConfig | None) -> ChunkIndex | None:
    index = _configurable(config).get("index")
    return index if index is not None and len(index) else None


def _call(node: str, messages: list, config: RunnableConfig | None) -> LLMResult:
    """Call the model for a node: tag errors with the node name and record usage."""
    try:
        result = call_llm(messages)
    except Exception as e:
        raise NodeError(node, e) from e
    recorder: UsageRecorder | None = _configurable(config).get("usage_recorder")
    if recorder:
        recorder(node, result)
    return result


def scope_check(state: AgentState, config: RunnableConfig) -> dict:
    _enter("scope_check")
    index = _index(config)
    if index is not None and index.retrieve(state["question"], k=1):
        # The question matches the added source files: in scope, no model call needed.
        return {"in_scope": True}
    prompt = SCOPE_CHECK_PROMPT + (SCOPE_CHECK_WITH_SOURCES if index is not None else "")
    result = _call("scope_check", [SystemMessage(prompt), HumanMessage(state["question"])], config)
    # Check OUT_OF_SCOPE first because "IN_SCOPE" is a substring of it.
    # If the reply is unclear, allow the question rather than refuse a valid one.
    in_scope = "OUT_OF_SCOPE" not in result.text.upper()
    update: dict = {"in_scope": in_scope}
    if not in_scope:
        update["answer"] = msg("graph.out_of_scope")
    return update


# ---- RAG helpers -----------------------------------------------------------------------
def describe_locator(locator: dict) -> str:
    """English locator for the model prompt, e.g. "winter_driving.pptx, slide 4"."""
    parts = [locator["file"]]
    for key in ("slide", "page"):
        if key in locator:
            parts.append(f"{key} {locator[key]}")
    parts.append(f"paragraph {locator['paragraph']}")
    return ", ".join(parts)


def format_excerpts(hits: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] ({describe_locator(h['locator'])}, {h['kind']})\n{h['text']}" for i, h in enumerate(hits, 1)
    )


def source_label(locator: dict) -> str:
    """Short Hebrew source label from messages_he.json, e.g. "(winter_driving.pptx, <slide> 4)"."""
    for key in ("slide", "page"):
        if key in locator:
            return msg(f"rag.source.{key}", file=locator["file"], number=locator[key])
    return msg("rag.source.paragraph", file=locator["file"], number=locator["paragraph"])


def cited_hits(text: str, hits: list[dict]) -> list[dict]:
    """Hits cited as [n] in the answer, in order of first citation, without duplicates."""
    seen: list[int] = []
    for group in _CITATION.findall(text):
        for n in (int(x) for x in group.split(",")):
            if 1 <= n <= len(hits) and n not in seen:
                seen.append(n)
    return [hits[n - 1] for n in seen]


def strip_citations(text: str) -> str:
    return re.sub(r"[ \t]*" + _CITATION.pattern, "", text).strip()


def sources_line(hits: list[dict]) -> str:
    labels: list[str] = []
    for h in hits:
        label = source_label(h["locator"])
        if label not in labels:  # several chunks of one slide/page -> one source
            labels.append(label)
    return msg("rag.sources", sources=", ".join(labels))


def answer(state: AgentState, config: RunnableConfig) -> dict:
    _enter("answer")
    index = _index(config)
    if index is None:
        result = _call("answer", [SystemMessage(ANSWER_PROMPT), HumanMessage(state["question"])], config)
        return {"answer": result.text, "sources": []}

    hits = make_retrieve_tool(index).invoke({"query": state["question"], "k": RETRIEVE_K})
    log(f"[INFO] Retrieved {len(hits)} chunks")
    if not hits:
        return {"answer": msg("rag.not_found"), "sources": []}

    prompt = f"Excerpts:\n\n{format_excerpts(hits)}\n\nQuestion: {state['question']}"
    result = _call("answer", [SystemMessage(RAG_PROMPT), HumanMessage(prompt)], config)
    if NOT_FOUND in result.text.upper():
        return {"answer": msg("rag.not_found"), "sources": []}

    used = cited_hits(result.text, hits) or hits[:FALLBACK_SOURCES]
    text = strip_citations(result.text)
    return {
        "answer": f"{text}\n\n{sources_line(used)}",
        "sources": [h["locator"] for h in used],
    }


def route_after_scope_check(state: AgentState) -> str:
    return "answer" if state["in_scope"] else "end"

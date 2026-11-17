"""Graph state and node logic."""

import re
from typing import Callable, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from agent.console import log
from agent.errors import NodeError
from agent.llm import LONG_REQUEST_TIMEOUT_SECONDS, LLMResult, call_llm
from agent.messages import msg
from agent.tools import ChunkIndex, make_retrieve_tool
from agent.topics import (
    MAX_TOPICS,
    MIN_TOPICS,
    build_units,
    fit_to_budget,
    format_units,
    parse_topic_map,
    topic_map_json,
)

RETRIEVE_K = 5
FALLBACK_SOURCES = 2  # if the model cites nothing, show the top retrieved chunks
NOT_FOUND = "NOT_FOUND"

SCOPE_CHECK_PROMPT = (
    "You are a gatekeeper for an assistant that helps build e-learning lessons. "
    "Classify the user's message into exactly one category:\n"
    "BUILD_LESSON - a request to build, create or develop an e-learning lesson, course or "
    "training unit (for example: build me a lesson from the presentation).\n"
    "QUESTION - a question or request about e-learning, instructional design, teaching or "
    "learning.\n"
    "OUT_OF_SCOPE - anything else.\n"
    "Reply with exactly one word: BUILD_LESSON, QUESTION or OUT_OF_SCOPE. Do not explain."
)

SCOPE_CHECK_WITH_SOURCES = (
    "\nThe user has added source files for a lesson (for example training slides, "
    "procedures or policies); their names and the excerpt most relevant to the message "
    "are given with it. Questions about the content of this source material are QUESTION, "
    "and requests to build a lesson from it are BUILD_LESSON, even if the topic itself "
    "(for example gifts, driving or safety rules) is not about learning. Topics that are "
    "related neither to e-learning nor to this material are OUT_OF_SCOPE."
)
SCOPE_EXCERPT_CHARS = 600

INTENT_QUESTION = "question"
INTENT_BUILD = "build_lesson"
INTENT_OUT = "out_of_scope"
INTENT_NEEDS_FILES = "needs_files"  # build_lesson with no files in the session

TOPIC_MAP_ATTEMPTS = 3
TOPIC_MAP_PROMPT = (
    "You are an instructional designer planning an e-learning lesson from the user's source "
    "material. Group the material into {min_topics}-{max_topics} topics that together cover "
    "ALL the source files. Write each topic name in Hebrew (2-6 words). Write exactly one "
    "learning objective per topic, in Hebrew, as one sentence that starts with \"{prefix}\" "
    "followed by a measurable verb. For each topic list the ids of the source units it is "
    "based on (for example \"U3\"); use only the given ids and give every topic at least one. "
    "Reply with only a JSON object and no other text, in exactly this shape: "
    '{{"topics": [{{"name": "...", "objective": "...", "sources": ["U1", "U2"]}}]}}'
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
    intent: str  # question | build_lesson | out_of_scope | needs_files
    answer: str
    sources: list[dict]  # locators of the chunks the answer is based on
    topic_map: list[dict] | None  # [{"name", "objective", "sources": [locator, ...]}]
    feedback: str | None  # reviewer's change request from the topics gate
    approved: bool | None


def _enter(name: str) -> None:
    log(f"[INFO] Entering Node: {name}")


def _configurable(config: RunnableConfig | None) -> dict:
    return (config or {}).get("configurable") or {}


def _index(config: RunnableConfig | None) -> ChunkIndex | None:
    index = _configurable(config).get("index")
    return index if index is not None and len(index) else None


def _call(node: str, messages: list, config: RunnableConfig | None, timeout: float | None = None) -> LLMResult:
    """Call the model for a node: tag errors with the node name and record usage."""
    try:
        result = call_llm(messages, timeout=timeout) if timeout else call_llm(messages)
    except Exception as e:
        raise NodeError(node, e) from e
    recorder: UsageRecorder | None = _configurable(config).get("usage_recorder")
    if recorder:
        recorder(node, result)
    return result


def scope_check(state: AgentState, config: RunnableConfig) -> dict:
    _enter("scope_check")
    index = _index(config)
    if index is None:
        system, user = SCOPE_CHECK_PROMPT, state["question"]
    else:
        # Always ask the model; give it the session's files and the closest excerpt as context.
        system = SCOPE_CHECK_PROMPT + SCOPE_CHECK_WITH_SOURCES
        user = scope_context(state["question"], index)
    result = _call("scope_check", [SystemMessage(system), HumanMessage(user)], config)
    intent = classify_intent(result.text)
    if intent == INTENT_OUT:
        return {"intent": intent, "answer": msg("graph.out_of_scope")}
    if intent == INTENT_BUILD and index is None:
        return {"intent": INTENT_NEEDS_FILES, "answer": msg("scope.need_files")}
    return {"intent": intent}


def classify_intent(reply: str) -> str:
    """Map the gatekeeper's reply to an intent. An unclear reply counts as a question,
    so a valid request is not refused by mistake."""
    upper = reply.upper()
    if "BUILD_LESSON" in upper:
        return INTENT_BUILD
    if "OUT_OF_SCOPE" in upper:
        return INTENT_OUT
    return INTENT_QUESTION


def scope_context(question: str, index: ChunkIndex) -> str:
    """Scope-check input: file names in the session, the most relevant excerpt, the question."""
    files = list(dict.fromkeys(c["locator"]["file"] for c in index.chunks))
    hits = index.retrieve(question, k=1)
    if hits:
        text = hits[0]["text"][:SCOPE_EXCERPT_CHARS]
        excerpt = f"({describe_locator(hits[0]['locator'])})\n{text}"
    else:
        excerpt = "(none: no excerpt matches the question)"
    return (
        f"Source files in this session: {', '.join(files)}\n\n"
        f"Most relevant excerpt:\n{excerpt}\n\n"
        f"Question: {question}"
    )


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
    if "paragraph_end" in locator:
        return msg("rag.source.paragraphs", file=locator["file"], start=locator["paragraph"],
                   end=locator["paragraph_end"])
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


# ---- lesson building: topic map + approval gate ----------------------------------------
def topic_map(state: AgentState, config: RunnableConfig) -> dict:
    """Build (or revise, after gate feedback) a 4-8 topic map from ALL source chunks."""
    _enter("topic_map")
    index = _index(config)
    if index is None:  # routing only gets here with files; keep a clear error just in case
        raise NodeError("topic_map", ValueError("no source files in the session"))
    units = fit_to_budget(build_units(index.chunks))
    prefix = msg("topic_map.objective_prefix")
    system = TOPIC_MAP_PROMPT.format(min_topics=MIN_TOPICS, max_topics=MAX_TOPICS, prefix=prefix)
    user = f"Source units:\n\n{format_units(units)}\n\nUser request: {state['question']}"
    if state.get("topic_map") and state.get("feedback") is not None:
        feedback = state["feedback"] or "(no comment: propose a different, improved map)"
        user += (
            f"\n\nYour previous topic map:\n{topic_map_json(state['topic_map'], units)}\n\n"
            f"The reviewer asked for changes. Revise the map and apply this feedback: {feedback}"
        )

    messages: list = [SystemMessage(system), HumanMessage(user)]
    error = ""
    for _ in range(TOPIC_MAP_ATTEMPTS):
        result = _call("topic_map", messages, config, timeout=LONG_REQUEST_TIMEOUT_SECONDS)
        try:
            topics = parse_topic_map(result.text, units, prefix)
        except ValueError as e:
            error = str(e)
            messages += [
                AIMessage(result.text),
                HumanMessage(f"That reply was invalid: {error}. Reply again with only the corrected JSON object."),
            ]
            continue
        return {"topic_map": topics, "feedback": None, "approved": None}
    raise NodeError("topic_map", ValueError(f"no valid topic map after {TOPIC_MAP_ATTEMPTS} attempts: {error}"))


def gate_topics(state: AgentState, config: RunnableConfig) -> dict:
    """Human approval of the topic map. Pauses the graph with interrupt(); the CLI shows the
    map and resumes with {"approved": bool, "feedback": str}.
    Note: on resume LangGraph re-runs this node from the top, so its [INFO] line prints again."""
    _enter("gate_topics")
    reply = interrupt({"gate": "topics", "topics": state["topic_map"]})
    if not isinstance(reply, dict):
        reply = {"approved": False, "feedback": str(reply)}
    if reply.get("approved"):
        return {"approved": True, "answer": msg("gate.topics.approved")}
    return {"approved": False, "feedback": reply.get("feedback") or ""}


def route_after_scope_check(state: AgentState) -> str:
    intent = state.get("intent")
    if intent == INTENT_QUESTION:
        return "answer"
    if intent == INTENT_BUILD:
        return "topic_map"
    return "end"  # out_of_scope, or build_lesson without files (message already set)


def route_after_gate(state: AgentState) -> str:
    """Approved ends this stage; anything else goes back to topic_map. Never skips the gate."""
    return "end" if state.get("approved") else "topic_map"

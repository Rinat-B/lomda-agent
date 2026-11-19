"""Graph state and node logic."""

import json
import re
from typing import Callable, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from agent.console import log, say
from agent.errors import NodeError
from agent.guards import SOURCE_DATA_RULE, chunk_text_for_prompt, wrap_sources
from agent.llm import LONG_REQUEST_TIMEOUT_SECONDS, LLMResult, call_llm
from agent.messages import msg
from agent.structure import (
    COMPLIANCE,
    COMPLIANCE_MAX_WORDS,
    LESSON_TYPES,
    MAX_SCREENS,
    TRAINING,
    other_type,
    parse_structure,
    structure_json,
)
from agent.tools import ChunkIndex, make_retrieve_tool
from agent.writer import assemble_lesson, build_messages, group_excerpts, parse_group, screen_groups
from agent.topics import (
    Unit,
    describe_unit,
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
    + SOURCE_DATA_RULE
)
SCOPE_EXCERPT_CHARS = 600

INTENT_QUESTION = "question"
INTENT_BUILD = "build_lesson"
INTENT_OUT = "out_of_scope"
INTENT_NEEDS_FILES = "needs_files"  # build_lesson with no files in the session

TOPIC_MAP_ATTEMPTS = 3
TOPIC_MAP_PROMPT = (
    "You are an instructional designer planning a {type_name} e-learning lesson from the "
    "user's source material. {type_rules} Group the material into {min_topics}-{max_topics} "
    "topics that together cover ALL the source files. Write each topic name in Hebrew (2-6 "
    "words). Write exactly one learning objective per topic, in Hebrew, as one sentence that "
    "starts with \"{prefix}\" followed by a measurable verb. For each topic list the ids of "
    "the source units it is based on (for example \"U3\"); use only the given ids and give "
    "every topic at least one. The ids are internal: put them only in \"sources\", never in "
    "a name or an objective. "
    "Reply with only a JSON object and no other text, in exactly this shape: "
    '{{"topics": [{{"name": "...", "objective": "...", "sources": ["U1", "U2"]}}]}}'
)

# How the lesson type shapes the lesson (docs/SPEC.md, "two lesson types").
TYPE_NAMES = {TRAINING: "training", COMPLIANCE: "compliance"}
TOPIC_RULES = {
    TRAINING: "Training lesson: practical know-how; topics may follow tasks, risks and decisions.",
    COMPLIANCE: (
        "Compliance lesson: the owner's material is the only source of truth; topics follow "
        "the document's obligations and rules; never add information that is not in the material."
    ),
}
STRUCTURE_RULES = {
    TRAINING: (
        "Training lesson: warm, friendly tone; screens may be rich and long; propose interactive "
        "illustrations wherever they teach something a sentence cannot; sharp decision questions."
    ),
    COMPLIANCE: (
        "Compliance lesson: formal tone; use only the owner's material, never outside facts; keep "
        f"screens short: the content points of one screen together must not exceed "
        f"{COMPLIANCE_MAX_WORDS} words; illustrations only when essential (null on most screens)."
    ),
}

CLASSIFY_TYPE_ATTEMPTS = 2
CLASSIFY_TYPE_SAMPLE_CHARS = 5_000
CLASSIFY_TYPE_PROMPT = (
    "Decide which kind of e-learning lesson the user's source material calls for.\n"
    "TRAINING - teaches skills and know-how (for example safe winter driving, operating "
    "equipment, customer service).\n"
    "COMPLIANCE - teaches rules the organisation requires people to follow (for example a "
    "code of ethics, a policy, legal or regulatory obligations).\n"
    'Reply with only a JSON object: {"type": "training" or "compliance", "reason": "<short '
    'English reason>"}' + SOURCE_DATA_RULE
)

PLAN_ATTEMPTS = 3
PLAN_STRUCTURE_PROMPT = (
    "You are an instructional designer. Plan the screens of a {type_name} e-learning lesson "
    "from the approved topic map and the source units. {type_rules}\n"
    "Rules: every topic of the map gets at least one screen (one to three screens per topic, "
    "at most {max_screens} screens in total), in a logical order. For each screen give: "
    "number; title (Hebrew); topic (the exact topic name from the map); 2-4 content points "
    "(short Hebrew sentences, each with the ids of the source units it comes from, for "
    "example \"U3\"; use only the given ids). Source ids belong only in the \"sources\" "
    "fields: the ids and file names are internal and the reviewer never sees them, so never "
    "write them, or any reference like \"(slide 4)\", inside a text field. "
    "question_idea (Hebrew): the lesson exists so that people act correctly at work, so its "
    "questions must test whether the learner can use what they learned, not whether they "
    "remember it. Describe a concrete situation the learner could meet, with the relevant "
    "facts inside the situation, and ask what they should decide or do. A question whose "
    "answer is a number, date, name or a value read from a table only tests memory of the "
    "screen and can be answered without understanding, so do not ask for those; put such "
    "facts in the situation instead. For example, rather than asking for the minimum tread "
    "depth, say that this morning the learner found a tyre with 2.5 mm of tread and ask "
    "whether to go out on deliveries. illustration (Hebrew: an idea for an interactive illustration of "
    "this screen, or null for none); decision_diagram (true only if the screen teaches an "
    "if/then decision process that appears in the source). Also give 2-3 descriptions in "
    "Hebrew, in words, of a visual motif for the whole lesson.\n"
    "Reply with only a JSON object and no other text, in exactly this shape: "
    '{{"screens": [{{"number": 1, "title": "...", "topic": "...", "points": [{{"text": "...", '
    '"sources": ["U1"]}}], "question_idea": "...", "illustration": "..." or null, '
    '"decision_diagram": false}}], "motifs": ["...", "..."]}}'
) + SOURCE_DATA_RULE.replace("{", "{{").replace("}", "}}")

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
    + SOURCE_DATA_RULE
)

# Called as recorder(node_name, result) after every successful model call.
UsageRecorder = Callable[[str, LLMResult], None]

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


class AgentState(TypedDict, total=False):
    question: str
    intent: str  # question | build_lesson | out_of_scope | needs_files
    answer: str
    sources: list[dict]  # locators of the chunks the answer is based on
    lesson_type: str | None  # training | compliance (proposed by classify_type, confirmed at gate_type)
    topic_map: list[dict] | None  # [{"name", "objective", "sources": [locator, ...]}]
    structure: dict | None  # {"lesson_type", "screens": [...], "motifs": [...]}
    # Shared by the review gates (only one is open at a time); reset by each producing node.
    feedback: str | None  # reviewer's change request
    gate_answer: str | None  # answer to a reviewer's question (the reviewed item is unchanged)
    approved: bool | None
    # Stage 3: writing, assembly, saving.
    written_screens: list[dict] | None  # schema screens written so far
    written_groups: int | None  # how many screen groups are written
    lesson_title: str | None
    lesson_summary: list[str] | None
    lesson: dict | None  # the assembled, validated lesson.json
    output_folder: str | None
    saved_path: str | None  # index.html of the exported lesson


def _configurable(config: RunnableConfig | None) -> dict:
    return (config or {}).get("configurable") or {}


def _index(config: RunnableConfig | None) -> ChunkIndex | None:
    index = _configurable(config).get("index")
    return index if index is not None and len(index) else None


def _call(
    node: str,
    messages: list,
    config: RunnableConfig | None,
    timeout: float | None = None,
    role: str = "writer",
    models: list[str] | None = None,
) -> LLMResult:
    """Call the model for a node: tag errors with the node name and record usage.
    `role` picks the model list in models.yaml (writer / checker / small); `models`
    overrides it with specific models."""
    kwargs: dict = {"role": role}
    if timeout:
        kwargs["timeout"] = timeout
    if models:
        kwargs["models"] = models
    try:
        result = call_llm(messages, **kwargs)
    except Exception as e:
        raise NodeError(node, e) from e
    recorder: UsageRecorder | None = _configurable(config).get("usage_recorder")
    if recorder:
        recorder(node, result)
    return result


def scope_check(state: AgentState, config: RunnableConfig) -> dict:
    index = _index(config)
    if index is None:
        system, user = SCOPE_CHECK_PROMPT, state["question"]
    else:
        # Always ask the model; give it the session's files and the closest excerpt as context.
        system = SCOPE_CHECK_PROMPT + SCOPE_CHECK_WITH_SOURCES
        user = scope_context(state["question"], index)
    result = _call("scope_check", [SystemMessage(system), HumanMessage(user)], config, role="small")
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
        text = chunk_text_for_prompt(hits[0])[:SCOPE_EXCERPT_CHARS]
        excerpt = wrap_sources(f"({describe_locator(hits[0]['locator'])})\n{text}")
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
    """Numbered excerpts for a prompt, wrapped as source data (see guards.SOURCE_DATA_RULE)."""
    return wrap_sources("\n\n".join(
        f"[{i}] ({describe_locator(h['locator'])}, {h['kind']})\n{chunk_text_for_prompt(h)}"
        for i, h in enumerate(hits, 1)
    ))


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


# ---- lesson building: type, topic map, structure, and their gates ------------------------
def _units(config: RunnableConfig | None, node: str) -> list[Unit]:
    index = _index(config)
    if index is None:  # routing only gets here with files; keep a clear error just in case
        raise NodeError(node, ValueError("no source files in the session"))
    return fit_to_budget(build_units(index.chunks))


def _lesson_type(state: AgentState) -> str:
    return state.get("lesson_type") if state.get("lesson_type") in LESSON_TYPES else TRAINING


def _structured_call(node: str, messages: list, config, parse, attempts: int):
    """Call the writer until parse(reply) succeeds; each failure is sent back as feedback."""
    error = ""
    for _ in range(attempts):
        result = _call(node, messages, config, timeout=LONG_REQUEST_TIMEOUT_SECONDS)
        try:
            return parse(result.text)
        except ValueError as e:
            error = str(e)
            messages += [
                AIMessage(result.text),
                HumanMessage(f"That reply was invalid: {error}. Reply again with only the corrected JSON object."),
            ]
    raise NodeError(node, ValueError(f"no valid reply after {attempts} attempts: {error}"))


def classify_type(state: AgentState, config: RunnableConfig) -> dict:
    """Small model proposes training or compliance from a sample of the material."""
    units = _units(config, "classify_type")
    files = list(dict.fromkeys(u.locator["file"] for u in units))
    sample, size = [], 0
    for u in units:  # a slice of every file, so no file is left out
        if size > CLASSIFY_TYPE_SAMPLE_CHARS:
            break
        text = u.text[: max(200, CLASSIFY_TYPE_SAMPLE_CHARS // max(1, len(units)))]
        sample.append(f"({describe_unit(u)})\n{text}")
        size += len(text)
    user = (
        f"Files: {', '.join(files)}\n\nSample of the material:\n{wrap_sources(chr(10).join(sample))}\n\n"
        f"User request: {state['question']}"
    )
    messages: list = [SystemMessage(CLASSIFY_TYPE_PROMPT), HumanMessage(user)]
    for _ in range(CLASSIFY_TYPE_ATTEMPTS):
        result = _call("classify_type", messages, config, role="small")
        match = re.search(r"\{.*\}", result.text, re.DOTALL)
        try:
            data = json.loads(match.group(0)) if match else {}
        except json.JSONDecodeError:
            data = {}
        proposed = str(data.get("type", "")).strip().lower()
        if proposed in LESSON_TYPES:
            return {"lesson_type": proposed}
        messages += [AIMessage(result.text), HumanMessage('Reply with only {"type": "training" or "compliance", "reason": "..."}.')]
    return {"lesson_type": TRAINING}  # unclear: propose training; the user confirms or switches


def gate_type(state: AgentState, config: RunnableConfig) -> dict:
    """Short [Y/N] gate: confirm the proposed type, or switch to the other one."""
    proposed = _lesson_type(state)
    reply = interrupt({"gate": "type", "lesson_type": proposed})
    confirmed = reply.get("confirm", True) if isinstance(reply, dict) else bool(reply)
    lesson_type = proposed if confirmed else other_type(proposed)
    say(msg("gate.type.chosen", type=msg(f"lesson_types.{lesson_type}")))  # after interrupt: printed once
    return {"lesson_type": lesson_type}


def topic_map(state: AgentState, config: RunnableConfig) -> dict:
    """Build (or revise, after gate feedback) a 4-8 topic map from ALL source chunks."""
    units = _units(config, "topic_map")
    lesson_type = _lesson_type(state)
    prefix = msg("topic_map.objective_prefix")
    system = TOPIC_MAP_PROMPT.format(
        type_name=TYPE_NAMES[lesson_type], type_rules=TOPIC_RULES[lesson_type],
        min_topics=MIN_TOPICS, max_topics=MAX_TOPICS, prefix=prefix,
    ) + SOURCE_DATA_RULE
    user = f"Source units:\n\n{format_units(units)}\n\nUser request: {state['question']}"
    if state.get("topic_map") and state.get("feedback") is not None:
        feedback = state["feedback"] or "(no comment: propose a different, improved map)"
        user += (
            f"\n\nYour previous topic map:\n{topic_map_json(state['topic_map'], units)}\n\n"
            f"The reviewer asked for changes. Revise the map and apply this feedback: {feedback}"
        )
    topics = _structured_call(
        "topic_map", [SystemMessage(system), HumanMessage(user)], config,
        lambda text: parse_topic_map(text, units, prefix), TOPIC_MAP_ATTEMPTS,
    )
    return {"topic_map": topics, "feedback": None, "approved": None, "gate_answer": None}


def plan_structure(state: AgentState, config: RunnableConfig) -> dict:
    """Stage 2: screens (topic, objective, points with sources, question and illustration
    ideas, decision diagram) and 2-3 motif descriptions, shaped by the lesson type."""
    units = _units(config, "plan_structure")
    lesson_type = _lesson_type(state)
    topics = state["topic_map"]
    system = PLAN_STRUCTURE_PROMPT.format(
        type_name=TYPE_NAMES[lesson_type], type_rules=STRUCTURE_RULES[lesson_type], max_screens=MAX_SCREENS,
    )
    map_json = json.dumps(
        {"topics": [{"name": t["name"], "objective": t["objective"]} for t in topics]}, ensure_ascii=False
    )
    user = f"Approved topic map (JSON):\n{map_json}\n\nSource units:\n\n{format_units(units)}"
    if state.get("structure") and state.get("feedback") is not None:
        feedback = state["feedback"] or "(no comment: propose a different, improved structure)"
        user += (
            f"\n\nYour previous structure:\n{structure_json(state['structure'], units)}\n\n"
            f"The reviewer asked for changes. Revise the structure and apply this feedback: {feedback}"
        )
    structure = _structured_call(
        "plan_structure", [SystemMessage(system), HumanMessage(user)], config,
        lambda text: parse_structure(text, units, topics, lesson_type), PLAN_ATTEMPTS,
    )
    return {"structure": structure, "feedback": None, "approved": None, "gate_answer": None}


# ---- review gates (topic map, structure) ------------------------------------------------
GATE_CLASSIFY_ATTEMPTS = 2  # one retry for the structured reply
GATE_CLASSIFY_PROMPT = (
    "You interpret a reviewer's reply at the {subject} approval gate of an e-learning lesson "
    "builder. Glossary for this gate: {glossary}\n"
    "Classify the reply as exactly one of:\n"
    "APPROVE - accepts the {subject} as it is.\n"
    "CHANGE - asks to add, remove, merge, split, rename, reorder or reword something.\n"
    "QUESTION - asks something about the {subject} or the material without requesting a change.\n"
    "Reply with only a JSON object: "
    '{{"decision": "APPROVE" | "CHANGE" | "QUESTION", "change": "<for CHANGE: what to change, '
    'in the reviewer\'s own language; otherwise empty>"}}'
)
GATE_QUESTION_PROMPT = (
    "You help a reviewer who is approving the {subject} of an e-learning lesson. Answer the "
    "question briefly (one to three sentences) in Hebrew, using only the {subject} and the "
    "source excerpts provided. Do not change the {subject} and do not propose a new one. If "
    "the answer is not there, say so briefly."
) + SOURCE_DATA_RULE.replace("{", "{{").replace("}", "}}")
GATE_EXCERPTS_K = 4

# Glossary terms per gate: English meaning, with the Hebrew word the reviewer may use
# (loaded from messages_he.json at runtime: gate.glossary.<gate>.<term>).
GLOSSARY = {
    "topics": {
        "map": "the whole topic map",
        "topic": "one row of the map",
        "objective": "the single learning objective of a topic",
        "sources": "the slides, pages or paragraphs a topic is based on",
    },
    "structure": {
        "structure": "the whole list of screens and the motifs",
        "screen": "one screen of the lesson (number, title, topic, points, question, illustration, diagram)",
        "points": "the content points of a screen",
        "question": "the question idea of a screen",
        "illustration": "the illustration field of ONE screen; it is NOT the motif",
        "diagram": "the decision-diagram flag of a screen",
        "motif": "the visual motif descriptions for the whole lesson (not a screen's illustration)",
    },
}
SUBJECTS = {"topics": "topic map", "structure": "lesson structure"}


def glossary_text(gate: str) -> str:
    parts = []
    for term, meaning in GLOSSARY[gate].items():
        hebrew = msg(f"gate.glossary.{gate}.{term}")
        parts.append(f"'{term}' (Hebrew: '{hebrew}') = {meaning}")
    return "; ".join(parts) + "."


def _topics_for_prompt(topics: list[dict]) -> str:
    lines = []
    for i, t in enumerate(topics, 1):
        sources = "; ".join(describe_unit(Unit("", loc, "")) for loc in t["sources"])
        lines.append(f"{i}. {t['name']} | objective: {t['objective']} | sources: {sources}")
    return "\n".join(lines)


def _structure_for_prompt(structure: dict) -> str:
    lines = [f"Lesson type: {structure['lesson_type']}"]
    for s in structure["screens"]:
        points = " / ".join(p["text"] for p in s["points"])
        lines.append(
            f"Screen {s['number']}: {s['title']} | topic: {s['topic']} | points: {points} | "
            f"question: {s['question_idea']} | illustration: {s['illustration'] or 'none'} | "
            f"decision diagram: {'yes' if s['decision_diagram'] else 'no'}"
        )
    lines.append("Motifs: " + " / ".join(structure["motifs"]))
    return "\n".join(lines)


def _gate_item_text(gate: str, state: AgentState) -> str:
    return _topics_for_prompt(state["topic_map"]) if gate == "topics" else _structure_for_prompt(state["structure"])


def classify_gate_reply(text: str, gate: str, item_text: str, config: RunnableConfig | None) -> tuple[str, str]:
    """Free text at a gate -> ("approve" | "change" | "question", change text).
    Falls back to a change request with the whole text if the model's reply is unusable."""
    subject = SUBJECTS[gate]
    messages: list = [
        SystemMessage(GATE_CLASSIFY_PROMPT.format(subject=subject, glossary=glossary_text(gate))),
        HumanMessage(f"Current {subject}:\n{item_text}\n\nReviewer reply: {text}"),
    ]
    for _ in range(GATE_CLASSIFY_ATTEMPTS):
        result = _call(f"gate_{gate}", messages, config, role="small")
        match = re.search(r"\{.*\}", result.text, re.DOTALL)
        try:
            data = json.loads(match.group(0)) if match else {}
        except json.JSONDecodeError:
            data = {}
        decision = str(data.get("decision", "")).strip().upper()
        if decision in ("APPROVE", "CHANGE", "QUESTION"):
            change = str(data.get("change") or "").strip() or text
            return decision.lower(), change if decision == "CHANGE" else ""
        messages += [
            AIMessage(result.text),
            HumanMessage('Invalid reply. Answer with only the JSON object {"decision": ..., "change": ...}.'),
        ]
    return "change", text


def answer_gate_question(question: str, gate: str, item_text: str, config: RunnableConfig | None) -> str:
    """A short answer from the reviewed item and the closest excerpts; nothing is changed."""
    index = _index(config)
    hits = index.retrieve(question, k=GATE_EXCERPTS_K) if index is not None else []
    excerpts = format_excerpts(hits) if hits else "(no matching excerpts)"
    subject = SUBJECTS[gate]
    user = f"{subject.capitalize()}:\n{item_text}\n\nSource excerpts:\n{excerpts}\n\nQuestion: {question}"
    system = GATE_QUESTION_PROMPT.format(subject=subject)
    return _call(f"gate_{gate}", [SystemMessage(system), HumanMessage(user)], config).text


def _review_gate(gate: str, state: AgentState, config: RunnableConfig) -> dict:
    """Shared logic of the review gates. Pauses with interrupt(); the CLI shows the item (or
    the answer to the reviewer's last question) and resumes with either
    {"approved": bool, "feedback": str} for Y/N answers (no model call), or
    {"text": str} for free text, which a small model classifies as approve / change / question.
    A question is answered and the gate asks again with the item unchanged.
    On resume LangGraph re-runs the node from the top: nothing may be printed before
    interrupt(); the CLI prints the "[INFO] Entering Node" lines."""
    item_key = "topic_map" if gate == "topics" else "structure"
    request: dict = {"gate": gate, item_key: state[item_key]}
    if state.get("gate_answer"):
        request["answer"] = state["gate_answer"]
    reply = interrupt(request)
    if not isinstance(reply, dict):
        reply = {"text": str(reply)}

    if "text" in reply:
        item_text = _gate_item_text(gate, state)
        decision, change = classify_gate_reply(reply["text"], gate, item_text, config)
        if decision == "question":
            return {"approved": None, "gate_answer": answer_gate_question(reply["text"], gate, item_text, config)}
        reply = {"approved": decision == "approve", "feedback": change}

    if reply.get("approved"):
        return {"approved": True, "feedback": None, "gate_answer": None}
    return {"approved": False, "feedback": reply.get("feedback") or "", "gate_answer": None}


def gate_topics(state: AgentState, config: RunnableConfig) -> dict:
    """Gate 1: approve the topic map (change -> topic_map, approve -> plan_structure)."""
    update = _review_gate("topics", state, config)
    if update.get("approved"):
        say(msg("gate.topics.approved"))  # after interrupt: printed once
        update["approved"] = None  # the structure gate decides the end of this stage
    return update


def gate_structure(state: AgentState, config: RunnableConfig) -> dict:
    """Gate 2: approve the lesson structure (change -> plan_structure, approve -> write_lesson)."""
    update = _review_gate("structure", state, config)
    if update.get("approved"):
        say(msg("gate.structure.approved"))  # after interrupt: printed once
        update["approved"] = None  # the save gate decides the end of this stage
    return update


# ---- stage 3: writing ---------------------------------------------------------------------------
WRITE_ATTEMPTS_PER_MODEL = 2  # the first try plus one retry with the validation error
# Writing 3-4 screens is the longest generation: free models took 60-160 s in
# evals/bench_writer.py, so the writer gets more than LONG_REQUEST_TIMEOUT_SECONDS.
WRITE_TIMEOUT_SECONDS = 240


def _groups(state: AgentState) -> list[tuple[int, int]]:
    return screen_groups(len(state["structure"]["screens"]))


def node_detail(node: str, state: dict) -> str | None:
    """Extra text for the "[INFO] Entering Node" line, e.g. write_lesson (screens 1-4)."""
    if node == "write_lesson" and state.get("structure"):
        groups = _groups(state)
        done = state.get("written_groups") or 0
        if done < len(groups):
            first, last = groups[done]
            return f"screens {first}-{last}" if first != last else f"screen {first}"
    return None


def _write_group(messages: list, config: RunnableConfig, parse) -> dict:
    """Ask the writer models in order. A model gets its validation error back once; if the
    reply is still invalid (or the model fails), the next writer model is tried."""
    from agent.errors import AllModelsFailedError
    from agent.llm import load_models, unavailable_models

    skipped = unavailable_models()
    candidates = [m for m in load_models("writer") if m not in skipped] or load_models("writer")
    problems: list[str] = []
    for i, model in enumerate(candidates):
        attempt_messages = list(messages)
        for _ in range(WRITE_ATTEMPTS_PER_MODEL):
            try:
                result = _call("write_lesson", attempt_messages, config,
                               timeout=WRITE_TIMEOUT_SECONDS, models=[model])
            except NodeError as e:
                if isinstance(e.cause, AllModelsFailedError):
                    problems.append(f"{model}: {e.cause}")
                    break  # this model is down or slow: try the next one
                raise  # missing key, 401, daily quota...: no point trying other models
            try:
                return parse(result.text)
            except ValueError as err:
                problems.append(f"{model}: {err}")
                attempt_messages = attempt_messages + [
                    AIMessage(result.text),
                    HumanMessage(f"That reply was invalid: {err}. Reply again with only the corrected JSON object."),
                ]
        else:
            if i + 1 < len(candidates):
                log(msg("llm.invalid_output", model=model, next_model=candidates[i + 1]))
    raise NodeError("write_lesson", ValueError("no writer model produced valid screens: " + " | ".join(problems)))


def write_lesson(state: AgentState, config: RunnableConfig) -> dict:
    """Write the next group of 3-4 screens (one model call when the reply is valid)."""
    index = _index(config)
    if index is None:
        raise NodeError("write_lesson", ValueError("no source files in the session"))
    structure = state["structure"]
    groups = _groups(state)
    done = state.get("written_groups") or 0
    first, last = groups[done]
    plan = structure["screens"][first - 1:last]
    excerpts = group_excerpts(plan, index.chunks)
    lesson_type = _lesson_type(state)
    is_first, is_last = done == 0, done == len(groups) - 1
    (_, system), (_, user) = build_messages(plan, excerpts, lesson_type, is_first, is_last)
    result = _write_group(
        [SystemMessage(system), HumanMessage(user)], config,
        lambda text: parse_group(text, plan, excerpts, lesson_type, is_first, is_last),
    )
    update: dict = {
        "written_screens": (state.get("written_screens") or []) + result["screens"],
        "written_groups": done + 1,
    }
    if is_first:
        update["lesson_title"] = result["lesson_title"]
    if is_last:
        update["lesson_summary"] = result["summary"]
    return update


def route_after_write(state: AgentState) -> str:
    return "write_lesson" if (state.get("written_groups") or 0) < len(_groups(state)) else "assemble_lesson"


def assemble_lesson_node(state: AgentState, config: RunnableConfig) -> dict:
    """Assemble and validate lesson.json, and pick the output folder for the save gate."""
    from agent.persistence import unique_output_folder

    try:
        lesson = assemble_lesson(
            state["written_screens"], _lesson_type(state), state.get("lesson_title") or "",
            state.get("lesson_summary") or [],
        )
    except ValueError as e:
        raise NodeError("assemble_lesson", e) from e
    index = _index(config)
    first_file = index.chunks[0]["locator"]["file"] if index is not None and index.chunks else "lesson"
    return {"lesson": lesson, "output_folder": str(unique_output_folder(first_file))}


def gate_save(state: AgentState, config: RunnableConfig) -> dict:
    """[Y/N] before writing files: Y exports the lesson folder (the CLI then opens it)."""
    from pathlib import Path

    from agent.lesson_export import export_lesson

    folder = Path(state["output_folder"])
    reply = interrupt({"gate": "save", "folder": str(folder)})
    if not (isinstance(reply, dict) and reply.get("confirm")):
        return {"approved": True, "saved_path": None, "answer": msg("lesson.not_saved")}
    try:
        index_html = export_lesson(state["lesson"], folder, placeholders=True)
    except Exception as e:
        raise NodeError("gate_save", e) from e
    return {"approved": True, "saved_path": str(index_html), "answer": msg("lesson.saved")}


def route_after_scope_check(state: AgentState) -> str:
    intent = state.get("intent")
    if intent == INTENT_QUESTION:
        return "answer"
    if intent == INTENT_BUILD:
        return "classify_type"
    return "end"  # out_of_scope, or build_lesson without files (message already set)


def _route_after_review(state: AgentState, gate_node: str, redo_node: str, next_node: str) -> str:
    """A question returns to the same gate (item unchanged); a change request goes back to
    the node that produced the item; approval moves on. Nothing skips a gate."""
    if state.get("gate_answer"):
        return gate_node
    if state.get("feedback") is not None:
        return redo_node
    return next_node


def route_after_gate_topics(state: AgentState) -> str:
    return _route_after_review(state, "gate_topics", "topic_map", "plan_structure")


def route_after_gate_structure(state: AgentState) -> str:
    return _route_after_review(state, "gate_structure", "plan_structure", "write_lesson")

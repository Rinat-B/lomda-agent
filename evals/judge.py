"""LLM-as-judge for the evals: one fixed model scores a lesson against evals/rubric.md.

judge_lesson() returns {"relevance": {"score": 1-5, "reason": "..."}, ..., "average": x}.
judge_node(state, config) is the same as a LangGraph node: it reads state["lesson"] and the
session's source chunks (config["configurable"]["index"]) and returns {"judge": {...}}.
"""

import re
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

ROOT = Path(__file__).resolve().parent.parent
RUBRIC_PATH = ROOT / "evals" / "rubric.md"
CRITERIA = ("relevance", "accuracy", "completeness", "format")
JUDGE_ATTEMPTS = 2
JUDGE_TIMEOUT_SECONDS = 240


def default_judge_model() -> str:
    """The balanced profile's first checker (Gemini Flash): one fixed judge for every eval."""
    from agent.llm import load_config

    return load_config()["profiles"]["balanced"]["checker"][0]


JUDGE_PROMPT = (
    "You are the judge of an evaluation of e-learning lessons written in Hebrew by different "
    "models from the same source material. Score the lesson strictly by the rubric below, "
    "comparing every claim with the source excerpts. Be consistent: the same flaw costs the same "
    "in every lesson, whoever wrote it. Reply with only a JSON object: "
    '{"relevance": {"score": 1-5, "reason": "<one or two English sentences>"}, '
    '"accuracy": {...}, "completeness": {...}, "format": {...}}.\n\nRUBRIC:\n'
)


def build_messages(lesson: dict, chunks: list[dict], lesson_type: str | None = None,
                   schema_valid: bool | None = None) -> list:
    from agent.checkers import render_lesson, render_source
    from agent.guards import SOURCE_DATA_RULE

    rubric = RUBRIC_PATH.read_text(encoding="utf-8")
    notes = []
    if lesson_type:
        notes.append(f"Requested lesson type: {lesson_type}.")
    if schema_valid is not None:
        notes.append(f"Schema validation: {'valid' if schema_valid else 'INVALID'}.")
    user = (f"Source excerpts:\n{render_source(chunks)}\n\n{' '.join(notes)}\n\n"
            f"The lesson:\n{render_lesson(lesson)}")
    return [SystemMessage(JUDGE_PROMPT + rubric + SOURCE_DATA_RULE), HumanMessage(user)]


def parse_scores(text: str) -> dict:
    from agent.checkers import _json_object

    data = _json_object(text, "relevance")
    out = {}
    for c in CRITERIA:
        item = data.get(c)
        if not isinstance(item, dict):
            raise ValueError(f'"{c}" is missing')
        try:
            score = int(round(float(item.get("score"))))
        except (TypeError, ValueError) as e:
            raise ValueError(f'"{c}.score" is not a number') from e
        if not 1 <= score <= 5:
            raise ValueError(f'"{c}.score" must be 1-5')
        out[c] = {"score": score, "reason": re.sub(r"\s+", " ", str(item.get("reason") or "")).strip()}
    out["average"] = round(sum(out[c]["score"] for c in CRITERIA) / len(CRITERIA), 2)
    return out


def judge_lesson(lesson: dict, chunks: list[dict], call=None, model: str | None = None,
                 lesson_type: str | None = None, schema_valid: bool | None = None) -> tuple[dict, list]:
    """(scores, LLM results). `call(messages, timeout=..., models=[...])` defaults to call_llm."""
    from agent.llm import call_llm

    call = call or call_llm
    model = model or default_judge_model()
    messages = build_messages(lesson, chunks, lesson_type, schema_valid)
    results, error = [], ""
    for _ in range(JUDGE_ATTEMPTS):
        result = call(messages, timeout=JUDGE_TIMEOUT_SECONDS, models=[model])
        results.append(result)
        try:
            scores = parse_scores(result.text)
            scores["model"] = result.model
            return scores, results
        except ValueError as e:
            error = str(e)
            messages = messages + [AIMessage(result.text),
                                   HumanMessage(f"That reply was invalid: {error}. Reply again with only the JSON object.")]
    raise ValueError(f"judge: no valid reply after {JUDGE_ATTEMPTS} attempts: {error}")


def judge_node(state: dict, config: dict | None = None) -> dict:
    """The judge as a graph node: {"judge": scores} for state["lesson"]."""
    configurable = (config or {}).get("configurable") or {}
    index = configurable.get("index")
    chunks = index.chunks if index is not None else []
    scores, results = judge_lesson(state["lesson"], chunks, call=configurable.get("judge_call"),
                                   model=configurable.get("judge_model"), lesson_type=state.get("lesson_type"))
    recorder = configurable.get("usage_recorder")
    for r in results:
        if recorder:
            recorder("judge", r)
    return {"judge": scores}

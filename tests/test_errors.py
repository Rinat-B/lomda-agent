import httpx
import openai
import pytest

import agent.errors as errors
import agent.llm as llm
from agent.cli import run_question, run_turn
from agent.commands import CommandHandler
from agent.errors import AllModelsFailedError, MissingApiKeyError, NodeError, classify, report_error
from agent.messages import msg
from tests.conftest import make_result

REQUEST = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")


def api_error(status: int) -> openai.APIStatusError:
    response = httpx.Response(status, request=REQUEST)
    cls = {401: openai.AuthenticationError, 429: openai.RateLimitError}.get(status, openai.APIStatusError)
    return cls(message=f"status {status}", response=response, body=None)


def offline_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=REQUEST)


@pytest.fixture(autouse=True)
def log_to_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(errors, "log_dir", lambda: tmp_path / "logs")
    monkeypatch.setattr(errors, "_logger", errors.logging.getLogger(f"test_{tmp_path.name}"))
    errors.setup_logging(debug=False)
    return tmp_path / "logs" / errors.LOG_FILE_NAME


# ---- classification ---------------------------------------------------------------------
@pytest.mark.parametrize(
    "exc, kind",
    [
        (MissingApiKeyError(), "missing_key"),
        (api_error(401), "unauthorized"),
        (api_error(402), "no_credit"),
        (AllModelsFailedError([("a", api_error(429)), ("b", api_error(429))]), "rate_limited"),
        (AllModelsFailedError([("a", offline_error()), ("b", offline_error())]), "offline"),
        (AllModelsFailedError([("a", api_error(429)), ("b", api_error(500))]), "models_unavailable"),
        (AllModelsFailedError([("a", openai.APITimeoutError(request=REQUEST)), ("b", api_error(429))]),
         "models_unavailable"),
        (AllModelsFailedError([("a", openai.APITimeoutError(request=REQUEST))]), "models_unavailable"),
        (AllModelsFailedError([("a", api_error(429)), ("b", api_error(400))]), "unknown"),
        (ValueError("boom"), "unknown"),
    ],
)
def test_classify(exc, kind):
    assert classify(NodeError("answer", exc)) == kind


def test_classify_wrapped_langchain_style_error():
    """langchain-openai re-raises API errors `from e`; the status is found via __cause__."""
    try:
        try:
            raise api_error(402)
        except openai.APIStatusError as e:
            raise RuntimeError("wrapped") from e
    except RuntimeError as wrapped:
        assert classify(wrapped) == "no_credit"


# ---- [ERROR] line + log file --------------------------------------------------------------
def test_report_error_format_and_log(log_to_tmp, output):
    try:
        raise NodeError("scope_check", api_error(401))
    except NodeError as e:
        line = report_error(e)
    expected = msg(
        "errors.format",
        node="scope_check",
        what=msg("errors.unauthorized.what"),
        todo=msg("errors.unauthorized.todo"),
    )
    assert line == expected
    assert line.startswith("[ERROR] Node: scope_check - ")
    out = output()
    assert expected in out
    assert "Traceback" not in out  # traceback goes to the log only
    assert "Traceback" in log_to_tmp.read_text(encoding="utf-8")


def test_debug_shows_traceback(log_to_tmp, output):
    errors.setup_logging(debug=True)
    try:
        raise ValueError("boom")
    except ValueError as e:
        report_error(e, node="cli")
    assert "Traceback" in output()


# ---- model fallback (no real calls) -------------------------------------------------------
class FakeClient:
    def __init__(self, outcome):
        self.outcome = outcome

    def invoke(self, messages):
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class FakeResponse:
    content = "answer"
    response_metadata = {"token_usage": {"prompt_tokens": 3, "completion_tokens": 4, "cost": 0}}


def use_models(monkeypatch, outcomes: dict):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(llm, "load_models", lambda role="writer": list(outcomes))
    calls = []

    def client(model, timeout=None):
        calls.append(model)
        return FakeClient(outcomes[model])

    monkeypatch.setattr(llm, "_client", client)
    return calls


def test_fallback_on_429(monkeypatch):
    calls = use_models(monkeypatch, {"a": api_error(429), "b": FakeResponse()})
    result = llm.call_llm([])
    assert result.model == "b" and calls == ["a", "b"]


def test_all_429_raises_all_models_failed(monkeypatch):
    use_models(monkeypatch, {"a": api_error(429), "b": api_error(429)})
    with pytest.raises(AllModelsFailedError) as info:
        llm.call_llm([])
    assert classify(info.value) == "rate_limited"


@pytest.mark.parametrize("status", [401, 402])
def test_no_fallback_on_account_errors(monkeypatch, status):
    calls = use_models(monkeypatch, {"a": api_error(status), "b": FakeResponse()})
    with pytest.raises(openai.APIStatusError):
        llm.call_llm([])
    assert calls == ["a"]


def timeout_error() -> openai.APITimeoutError:
    return openai.APITimeoutError(request=REQUEST)


def test_timeout_is_30_seconds():
    assert llm.REQUEST_TIMEOUT_SECONDS == 30


@pytest.mark.parametrize("failure", [api_error(503), timeout_error()], ids=["503", "timeout"])
def test_unavailable_model_is_skipped_for_the_rest_of_the_conversation(monkeypatch, failure, output):
    calls = use_models(monkeypatch, {"a": failure, "b": FakeResponse()})
    assert llm.call_llm([]).model == "b" and calls == ["a", "b"]
    assert "a" in llm.unavailable_models()
    assert msg("llm.marked_unavailable", model="a", error=llm._short_error(failure), next_model="b") in output()
    calls.clear()
    assert llm.call_llm([]).model == "b"
    assert calls == ["b"]  # "a" skipped without waiting


def test_429_does_not_mark_unavailable(monkeypatch):
    calls = use_models(monkeypatch, {"a": api_error(429), "b": FakeResponse()})
    llm.call_llm([])
    llm.call_llm([])
    assert calls == ["a", "b", "a", "b"] and llm.unavailable_models() == {}


def test_when_all_models_are_unavailable_they_are_tried_again(monkeypatch):
    calls = use_models(monkeypatch, {"a": api_error(503), "b": api_error(503)})
    with pytest.raises(AllModelsFailedError):
        llm.call_llm([])
    assert set(llm.unavailable_models()) == {"a", "b"}
    calls.clear()
    with pytest.raises(AllModelsFailedError):
        llm.call_llm([])
    assert calls == ["a", "b"]  # not an instant failure with zero attempts


def test_new_conversation_resets_unavailable_models(monkeypatch, session):
    use_models(monkeypatch, {"a": api_error(503), "b": FakeResponse()})
    llm.call_llm([])
    assert llm.unavailable_models()
    CommandHandler(session).handle("/new")
    assert llm.unavailable_models() == {}


def test_models_yaml_order(monkeypatch):
    monkeypatch.delenv("PROFILE", raising=False)
    assert llm.load_models("writer")[0] == "nvidia/nemotron-3.5-super-vl:free"


def test_missing_key(monkeypatch):
    monkeypatch.setattr(llm, "load_models", lambda role="writer": ["a"])
    with pytest.raises(MissingApiKeyError):
        llm.call_llm([])


# ---- the CLI keeps running after errors ---------------------------------------------------
class _Snapshot:
    def __init__(self, values):
        self.values, self.interrupts, self.next = values, (), ()


class FailingGraph:
    """Stands in for the compiled graph: its run raises."""

    def __init__(self, exc):
        self.exc = exc

    def stream(self, payload, config=None, stream_mode=None):
        raise self.exc

    def get_state(self, config):
        return _Snapshot({})


class AnsweringGraph:
    """Stands in for the compiled graph: records usage and answers, no gates."""

    def __init__(self):
        self.values = {}

    def stream(self, payload, config=None, stream_mode=None):
        record = config["configurable"]["usage_recorder"]
        for node in ("scope_check", "answer"):
            yield {"id": node, "name": node, "input": {}, "triggers": []}
        record("scope_check", make_result("a/free", 10, 1))
        record("answer", make_result("a/free", 20, 30))
        self.values = {"question": payload["question"], "intent": "question", "answer": "**answer**"}

    def get_state(self, config):
        return _Snapshot(self.values)


def test_error_does_not_stop_the_loop(session, output):
    handler = CommandHandler(session)
    graph = FailingGraph(NodeError("answer", AllModelsFailedError([("a", offline_error())])))
    assert run_turn("how do I write a quiz?", handler, graph) is True
    assert "[ERROR] Node: answer - " + msg("errors.offline.what") in output()
    assert run_turn("/files", handler, graph) is True


def test_successful_question_records_cost(session, output):
    run_question("q", session, AnsweringGraph())
    assert len(session.usage) == 2
    assert "[INFO] Entering Node: scope_check" in output() and "[INFO] Entering Node: answer" in output()
    expected = msg("cli.cost_line", calls=2, prompt_tokens=30, completion_tokens=31, total_tokens=61, cost=0.0)
    assert expected in output()

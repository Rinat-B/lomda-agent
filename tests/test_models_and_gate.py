"""Model roles/profiles, 402 -> free profile, the OpenRouter list check, and free-text
replies at the topics gate (classified by a small model). No real model or network calls."""

import json
import time
from pathlib import Path

import httpx
import openai
import pytest

import agent.llm as llm
import agent.nodes as nodes
from agent.cli import run_question
from agent.graph import build_graph
from agent.messages import msg
from agent.session import Session, SourceDoc
from tests.test_lesson_flow import PREFIX, deck_chunks, rendered, scripted, structure_json_reply, topics_json, writer_reply

REQUEST = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")


def api_error(status: int) -> openai.APIStatusError:
    return openai.APIStatusError(message=f"status {status}", response=httpx.Response(status, request=REQUEST),
                                 body=None)


class OkResponse:
    content = "ok"
    response_metadata = {"token_usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0}}


# ---- roles and profiles --------------------------------------------------------------------
def test_every_profile_has_every_role():
    config = llm.load_config()
    assert set(config["profiles"]) >= {"free", "balanced"}
    for roles in config["profiles"].values():
        assert set(roles) >= set(llm.ROLES) and all(roles[r] for r in llm.ROLES)


def test_free_profile_is_the_default(monkeypatch):
    monkeypatch.delenv("PROFILE", raising=False)
    assert llm.active_profile() == "free"
    assert all(m.endswith(":free") for r in llm.ROLES for m in llm.load_models(r))


def test_balanced_profile_models(monkeypatch):
    monkeypatch.setenv("PROFILE", "balanced")
    assert llm.load_models("writer")[0] == "anthropic/claude-sonnet-5.5"
    assert llm.load_models("checker")[0].startswith("google/gemini-") and "flash" in llm.load_models("checker")[0]
    assert "flash-lite" in llm.load_models("small")[0]


def test_unknown_profile_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("PROFILE", "nonsense")
    assert llm.active_profile() == "free"


def test_writer_and_checker_are_different_families(monkeypatch):
    for profile in ("free", "balanced"):
        monkeypatch.setenv("PROFILE", profile)
        assert llm.load_models("writer")[0].split("/")[0] != llm.load_models("checker")[0].split("/")[0]


def test_unknown_role_is_an_error():
    with pytest.raises(ValueError):
        llm.load_models("poet")


# ---- 402 -> free profile ------------------------------------------------------------------
def fake_clients(monkeypatch, outcome_for):
    calls = []

    def client(model, timeout=None):
        calls.append(model)

        class C:
            def invoke(self, messages):
                result = outcome_for(model)
                if isinstance(result, BaseException):
                    raise result
                return result

        return C()

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(llm, "_client", client)
    return calls


def test_402_in_balanced_switches_to_free_and_retries(monkeypatch, output):
    monkeypatch.setenv("PROFILE", "balanced")
    calls = fake_clients(monkeypatch, lambda m: api_error(402) if not m.endswith(":free") else OkResponse())
    result = llm.call_llm([], role="writer")
    assert result.model.endswith(":free")
    assert calls[0] == "anthropic/claude-sonnet-5.5" and calls[1].endswith(":free")
    assert llm.active_profile() == "free"  # for the rest of the session
    assert msg("llm.switched_to_free", profile="balanced") in output()
    calls.clear()
    llm.call_llm([], role="small")
    assert all(m.endswith(":free") for m in calls)  # no more paid attempts


def test_402_in_free_profile_is_an_error(monkeypatch):
    monkeypatch.delenv("PROFILE", raising=False)
    calls = fake_clients(monkeypatch, lambda m: api_error(402))
    with pytest.raises(openai.APIStatusError):
        llm.call_llm([], role="writer")
    assert len(calls) == 1


def test_401_stops_without_switching(monkeypatch):
    monkeypatch.setenv("PROFILE", "balanced")
    calls = fake_clients(monkeypatch, lambda m: api_error(401))
    with pytest.raises(openai.APIStatusError):
        llm.call_llm([], role="writer")
    assert len(calls) == 1 and llm.active_profile() == "balanced"


# ---- OpenRouter model list check -------------------------------------------------------------
def all_configured() -> set[str]:
    return {m for roles in llm.load_config()["profiles"].values() for ms in roles.values() for m in ms}


def test_check_filters_unlisted_models_and_writes_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("PROFILE", raising=False)
    gone = "nvidia/nemotron-3.5-super-vl:free"
    listed = all_configured() - {gone}
    missing = llm.check_available_models(tmp_path / "cache.json", fetch=lambda: listed)
    assert missing == [gone]
    assert gone not in llm.load_models("writer")
    cached = json.loads((tmp_path / "cache.json").read_text(encoding="utf-8"))
    assert set(cached["ids"]) == listed


def test_check_uses_a_fresh_cache_without_fetching(tmp_path):
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"fetched_at": time.time() - 3600, "ids": sorted(all_configured())}))

    def must_not_fetch():
        raise AssertionError("fetched although the cache is fresh")

    assert llm.check_available_models(cache, fetch=must_not_fetch) == []


def test_check_refetches_a_stale_cache(tmp_path):
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"fetched_at": time.time() - 2 * 86400, "ids": []}))
    fetched = []
    llm.check_available_models(cache, fetch=lambda: fetched.append(1) or all_configured())
    assert fetched == [1]


def test_check_failure_does_not_block(tmp_path, monkeypatch):
    monkeypatch.delenv("PROFILE", raising=False)

    def offline():
        raise httpx.ConnectError("no network")

    assert llm.check_available_models(tmp_path / "cache.json", fetch=offline) is None
    assert llm.load_models("writer") == llm.load_config()["profiles"]["free"]["writer"]  # unfiltered


def test_check_never_empties_a_role(tmp_path, monkeypatch):
    monkeypatch.delenv("PROFILE", raising=False)
    llm.check_available_models(tmp_path / "cache.json", fetch=lambda: set())
    assert llm.load_models("writer")  # nothing listed -> keep the configured list


# ---- free text at the topics gate --------------------------------------------------------------
class GateModel:
    """Fake models: gatekeeper, topic mapper, and the gate's classifier and question answerer.
    The classifier reads markers from the reviewer's reply: APPROVE_ME, CHANGE_ME, QUESTION_ME."""

    def __init__(self):
        self.calls: list[tuple[str, str, str]] = []  # (kind, role, user text)

    def __call__(self, messages, **kwargs):
        system, user = messages[0].content, messages[-1].content
        role = kwargs.get("role", "writer")
        if system.startswith("You are a gatekeeper"):
            self.calls.append(("scope", role, user))
            return _result("BUILD_LESSON")
        if system.startswith("Decide which kind"):
            self.calls.append(("classify_type", role, user))
            return _result('{"type": "training", "reason": "fake"}')
        if system.startswith("You are an instructional designer. Plan the screens"):
            self.calls.append(("plan_structure", role, messages[1].content))
            return _result(structure_json_reply(messages[1].content))
        if system.startswith("You are an instructional designer"):
            self.calls.append(("topic_map", role, messages[1].content))
            extra = "TIRES" if "reviewer asked" in messages[1].content.lower() else None
            return _result(topics_json(4, extra_name=extra))
        if system.startswith("You interpret"):
            self.calls.append(("classify", role, user))
            if "APPROVE_ME" in user:
                return _result('{"decision": "APPROVE", "change": ""}')
            if "CHANGE_ME" in user:
                return _result('Sure: {"decision": "CHANGE", "change": "add a topic on tires"}')
            if "QUESTION_ME" in user:
                return _result('{"decision": "QUESTION", "change": ""}')
            return _result("not json")
        if system.startswith("You write the screens"):
            self.calls.append(("write", role, user))
            return _result(writer_reply(system, messages[1].content))
        if system.startswith("You help a reviewer"):
            self.calls.append(("answer", role, user))
            return _result("ANSWER-FROM-MAP")
        raise AssertionError(f"unexpected prompt: {system[:60]}")

    def kinds(self):
        return [k for k, _, _ in self.calls]


def _result(text):
    from tests.conftest import make_result

    return make_result(text=text)


@pytest.fixture
def gate_model(monkeypatch):
    model = GateModel()
    monkeypatch.setattr(nodes, "call_llm", model)
    return model


def run_with(answers: list[str]) -> Session:
    session = Session()
    session.add_file(Path("deck.pptx"), SourceDoc("slide", 2, deck_chunks(2)))
    run_question("build a lesson", session, build_graph(), ask_fn=scripted(answers))
    return session


@pytest.mark.parametrize("answer", ["Y", "N add tires"])
def test_y_and_n_need_no_model_call(gate_model, output, answer):
    # type gate, topics gate (+ approval after a change), structure gate
    run_with(["Y", answer] + (["Y"] if answer.startswith("N") else []) + ["Y"])
    assert "classify" not in gate_model.kinds() and "answer" not in gate_model.kinds()


def test_free_text_approval(gate_model, output):
    session = run_with(["Y", "looks great APPROVE_ME", "Y"])
    assert gate_model.kinds() == ["scope", "classify_type", "topic_map", "classify", "plan_structure", "write"]
    assert msg("gate.topics.approved") in output() and session.last_run['topic_map'] is not None


def test_free_text_change_uses_the_extracted_change(gate_model, output):
    session = run_with(["Y", "CHANGE_ME please", "Y", "Y"])
    topic_prompts = [u for k, _, u in gate_model.calls if k == "topic_map"]
    assert len(topic_prompts) == 2
    assert "apply this feedback: add a topic on tires" in topic_prompts[1]
    assert session.last_run['topic_map'][-1]["name"] == "TIRES"


def test_question_is_answered_and_the_gate_asks_again(gate_model, output):
    session = run_with(["Y", "QUESTION_ME why is slide 2 here?", "Y", "Y"])
    kinds = gate_model.kinds()
    assert kinds[:5] == ["scope", "classify_type", "topic_map", "classify", "answer"]  # no new map
    assert kinds.count("topic_map") == 1
    out = output()
    assert "ANSWER-FROM-MAP" in out
    assert out.count(rendered("gate.topics.title", count=4)) == 1  # the map is not reprinted
    assert session.last_run['topic_map'] is not None and len(session.last_run['topic_map']) == 4
    answer_prompt = next(u for k, _, u in gate_model.calls if k == "answer")
    assert "Topic map:" in answer_prompt and "Question: QUESTION_ME why is slide 2 here?" in answer_prompt


def test_classifier_uses_the_small_role(gate_model, output):
    run_with(["Y", "APPROVE_ME", "Y"])
    roles = {k: r for k, r, _ in gate_model.calls}
    assert roles["classify"] == "small" and roles["scope"] == "small" and roles["topic_map"] == "writer"


def test_unusable_classifier_reply_is_retried_then_treated_as_change(gate_model, output):
    session = run_with(["Y", "something unclear", "Y", "Y"])
    assert gate_model.kinds().count("classify") == 2  # one retry
    topic_prompts = [u for k, _, u in gate_model.calls if k == "topic_map"]
    assert "apply this feedback: something unclear" in topic_prompts[1]
    assert session.last_run['topic_map'] is not None


# ---- OpenRouter's daily free-model quota ---------------------------------------------------------
RESET_MS = 1795046400000  # 19.11.2026 00:00 UTC, as OpenRouter returned it


def quota_error(with_headers: bool = True) -> openai.RateLimitError:
    """The 429 OpenRouter returns when the free-models-per-day quota is used up."""
    body = {
        "message": "Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests per day",
        "code": 429,
        "metadata": {"headers": {"X-RateLimit-Limit": "50", "X-RateLimit-Remaining": "0",
                                 "X-RateLimit-Reset": str(RESET_MS)},
                     "limit_source": "openrouter_free_tier_daily"},
    }
    headers = {"x-ratelimit-limit": "50", "x-ratelimit-reset": str(RESET_MS)} if with_headers else {}
    response = httpx.Response(429, headers=headers, request=REQUEST)
    return openai.RateLimitError(message=body["message"], response=response, body=body)


@pytest.mark.parametrize("with_headers", [True, False], ids=["headers", "body only"])
def test_daily_quota_is_recognised_with_limit_and_reset(with_headers):
    from agent.errors import AllModelsFailedError, NodeError, classify, daily_quota_info

    err = quota_error(with_headers)
    assert daily_quota_info(err) == {"limit": 50, "reset_ms": RESET_MS}
    assert classify(err) == "daily_quota"
    assert classify(NodeError("topic_map", err)) == "daily_quota"
    assert classify(AllModelsFailedError([("a:free", err)])) == "daily_quota"


def test_an_ordinary_429_is_still_rate_limited():
    from agent.errors import classify, daily_quota_info

    busy = openai.RateLimitError(message="Provider returned error",
                                 response=httpx.Response(429, request=REQUEST), body={"message": "busy"})
    assert daily_quota_info(busy) is None and classify(busy) == "rate_limited"


def test_daily_quota_message_has_limit_reset_and_the_credit_option(monkeypatch):
    import agent.errors as errors
    from agent.errors import NodeError, format_error

    monkeypatch.setattr(errors, "quota_reset_text", lambda ms: f"RESET<{ms}>")
    line = format_error("scope_check", NodeError("scope_check", quota_error()))
    assert line == msg(
        "errors.format",
        node="scope_check",
        what=msg("errors.daily_quota.what", limit=50),
        todo=msg("errors.daily_quota.todo", reset=f"RESET<{RESET_MS}>"),
    )
    assert "10$" in line and "1000" in line and "openrouter.ai/settings/credits" in line


def test_quota_reset_text_today_tomorrow_date_unknown():
    from datetime import datetime, timedelta, timezone

    from agent.errors import quota_reset_text

    reset = datetime.fromtimestamp(RESET_MS / 1000, tz=timezone.utc).astimezone()
    time_text = reset.strftime("%H:%M")
    assert quota_reset_text(RESET_MS, now=reset) == msg("errors.daily_quota.reset_today", time=time_text)
    assert quota_reset_text(RESET_MS, now=reset - timedelta(days=1)) == msg(
        "errors.daily_quota.reset_tomorrow", time=time_text)
    assert quota_reset_text(RESET_MS, now=reset - timedelta(days=5)) == msg(
        "errors.daily_quota.reset_date", date=reset.strftime("%d.%m.%Y"), time=time_text)
    assert quota_reset_text(None) == msg("errors.daily_quota.reset_unknown")


def test_quota_stops_at_the_first_free_model(monkeypatch, output):
    monkeypatch.setattr(llm, "load_models", lambda role="writer": ["a:free", "b:free", "c:free"])
    calls = fake_clients(monkeypatch, lambda m: quota_error())
    with pytest.raises(openai.RateLimitError):
        llm.call_llm([], role="small")
    assert calls == ["a:free"]  # no pointless attempts on the other free models
    assert "b:free" not in output()  # no "switching to the next model" lines


def test_quota_skips_free_models_but_still_tries_a_paid_one(monkeypatch):
    monkeypatch.setattr(llm, "load_models", lambda role="writer": ["a:free", "b:free", "paid/model"])
    calls = fake_clients(monkeypatch, lambda m: quota_error() if m.endswith(":free") else OkResponse())
    assert llm.call_llm([], role="small").model == "paid/model"
    assert calls == ["a:free", "paid/model"]

"""OpenRouter model calls: roles and profiles from models.yaml, with automatic fallback.

- call_llm(messages, role=...) tries the role's models in order (same role, next model).
- 401 stops at once (bad key). 402 (no credit) in a paid profile switches the session to
  the free profile, says so, and retries there; 402 in the free profile is an error.
- A model that returns 503 or times out is skipped for the rest of the conversation.
- check_available_models() compares the lists with OpenRouter's model list (cached for a
  day in data/); ids OpenRouter does not list are skipped. If the check fails, nothing
  is filtered and the agent carries on.
"""

import json
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from agent.console import log
from agent.errors import AllModelsFailedError, MissingApiKeyError, daily_quota_info, status_code
from agent.messages import msg

MODELS_PATH = Path(__file__).parent / "models.yaml"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODELS_URL = f"{OPENROUTER_BASE_URL}/models"
MODELS_CACHE_SECONDS = 24 * 60 * 60
REQUEST_TIMEOUT_SECONDS = 30
# Long structured generations (a whole topic map) need more time than a short answer.
LONG_REQUEST_TIMEOUT_SECONDS = 120
ROLES = ("writer", "checker", "small")
FREE_PROFILE = "free"
# A model that fails like this is skipped for the rest of the conversation.
UNAVAILABLE_STATUS = {503}

# model id -> short reason; reset by /new (reset_unavailable_models) or a restart.
_unavailable: dict[str, str] = {}
# Set after a 402 in a paid profile: the session continues on the free profile.
_profile_override: str | None = None
# Model ids OpenRouter currently lists; None = not checked (no filtering).
_listed: set[str] | None = None

__all__ = [
    "LLMResult",
    "call_llm",
    "load_models",
    "active_profile",
    "check_available_models",
    "reset_unavailable_models",
    "unavailable_models",
    "AllModelsFailedError",
    "MissingApiKeyError",
]


# ---- configuration ----------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_config() -> dict:
    with MODELS_PATH.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    profiles = config.get("profiles") or {}
    for name, roles in profiles.items():
        for role in ROLES:
            if not (roles or {}).get(role):
                raise ValueError(f"{MODELS_PATH}: profile {name!r} has no models for role {role!r}")
    if FREE_PROFILE not in profiles:
        raise ValueError(f"{MODELS_PATH}: a {FREE_PROFILE!r} profile is required")
    return config


def configured_profile() -> str:
    """PROFILE from the environment if it names a known profile, else the default."""
    config = load_config()
    wanted = (os.getenv("PROFILE") or "").strip().lower()
    return wanted if wanted in config["profiles"] else config.get("default_profile", FREE_PROFILE)


def active_profile() -> str:
    return _profile_override or configured_profile()


def load_models(role: str = "writer") -> list[str]:
    """The role's models in the active profile, without ids OpenRouter does not list."""
    if role not in ROLES:
        raise ValueError(f"unknown model role: {role}")
    models = list(load_config()["profiles"][active_profile()][role])
    if _listed is not None:
        listed = [m for m in models if m in _listed]
        return listed or models  # never leave a role empty because of the check
    return models


# ---- session state ----------------------------------------------------------------------
def unavailable_models() -> dict[str, str]:
    return dict(_unavailable)


def reset_unavailable_models() -> None:
    _unavailable.clear()


def reset_model_state() -> None:
    """Forget unavailable models, the profile switch, and the listing check (tests)."""
    global _profile_override, _listed
    _unavailable.clear()
    _profile_override = None
    _listed = None


# ---- startup check against OpenRouter ----------------------------------------------------
def _fetch_model_ids() -> set[str]:
    import httpx

    response = httpx.get(MODELS_URL, timeout=15)
    response.raise_for_status()
    return {m["id"] for m in response.json()["data"]}


def check_available_models(cache_path: Path, fetch=_fetch_model_ids, now: float | None = None) -> list[str] | None:
    """Compare models.yaml with OpenRouter's list (cached for a day at cache_path).

    Returns the configured ids OpenRouter does not list (they will be skipped), or None
    if the check could not be made, in which case nothing is filtered.
    """
    global _listed
    now = time.time() if now is None else now
    ids: set[str] | None = None
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if now - cached["fetched_at"] < MODELS_CACHE_SECONDS:
            ids = set(cached["ids"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    if ids is None:
        try:
            ids = fetch()
        except Exception:
            _listed = None
            return None
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({"fetched_at": now, "ids": sorted(ids)}), encoding="utf-8")
        except OSError:
            pass
    _listed = ids
    configured = {m for roles in load_config()["profiles"].values() for ms in roles.values() for m in ms}
    return sorted(configured - ids)


# ---- calls ------------------------------------------------------------------------------
def _is_timeout(e: BaseException) -> bool:
    import httpx
    import openai

    chain, seen = [], set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        chain.append(e)
        e = e.__cause__ or e.__context__
    return any(isinstance(x, (openai.APITimeoutError, httpx.TimeoutException, TimeoutError)) for x in chain)


def _marks_unavailable(e: BaseException) -> bool:
    return status_code(e) in UNAVAILABLE_STATUS or _is_timeout(e)


@dataclass
class LLMResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost: float


def _client(model: str, timeout: float = REQUEST_TIMEOUT_SECONDS) -> ChatOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise MissingApiKeyError("OPENROUTER_API_KEY is not set")
    # max_retries=0: on failure we move to the next model instead of retrying the same one.
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        max_retries=0,
        timeout=timeout,
    )


def _short_error(e: Exception) -> str:
    code = status_code(e)
    return str(code) if code else type(e).__name__


def _switch_to_free(error: BaseException) -> bool:
    """After a 402 in a paid profile, continue the session on the free profile."""
    global _profile_override
    if active_profile() == FREE_PROFILE:
        return False
    previous = active_profile()
    _profile_override = FREE_PROFILE
    log(msg("llm.switched_to_free", profile=previous))
    return True


def call_llm(messages: list[BaseMessage], timeout: float = REQUEST_TIMEOUT_SECONDS, role: str = "writer") -> LLMResult:
    """Call the role's first available model that succeeds, in models.yaml order.

    Models marked unavailable (503 or timeout earlier in the conversation) are skipped.
    If every model is marked, all of them are tried again rather than failing at once.
    `timeout` is per model attempt; long generations pass LONG_REQUEST_TIMEOUT_SECONDS.
    """
    all_models = load_models(role)
    models = [m for m in all_models if m not in _unavailable] or all_models
    failures: list[tuple[str, BaseException]] = []
    free_quota_hit = False
    for i, model in enumerate(models):
        if free_quota_hit and model.endswith(":free"):
            continue  # the daily free quota is per account: every free model would fail
        try:
            response = _client(model, timeout).invoke(messages)
        except MissingApiKeyError:
            raise
        except Exception as e:
            code = status_code(e)
            if code == 401:
                raise
            if code == 402:
                if _switch_to_free(e):
                    return call_llm(messages, timeout=timeout, role=role)
                raise
            if daily_quota_info(e) is not None:
                free_quota_hit = True
                if all(m.endswith(":free") for m in models[i + 1:]):
                    raise  # nothing left that the quota does not block
                failures.append((model, e))
                continue
            failures.append((model, e))
            marked = _marks_unavailable(e)
            if marked:
                _unavailable[model] = _short_error(e)
            next_model = models[i + 1] if i + 1 < len(models) else None
            if next_model:
                key = "llm.marked_unavailable" if marked else "llm.switch_model"
                log(msg(key, model=model, error=_short_error(e), next_model=next_model))
            continue
        _unavailable.pop(model, None)  # it answered: available again

        usage = response.response_metadata.get("token_usage") or {}
        return LLMResult(
            text=str(response.content).strip(),
            model=model,
            prompt_tokens=usage.get("prompt_tokens", 0) or 0,
            completion_tokens=usage.get("completion_tokens", 0) or 0,
            cost=float(usage.get("cost", 0) or 0),
        )
    raise AllModelsFailedError(failures)

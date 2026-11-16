"""OpenRouter model calls with automatic fallback across the models in models.yaml."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from agent.console import log
from agent.errors import AllModelsFailedError, MissingApiKeyError, status_code
from agent.messages import msg

MODELS_PATH = Path(__file__).parent / "models.yaml"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
REQUEST_TIMEOUT_SECONDS = 90
# Account-level failures: every model would fail the same way, so do not fall back.
NO_FALLBACK_STATUS = {401, 402}

__all__ = ["LLMResult", "call_llm", "load_models", "AllModelsFailedError", "MissingApiKeyError"]


@dataclass
class LLMResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost: float


@lru_cache(maxsize=1)
def load_models() -> list[str]:
    with MODELS_PATH.open(encoding="utf-8") as f:
        models = yaml.safe_load(f)["models"]
    if not models:
        raise ValueError(f"No models listed in {MODELS_PATH}")
    return list(models)


def _client(model: str) -> ChatOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise MissingApiKeyError("OPENROUTER_API_KEY is not set")
    # max_retries=0: on failure we move to the next model instead of retrying the same one.
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        max_retries=0,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def _short_error(e: Exception) -> str:
    code = status_code(e)
    return str(code) if code else type(e).__name__


def call_llm(messages: list[BaseMessage]) -> LLMResult:
    """Call the first model that succeeds, in models.yaml order."""
    models = load_models()
    failures: list[tuple[str, BaseException]] = []
    for i, model in enumerate(models):
        try:
            response = _client(model).invoke(messages)
        except MissingApiKeyError:
            raise
        except Exception as e:
            if status_code(e) in NO_FALLBACK_STATUS:
                raise
            failures.append((model, e))
            next_model = models[i + 1] if i + 1 < len(models) else None
            if next_model:
                log(msg("llm.switch_model", model=model, error=_short_error(e), next_model=next_model))
            continue

        usage = response.response_metadata.get("token_usage") or {}
        return LLMResult(
            text=str(response.content).strip(),
            model=model,
            prompt_tokens=usage.get("prompt_tokens", 0) or 0,
            completion_tokens=usage.get("completion_tokens", 0) or 0,
            cost=float(usage.get("cost", 0) or 0),
        )
    raise AllModelsFailedError(failures)

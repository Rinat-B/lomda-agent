"""Shared fixtures. No test makes a real model or network call."""

import io

import pytest
from rich.console import Console

import agent.console as console_mod
from agent.llm import LLMResult
from agent.session import Session


@pytest.fixture(autouse=True)
def plain_console(monkeypatch):
    """Capture all CLI output as plain logical-order text (bidi off, no colors)."""
    monkeypatch.setenv("HEBREW_BIDI", "off")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    buf = io.StringIO()
    monkeypatch.setattr(console_mod, "console", Console(file=buf, width=200, color_system=None))
    return buf


@pytest.fixture
def output(plain_console):
    """Callable returning everything printed so far."""
    return plain_console.getvalue


@pytest.fixture
def session():
    return Session()


def make_result(model="m/free", prompt=10, completion=5, cost=0.0, text="ok") -> LLMResult:
    return LLMResult(text=text, model=model, prompt_tokens=prompt, completion_tokens=completion, cost=cost)

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


class FakeMCP:
    """Stands in for MCPManager: same interface, calls the reader in-process, no subprocess."""

    def __init__(self):
        from agent.mcp_client import SERVERS, ServerState

        self.specs = dict(SERVERS)
        self.states = {name: ServerState(spec) for name, spec in self.specs.items()}
        self.credentials: dict[str, str] = {}
        self.calls: list[tuple[str, str, dict]] = []

    def ensure_all(self):
        for state in self.states.values():
            state.stack = object()  # mark as connected
            state.tools = {"read_source": None}

    def call(self, server, tool, args):
        from agent.mcp_servers.source_reader import read_source_file

        self.calls.append((server, tool, args))
        return read_source_file(args["path"])

    def needs_auth(self, name):
        return self.specs[name].auth_env is not None

    def set_credentials(self, name, secret):
        self.credentials[name] = secret


@pytest.fixture
def fake_mcp():
    return FakeMCP()


def make_result(model="m/free", prompt=10, completion=5, cost=0.0, text="ok") -> LLMResult:
    return LLMResult(text=text, model=model, prompt_tokens=prompt, completion_tokens=completion, cost=cost)

"""MCP wiring: the real source-reader server over stdio, and the auth mechanism.

These start a real subprocess (python -m agent.cli --mcp source-reader). No model calls.
"""

from pathlib import Path

import pytest

from agent.mcp_client import MCPManager, ServerSpec, parse_tool_result

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def manager():
    m = MCPManager()
    yield m
    m.close()


def test_real_server_lists_read_source_tool(manager):
    state = manager.ensure("source-reader")
    assert state.connected
    assert list(state.tools) == ["read_source"]


def test_real_server_reads_pptx(manager):
    result = manager.call("source-reader", "read_source", {"path": str(FIXTURES / "sample.pptx")})
    assert result["ok"] and result["units"] == 3 and len(result["chunks"]) == 9
    assert result["chunks"][0]["locator"] == {"file": "sample.pptx", "slide": 1, "paragraph": 1}


def test_real_server_survives_bad_files(manager):
    bad = manager.call("source-reader", "read_source", {"path": str(FIXTURES / "corrupt.pptx")})
    assert bad["ok"] is False and bad["error"] == "corrupt"
    good = manager.call("source-reader", "read_source", {"path": str(FIXTURES / "sample.txt")})
    assert good["ok"] is True  # same server process still answers


def test_unknown_server_raises():
    from agent.mcp_client import MCPError

    with pytest.raises(MCPError):
        MCPManager().ensure("nope")


def test_credentials_are_passed_as_env_and_kept_in_memory():
    spec = ServerSpec("source-reader", auth_env="SOURCE_READER_TOKEN")
    m = MCPManager({"source-reader": spec})
    assert "SOURCE_READER_TOKEN" not in m.connection("source-reader")["env"]
    m.set_credentials("source-reader", "tok-123")  # not connected: just stored
    assert m.credentials == {"source-reader": "tok-123"}
    assert m.connection("source-reader")["env"]["SOURCE_READER_TOKEN"] == "tok-123"
    assert m.needs_auth("source-reader")


def test_set_credentials_reconnects_a_connected_server_that_needs_auth():
    spec = ServerSpec("source-reader", auth_env="SOURCE_READER_TOKEN")
    m = MCPManager({"source-reader": spec})
    try:
        first = m.ensure("source-reader").stack
        m.set_credentials("source-reader", "tok-456")
        state = m.states["source-reader"]
        assert state.connected and state.stack is not first  # restarted with the new env
    finally:
        m.close()


@pytest.mark.parametrize(
    "content, expected",
    [
        ('{"ok": true}', {"ok": True}),
        ([{"type": "text", "text": '{"a": 1}'}], {"a": 1}),
        ("not json", "not json"),
    ],
)
def test_parse_tool_result(content, expected):
    assert parse_tool_result(content) == expected

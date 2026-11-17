"""MCP servers used by the agent, connected through langchain-mcp-adapters.

Each server runs as a stdio subprocess of this program:
  - frozen exe:   agent.exe --mcp <name>
  - from source:  python -m agent.cli --mcp <name>
so no separate Python install is needed for the exe.

Sessions stay open for the whole CLI run (one subprocess per server), on a private
asyncio loop in a background thread, so the synchronous CLI can call tools directly.
Credentials from /mcp auth live only in memory and reach a server as an environment
variable (ServerSpec.auth_env) when it is (re)started.
"""

import asyncio
import json
import os
import sys
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONNECT_TIMEOUT_SECONDS = 60
CALL_TIMEOUT_SECONDS = 180


@dataclass(frozen=True)
class ServerSpec:
    name: str
    # Environment variable that receives the /mcp auth secret; None = no auth needed.
    auth_env: str | None = None


SERVERS: dict[str, ServerSpec] = {
    "source-reader": ServerSpec("source-reader"),
}


class MCPError(RuntimeError):
    """An MCP server could not be started or a tool call failed."""

    def __init__(self, server: str, message: str):
        super().__init__(f"{server}: {message}")
        self.server = server


def run_server(name: str) -> None:
    """Entry point for `--mcp <name>`: serve over stdio until stdin closes."""
    if name not in SERVERS:
        raise SystemExit(f"unknown MCP server: {name}")
    import io

    from agent.mcp_servers import source_reader

    # The stdio transport wraps sys.stdout.buffer and closes it on shutdown. The
    # PyInstaller bootloader flushes sys.stdout / sys.__stdout__ at exit, which would then
    # print "I/O operation on closed file". Give the server a duplicate of the stdout
    # handle to close instead, and restore the untouched original afterwards.
    original_stdout = sys.stdout
    sys.stdout = io.TextIOWrapper(os.fdopen(os.dup(original_stdout.fileno()), "wb"), encoding="utf-8")
    try:
        source_reader.main()
    finally:
        sys.stdout = original_stdout


def server_command(name: str) -> tuple[str, list[str]]:
    if getattr(sys, "frozen", False):
        return sys.executable, ["--mcp", name]
    return sys.executable, ["-m", "agent.cli", "--mcp", name]


@dataclass
class ServerState:
    spec: ServerSpec
    stack: AsyncExitStack | None = None
    tools: dict[str, Any] = field(default_factory=dict)  # tool name -> LangChain BaseTool
    error: str | None = None

    @property
    def connected(self) -> bool:
        return self.stack is not None


def parse_tool_result(content: Any) -> Any:
    """LangChain MCP tools return a string or a list of content blocks; decode the JSON."""
    if isinstance(content, list):
        texts = [b.get("text", "") if isinstance(b, dict) else str(b) for b in content]
        content = "".join(texts)
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content
    return content


class MCPManager:
    def __init__(self, specs: dict[str, ServerSpec] | None = None):
        self.specs = dict(specs if specs is not None else SERVERS)
        self.states = {name: ServerState(spec) for name, spec in self.specs.items()}
        self.credentials: dict[str, str] = {}  # session only, never written to disk
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    # ---- event loop -----------------------------------------------------------------
    def _run(self, coro, timeout: float):
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._loop.run_forever, name="mcp-loop", daemon=True)
            self._thread.start()
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    # ---- connections ----------------------------------------------------------------
    def connection(self, name: str) -> dict:
        """The stdio connection config for a server, including session credentials."""
        spec = self.specs[name]
        command, args = server_command(name)
        env = dict(os.environ)
        if spec.auth_env and name in self.credentials:
            env[spec.auth_env] = self.credentials[name]
        return {
            "transport": "stdio",
            "command": command,
            "args": args,
            "env": env,
            "cwd": str(PROJECT_ROOT) if not getattr(sys, "frozen", False) else None,
        }

    async def _connect(self, name: str) -> None:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_mcp_adapters.tools import load_mcp_tools

        state = self.states[name]
        stack = AsyncExitStack()
        try:
            client = MultiServerMCPClient({name: self.connection(name)})
            session = await stack.enter_async_context(client.session(name))
            tools = await load_mcp_tools(session, server_name=name)
        except BaseException:
            await stack.aclose()
            raise
        state.stack, state.tools, state.error = stack, {t.name: t for t in tools}, None

    async def _disconnect(self, name: str) -> None:
        state = self.states[name]
        if state.stack is not None:
            stack, state.stack, state.tools = state.stack, None, {}
            try:
                await stack.aclose()
            except Exception:
                pass

    def ensure(self, name: str) -> ServerState:
        """Connect to a server if needed. Raises MCPError if it cannot start."""
        if name not in self.states:
            raise MCPError(name, "unknown server")
        state = self.states[name]
        if not state.connected:
            try:
                self._run(self._connect(name), CONNECT_TIMEOUT_SECONDS)
            except Exception as e:
                state.error = f"{type(e).__name__}: {e}"
                raise MCPError(name, state.error) from e
        return state

    def ensure_all(self) -> None:
        for name in self.states:
            try:
                self.ensure(name)
            except MCPError:
                pass  # recorded in state.error, shown by /mcp

    def call(self, server: str, tool: str, args: dict) -> Any:
        state = self.ensure(server)
        if tool not in state.tools:
            raise MCPError(server, f"no tool named {tool}")
        try:
            content = self._run(state.tools[tool].ainvoke(args), CALL_TIMEOUT_SECONDS)
        except Exception as e:
            raise MCPError(server, f"{tool} failed: {type(e).__name__}: {e}") from e
        return parse_tool_result(content)

    # ---- auth -----------------------------------------------------------------------
    def needs_auth(self, name: str) -> bool:
        return self.specs[name].auth_env is not None

    def set_credentials(self, name: str, secret: str) -> None:
        """Keep a secret for this session; reconnect the server so it receives it."""
        if name not in self.specs:
            raise MCPError(name, "unknown server")
        self.credentials[name] = secret
        if self.states[name].connected and self.needs_auth(name):
            self._run(self._disconnect(name), CONNECT_TIMEOUT_SECONDS)
            self.ensure(name)

    # ---- shutdown -------------------------------------------------------------------
    def close(self) -> None:
        if self._loop is None:
            return
        for name in self.states:
            try:
                self._run(self._disconnect(name), 10)
            except Exception:
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
        self._loop = None

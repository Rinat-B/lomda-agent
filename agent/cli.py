"""Interactive CLI: read a line, run a command or send it through the agent graph."""

import argparse
from typing import Callable

from langgraph.types import Command

from agent.commands import CommandHandler
from agent.config import configure_stdio, ensure_openrouter_key, load_env
from agent.console import ask, log, say, say_path, show_markdown
from agent.errors import report_error, setup_logging
from agent.gates import GateExit, ask_gate
from agent.mcp_client import MCPManager, run_server
from agent.messages import msg
from agent.session import Session
from agent.tracing import flush_traces, setup_tracing


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="agent", description="E-learning lesson agent")
    parser.add_argument("--debug", action="store_true", help="show full tracebacks on screen")
    parser.add_argument("--mcp", metavar="SERVER", help="run a built-in MCP server over stdio (internal)")
    return parser.parse_args(argv)


# Every run starts from a clean state: the checkpointer keeps a thread's previous values.
FRESH_RUN = {
    "intent": None,
    "answer": None,
    "sources": [],
    "lesson_type": None,
    "topic_map": None,
    "structure": None,
    "feedback": None,
    "approved": None,
    "gate_answer": None,
    "written_screens": None,
    "written_groups": None,
    "lesson_title": None,
    "lesson_summary": None,
    "lesson": None,
    "output_folder": None,
    "saved_path": None,
}


def open_preview(index_html: str) -> None:
    """Open the exported lesson in the default browser."""
    import webbrowser
    from pathlib import Path

    webbrowser.open(Path(index_html).resolve().as_uri())


def graph_config(session: Session) -> dict:
    return {
        "configurable": {
            "thread_id": session.thread_id,
            "usage_recorder": session.record_usage,
            "index": session.index,
        }
    }


def stream_until_pause(graph, payload, config: dict, resumed_node: str | None = None) -> None:
    """Run the graph and print "[INFO] Entering Node: <name>" when each node starts.

    On resume, LangGraph runs the paused node again from the top; that re-entry is the
    first node to start, and it is not printed a second time.
    """
    skip = resumed_node
    for task in graph.stream(payload, config, stream_mode="tasks"):
        if "input" not in task:  # a task result, not a task start
            continue
        if skip is not None and task["name"] == skip:
            skip = None
            continue
        skip = None
        from agent.nodes import node_detail

        detail = node_detail(task["name"], task.get("input") or {})
        log(f"[INFO] Entering Node: {task['name']}" + (f" ({detail})" if detail else ""))


def run_graph(session: Session, graph, payload, ask_fn: Callable[[str], str] = ask, store=None) -> bool:
    """Run (payload) or continue (payload=None) the session's conversation, answering gates
    on the way. Errors are reported and never escape. Returns False if the user asked to quit."""
    start = len(session.usage)
    config = graph_config(session)
    keep_running = True
    try:
        if payload is not None:
            stream_until_pause(graph, payload, config)
        while True:
            snapshot = graph.get_state(config)
            if not snapshot.interrupts:
                break
            paused_node = snapshot.next[0] if snapshot.next else None
            resume = ask_gate(snapshot.interrupts[0].value, ask_fn, show_markdown, say)
            stream_until_pause(graph, Command(resume=resume), config, resumed_node=paused_node)
    except (GateExit, EOFError):
        say(msg("gate.paused"), style="yellow")
        return False
    except KeyboardInterrupt:
        paused = bool(graph.get_state(config).interrupts)
        say(msg("gate.paused" if paused else "cli.cancelled"), style="yellow")
        return True
    except Exception as e:
        report_error(e)
        return True
    finally:
        if store is not None:
            store.save(session)

    values = graph.get_state(config).values
    if values.get("approved"):  # the lesson flow reached its end (the save gate)
        session.lesson_type = values.get("lesson_type")
        session.topic_map = values.get("topic_map")
        session.structure = values.get("structure")
    if values.get("answer"):
        show_markdown(values["answer"])
    if values.get("saved_path"):
        say_path(values["saved_path"])
        open_preview(values["saved_path"])
    totals = session.totals(session.usage[start:])
    log(
        msg(
            "cli.cost_line",
            calls=totals.calls,
            prompt_tokens=totals.prompt_tokens,
            completion_tokens=totals.completion_tokens,
            total_tokens=totals.total_tokens,
            cost=totals.cost,
        )
    )
    return keep_running


def run_question(question: str, session: Session, graph, ask_fn: Callable[[str], str] = ask, store=None) -> bool:
    """Run one question through the graph from a fresh state."""
    return run_graph(session, graph, {**FRESH_RUN, "question": question}, ask_fn, store)


def run_turn(line: str, handler: CommandHandler, graph) -> bool:
    """Handle one input line. Returns False when the user asked to quit."""
    result = handler.handle(line)
    if result.quit:
        return False
    if result.resume:
        return run_graph(handler.session, graph, None, store=handler.store)
    if result.question:
        return run_question(result.question, handler.session, graph, store=handler.store)
    return True


def check_models() -> None:
    """Say which model profile is active, and skip configured models OpenRouter does not
    list (checked at most once a day; if the check fails, carry on without it)."""
    from agent.llm import active_profile, check_available_models
    from agent.persistence import data_dir

    log(msg("llm.profile", profile=active_profile()))
    missing = check_available_models(data_dir() / "models_cache.json")
    if missing:
        log(msg("llm.not_listed", models=", ".join(missing)))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mcp:
        # MCP server mode: stdout carries the protocol, so nothing else may run or print.
        run_server(args.mcp)
        return

    from agent.graph import build_graph  # heavy imports, not needed in MCP server mode
    from agent.persistence import ConversationStore, open_checkpointer

    configure_stdio()
    load_env()
    log_file = setup_logging(debug=args.debug)
    if args.debug:
        log(msg("cli.debug_on", log_file=str(log_file)))

    if not ensure_openrouter_key():
        say(msg("cli.no_key_warning"), style="yellow")

    tracing_enabled = setup_tracing()
    check_models()
    graph = build_graph(open_checkpointer())
    mcp = MCPManager()
    handler = CommandHandler(Session(), mcp=mcp, store=ConversationStore(), graph=graph)
    say(msg("cli.welcome"))

    while True:
        try:
            print()
            line = ask(msg("cli.question_prompt"), style="bold cyan")
        except (EOFError, KeyboardInterrupt):
            break
        try:
            if not run_turn(line, handler, graph):
                break
        except KeyboardInterrupt:
            say(msg("cli.cancelled"), style="yellow")
        except Exception as e:
            report_error(e, node="cli")

    try:
        mcp.close()
    except Exception as e:
        report_error(e, node="mcp")
    if tracing_enabled:
        try:
            flush_traces()
        except Exception as e:
            report_error(e, node="tracing")
    say(msg("cli.goodbye"))


if __name__ == "__main__":
    main()

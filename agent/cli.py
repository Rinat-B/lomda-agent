"""Interactive CLI: read a line, run a command or send it through the agent graph."""

import argparse

from agent.commands import CommandHandler
from agent.config import configure_stdio, ensure_openrouter_key, load_env
from agent.console import ask, log, say, show_markdown
from agent.errors import report_error, setup_logging
from agent.mcp_client import MCPManager, run_server
from agent.messages import msg
from agent.session import Session
from agent.tracing import flush_traces, setup_tracing


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="agent", description="E-learning lesson agent")
    parser.add_argument("--debug", action="store_true", help="show full tracebacks on screen")
    parser.add_argument("--mcp", metavar="SERVER", help="run a built-in MCP server over stdio (internal)")
    return parser.parse_args(argv)


def run_question(question: str, session: Session, graph) -> None:
    """Run one question through the graph; errors are reported and never escape."""
    start = len(session.usage)
    try:
        state = graph.invoke(
            {"question": question},
            config={"configurable": {"usage_recorder": session.record_usage}},
        )
    except KeyboardInterrupt:
        say(msg("cli.cancelled"), style="yellow")
        return
    except Exception as e:
        report_error(e)
        return

    show_markdown(state["answer"])
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


def run_turn(line: str, handler: CommandHandler, graph) -> bool:
    """Handle one input line. Returns False when the user asked to quit."""
    result = handler.handle(line)
    if result.quit:
        return False
    if result.question:
        run_question(result.question, handler.session, graph)
    return True


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mcp:
        # MCP server mode: stdout carries the protocol, so nothing else may run or print.
        run_server(args.mcp)
        return

    from agent.graph import build_graph  # heavy import, not needed in MCP server mode

    configure_stdio()
    load_env()
    log_file = setup_logging(debug=args.debug)
    if args.debug:
        log(msg("cli.debug_on", log_file=str(log_file)))

    if not ensure_openrouter_key():
        say(msg("cli.no_key_warning"), style="yellow")

    tracing_enabled = setup_tracing()
    graph = build_graph()
    mcp = MCPManager()
    handler = CommandHandler(Session(), mcp=mcp)
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

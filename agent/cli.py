"""Interactive CLI: ask questions in a loop and run each one through the agent graph."""

import sys

from rich.markup import escape

from agent.config import configure_stdio, ensure_openrouter_key, load_env
from agent.console import console, log
from agent.graph import build_graph
from agent.llm import AllModelsFailedError, MissingApiKeyError
from agent.messages import msg
from agent.tracing import flush_traces, setup_tracing

QUIT_WORDS = {"quit", "exit"}


def main() -> None:
    configure_stdio()
    load_env()
    if not ensure_openrouter_key():
        console.print(f"[red]{escape(msg('cli.missing_api_key'))}[/red]")
        sys.exit(1)

    tracing_enabled = setup_tracing()
    graph = build_graph()
    console.print(escape(msg("cli.welcome")))

    while True:
        try:
            question = console.input(f"\n[bold cyan]{escape(msg('cli.question_prompt'))}[/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in QUIT_WORDS:
            break
        if not question:
            console.print(f"[yellow]{escape(msg('cli.no_question'))}[/yellow]")
            continue

        try:
            state = graph.invoke({"question": question})
        except MissingApiKeyError:
            console.print(f"[red]{escape(msg('cli.missing_api_key'))}[/red]")
            sys.exit(1)
        except AllModelsFailedError as e:
            console.print(f"[red]{escape(msg('cli.all_models_failed', error=e))}[/red]")
            continue

        console.print(state["answer"], markup=False)
        log(
            msg(
                "cli.cost_line",
                calls=len(state.get("models_used", [])),
                prompt_tokens=state.get("prompt_tokens", 0),
                completion_tokens=state.get("completion_tokens", 0),
                total_tokens=state.get("prompt_tokens", 0) + state.get("completion_tokens", 0),
                cost=state.get("cost", 0.0),
            )
        )

    if tracing_enabled:
        flush_traces()
    console.print(escape(msg("cli.goodbye")))


if __name__ == "__main__":
    main()

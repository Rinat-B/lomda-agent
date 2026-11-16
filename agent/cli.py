"""Interactive CLI: ask questions in a loop and run each one through the agent graph."""

import os
import sys

from dotenv import load_dotenv
from rich.markup import escape

from agent.console import console, log
from agent.graph import build_graph
from agent.llm import AllModelsFailedError, MissingApiKeyError
from agent.messages import msg

QUIT_WORDS = {"quit", "exit"}


def main() -> None:
    load_dotenv()
    if not os.getenv("OPENROUTER_API_KEY"):
        console.print(f"[red]{escape(msg('cli.missing_api_key'))}[/red]")
        sys.exit(1)

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

    console.print(escape(msg("cli.goodbye")))


if __name__ == "__main__":
    main()

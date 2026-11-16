"""Interactive CLI: ask questions in a loop and run each one through the agent graph."""

import sys

from agent.config import configure_stdio, ensure_openrouter_key, load_env
from agent.console import ask, log, say, show_markdown
from agent.graph import build_graph
from agent.llm import AllModelsFailedError, MissingApiKeyError
from agent.messages import msg
from agent.tracing import flush_traces, setup_tracing

QUIT_WORDS = {"quit", "exit"}


def main() -> None:
    configure_stdio()
    load_env()
    if not ensure_openrouter_key():
        say(msg("cli.missing_api_key"), style="red")
        sys.exit(1)

    tracing_enabled = setup_tracing()
    graph = build_graph()
    say(msg("cli.welcome"))

    while True:
        try:
            print()
            question = ask(msg("cli.question_prompt"), style="bold cyan").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in QUIT_WORDS:
            break
        if not question:
            say(msg("cli.no_question"), style="yellow")
            continue

        try:
            state = graph.invoke({"question": question})
        except MissingApiKeyError:
            say(msg("cli.missing_api_key"), style="red")
            sys.exit(1)
        except AllModelsFailedError as e:
            say(msg("cli.all_models_failed", error=e), style="red")
            continue

        show_markdown(state["answer"])
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
    say(msg("cli.goodbye"))


if __name__ == "__main__":
    main()

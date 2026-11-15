"""Minimal CLI: ask one question, get an answer from an OpenRouter model, print usage/cost."""

import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.markup import escape

from agent.messages import msg

DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

SYSTEM_PROMPT = (
    "You are a helpful assistant for building e-learning lessons. "
    "Always answer in Hebrew, clearly and concisely."
)


def main() -> None:
    load_dotenv()
    console = Console()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        console.print(f"[red]{escape(msg('cli.missing_api_key'))}[/red]")
        sys.exit(1)

    model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
    llm = ChatOpenAI(model=model, api_key=api_key, base_url=OPENROUTER_BASE_URL)

    question = console.input(f"[bold cyan]{escape(msg('cli.question_prompt'))}[/bold cyan]").strip()
    if not question:
        console.print(f"[yellow]{escape(msg('cli.no_question'))}[/yellow]")
        return

    try:
        response = llm.invoke([SystemMessage(SYSTEM_PROMPT), HumanMessage(question)])
    except Exception as e:  # free models are often rate-limited (429)
        console.print(f"[red]{escape(msg('cli.request_failed', model=model, error=e))}[/red]")
        console.print(msg("cli.request_failed_tip"), markup=False)
        sys.exit(1)
    console.print(response.content, markup=False)

    usage = response.response_metadata.get("token_usage") or {}
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
    cost = usage.get("cost", 0) or 0
    console.print(
        msg(
            "cli.cost_line",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost,
        ),
        markup=False,
        soft_wrap=True,
    )


if __name__ == "__main__":
    main()

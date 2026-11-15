"""Minimal CLI: ask one question, get an answer from an OpenRouter model, print usage/cost."""

import os
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from rich.console import Console

DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def main() -> None:
    load_dotenv()
    console = Console()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        console.print("[red]OPENROUTER_API_KEY is missing. Copy .env.example to .env and set it.[/red]")
        sys.exit(1)

    model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
    llm = ChatOpenAI(model=model, api_key=api_key, base_url=OPENROUTER_BASE_URL)

    question = console.input("[bold cyan]Question:[/bold cyan] ").strip()
    if not question:
        console.print("[yellow]No question given.[/yellow]")
        return

    try:
        response = llm.invoke(question)
    except Exception as e:  # free models are often rate-limited (429)
        console.print(f"[red]Request to {model} failed:[/red] {e}")
        console.print("Tip: set OPENROUTER_MODEL in .env to another ':free' model.", markup=False)
        sys.exit(1)
    console.print(response.content)

    usage = response.response_metadata.get("token_usage") or {}
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
    cost = usage.get("cost", 0) or 0
    console.print(
        f"[COST] model={model} prompt={prompt_tokens} completion={completion_tokens} "
        f"total={total_tokens} tokens, cost=${cost:.6f}",
        markup=False,
        soft_wrap=True,
    )


if __name__ == "__main__":
    main()

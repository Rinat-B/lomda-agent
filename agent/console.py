"""Shared rich console and log-line helpers."""

from rich.console import Console

console = Console()


def log(line: str) -> None:
    """Print a plain log line (no rich markup, no wrapping)."""
    console.print(line, markup=False, soft_wrap=True)

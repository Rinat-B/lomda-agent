"""LangSmith tracing setup. LangChain/LangGraph trace automatically once these env vars are set."""

import os

from agent.console import log
from agent.messages import msg

DEFAULT_PROJECT = "lomda-agent"


def setup_tracing() -> bool:
    """Enable LangSmith tracing if an API key is present; otherwise disable it and say so once."""
    if not os.getenv("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_TRACING"] = "false"
        log(msg("tracing.disabled"))
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", DEFAULT_PROJECT)
    log(msg("tracing.enabled", project=os.environ["LANGSMITH_PROJECT"]))
    return True


def flush_traces() -> None:
    """Wait for pending traces to upload before the process exits."""
    from langchain_core.tracers.langchain import wait_for_all_tracers

    wait_for_all_tracers()

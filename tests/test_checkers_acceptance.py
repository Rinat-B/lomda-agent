"""Acceptance: the checkers find at least 80% of the hand-listed problems in a known-bad lesson.

Real model calls (role "checker"), so it is marked slow and skipped by default and in CI.
Run it with:  .venv\\Scripts\\python -m pytest -m slow -s
The full report is written to evals/results/checker_acceptance.txt.
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
REQUIRED_SHARE = 0.8

pytestmark = pytest.mark.slow


def test_checkers_find_most_known_problems():
    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY is not set")
    from evals.checker_eval import run

    issues, matched, report, text = run()  # the report is also in evals/results/checker_acceptance.txt
    caught = [i["id"] for i in issues if matched[i["id"]]]
    assert len(caught) >= REQUIRED_SHARE * len(issues), text

"""Acceptance: the checkers find at least 80% of the hand-listed problems in a known-bad lesson.

Real model calls (role "checker"), so it is marked slow and skipped by default and in CI.
Run it with:  .venv\\Scripts\\python -m pytest -m slow -s
The full report is written to evals/results/checker_acceptance_<profile>_<lesson>.txt.
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
REQUIRED_SHARE = 0.8
# questions 6 and 7 of the first lesson: more than one defensible answer
MUST_CATCH = {"known_bad_lesson.json": ("I10", "I11")}
LESSONS = ("known_bad_lesson.json", "known_bad_lesson_2.json")

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("lesson_name", LESSONS)
def test_checkers_find_most_known_problems(lesson_name):
    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY is not set")
    from evals.checker_eval import run

    issues, matched, report, text = run(lesson_name)  # also in evals/results/checker_acceptance_<profile>_<lesson>.txt
    caught = [i["id"] for i in issues if matched[i["id"]]]
    assert len(caught) >= REQUIRED_SHARE * len(issues), text
    assert set(MUST_CATCH.get(lesson_name, ())) <= set(caught), text

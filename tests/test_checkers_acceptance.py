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
# question 6 of the first lesson: more than one defensible answer. Required only in the balanced
# profile; in free the bar is the 80% share.
STRICT_PROFILE = "balanced"
MUST_CATCH = {"known_bad_lesson.json": ("I10",)}
# question 7 ("after the curve" vs "right after the curve"): a known limitation, no profile
# catches it (docs/SPEC.md, "single_answer"; evals/results/README.md). Expected to fail.
KNOWN_LIMITATION = {"known_bad_lesson.json": ("I11",)}
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
    from agent.llm import active_profile

    if active_profile() == STRICT_PROFILE:  # free models catch these only about half the time
        assert set(MUST_CATCH.get(lesson_name, ())) <= set(caught), text
        missed = [i for i in KNOWN_LIMITATION.get(lesson_name, ()) if i not in caught]
        if missed:
            pytest.xfail(f"{', '.join(missed)} (question 7): known limitation of the single-answer checker, "
                         "see docs/SPEC.md and evals/results/README.md")

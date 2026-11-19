"""Acceptance eval for the lesson checkers on a lesson with known problems.

Runs agent/checkers.py on each known-bad lesson in tests/fixtures (written from
evals/fixtures/short_winter.pptx) and matches the findings to the problems listed by hand in
tests/fixtures/known_bad_issues.json (one list per lesson). Used by tests/test_checkers_acceptance.py (slow, real
model) and runnable on its own:

    .venv\\Scripts\\python -m evals.checker_eval
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
FIRST_LESSON = "known_bad_lesson.json"
ISSUES_PATH = ROOT / "tests" / "fixtures" / "known_bad_issues.json"
SOURCE_PATH = ROOT / "evals" / "fixtures" / "short_winter.pptx"
RESULTS_DIR = ROOT / "evals" / "results"


def result_path(lesson_name: str = FIRST_LESSON) -> Path:
    """One report per profile and lesson: checker_acceptance_free_known_bad_lesson.txt, ..."""
    from agent.llm import active_profile

    return RESULTS_DIR / f"checker_acceptance_{active_profile()}_{Path(lesson_name).stem}.txt"


def lesson_names() -> list[str]:
    return [entry["lesson"] for entry in json.loads(ISSUES_PATH.read_text(encoding="utf-8"))["lessons"]]


def load_issues(lesson_name: str = FIRST_LESSON) -> list[dict]:
    for entry in json.loads(ISSUES_PATH.read_text(encoding="utf-8"))["lessons"]:
        if entry["lesson"] == lesson_name:
            return entry["issues"]
    raise KeyError(lesson_name)


def finding_matches(issue: dict, finding) -> bool:
    from agent.tools import normalize

    screens = issue.get("screens")
    if screens is not None and finding.screen not in screens:
        return False
    if finding.category not in issue["categories"]:
        return False
    keywords = issue.get("keywords")
    if not keywords:
        return True
    text = normalize(" ".join((finding.evidence, finding.problem, finding.suggestion, finding.where)))
    return any(normalize(k) in text for k in keywords)


def match_issues(issues: list[dict], findings: list) -> dict[str, list]:
    """Issue id -> the findings that match it (empty list = missed)."""
    return {i["id"]: [f for f in findings if finding_matches(i, f)] for i in issues}


def unmatched_findings(issues: list[dict], findings: list) -> list:
    return [f for f in findings if not any(finding_matches(i, f) for i in issues)]


def format_report(issues: list[dict], matched: dict[str, list], extra: list, report, seconds: float) -> str:
    caught = sum(1 for i in issues if matched[i["id"]])
    from agent.llm import active_profile

    cost = sum(c.cost for c in report.calls)
    lines = [f"profile {active_profile()}: caught {caught}/{len(issues)} ({caught / len(issues):.0%}) "
             f"in {seconds:.0f} s, {len(report.calls)} model calls, cost ${cost:.4f}"]
    if report.failed:
        lines.append(f"failed checks: {report.failed}")
    for i in issues:
        hits = matched[i["id"]]
        status = "CAUGHT" if hits else "MISSED"
        by = ", ".join(sorted({f.check for f in hits}))
        lines.append(f"{status} {i['id']} {i['description']}" + (f"  [by: {by}]" if hits else ""))
    lines.append(f"\nfindings not in the list ({len(extra)}):")
    for f in extra:
        lines.append(f"- [{f.check}/{f.category}] screen {f.screen} {f.where}: {f.problem} | {f.evidence[:100]}")
    for c in report.calls:
        lines.append(f"call: {c.model} in {c.prompt_tokens} / out {c.completion_tokens} tokens, ${c.cost:.4f}")
    return "\n".join(lines)


def run(lesson_name: str = FIRST_LESSON):
    from agent.checkers import run_checkers
    from agent.mcp_servers.source_reader import read_pptx

    lesson = json.loads((FIXTURES / lesson_name).read_text(encoding="utf-8"))
    _, chunks = read_pptx(SOURCE_PATH)
    issues = load_issues(lesson_name)
    start = time.monotonic()
    report = run_checkers(lesson, chunks)
    seconds = time.monotonic() - start
    matched = match_issues(issues, report.findings)
    text = format_report(issues, matched, unmatched_findings(issues, report.findings), report, seconds)
    path = result_path(lesson_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    return issues, matched, report, text


if __name__ == "__main__":
    from dotenv import load_dotenv

    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(ROOT / ".env")
    for name in lesson_names():
        print(f"## {name}")
        print(run(name)[3])

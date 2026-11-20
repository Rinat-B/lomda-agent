"""Stage 3 review: the evaluator's bookkeeping around agent/checkers.py.

- first evaluation: all checkers on the whole lesson; every finding gets an id and status "open";
- fix rounds (at most MAX_FIX_ROUNDS): only the screens with open findings are rewritten;
- after a fix, every finding is checked again by the checker that found it, on the rewritten
  screens only: code checks run again on the whole lesson, and each model check that had a
  finding on a rewritten screen runs again on those screens. A finding is "fixed" only when
  its checker no longer reports it (same check, screen and category); whatever the re-run
  reports that is new becomes a new open finding.
- a call budget: a fix round starts only if it fits (fix calls + re-check calls).
- coverage: the share of the source's key points (extracted once by the small model) that
  the lesson teaches, measured in code.
"""

import json
import re

from agent.checkers import (
    batch_screens,
    CODE_CHECKS,
    MODEL_CHECKS,
    Finding,
    _content_words,
    _items,
    _json_object,
    _numbers,
    _similarity,
    _text,
    block_texts,
    run_checkers,
)
from agent.guards import SOURCE_DATA_RULE, chunk_text_for_prompt, wrap_sources

MAX_FIX_ROUNDS = 2
FIX_GROUP_MAX = 4
OWNER_CHECK = "owner"
CALL_BUDGET = 29  # model calls for a whole lesson (fewer than 30)
COVERAGE_THRESHOLD = 85
POINT_MATCH = 0.7  # share of a key point's words found on one screen for it to count as taught


def finding_dict(f: Finding, fid: str, round_: int) -> dict:
    return {**f.model_dump(), "id": fid, "status": "open", "found_round": round_, "fixed_round": None}


def _next_id(findings: list[dict]) -> int:
    return 1 + max((int(f["id"][1:]) for f in findings if f["id"][1:].isdigit()), default=0)


def first_evaluation(lesson: dict, chunks: list[dict], call) -> tuple[list[dict], dict, int]:
    """(findings, failed checks, model calls) of the full check."""
    report = run_checkers(lesson, chunks, call=call)
    findings = [finding_dict(f, f"F{i}", 0) for i, f in enumerate(report.findings, 1)]
    return findings, report.failed, len(report.calls)


def is_code_finding(f: dict) -> bool:
    return f["check"] in CODE_CHECKS


def _code_key(f: dict) -> tuple:
    return f["check"], f["category"], f.get("screen"), f.get("where", ""), f.get("evidence", "")


def reconcile_code_findings(findings: list[dict], lesson: dict, round_: int) -> list[dict]:
    """Code checks again on the whole lesson: gone -> fixed, still there -> open, new -> added."""
    fresh = [f for check in CODE_CHECKS.values() for f in check(lesson)]
    fresh_keys = {_code_key(f.model_dump()) for f in fresh}
    out, open_keys = [], set()
    for f in findings:
        if is_code_finding(f) and f["status"] == "open":
            if _code_key(f) in fresh_keys:
                open_keys.add(_code_key(f))
            else:
                f = {**f, "status": "fixed", "fixed_round": round_}
        out.append(f)
    next_id = _next_id(out)
    for f in fresh:
        if _code_key(f.model_dump()) not in open_keys:
            out.append(finding_dict(f, f"F{next_id}", round_))
            next_id += 1
    return out


def actionable(findings: list[dict]) -> list[dict]:
    """Open findings tied to a screen: the writer can fix them."""
    return [f for f in findings if f["status"] == "open" and f.get("screen")]


def fix_groups(screens: list[int], max_size: int = FIX_GROUP_MAX) -> list[list[int]]:
    """Screens to rewrite, in order, a few per writer call."""
    ordered = sorted(set(screens))
    return [ordered[i:i + max_size] for i in range(0, len(ordered), max_size)]


def _model_check_name(f: dict) -> str | None:
    return f["check"] if f["check"] in MODEL_CHECKS else None


def recheck_jobs(findings: list[dict], screens: list[int]) -> list[tuple[str, tuple[int, ...]]]:
    """The model check jobs that re-check the open findings on `screens`: each check that
    found something there runs again on exactly those screens (in batches where it batches)."""
    per_check: dict[str, set[int]] = {}
    for f in findings:
        name = _model_check_name(f)
        if name and f["status"] == "open" and f.get("screen") in screens:
            per_check.setdefault(name, set()).add(f["screen"])
    jobs = []
    for name in MODEL_CHECKS:  # a stable order
        if name not in per_check:
            continue
        numbers = sorted(per_check[name])
        size = batch_screens(MODEL_CHECKS[name]["format"])
        if size:
            jobs += [(name, tuple(numbers[i:i + size])) for i in range(0, len(numbers), size)]
        else:
            jobs.append((name, tuple(numbers)))
    return jobs


def round_cost(findings: list[dict]) -> int:
    """Model calls a fix round would take: the writer calls and the re-check calls."""
    screens = sorted({f["screen"] for f in actionable(findings)})
    return len(fix_groups(screens)) + len(recheck_jobs(findings, screens))


def _same(old: dict, new: Finding) -> bool:
    return old["check"] == new.check and old.get("screen") == new.screen and old["category"] == new.category


def verify_fixes(findings: list[dict], lesson: dict, chunks: list[dict], fixed: list[int], round_: int,
                 call) -> tuple[list[dict], dict, int]:
    """After a fix round: every finding on a rewritten screen is checked again by its checker."""
    jobs = recheck_jobs(findings, fixed)  # before reconciling: the old open findings decide the jobs
    findings = reconcile_code_findings(findings, lesson, round_)
    report = run_checkers(lesson, chunks, call=call, code=False, jobs_list=jobs) if jobs else None
    failed = dict(report.failed) if report else {}
    fresh = report.findings if report else []
    failed_checks = {label.split(" ", 1)[0] for label in failed}
    out = []
    for f in findings:
        name = _model_check_name(f)
        if (name and f["status"] == "open" and f.get("screen") in fixed and name not in failed_checks
                and not any(_same(f, n) for n in fresh)):
            f = {**f, "status": "fixed", "fixed_round": round_}
        elif f["check"] == OWNER_CHECK and f["status"] == "open" and f.get("screen") in fixed:
            f = {**f, "status": "fixed", "fixed_round": round_}  # the owner sees the result at gate 3
        out.append(f)
    next_id = _next_id(out)
    for n in fresh:
        if not any(_same(f, n) and f["status"] == "open" for f in out):
            out.append(finding_dict(n, f"F{next_id}", round_))
            next_id += 1
    return out, failed, len(report.calls) if report else 0


def owner_findings(feedback: str, screen_count: int, findings: list[dict], round_: int) -> list[dict]:
    """The content owner's change request at gate 3, as findings on the screens it names
    (numbers in the text), or on every screen when it names none."""
    numbers = sorted({int(n) for n in re.findall(r"\d+", feedback) if 1 <= int(n) <= screen_count})
    screens = numbers or list(range(1, screen_count + 1))
    next_id = _next_id(findings)
    return findings + [
        {"check": OWNER_CHECK, "category": "other", "screen": s, "where": "", "evidence": "",
         "problem": feedback, "suggestion": "", "id": f"F{next_id + i}", "status": "open",
         "found_round": round_, "fixed_round": None}
        for i, s in enumerate(screens)
    ]


# ---- coverage of the source's key points --------------------------------------------------------
KEY_POINTS_PROMPT = (
    "You list the key points of a content owner's source material for an e-learning lesson: "
    "every rule, obligation, threshold or number, procedure step and condition that an employee "
    "must know. One point per fact, as a short Hebrew sentence close to the source's wording, "
    "with the place it comes from. Leave out titles, greetings and decoration. Reply with only a "
    'JSON object: {"points": [{"text": "...", "source": "<file, slide/page/paragraph>"}]}.'
) + SOURCE_DATA_RULE
KEY_POINTS_SOURCE_CHARS = 20_000


def key_points_messages(chunks: list[dict]) -> tuple[str, str]:
    lines, size = [], 0
    for c in chunks:
        loc = c["locator"]
        where = ", ".join(f"{k} {loc[k]}" for k in ("slide", "page", "paragraph") if k in loc)
        text = chunk_text_for_prompt(c)
        if size + len(text) > KEY_POINTS_SOURCE_CHARS:
            break
        lines.append(f"({loc.get('file', '')}, {where}) {text}")
        size += len(text)
    return KEY_POINTS_PROMPT, f"Source material:\n{wrap_sources(chr(10).join(lines))}"


def parse_key_points(text: str) -> list[dict]:
    points = [{"text": _text(p.get("text")), "source": _text(p.get("source"))}
              for p in _items(_json_object(text, "points"), "points")]
    points = [p for p in points if p["text"]]
    if not points:
        raise ValueError("the points list is empty")
    return points


def screen_text(screen: dict) -> str:
    return " ".join(t for b in screen["blocks"] for t in block_texts(b))


def key_point_coverage(points: list[dict], lesson: dict) -> tuple[list[dict], list[dict]]:
    """(taught points, missing points): a point is taught when one screen holds most of its
    words (in any Hebrew prefix form) and all of its numbers."""
    screens = [(_content_words(screen_text(s)), _numbers(screen_text(s))) for s in lesson["screens"]]
    taught, missing = [], []
    for p in points:
        words, numbers = _content_words(p["text"]), _numbers(p["text"])
        if not words:
            continue
        pool_hit = any(_similarity(words, sw) >= POINT_MATCH and numbers <= sn and len(words) <= len(sw)
                       for sw, sn in screens)
        (taught if pool_hit else missing).append(p)
    return taught, missing


def coverage_summary(points: list[dict] | None, lesson: dict) -> dict | None:
    if not points:
        return None
    taught, missing = key_point_coverage(points, lesson)
    total = len(taught) + len(missing)
    return {"taught": len(taught), "total": total, "missing": [p["text"] for p in missing],
            "percent": round(100 * len(taught) / total) if total else 100}


def dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False)

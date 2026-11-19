"""Stage 3 review: the evaluator's bookkeeping around agent/checkers.py.

- first evaluation: all checkers on the whole lesson; every finding gets an id and status "open";
- fix rounds (at most MAX_FIX_ROUNDS): only the screens with open findings are rewritten;
- after a fix: code checks run again on the whole lesson (their findings are reconciled by
  content), and one model call per fixed group verifies the old findings of those screens and
  reports new problems in them. Whatever is still open goes to gate 3 and report.md.
"""

import json
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.checkers import (
    CATEGORIES,
    CHECK_TIMEOUT_SECONDS,
    CODE_CHECKS,
    COMMON,
    MODEL_ATTEMPTS,
    Finding,
    _items,
    _json_object,
    _screen,
    _text,
    cited_chunks,
    render_lesson,
    render_source,
    run_checkers,
)
from agent.guards import SOURCE_DATA_RULE

MAX_FIX_ROUNDS = 2
FIX_GROUP_MAX = 4
OWNER_CHECK = "owner"


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


VERIFY_PROMPT = COMMON + (
    " Your task: some screens were rewritten to fix review findings. You get the source "
    "material, the new version of those screens, and the findings (with ids) that were reported "
    "on the old version. For every finding, say whether the new version resolves it. Then report "
    "real new problems in these screens only, in the same terms as a reviewer: language, "
    "grounding in the source, questions with more than one defensible answer, recall questions, "
    "weak distractors, repetition, illustrations that add nothing, diagram logic. Reply with only "
    'a JSON object: {"resolved": [{"id": "F1", "resolved": true or false, "reason": "<Hebrew>"}], '
    '"new_findings": [{"screen": <n>, "where": "block <n> (<type>)", "category": "<one of: '
    + ", ".join(c for c in CATEGORIES if c != "other") + '>", "evidence": "...", "problem": "...", '
    '"suggestion": "..."}]}.'
) + SOURCE_DATA_RULE


def _subset(lesson: dict, screens: list[int]) -> dict:
    """The lesson with only some screens (numbers are kept in the rendered labels)."""
    keep = set(screens)
    return {**lesson, "screens": [s if i in keep else {**s, "blocks": []}
                                  for i, s in enumerate(lesson["screens"], 1)]}


def verify_group(lesson: dict, chunks: list[dict], screens: list[int], old: list[dict], call) -> tuple[dict, list[Finding], int]:
    """One call: {finding id: (resolved, reason)} for the old findings, and new findings."""
    subset = _subset(lesson, screens)
    text = "\n".join(line for line in render_lesson(subset).splitlines())
    items = [{"id": f["id"], "screen": f.get("screen"), "where": f.get("where", ""), "category": f["category"],
              "problem": f.get("problem", ""), "quote": f.get("evidence", "")} for f in old]
    user = (f"Source material:\n{render_source(cited_chunks(subset, chunks))}\n\n"
            f"New version of screens {screens}:\n{text}\n\nFindings on the old version (JSON):\n"
            f"{json.dumps(items, ensure_ascii=False, indent=1)}")
    messages: list = [SystemMessage(VERIFY_PROMPT), HumanMessage(user)]
    error, calls = "", 0
    for _ in range(MODEL_ATTEMPTS):
        result = call(messages, timeout=CHECK_TIMEOUT_SECONDS, role="checker")
        calls += 1
        try:
            data = _json_object(result.text, "resolved")
            resolved = {str(r.get("id")): (r.get("resolved") is True or str(r.get("resolved")).lower() == "true",
                                           _text(r.get("reason"))) for r in _items(data, "resolved")}
            new = []
            for raw in data.get("new_findings") or []:
                if not isinstance(raw, dict):
                    continue
                screen = _screen(raw.get("screen"))
                if screen not in screens:
                    continue
                new.append(Finding(check="verify", category=str(raw.get("category", "other")), screen=screen,
                                   **{k: _text(raw.get(k)) for k in ("where", "evidence", "problem", "suggestion")}))
            return resolved, new, calls
        except ValueError as e:
            error = str(e)
            messages += [AIMessage(result.text),
                         HumanMessage(f"That reply was invalid: {error}. Reply again with only the JSON object.")]
    raise ValueError(f"verify: no valid reply after {MODEL_ATTEMPTS} attempts: {error}")


def verify_fixes(findings: list[dict], lesson: dict, chunks: list[dict], fixed: list[int], round_: int,
                 call) -> tuple[list[dict], dict, int]:
    """After a fix round: reconcile code findings, verify the model findings of the fixed screens."""
    findings = reconcile_code_findings(findings, lesson, round_)
    failed: dict = {}
    calls = 0
    groups = fix_groups(fixed)
    old_by_group = [[f for f in findings if f["status"] == "open" and not is_code_finding(f)
                     and f["check"] != OWNER_CHECK and f.get("screen") in g] for g in groups]
    results: list = [None] * len(groups)

    def run(i: int):
        results[i] = verify_group(lesson, chunks, groups[i], old_by_group[i], call)

    with ThreadPoolExecutor(max_workers=max(1, len(groups))) as pool:
        futures = [pool.submit(run, i) for i in range(len(groups))]
        for i, fut in enumerate(futures):
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001 - the findings simply stay open
                failed[f"verify {groups[i]}"] = f"{type(e).__name__}: {e}"
    next_id = _next_id(findings)
    for res in results:
        if res is None:
            continue
        resolved, new, n = res
        calls += n
        for k, f in enumerate(findings):
            if f["id"] in resolved and f["status"] == "open" and resolved[f["id"]][0]:
                findings[k] = {**f, "status": "fixed", "fixed_round": round_, "resolution": resolved[f["id"]][1]}
        for nf in new:
            findings.append(finding_dict(nf, f"F{next_id}", round_))
            next_id += 1
    # findings of the owner's request are done once their screens were rewritten
    for k, f in enumerate(findings):
        if f["check"] == OWNER_CHECK and f["status"] == "open" and f.get("screen") in fixed:
            findings[k] = {**f, "status": "fixed", "fixed_round": round_}
    return findings, failed, calls


def owner_findings(feedback: str, screen_count: int, findings: list[dict], round_: int) -> list[dict]:
    """The content owner's change request at gate 3, as findings on the screens it names
    (numbers in the text), or on every screen when it names none."""
    import re

    numbers = sorted({int(n) for n in re.findall(r"\d+", feedback) if 1 <= int(n) <= screen_count})
    screens = numbers or list(range(1, screen_count + 1))
    next_id = _next_id(findings)
    return findings + [
        {"check": OWNER_CHECK, "category": "other", "screen": s, "where": "", "evidence": "",
         "problem": feedback, "suggestion": "", "id": f"F{next_id + i}", "status": "open",
         "found_round": round_, "fixed_round": None}
        for i, s in enumerate(screens)
    ]


def coverage(structure: dict, lesson: dict) -> tuple[int, int]:
    """(planned content points whose sources the lesson cites, all planned points)."""
    from agent.writer import chunk_matches

    cited = [loc for s in lesson["screens"] for b in s["blocks"] for loc in b.get("sources", [])]
    points = [p for s in structure["screens"] for p in s["points"]]
    covered = sum(1 for p in points if any(chunk_matches(c, loc) or c == loc for loc in p["sources"] for c in cited))
    return covered, len(points)

"""The illustration logic checker: one model call per built illustration, with its code, its
planned description and the source excerpts of its screen (plus verified web facts).

The model does not give a verdict; it fills a table from the page, and the code turns the
table into findings (as the single-answer checker does):

- decisions: every threshold, correct answer or recommended action in the code, next to the
  source rule with the same condition; `same` is false when the threshold or the action differs,
  or when the source has no such rule (category illustration_rule);
- numbers: every number or fact shown to the learner: not in the source (a computed or converted
  number is not in the source), or shown without the condition the source ties it to
  (illustration_number);
- choices: every combination the learner can select; one the source has no data for
  (illustration_choice);
- cause_effect: what the learner changes and what the page shows (illustration_cause);
- feedback: wrong, tells the learner to drop caution, or reveals the answer after a mistake
  instead of a hint and another try (illustration_feedback);
- situation: a situation the source's rules tell apart, shown without its name
  (illustration_situation).

A finding enters the fix loop like any other: fix_screens rebuilds the page with the findings,
and the same check runs again on the new page.
"""

import json
import re

from agent.checkers import Finding, _json_object, render_source
from agent.guards import SOURCE_DATA_RULE
from agent.messages import msg

CHECK = "illustration_logic"
CATEGORIES = ("illustration_rule", "illustration_number", "illustration_choice", "illustration_cause",
              "illustration_feedback", "illustration_situation")
PAGE_BUDGET_CHARS = 16_000
ATTEMPTS = 2
TIMEOUT_SECONDS = 180  # a long page and a full table

PROMPT = (
    "You check the LOGIC of one interactive illustration from a Hebrew lesson: an HTML page with "
    "JavaScript. Compare it with the source excerpts. Do not judge looks, accessibility or code "
    "style. Read the visible text, the controls (with their options and ranges; a disabled option "
    "cannot be chosen) and every branch of the script.\n\n"
    "Fill this table, in this order:\n"
    "1. source_rules: each if-then rule in the excerpts that the illustration teaches or touches "
    "(condition, action, a short exact quote).\n"
    "2. decisions: what the page TEACHES as right: the option it calls correct, an action it "
    "recommends, a safe/unsafe verdict tied to a rule. For each: code_condition (quote the code or "
    "page), code_action, source_rule (the condition and action of the source rule for the SAME "
    "situation; 'none' if the source has no rule for it), same (true only if the threshold AND the "
    "action are those of that rule). Compare with the rule whose condition is the same situation, "
    "not with another rule that happens to contain a similar number. A recommendation the source "
    "does not give (for example to go on with no change) is same=false with source_rule 'none'. "
    "Simulation mechanics are NOT decisions: computing a result from the source's data (a crash "
    "when the gap is shorter than the stopping distance, stopping safely when it is longer, a spin "
    "when the learner breaks the rule, a post counted when it is within the visibility) belongs in "
    "cause_effect, and code that blocks a choice the source has no data for is right, not a "
    "decision. The plain complement of a source rule is that rule (if 'fewer than two posts means "
    "under 100 m', then 'two posts or more means not under 100 m' is same=true); a recommended "
    "ACTION the source does not give is not a complement.\n"
    "3. numbers: the facts about the world the page states: distances, times, thresholds, rates, "
    "years. NOT facts: a value the learner sets (a slider's position, the gap they chose), what the "
    "learner counts in the drawing, a name of the scene (a wet road), or a result the page computes "
    "from the learner's input inside the simulation. For each fact: text (as shown), in_source (the "
    "excerpts state it; a number the code converts or derives, e.g. a speed turned into metres per "
    "second, is NOT in the source), condition_missing: the source gives some numbers only under a "
    "condition (a speed, a vehicle, a road state); read ALL the visible text of the page, and if the "
    "condition appears nowhere on it, write the condition; else \"\". A condition fixed in the code "
    "but never shown to the learner is missing.\n"
    "4. choices: only selections that pick the source's DATA (for example vehicle x road picks a "
    "stopping distance). A free setting of a continuous value (a slider) is not such a choice. For "
    "each: combination, selectable (a disabled option cannot be selected), source_has_data.\n"
    "5. cause_effect: each relation the learner changes and watches. For each: cause, effect_in_page, "
    "correct (per the source and plain physics: more fog -> fewer posts visible, a wet road -> a "
    "longer stop). Trace the code: which way does the effect move when the cause grows?\n"
    "6. feedback: every message after an answer or an action. For each: case, text, correct (false "
    "only when it contradicts the source or teaches an action the source does not give; showing the "
    "consequence of breaking a rule, such as a crash or a spin, is not wrong), drops_caution (it says "
    "caution is not needed, to go on unchanged, or to skip a safety step), reveals_answer (after a "
    "WRONG answer it states the right answer or action, instead of a hint and another try; showing "
    "the result of the learner's own setting, such as a crash with the distances, is not revealing).\n"
    "7. situation: if the source's rules tell situations apart (for example the rear slides vs. "
    "the front does not respond to the wheel) and the page shows ONE of those situations without "
    "saying which (a condition that only changes a number, such as tyre tread, is not a situation), "
    "name_missing = the missing name; else \"\".\n\n"
    "Copy texts exactly as on the page (Hebrew). Write every explanation in Hebrew.\n"
    "Reply with only this JSON object:\n"
    '{"source_rules": [{"condition": "...", "action": "...", "quote": "..."}], '
    '"decisions": [{"code_condition": "...", "code_action": "...", "source_rule": "...", "same": true, "why": "..."}], '
    '"numbers": [{"text": "...", "in_source": true, "condition_missing": ""}], '
    '"choices": [{"combination": "...", "selectable": true, "source_has_data": true}], '
    '"cause_effect": [{"cause": "...", "effect_in_page": "...", "correct": true}], '
    '"feedback": [{"case": "...", "text": "...", "correct": true, "drops_caution": false, "reveals_answer": false}], '
    '"situation": {"name_missing": ""}}\n\n'
    + SOURCE_DATA_RULE
)

_STYLE = re.compile(r"<style\b.*?</style>", re.DOTALL | re.IGNORECASE)
_PATH_DATA = re.compile(r'\s(d|points)="[^"]{40,}"')


def page_for_prompt(html: str) -> str:
    """The page without CSS and long drawing data (they carry no logic)."""
    text = _PATH_DATA.sub(r' \1="..."', _STYLE.sub("<style>...</style>", html))
    return text[:PAGE_BUDGET_CHARS]


def messages(html: str, description: str, title: str, excerpts: list[dict], web_facts: list[str] | None = None):
    from langchain_core.messages import HumanMessage, SystemMessage

    facts = "\n".join(f"- {f}" for f in web_facts or []) or "(none)"
    user = (f"Illustration title: {title}\nPlanned description: {description}\n\n"
            f"Source excerpts:\n{render_source(excerpts)}\n\nVerified web facts:\n{facts}\n\n"
            f"The page:\n```html\n{page_for_prompt(html)}\n```")
    return [SystemMessage(PROMPT), HumanMessage(user)]


def _false(value) -> bool:
    return value is False or (isinstance(value, str) and value.strip().lower() in ("false", "no"))


def _true(value) -> bool:
    return value is True or (isinstance(value, str) and value.strip().lower() in ("true", "yes"))


def _rows(data: dict, key: str) -> list[dict]:
    rows = data.get(key)
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []


def findings_from_table(data: dict, screen: int | None, where: str) -> list[Finding]:
    """The code's verdict on the model's table."""
    out: list[Finding] = []

    def add(category: str, evidence: str, problem: str, suggestion: str = "") -> None:
        out.append(Finding(check=CHECK, category=category, screen=screen, where=where,
                           evidence=str(evidence or "")[:300], problem=problem, suggestion=suggestion))

    for d in _rows(data, "decisions"):
        if _false(d.get("same")):
            add("illustration_rule", d.get("code_condition", ""),
                msg("illustration_logic.rule", action=d.get("code_action", ""), rule=d.get("source_rule", "") or "-",
                    why=d.get("why", "")))
    for n in _rows(data, "numbers"):
        if _false(n.get("in_source")):
            add("illustration_number", n.get("text", ""), msg("illustration_logic.not_in_source", text=n.get("text", "")))
        elif str(n.get("condition_missing") or "").strip():
            add("illustration_number", n.get("text", ""),
                msg("illustration_logic.condition_missing", text=n.get("text", ""), condition=n["condition_missing"]))
    for c in _rows(data, "choices"):
        if _true(c.get("selectable")) and _false(c.get("source_has_data")):
            add("illustration_choice", c.get("combination", ""),
                msg("illustration_logic.no_data", combination=c.get("combination", "")))
    for c in _rows(data, "cause_effect"):
        if _false(c.get("correct")):
            add("illustration_cause", c.get("effect_in_page", ""),
                msg("illustration_logic.cause", cause=c.get("cause", ""), effect=c.get("effect_in_page", "")))
    for f in _rows(data, "feedback"):
        problems = []
        if _false(f.get("correct")):
            problems.append(msg("illustration_logic.feedback_wrong"))
        if _true(f.get("drops_caution")):
            problems.append(msg("illustration_logic.feedback_caution"))
        if _true(f.get("reveals_answer")):
            problems.append(msg("illustration_logic.feedback_reveals"))
        if problems:
            add("illustration_feedback", f.get("text", ""), " ".join(problems))
    situation = data.get("situation") if isinstance(data.get("situation"), dict) else {}
    missing = str(situation.get("name_missing") or "").strip()
    if missing:
        add("illustration_situation", "", msg("illustration_logic.situation", name=missing))
    return _merge(out)


def _merge(findings: list[Finding]) -> list[Finding]:
    """One finding per category (the problems joined), so the fix and the re-check match them."""
    by_category: dict[str, Finding] = {}
    for f in findings:
        if f.category not in by_category:
            by_category[f.category] = f
        else:
            first = by_category[f.category]
            by_category[f.category] = first.model_copy(update={"problem": f"{first.problem} | {f.problem}",
                                                               "evidence": first.evidence or f.evidence})
    return list(by_category.values())


def check_page(html: str, description: str, title: str, excerpts: list[dict], call, screen: int | None,
               where: str, web_facts: list[str] | None = None) -> tuple[list[Finding], dict | None, list]:
    """(findings, the model's table, results). One call; an invalid reply gets one retry.
    Raises ValueError if the reply is still invalid."""
    from langchain_core.messages import AIMessage, HumanMessage

    prompt = messages(html, description, title, excerpts, web_facts)
    results = []
    error = ""
    for _ in range(ATTEMPTS):
        result = call(prompt, timeout=TIMEOUT_SECONDS)
        results.append(result)
        try:
            data = _json_object(result.text, "decisions")
            return findings_from_table(data, screen, where), data, results
        except ValueError as e:
            error = str(e)
            prompt = prompt + [AIMessage(result.text), HumanMessage(
                f"That reply was invalid: {error}. Reply again with only the JSON object.")]
    raise ValueError(error)


def illustration_jobs(lesson: dict, pages: dict[str, str]) -> list[tuple[int, dict]]:
    """(screen number, illustration block) for every block whose page was built."""
    jobs = []
    for screen in lesson["screens"]:
        for block in screen["blocks"]:
            if block["type"] == "illustration" and block.get("file") in pages:
                jobs.append((int(screen["id"][1:]), block))
    return jobs


def screen_excerpts(lesson: dict, number: int, chunks: list[dict]) -> list[dict]:
    """The chunks the illustration's screen cites (its excerpts), in source order."""
    from agent.checkers import cited_chunks

    screen = next(s for s in lesson["screens"] if int(s["id"][1:]) == number)
    return cited_chunks({"screens": [screen]}, chunks)


def where_of(block: dict) -> str:
    return f"illustration {block.get('file', '')}"


def table_json(data: dict | None) -> str:
    return json.dumps(data or {}, ensure_ascii=False, indent=1)


def run_logic_checks(lesson: dict, pages: dict[str, str], chunks: list[dict], call,
                     files: list[str] | None = None, workers: int = 3) -> tuple[list[Finding], dict, int]:
    """(findings, failed {label: error}, model calls) for the built illustrations (or `files`).
    A failed check gets one more try in sequence, then is recorded, not raised."""
    from concurrent.futures import ThreadPoolExecutor

    jobs = [(n, b) for n, b in illustration_jobs(lesson, pages) if files is None or b["file"] in files]
    findings: list[Finding] = []
    failed: dict[str, str] = {}
    calls = 0

    def one(job):
        number, block = job
        return check_page(pages[block["file"]], block.get("description", ""), block.get("title", ""),
                          screen_excerpts(lesson, number, chunks), call, number, where_of(block))

    def label(job) -> str:
        return f"{CHECK} {job[0]}"

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [(job, pool.submit(one, job)) for job in jobs]
    retry = []
    for job, future in futures:
        try:
            found, _, results = future.result()
            findings += found
            calls += len(results)
        except Exception as e:  # noqa: BLE001 - one more try below
            retry.append((job, e))
    for job, _ in retry:
        try:
            found, _, results = one(job)
            findings += found
            calls += len(results)
        except Exception as e:  # noqa: BLE001 - recorded; the other checks go on
            failed[label(job)] = f"{type(e).__name__}: {e}"
    return findings, failed, calls


def logic_files(findings: list[dict], lesson: dict) -> dict[str, list[dict]]:
    """Illustration file -> its open logic findings (the pages a fix round rebuilds)."""
    files = {int(s["id"][1:]): [b["file"] for b in s["blocks"] if b["type"] == "illustration"] for s in lesson["screens"]}
    out: dict[str, list[dict]] = {}
    for f in findings:
        if f["check"] == CHECK and f["status"] == "open" and f.get("screen"):
            where = f.get("where", "")
            file = where.removeprefix("illustration ").strip() or next(iter(files.get(f["screen"], [])), "")
            if file:
                out.setdefault(file, []).append(f)
    return out

"""report.md for the content owner: coverage, review rounds, open and fixed findings,
sources per screen, models, cost. All the Hebrew text comes from messages_he.json."""

import re
from datetime import date

from agent.evaluation import COVERAGE_THRESHOLD, MAX_FIX_ROUNDS, coverage_summary
from agent.messages import msg, msg_or_none

REPORT_FILE = "report.md"


def _category(name: str) -> str:
    try:
        return msg(f"report.category.{name}")
    except KeyError:
        return name


def _one_line(text: str) -> str:
    return " ".join(str(text or "").split())


_BLOCK = re.compile(r"block (\d+)(?: \(([a-z_]+)\))?")
_ALTERNATE = re.compile(r",?\s*alternate\b")
_MAIN = re.compile(r",\s*main\b")
# labels of the checker prompts ("[screen 2, block 1, paragraph]") that models copy into quotes
_PROMPT_LABEL = re.compile(r"\[screen \d+, block \d+, [a-z_]+\]\s*(?:alternate question \(shown on retry\)|question)?:?\s*")


def where_text(where: str) -> str:
    """ "block 5 (multiple_choice), alternate" in the reader's language."""
    def block(m: re.Match) -> str:
        kind = m.group(2)
        name = msg_or_none(f"report.block_types.{kind}") if kind else None
        return msg("report.block", number=m.group(1), type=name or kind or "-")

    text = _MAIN.sub("", where or "")
    text = _ALTERNATE.sub(", " + msg("report.alternate"), text)
    return _BLOCK.sub(block, text).strip(" ,") or "-"


def quote_text(evidence: str) -> str:
    return _one_line(_PROMPT_LABEL.sub("", evidence or ""))


def check_label(label: str) -> str:
    """ "single_answer 1, 2, 3" -> the check's Hebrew name and its screens."""
    name, _, screens = label.partition(" ")
    hebrew = msg_or_none(f"report.checks.{name}") or name
    return msg("report.check_screens", check=hebrew, screens=screens) if screens else hebrew


def _finding_lines(f: dict) -> list[str]:
    category = _category(f["category"])
    if f.get("screen"):
        lines = [msg("report.item", screen=f["screen"], where=where_text(f.get("where", "")), category=category,
                     problem=_one_line(f.get("problem")))]
    else:
        lines = [msg("report.item_lesson", category=category, problem=_one_line(f.get("problem")))]
    if quote_text(f.get("evidence", "")):
        lines.append(msg("report.quote", text=quote_text(f["evidence"])[:300]))
    if f.get("suggestion"):
        lines.append(msg("report.suggestion", text=_one_line(f["suggestion"])))
    return lines


def build_report(state: dict, usage: list, source_label, elapsed_minutes: int | None = None) -> str:
    """The report text. `usage` is a list of UsageRecord; `source_label(locator)` names a source."""
    lesson = state["lesson"]
    findings = state.get("findings") or []
    review = state.get("review") or {}
    open_ = [f for f in findings if f["status"] == "open"]
    fixed = [f for f in findings if f["status"] == "fixed"]
    files = list(dict.fromkeys(loc["file"] for s in lesson["screens"] for b in s["blocks"]
                               for loc in b.get("sources", [])))
    lines = [
        msg("report.title", title=lesson["title"]), "",
        msg("report.type", type=msg(f"lesson_types.{lesson['lesson_type']}")),
        msg("report.date", date=date.today().strftime("%d.%m.%Y")),
        msg("report.files", files=", ".join(files) or "-"),
        msg("report.screens", count=len(lesson["screens"])), "",
    ]
    cov = coverage_summary(state.get("key_points"), lesson)
    lines += [msg("report.coverage_title"), ""]
    if cov:
        lines.append(msg("report.coverage", taught=cov["taught"], total=cov["total"], percent=cov["percent"]))
        if cov["percent"] < COVERAGE_THRESHOLD:
            lines.append(msg("report.coverage_low", threshold=COVERAGE_THRESHOLD))
        if cov["missing"]:
            lines += ["", msg("report.missing_title")]
            lines += [msg("report.missing_item", point=p) for p in cov["missing"]]
    else:
        lines.append(msg("report.coverage_unknown"))
    lines.append("")

    lines += [msg("report.checks_title"), "",
              msg("report.rounds", rounds=state.get("review_round") or 0, max=MAX_FIX_ROUNDS),
              msg("report.counts", found=len(findings), fixed=len(fixed), open=len(open_))]
    if state.get("writer_models"):
        lines.append(msg("report.writers", models=", ".join(dict.fromkeys(state["writer_models"]))))
    checker_models = list(dict.fromkeys(r.model for r in usage if r.node in ("evaluator",)))
    if checker_models:
        lines.append(msg("report.checkers", models=", ".join(checker_models)))
    if review.get("same_family"):
        lines.append(msg("report.same_family", families=", ".join(review["same_family"])))
    for check, error in (review.get("failed") or {}).items():
        lines.append(msg("report.failed", check=check_label(check)))
    if review.get("fix_failed"):
        lines.append(msg("report.fix_failed", screens=", ".join(str(n) for n in review["fix_failed"])))
    if review.get("budget_stop"):
        lines.append(msg("report.budget_stop", **review["budget_stop"]))
    lines.append("")

    lines += [msg("report.open_title"), ""]
    if open_:
        for f in sorted(open_, key=lambda f: (f.get("screen") or 0)):
            lines += _finding_lines(f)
    else:
        lines.append(msg("report.no_open"))
    lines += ["", msg("report.fixed_title"), ""]
    if fixed:
        for f in sorted(fixed, key=lambda f: (f.get("screen") or 0)):
            lines.append(msg("report.fixed_item", screen=f.get("screen") or "-", category=_category(f["category"]),
                             round=f.get("fixed_round") or "-", problem=_one_line(f.get("problem"))))
    else:
        lines.append(msg("report.no_fixed"))

    lines += ["", msg("report.sources_title"), ""]
    for s in lesson["screens"]:
        locs = [loc for b in s["blocks"] for loc in b.get("sources", [])]
        labels = ", ".join(dict.fromkeys(source_label(loc).strip("()") for loc in locs)) or "-"
        lines.append(msg("report.source_item", number=int(s["id"][1:]), title=s["title"], sources=labels))

    calls = len(usage)
    tokens = sum(r.prompt_tokens + r.completion_tokens for r in usage)
    cost = sum(r.cost for r in usage)
    lines += ["", msg("report.cost_title"), "", msg("report.cost", calls=calls, tokens=f"{tokens:,}", cost=cost)]
    by_node: dict[str, list] = {}
    for r in usage:
        by_node.setdefault(r.node, []).append(r)
    for node, records in by_node.items():
        lines.append(msg("report.cost_node", node=msg_or_none(f"report.nodes.{node}") or node, calls=len(records),
                         cost=sum(r.cost for r in records)))
    if elapsed_minutes is not None:
        lines += ["", msg("report.elapsed", minutes=elapsed_minutes)]
    return "\n".join(lines).rstrip() + "\n"

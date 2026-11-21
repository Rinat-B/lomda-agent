"""Interactive illustrations: one writer call per illustration builds a self-contained HTML page
(HTML, CSS and JS in one file, no libraries, no network), cleaned in code before it is saved, and
checked in the installed Edge (Code-Exec) after the owner's [Y/N].

The player loads every illustration in an <iframe sandbox="allow-scripts">, so the page runs
without access to the lesson, storage or the network. The check (Playwright, channel msedge,
network blocked) runs at 390 and 1280 px: it loads, no console errors, no sideways scrolling,
Tab reaches every control, and using a control changes something on the page.
"""

import json
import re
import tempfile
from pathlib import Path

ILLUSTRATION_TIMEOUT_SECONDS = 240
CHECK_TIMEOUT_MS = 20_000
WIDTHS = (390, 1280)
NOT_POSSIBLE = "NOT_POSSIBLE"

BUILD_PROMPT = (
    "You build one interactive illustration for a screen of a Hebrew e-learning lesson: a single, "
    "self-contained HTML file (HTML, CSS and JavaScript inline in one file). The learner does "
    "something (moves a slider, chooses an option, presses buttons, drags with the keyboard too) and "
    "sees the result change, as the illustration's description says. It must teach the screen's "
    "decision or relation, using only the facts and numbers of the screen text below; do not invent "
    "facts.\n"
    "Rules: <html dir=\"rtl\" lang=\"he\">, every visible text in Hebrew; no libraries, no external "
    "files, fonts or images, no network (no fetch, XMLHttpRequest, WebSocket, import, links to other "
    "sites) and no eval or new Function; no localStorage, sessionStorage or cookies (it runs in a "
    "sandbox). Use real <button>, <input> and <select> elements with visible labels, so that Tab and "
    "Enter/Space/arrows work; a visible focus outline. It must fit a 390 px wide frame without "
    "sideways scrolling (use max-width: 100%, flexible layout, no fixed widths over 360 px) and a "
    "height of about {height} px. Use these colours: {palette}. Show the result in text too (an "
    "aria-live region), not only as a drawing.\n"
    "If the description cannot become a real interaction where the learner acts and sees a result "
    f"(for example it only shows a card or a list), reply with only {NOT_POSSIBLE}: <short English reason>. "
    "Otherwise reply with only the HTML file, starting with <!doctype html>."
)

FIX_PROMPT = (
    "The interactive illustration you built failed checks. Fix every problem listed and reply with "
    "only the corrected, complete HTML file (same rules as before), starting with <!doctype html>.\n"
    "Problems:\n{problems}"
)

REDESIGN_PROMPT = (
    "You design an interactive illustration for a screen of a Hebrew e-learning lesson. The content "
    "owner rejected the previous idea and asked for a different one. Propose a genuinely new idea: a "
    "different thing the learner does and a different result they see, not a variation of the "
    "rejected idea. It must teach the screen's decision or relation, using only the screen text. "
    'Reply with only a JSON object: {"title": "<Hebrew title>", "description": "<Hebrew: what the '
    'learner sees and does>"}.'
)


# ---- cleaning ------------------------------------------------------------------------------------
# Removed quietly: references to outside resources that the page can live without.
_STRIP = [
    (re.compile(r"<script\b[^>]*\bsrc\s*=[^>]*>\s*</script\s*>", re.I), "an external script"),
    (re.compile(r"<link\b[^>]*>", re.I), "an external stylesheet or link"),
    (re.compile(r"<meta\b[^>]*http-equiv\s*=\s*[\"']?refresh[^>]*>", re.I), "a meta refresh"),
    (re.compile(r"<base\b[^>]*>", re.I), "a base element"),
    (re.compile(r"<(iframe|object|embed)\b[^>]*>(.*?</\1\s*>)?", re.I | re.S), "an embedded frame or object"),
]
# Refused: code that needs the network, loads code, or evaluates strings. The page is sent back.
_FORBIDDEN = [
    (re.compile(r"\bfetch\s*\("), "fetch()"),
    (re.compile(r"\bXMLHttpRequest\b"), "XMLHttpRequest"),
    (re.compile(r"\bWebSocket\b"), "WebSocket"),
    (re.compile(r"\bEventSource\b"), "EventSource"),
    (re.compile(r"\bsendBeacon\b"), "navigator.sendBeacon"),
    (re.compile(r"(?<![\w.$])import\s*[\s({*'\"]"), "import"),
    (re.compile(r"\beval\s*\("), "eval()"),
    (re.compile(r"\bnew\s+Function\b"), "new Function"),
    (re.compile(r"\bset(Timeout|Interval)\s*\(\s*[\"'`]"), "setTimeout/setInterval with a string"),
    (re.compile(r"\b(localStorage|sessionStorage|document\.cookie|indexedDB)\b"), "browser storage"),
    (re.compile(r"\bwindow\.open\s*\("), "window.open()"),
]
_EXTERNAL_URL = re.compile(r"""(?:href|src|action|formaction|xlink:href)\s*=\s*["']?\s*(?:https?:)?//""", re.I)
_CSS_URL = re.compile(r"url\(\s*[\"']?\s*(?:https?:)?//", re.I)


def sanitize(page: str) -> tuple[str, list[str], list[str]]:
    """(cleaned page, what was removed, problems that make the page unusable)."""
    removed: list[str] = []
    for pattern, what in _STRIP:
        if pattern.search(page):
            page = pattern.sub("", page)
            removed.append(what)
    problems = [what for pattern, what in _FORBIDDEN if pattern.search(page)]
    if _EXTERNAL_URL.search(page) or _CSS_URL.search(page):
        problems.append("a link or resource on another site")
    if not re.search(r"<html\b", page, re.I):
        problems.append("not a complete HTML page")
    return page, removed, problems


def extract_html(text: str) -> str | None:
    """The HTML page in a model reply (fenced or bare); None for NOT_POSSIBLE or no page."""
    if text.strip().upper().startswith(NOT_POSSIBLE):
        return None
    fenced = re.search(r"```(?:html)?\s*(.*?)```", text, re.S | re.I)
    body = fenced.group(1) if fenced else text
    start = re.search(r"<!doctype html|<html\b", body, re.I)
    return body[start.start():].strip() if start else None


def not_possible_reason(text: str) -> str | None:
    t = text.strip()
    return t.split(":", 1)[1].strip() if t.upper().startswith(NOT_POSSIBLE) and ":" in t else (
        "" if t.upper().startswith(NOT_POSSIBLE) else None)


def screen_teaching_text(screen: dict) -> str:
    parts = []
    for b in screen["blocks"]:
        if b["type"] in ("paragraph", "did_you_know"):
            parts.append(b["text"])
        elif b["type"] == "list":
            parts += [f"- {i}" for i in b["items"]]
        elif b["type"] == "stop_and_think":
            parts.append(b["explanation"])
    return "\n".join(parts)


def build_messages(screen: dict, block: dict, palette: dict) -> tuple[str, str]:
    from agent.guards import SOURCE_DATA_RULE

    system = BUILD_PROMPT.format(height=block.get("height") or 320,
                                 palette=", ".join(f"{k} {v}" for k, v in palette.items())) + SOURCE_DATA_RULE
    user = (f"Screen title: {screen['title']}\nIllustration title: {block['title']}\n"
            f"Illustration description: {block['description']}\n\nScreen text:\n<source_material>\n"
            f"{screen_teaching_text(screen)}\n</source_material>")
    return system, user


# ---- the browser check (Code-Exec) ----------------------------------------------------------------
class NoBrowser(Exception):
    """Playwright or the installed Edge is not available: the check is skipped."""


ACTIVATE_JS = """
() => {
  const controls = Array.from(document.querySelectorAll(
    'button, input, select, textarea, [tabindex]:not([tabindex="-1"]), a[href]'))
    .filter(e => !e.disabled && e.offsetParent !== null);
  return controls.length;
}
"""

SNAPSHOT_JS = """
() => document.body.innerText + '|' + Array.from(document.querySelectorAll('input, select, textarea'))
  .map(e => e.value + ':' + e.checked).join(',') + '|' + document.body.innerHTML.length
"""


def _check_page(page, url: str, width: int, height: int) -> list[str]:
    problems: list[str] = []
    errors: list[str] = []
    page.on("console", lambda m: m.type == "error" and errors.append(m.text))
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.set_viewport_size({"width": width, "height": max(height, 300)})
    page.goto(url, timeout=CHECK_TIMEOUT_MS)
    page.wait_for_timeout(300)
    overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
    if overflow > 1:
        problems.append(f"{width} px: sideways scrolling ({overflow} px too wide)")
    count = page.evaluate(ACTIVATE_JS)
    if count == 0:
        problems.append(f"{width} px: no control the learner can use (button, input, select)")
    else:
        page.evaluate("document.activeElement && document.activeElement.blur()")
        reached = set()
        for _ in range(count + 5):
            page.keyboard.press("Tab")
            reached.add(page.evaluate(
                "(() => { const e = document.activeElement; if (!e || e === document.body) return null;"
                " if (!e.dataset.probe) e.dataset.probe = String(Math.random()); return e.dataset.probe; })()"))
        reached.discard(None)
        if len(reached) < count:
            problems.append(f"{width} px: Tab reaches {len(reached)} of {count} controls")
        before = page.evaluate(SNAPSHOT_JS)
        _use_first_control(page)
        page.wait_for_timeout(300)
        if page.evaluate(SNAPSHOT_JS) == before:
            problems.append(f"{width} px: using a control changes nothing on the page")
    problems += [f"{width} px: console error: {e[:200]}" for e in dict.fromkeys(errors)]
    return problems


def _use_first_control(page) -> None:
    """Operate every control once, as a learner would with the keyboard or a click."""
    handles = page.query_selector_all("button, input, select")
    for h in handles[:6]:
        try:
            tag = h.evaluate("e => e.tagName.toLowerCase() + ':' + (e.type || '')")
            if tag.startswith("select"):
                options = h.evaluate("e => e.options.length")
                if options > 1:
                    h.select_option(index=options - 1)
            elif tag in ("input:range", "input:number"):
                h.focus()
                for _ in range(5):
                    page.keyboard.press("ArrowRight")
            elif tag.startswith("input:text") or tag.startswith("textarea"):
                h.fill("5")
            else:
                h.click(timeout=2000)
        except Exception:  # noqa: BLE001 - a control that cannot be used is caught by the change test
            continue


def check_illustrations(pages: dict[str, str], heights: dict[str, int]) -> dict[str, list[str]]:
    """{file: problems} for every page ([] = passed). Raises NoBrowser without Playwright/Edge."""
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:  # noqa: BLE001
        raise NoBrowser(f"Playwright is not available ({type(e).__name__})") from e
    results: dict[str, list[str]] = {}
    with tempfile.TemporaryDirectory(prefix="lesson-check-") as tmp:
        try:
            with sync_playwright() as p:
                try:
                    browser = p.chromium.launch(channel="msedge")
                except Exception as e:  # noqa: BLE001
                    raise NoBrowser(f"Edge is not available ({type(e).__name__})") from e
                try:
                    for file, html in pages.items():
                        path = Path(tmp) / Path(file).name
                        path.write_text(html, encoding="utf-8")
                        problems: list[str] = []
                        for width in WIDTHS:
                            context = browser.new_context(offline=True)
                            context.route("**/*", lambda route: route.continue_()
                                          if route.request.url.startswith("file:") else route.abort())
                            page = context.new_page()
                            try:
                                problems += _check_page(page, path.resolve().as_uri(), width, heights.get(file, 320))
                            except Exception as e:  # noqa: BLE001 - a page that hangs or crashes fails
                                problems.append(f"{width} px: the page did not load ({type(e).__name__})")
                            finally:
                                context.close()
                        results[file] = problems
                finally:
                    browser.close()
        except NoBrowser:
            raise
        except Exception as e:  # noqa: BLE001 - the driver itself failed
            raise NoBrowser(f"the browser check could not start ({type(e).__name__}: {e})") from e
    return results


def is_same_idea(new: dict, rejected: dict) -> bool:
    """A redesign that is the rejected idea again (the same title, or mostly the same words)."""
    from agent.checkers import _content_words, _similarity

    if normalize(new.get("title")) and normalize(new.get("title")) == normalize(rejected.get("title")):
        return True
    return _similarity(_content_words(new.get("description", "")), _content_words(rejected.get("description", ""))) >= 0.6


def normalize(text) -> str:
    return " ".join(str(text or "").split()).lower()


def parse_design(text: str) -> dict:
    from agent.checkers import _json_object

    data = _json_object(text, "description")
    title, description = str(data.get("title") or "").strip(), str(data.get("description") or "").strip()
    if len(title) < 2 or len(description) < 10:
        raise ValueError("title and description are required")
    return {"title": title, "description": description}


def dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False)

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
    "Logic: an illustration of a decision rule uses exactly the thresholds and the actions of that "
    "rule in the screen text, for the same situation (not a number borrowed from another rule), and "
    "recommends nothing the text does not say (never 'go on with no change'). It names the kind of "
    "situation it shows when the text tells situations apart (for example 'the rear slides to the "
    "left', not just 'a skid to the left'). Every number is shown with its condition (the speed, the "
    "vehicle, the road); no number of your own (no conversions or derived values). The learner can "
    "choose only values the text has data for: leave out a combination with no data. More of a cause "
    "moves the effect the right way (more fog, fewer posts seen). Feedback is true to the text, never "
    "tells the learner to drop caution, and after a wrong choice gives a hint and lets them try again, "
    "without stating the right answer: the feedback after a mistake holds only the hint, and no "
    "instruction or rule that reads like the solution is shown next to it (information that is not "
    "feedback goes in its own place, away from the answer area). A value that follows from the "
    "learner's own choice (the visibility from the number of posts seen) is computed and shown from "
    "that choice, never offered as a separate control the learner could set into a contradiction. "
    "A state of the scene (a pedal held down) is shown as text, not as a pressed button; buttons are "
    "actions the learner takes.\n"
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

# A logic finding (agent/illustration_logic.py): the page, the screen text and the findings.
LOGIC_FIX_PROMPT = (
    "The interactive illustration below teaches a screen of a Hebrew e-learning lesson, and a check "
    "against the source found logic problems in it. Fix every problem: use exactly the source's "
    "thresholds and actions for the situation shown, show every number with its condition, add no "
    "number of your own, offer only choices the source has data for, keep cause and effect the right "
    "way, compute a value that follows from the learner's choice instead of offering it as a separate "
    "control, and after a wrong choice give only a hint and another try (no instruction next to it "
    "that reads like the answer); never "
    "tell the learner to drop caution. Keep the same idea and the same technical rules (one HTML "
    "file, RTL, Hebrew, no network, keyboard). Reply with only the corrected, complete HTML file, "
    "starting with <!doctype html>.\n\nProblems:\n{problems}\n\nScreen text:\n{screen}"
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


# Tab stops, not elements: a group of radio buttons is one stop (the arrows move inside it).
ACTIVATE_JS = """
() => {
  const controls = Array.from(document.querySelectorAll(
    'button, input, select, textarea, [tabindex]:not([tabindex="-1"]), a[href]'))
    .filter(e => !e.disabled && e.offsetParent !== null);
  return new Set(controls.map((e, i) => e.type === 'radio' && e.name ? 'radio:' + e.name : 'c' + i)).size;
}
"""
FOCUS_KEY_JS = """
(() => { const e = document.activeElement; if (!e || e === document.body) return null;
  if (e.type === 'radio' && e.name) return 'radio:' + e.name;
  if (!e.dataset.probe) e.dataset.probe = String(Math.random()); return e.dataset.probe; })()
"""

# The text the learner reads as an answer: live regions (aria-live, status, alert, output) and any
# text outside the controls (labels and options are the controls' own words, not feedback).
FEEDBACK_JS = r"""
() => {
  const live = Array.from(document.querySelectorAll('[aria-live], [role=status], [role=alert], output'))
    .map(e => e.textContent.trim()).join(' | ');
  const copy = document.body.cloneNode(true);
  copy.querySelectorAll('button, select, option, input, textarea, label, script, style, [aria-live], [role=status], [role=alert], output')
    .forEach(e => e.remove());
  return live + ' || ' + copy.textContent.replace(/\s+/g, ' ').trim();
}
"""
# Everything that moves: markup with inline styles and SVG attributes, and running CSS animations.
STILL_JS = """
() => document.body.innerHTML.length + ':' + document.body.innerHTML.slice(-4000) + ':'
  + (document.getAnimations ? document.getAnimations().filter(a => a.playState === 'running').length : 0)
"""
# A button that looks chosen before the learner did anything: a state shown as a pressed button.
PRESSED_JS = r"""
() => Array.from(document.querySelectorAll('button, [role=button]'))
  .filter(e => e.getAttribute('aria-pressed') === 'true' || e.getAttribute('aria-selected') === 'true'
    || /(^|[\s_-])(active|selected|pressed|is-on|on)($|[\s_-])/i.test(e.className || ''))
  .map(e => e.textContent.trim().slice(0, 40))
"""
STILL_POLL_MS = 200
STILL_MAX_MS = 6000


def wait_until_still(page, max_ms: int = STILL_MAX_MS) -> None:
    """Wait until the page stops changing (animations by script or CSS ended), up to max_ms."""
    last, waited = None, 0
    while waited < max_ms:
        now = page.evaluate(STILL_JS)
        if now == last and now.endswith(":0"):
            return
        last = now
        page.wait_for_timeout(STILL_POLL_MS)
        waited += STILL_POLL_MS


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
            reached.add(page.evaluate(FOCUS_KEY_JS))
        reached.discard(None)
        if len(reached) < count:
            problems.append(f"{width} px: Tab reaches {len(reached)} of {count} controls")
        pressed = page.evaluate(PRESSED_JS)
        if pressed:
            problems.append(f"{width} px: a button looks chosen before the learner acted ({pressed[0]}); "
                            "show a state as text, not as a pressed button")
        wait_until_still(page)
        before = page.evaluate(FEEDBACK_JS)
        seen: list[str] = []

        def read_feedback() -> None:
            wait_until_still(page)
            seen.append(page.evaluate(FEEDBACK_JS))

        _use_first_control(page, after_each=read_feedback)
        if all(text == before for text in seen):  # a reset at the end does not hide a change on the way
            problems.append(f"{width} px: using the controls changes no text the learner reads (feedback or "
                            "the chosen state); a change in the drawing alone is not feedback")
    problems += [f"{width} px: console error: {e[:200]}" for e in dict.fromkeys(errors)]
    return problems


def _use_first_control(page, after_each=None) -> None:
    """Operate every control once, as a learner would with the keyboard or a click; after_each()
    runs after each one."""
    handles = page.query_selector_all("button, input, select")
    for h in handles[:6]:
        if after_each is not None and h.evaluate("e => e.disabled"):
            continue
        try:
            tag = h.evaluate("e => e.tagName.toLowerCase() + ':' + (e.type || '')")
            if tag.startswith("select"):
                options = h.evaluate("e => e.options.length")
                if options > 1:
                    h.select_option(index=options - 1)
            elif tag in ("input:range", "input:number"):
                h.focus()
                start = h.evaluate("e => e.value")
                for key in ("ArrowRight", "ArrowLeft"):  # in a right-to-left page the arrows are reversed
                    for _ in range(5):
                        page.keyboard.press(key)
                    if h.evaluate("e => e.value") != start:
                        break
            elif tag.startswith("input:text") or tag.startswith("textarea"):
                h.fill("5")
            else:
                h.click(timeout=2000)
        except Exception:  # noqa: BLE001 - a control that cannot be used is caught by the change test
            continue
        if after_each is not None:
            after_each()


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

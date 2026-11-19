"""The lesson player in a real browser (the installed Edge, via Playwright), from file://.

Checks at 390 px and 1280 px: loads, no console errors, no sideways scrolling, Tab reaches
every button; plus the spec rules: "next" waits for the whole screen, a wrong answer never
reveals the right one, failing lists only the wrong questions, retry shows the alternate
questions, and the certificate is created in the browser.
Skipped when Edge is not available.
"""

import json
from pathlib import Path

import pytest

from agent.lesson_export import export_lesson

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = ROOT / "examples" / "sample_lesson"
WIDTHS = (390, 1280)

playwright_api = pytest.importorskip("playwright.sync_api")


@pytest.fixture(scope="module")
def lesson() -> dict:
    return json.loads((SAMPLE_DIR / "lesson.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def lesson_url(lesson, tmp_path_factory) -> str:
    out = tmp_path_factory.mktemp("lesson")
    return export_lesson(lesson, out, source_dir=SAMPLE_DIR).resolve().as_uri()


@pytest.fixture(scope="module")
def browser():
    with playwright_api.sync_playwright() as p:
        try:
            b = p.chromium.launch(channel="msedge")
        except Exception as e:  # no Edge on this machine
            pytest.skip(f"Edge is not available: {e}")
        yield b
        b.close()


class Page:
    """A lesson page plus the console errors it produced."""

    def __init__(self, browser, url, width):
        self.page = browser.new_page(viewport={"width": width, "height": 800}, accept_downloads=True)
        self.errors: list[str] = []
        self.page.on("console", lambda m: m.type == "error" and self.errors.append(m.text))
        self.page.on("pageerror", lambda e: self.errors.append(str(e)))
        self.page.goto(url)

    def overflow(self) -> int:
        return self.page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")

    def next_disabled(self) -> bool:
        return self.page.get_attribute("#next", "aria-disabled") == "true"

    def scroll_to_end(self):
        self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        self.page.wait_for_timeout(150)

    def open_explanations(self):
        for button in self.page.locator(".think button:visible").all():
            button.click()

    def answer_questions(self, lesson, round_, correct: bool):
        """Answer every question on the current screen (right or wrong)."""
        for fieldset in self.page.locator("fieldset.mc").all():
            legend = fieldset.locator("legend").inner_text()
            q = question_shown(lesson, legend, round_)
            index = q["correct_index"] if correct else (q["correct_index"] + 1) % len(q["options"])
            fieldset.locator(".option").nth(index).click()
            fieldset.get_by_role("button").click()

    def finish_screen(self, lesson, round_, correct=True):
        self.open_explanations()
        self.answer_questions(lesson, round_, correct)
        self.scroll_to_end()
        assert not self.next_disabled()
        self.page.click("#next")
        self.page.wait_for_timeout(100)

    def close(self):
        self.page.close()


def question_shown(lesson, legend_text, round_):
    for s in lesson["screens"]:
        for b in s["blocks"]:
            if b["type"] == "multiple_choice":
                q = b["alternate"] if round_ % 2 else b
                if q["question"] in legend_text:
                    return q
    raise AssertionError(f"question not found: {legend_text}")


def all_question_texts(lesson, round_):
    return [(b["alternate"] if round_ % 2 else b) for s in lesson["screens"] for b in s["blocks"]
            if b["type"] == "multiple_choice"]


@pytest.fixture
def open_page(browser, lesson_url):
    pages = []

    def opener(width):
        p = Page(browser, lesson_url, width)
        pages.append(p)
        return p

    yield opener
    for p in pages:
        p.close()


# ---- basics at both widths ----------------------------------------------------------------------
@pytest.mark.parametrize("width", WIDTHS)
def test_every_screen_loads_without_errors_or_sideways_scrolling(open_page, lesson, width):
    p = open_page(width)
    assert p.page.inner_text("#lesson-title") == lesson["title"]
    for _ in lesson["screens"]:
        assert p.overflow() <= 0, "horizontal scrolling"
        p.finish_screen(lesson, 0)
    assert p.page.locator("h2").inner_text()  # the summary screen
    assert p.overflow() <= 0
    assert p.errors == []


@pytest.mark.parametrize("width", WIDTHS)
def test_tab_reaches_every_button(open_page, lesson, width):
    p = open_page(width)
    for screen_number in range(len(lesson["screens"])):
        buttons = p.page.evaluate(
            "Array.from(document.querySelectorAll('button')).filter(b => !b.hidden && b.offsetParent)"
            ".map((b, i) => { b.dataset.tabProbe = String(i); return String(i); })")
        reached = set()
        p.page.evaluate("document.activeElement && document.activeElement.blur()")
        for _ in range(len(buttons) * 3 + 25):
            p.page.keyboard.press("Tab")
            probe = p.page.evaluate("document.activeElement && document.activeElement.dataset.tabProbe")
            if probe is not None:
                reached.add(probe)
        assert reached == set(buttons), f"screen {screen_number + 1}: not reachable by Tab: {set(buttons) - reached}"
        p.finish_screen(lesson, 0)


# ---- spec rules -----------------------------------------------------------------------------------
def test_next_waits_for_the_whole_screen(open_page, lesson):
    p = open_page(390)
    assert p.next_disabled()
    # aria-disabled keeps the button focusable (so its hint is announced); a click is ignored.
    p.page.click("#next", force=True)
    assert p.page.inner_text("#screen-title") == lesson["screens"][0]["title"]
    assert p.page.inner_text("#next-hint")  # says why the button is locked
    p.scroll_to_end()
    assert p.next_disabled()  # the stop-and-think explanation is still closed
    p.open_explanations()
    assert not p.next_disabled()


def test_a_question_must_be_checked_before_next(open_page, lesson):
    p = open_page(1280)
    p.finish_screen(lesson, 0)  # screen 1
    p.scroll_to_end()
    assert p.next_disabled()  # screen 2 has an unanswered question
    p.answer_questions(lesson, 0, correct=True)
    assert not p.next_disabled()


def test_a_wrong_answer_never_reveals_the_right_one(open_page, lesson):
    p = open_page(390)
    p.finish_screen(lesson, 0)
    fieldset = p.page.locator("fieldset.mc").first
    q = question_shown(lesson, fieldset.locator("legend").inner_text(), 0)
    wrong = (q["correct_index"] + 1) % len(q["options"])
    fieldset.locator(".option").nth(wrong).click()
    fieldset.get_by_role("button").click()
    assert fieldset.locator(".feedback").inner_text() == q["feedback_incorrect"]
    assert q["feedback_correct"] not in p.page.content()
    marked = fieldset.locator(".option.chosen")
    assert marked.count() == 1 and q["options"][wrong] in marked.inner_text()
    correct_option = fieldset.locator(".option").nth(q["correct_index"])
    assert correct_option.get_attribute("class") == "option"  # nothing marks the right answer
    assert fieldset.locator("input:not([disabled])").count() == 0  # locked: no second try


def test_failing_lists_only_the_wrong_questions_and_retry_uses_alternates(open_page, lesson):
    p = open_page(390)
    for _ in lesson["screens"]:
        p.finish_screen(lesson, 0, correct=False)
    result = p.page.locator(".result-box").inner_text()
    questions = all_question_texts(lesson, 0)
    for q in questions:
        assert q["question"] in result  # which questions were wrong
        assert q["options"][q["correct_index"]] not in result  # but never the right answers
    assert p.page.locator(".wrong-list li").count() == len(questions)
    assert p.page.locator("#certificate").count() == 0

    p.page.click("#retry")
    assert p.page.inner_text("#screen-title") == lesson["screens"][0]["title"]
    p.finish_screen(lesson, 1)
    legend = p.page.locator("fieldset.mc legend").first.inner_text()
    alternate = lesson["screens"][1]["blocks"][2]["alternate"]["question"]
    assert alternate in legend  # the retry shows the alternate question
    assert p.page.locator(".option.chosen").count() == 0  # answers were reset


def test_passing_offers_a_certificate_created_in_the_browser(open_page, lesson):
    p = open_page(1280)
    for _ in lesson["screens"]:
        p.finish_screen(lesson, 0, correct=True)
    assert p.page.locator(".wrong-list").count() == 0
    p.page.click("#certificate")  # no name yet
    assert p.page.locator(".cert .feedback").inner_text()
    p.page.fill(".cert input", "Dana Levi")
    with p.page.expect_download() as info:
        p.page.click("#certificate")
    download = info.value
    assert download.suggested_filename == "certificate.png"
    assert Path(download.path()).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert p.errors == []


def test_decision_diagram_is_drawn_as_svg_with_a_text_alternative(open_page, lesson):
    p = open_page(390)
    p.finish_screen(lesson, 0)
    p.finish_screen(lesson, 0)
    diagram = next(b for b in lesson["screens"][2]["blocks"] if b["type"] == "decision_diagram")
    svg = p.page.locator(".diagram svg")
    assert svg.get_attribute("role") == "img"
    texts = p.page.evaluate("Array.from(document.querySelectorAll('.diagram svg text')).map(t => t.textContent)")
    joined = " ".join(texts)
    for node in diagram["nodes"]:
        for word in node["label"].split()[:2]:
            assert word in joined, word
    steps = p.page.locator(".diagram ol.sr-only li")
    assert steps.count() == len(diagram["edges"])
    assert svg.bounding_box()["width"] <= 390


def test_edge_labels_never_overlap_on_a_phone(browser, lesson, tmp_path):
    """A wide diagram (1 question, 4 actions, long labels) at 390 px: labels stay apart."""
    import copy

    data = copy.deepcopy(lesson)
    diagram = next(b for b in data["screens"][2]["blocks"] if b["type"] == "decision_diagram")
    diagram["nodes"] = [{"id": "q", "label": "What is skidding?", "kind": "question"}] + [
        {"id": f"a{i}", "label": f"Action number {i} for this case", "kind": "action"} for i in range(1, 5)]
    diagram["edges"] = [{"from": "q", "to": f"a{i}", "label": f"Long label for case {i}"} for i in range(1, 5)]
    url = export_lesson(data, tmp_path / "wide", source_dir=SAMPLE_DIR).resolve().as_uri()
    p = Page(browser, url, 390)
    try:
        p.finish_screen(data, 0)
        p.finish_screen(data, 0)
        boxes = [r.bounding_box() for r in p.page.locator(".diagram rect.edge-label").all()]
        assert len(boxes) == 4

        def overlap(a, b):
            return not (a["x"] + a["width"] <= b["x"] or b["x"] + b["width"] <= a["x"]
                        or a["y"] + a["height"] <= b["y"] or b["y"] + b["height"] <= a["y"])

        assert not any(overlap(a, b) for i, a in enumerate(boxes) for b in boxes[i + 1:])
        assert p.overflow() <= 0 and p.errors == []
    finally:
        p.close()


def test_question_can_be_answered_with_the_keyboard_only(open_page, lesson):
    p = open_page(390)
    p.finish_screen(lesson, 0)
    radio = p.page.locator("fieldset.mc input[type=radio]").first
    radio.focus()
    p.page.keyboard.press("Space")
    p.page.keyboard.press("Tab")  # from the radio group to the check button
    p.page.keyboard.press("Enter")
    assert p.page.locator("fieldset.mc .feedback").inner_text()

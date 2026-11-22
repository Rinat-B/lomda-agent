"""Interactive illustrations: the cleaner, the build node, the browser check paths, the redesign
at gate 3, and the player's sandbox. The real Edge check runs only when Edge is installed."""

import json
from pathlib import Path

import pytest

import agent.nodes as nodes
from agent import illustrations as il
from agent.messages import msg
from tests.conftest import make_result
from tests.test_lesson_flow import FAKE_ILLUSTRATION

ROOT = Path(__file__).resolve().parent.parent


# ---- cleaning ------------------------------------------------------------------------------------
def test_clean_page_passes_and_svg_namespace_is_allowed():
    page = FAKE_ILLUSTRATION.replace("<p", '<svg xmlns="http://www.w3.org/2000/svg"></svg><p')
    cleaned, removed, problems = il.sanitize(page)
    assert problems == [] and removed == [] and cleaned == page


def test_outside_resources_are_removed():
    page = ('<!doctype html><html><head><link rel="stylesheet" href="https://cdn.x/s.css">'
            '<script src="https://cdn.x/lib.js"></script><meta http-equiv="refresh" content="0;url=x"></head>'
            '<body><iframe src="https://x"></iframe><button>a</button></body></html>')
    cleaned, removed, problems = il.sanitize(page)
    assert "cdn.x" not in cleaned and "<iframe" not in cleaned and "refresh" not in cleaned
    assert len(removed) == 4 and problems == []


@pytest.mark.parametrize("code, what", [
    ("fetch('/x')", "fetch()"),
    ("new XMLHttpRequest()", "XMLHttpRequest"),
    ("new WebSocket('ws://x')", "WebSocket"),
    ("import('./m.js')", "import"),
    ("eval('1+1')", "eval()"),
    ("new Function('return 1')", "new Function"),
    ("localStorage.setItem('a', 1)", "browser storage"),
    ("setTimeout('go()', 10)", "setTimeout/setInterval with a string"),
])
def test_network_and_eval_code_is_refused(code, what):
    page = f"<!doctype html><html><body><button>a</button><script>{code}</script></body></html>"
    assert what in il.sanitize(page)[2]


def test_links_to_other_sites_are_refused():
    for page in ('<html><body><a href="https://evil.example">x</a></body></html>',
                 '<html><body><img src="//evil.example/p.png"></body></html>',
                 '<html><style>body{background:url(https://evil.example/b.png)}</style></html>'):
        assert "a link or resource on another site" in il.sanitize(page)[2]


def test_extract_html_and_not_possible():
    assert il.extract_html("Here:\n```html\n<!doctype html><html></html>\n```").startswith("<!doctype html>")
    assert il.extract_html("NOT_POSSIBLE: it is a list") is None
    assert il.not_possible_reason("NOT_POSSIBLE: it is a list") == "it is a list"
    assert il.not_possible_reason("<!doctype html>") is None


# ---- the build node ------------------------------------------------------------------------------
def _screen(n, illustration=True):
    blocks = [{"type": "paragraph", "text": f"Rule {n}.", "sources": [{"file": "d.pptx", "slide": 1}]}]
    if illustration:
        blocks.append({"type": "illustration", "title": f"Ill {n}", "description": f"Learner moves slider {n}",
                       "file": f"assets/illustrations/s{n}.html", "height": 320})
    return {"id": f"s{n}", "title": f"S{n}", "topic": "T", "blocks": blocks}


def _state(count):
    screens = [_screen(n) for n in range(1, count + 1)]
    plan = {"lesson_type": "training", "screens": [{"number": n, "illustration": f"idea {n}"} for n in range(1, count + 1)],
            "motifs": []}
    return {"written_screens": screens, "structure": plan, "lesson_type": "training"}


class Replies:
    def __init__(self, *texts):
        self.texts = list(texts)
        self.prompts = []

    def __call__(self, messages, **kw):
        self.prompts.append(messages)
        return make_result(text=self.texts.pop(0) if self.texts else FAKE_ILLUSTRATION)


def test_each_illustration_gets_its_own_call_and_the_cap_removes_the_rest(monkeypatch):
    model = Replies()
    monkeypatch.setattr(nodes, "call_llm", model)
    monkeypatch.setattr(nodes, "illustration_cap", lambda: 3)
    out = nodes.build_illustrations(_state(4), {})
    assert len(model.prompts) == 3  # one call per illustration, up to the cap
    assert sorted(out["illustrations"]) == [f"assets/illustrations/s{n}.html" for n in (1, 2, 3)]
    assert out["illustration_notes"]["assets/illustrations/s4.html"]["status"] == "removed"
    assert not any(b["type"] == "illustration" for b in out["written_screens"][3]["blocks"])
    assert out["structure"]["screens"][3]["illustration"] is None  # later rewrites do not add it back


def test_a_dirty_page_gets_one_retry_then_is_used(monkeypatch):
    dirty = FAKE_ILLUSTRATION.replace("<script>", "<script>fetch('https://x');")
    model = Replies(dirty, FAKE_ILLUSTRATION)
    monkeypatch.setattr(nodes, "call_llm", model)
    out = nodes.build_illustrations(_state(1), {})
    assert len(model.prompts) == 2 and "fetch()" in model.prompts[1][-1].content  # the retry names the problem
    assert out["illustrations"]["assets/illustrations/s1.html"] == FAKE_ILLUSTRATION


def test_a_description_that_cannot_be_interactive_removes_the_block(monkeypatch):
    monkeypatch.setattr(nodes, "call_llm", Replies("NOT_POSSIBLE: only shows a card"))
    out = nodes.build_illustrations(_state(1), {})
    note = out["illustration_notes"]["assets/illustrations/s1.html"]
    assert note["status"] == "removed" and "only shows a card" in note["reason"]
    assert out["illustrations"] == {}


def test_still_dirty_after_the_retry_becomes_static(monkeypatch):
    dirty = FAKE_ILLUSTRATION.replace("<script>", "<script>eval('1');")
    monkeypatch.setattr(nodes, "call_llm", Replies(dirty, dirty))
    out = nodes.build_illustrations(_state(1), {})
    assert out["illustration_notes"]["assets/illustrations/s1.html"]["status"] == "static"
    assert out["illustrations"] == {}


# ---- the browser check paths ---------------------------------------------------------------------
def _checked_state():
    state = _state(2)
    state["illustrations"] = {f"assets/illustrations/s{n}.html": FAKE_ILLUSTRATION for n in (1, 2)}
    state["illustration_notes"] = {f"assets/illustrations/s{n}.html": {"screen": n, "title": f"Ill {n}", "status": "built"}
                                   for n in (1, 2)}
    return state


def _answer(monkeypatch, confirm):
    monkeypatch.setattr(nodes, "interrupt", lambda request: {"confirm": confirm})


def test_n_skips_the_check_and_marks_it(monkeypatch):
    _answer(monkeypatch, False)
    called = []
    monkeypatch.setattr(il, "check_illustrations", lambda *a: called.append(a))
    out = nodes.browser_check(_checked_state(), {})
    assert called == [] and {n["check"] for n in out["illustration_notes"].values()} == {"not_checked"}


def test_no_browser_is_a_warning_not_a_crash(monkeypatch, output):
    _answer(monkeypatch, True)

    def no_browser(*a):
        raise il.NoBrowser("Edge is not available")

    monkeypatch.setattr(il, "check_illustrations", no_browser)
    out = nodes.browser_check(_checked_state(), {})
    assert {n["check"] for n in out["illustration_notes"].values()} == {"no_browser"}
    assert "Edge is not available" in output()


def test_a_failure_gets_one_fix_then_the_static_description(monkeypatch):
    _answer(monkeypatch, True)
    calls = []

    def check(pages, heights):
        calls.append(sorted(pages))
        return {f: (["390 px: no control the learner can use"] if f.endswith("s2.html") else []) for f in pages}

    monkeypatch.setattr(il, "check_illustrations", check)
    model = Replies(FAKE_ILLUSTRATION)  # the fix, which still fails
    monkeypatch.setattr(nodes, "call_llm", model)
    state = _checked_state()
    out = nodes.browser_check(state, {})
    notes = out["illustration_notes"]
    assert notes["assets/illustrations/s1.html"]["check"] == "passed"
    assert notes["assets/illustrations/s2.html"]["check"] == "failed"
    assert "assets/illustrations/s2.html" not in out["illustrations"]  # the player shows the description
    assert len(model.prompts) == 1 and "no control" in model.prompts[0][0].content  # one fix with the log
    summary = nodes.review_summary({**state, **out, "lesson": {"title": "T", "lesson_type": "training",
                                                                 "screens": state["written_screens"]}})
    assert any("Ill 2" in o["problem"] for o in summary["open"])  # shown at gate 3


def test_nothing_to_check_asks_nothing(monkeypatch):
    monkeypatch.setattr(nodes, "interrupt", lambda request: pytest.fail("must not ask"))
    assert nodes.browser_check({"illustrations": {}}, {}) == {}


# ---- redesign at gate 3 --------------------------------------------------------------------------
def test_another_illustration_idea_must_differ_from_the_rejected_one(monkeypatch):
    same = json.dumps({"title": "Ill 1", "description": "Learner moves slider 1"})
    new = json.dumps({"title": "Choose the gap", "description": "The learner picks a following distance and sees a car stop"})
    model = Replies(same, new, FAKE_ILLUSTRATION)
    monkeypatch.setattr(nodes, "call_llm", model)
    state = {**_state(1), "feedback": "x"}
    out = nodes._redesign_illustrations(state, {}, [1])
    block = out["written_screens"][0]["blocks"][1]
    assert block["title"] == "Choose the gap" and out["redesigned"] == [1]
    assert "rejected idea again" in model.prompts[1][-1].content  # the explicit note on the second try
    assert out["illustration_notes"]["assets/illustrations/s1.html"]["check"] == "not_checked"


def test_the_owner_illustration_request_is_recognised():
    word = msg("gate.glossary.review.illustration")
    assert nodes.owner_asks_for_illustration(f"x {word} y 2")
    assert not nodes.owner_asks_for_illustration("make screen 2 shorter")


# ---- the player and the export -------------------------------------------------------------------
def test_the_player_sandboxes_illustrations_with_scripts_only():
    js = (ROOT / "agent" / "assets" / "player" / "player.js").read_text(encoding="utf-8")
    assert 'sandbox: "allow-scripts"' in js and "allow-same-origin" not in js


def test_export_writes_the_built_page(tmp_path):
    from agent.lesson_export import export_lesson

    lesson = json.loads((ROOT / "examples" / "sample_lesson" / "lesson.json").read_text(encoding="utf-8"))
    file = next(b["file"] for s in lesson["screens"] for b in s["blocks"] if b["type"] == "illustration")
    export_lesson(lesson, tmp_path, placeholders=True, illustrations={file: FAKE_ILLUSTRATION})
    assert (tmp_path / file).read_text(encoding="utf-8") == FAKE_ILLUSTRATION


# ---- the real check in Edge ----------------------------------------------------------------------
def test_the_real_browser_check_in_edge():
    try:
        results = il.check_illustrations({"good.html": FAKE_ILLUSTRATION,
                                          "static.html": '<!doctype html><html><body><p>only text</p></body></html>',
                                          "wide.html": FAKE_ILLUSTRATION.replace("<body>", '<body><div style="width:2000px">x</div>')},
                                         {})
    except il.NoBrowser as e:
        pytest.skip(str(e))
    assert results["good.html"] == []
    assert any("no control" in p for p in results["static.html"])
    assert any("sideways" in p for p in results["wide.html"])


def test_a_radio_group_is_one_tab_stop_in_the_real_check():
    page = ('<!doctype html><html dir="rtl" lang="he"><body>'
            '<label><input type="radio" name="r" value="a" checked> a</label>'
            '<label><input type="radio" name="r" value="b"> b</label>'
            '<button id="b">go</button><p id="out">0</p>'
            '<script>document.getElementById("b").onclick = () => {document.getElementById("out").textContent = "1"};'
            '</script></body></html>')
    try:
        results = il.check_illustrations({"radio.html": page}, {})
    except il.NoBrowser as e:
        pytest.skip(str(e))
    assert results["radio.html"] == []


def test_the_browser_tab_shows_the_lesson_title_and_no_english(tmp_path):
    import re
    from html.parser import HTMLParser

    from agent.lesson_export import export_lesson

    lesson = json.loads((ROOT / "examples" / "sample_lesson" / "lesson.json").read_text(encoding="utf-8"))
    index = export_lesson(lesson, tmp_path, placeholders=True)
    page = index.read_text(encoding="utf-8")
    assert f"<title>{lesson['title']}</title>" in page

    class Text(HTMLParser):
        words: list[str] = []
        skip = False

        def handle_starttag(self, tag, attrs):
            self.skip = tag in ("script", "style")

        def handle_data(self, data):
            if not self.skip:
                self.words += re.findall(r"[A-Za-z]{2,}", data)

    for name in ("index.html", *(b["file"] for s in lesson["screens"] for b in s["blocks"] if b["type"] == "illustration")):
        parser = Text()
        parser.words = []
        parser.feed((tmp_path / name).read_text(encoding="utf-8"))
        assert parser.words == [], f"{name}: {parser.words}"


def _real_check(page: str) -> list[str]:
    try:
        return il.check_illustrations({"p.html": page}, {})["p.html"]
    except il.NoBrowser as e:
        pytest.skip(str(e))


def test_a_change_in_the_drawing_alone_is_not_feedback():
    page = ('<!doctype html><html dir="rtl" lang="he"><body><button id="b">go</button>'
            '<svg><rect id="r" width="10" height="10"/></svg>'
            '<script>document.getElementById("b").onclick = () => '
            'document.getElementById("r").setAttribute("width", "50");</script></body></html>')
    assert any("changes no text" in p for p in _real_check(page))


def test_a_state_shown_as_a_pressed_button_is_reported():
    page = ('<!doctype html><html dir="rtl" lang="he"><body>'
            '<button aria-pressed="true" id="g">gas: pressed</button><p id="o" aria-live="polite">x</p>'
            '<script>document.getElementById("g").onclick = () => document.getElementById("o").textContent = "y";'
            '</script></body></html>')
    assert any("looks chosen" in p for p in _real_check(page))


def test_the_feedback_is_read_after_the_animation_ends():
    """The answer appears only after a 1.5 s animation (requestAnimationFrame)."""
    page = ('<!doctype html><html dir="rtl" lang="he"><body><button id="b">go</button>'
            '<div id="car" style="left:0;position:relative">car</div><p id="o" aria-live="polite"></p><script>'
            'document.getElementById("b").onclick = () => { const t0 = performance.now();'
            ' function step(t) { const p = Math.min((t - t0) / 1500, 1);'
            '  document.getElementById("car").style.left = (p * 100) + "px";'
            '  if (p < 1) requestAnimationFrame(step); else document.getElementById("o").textContent = "stopped"; }'
            ' requestAnimationFrame(step); };</script></body></html>')
    assert _real_check(page) == []


def test_a_slider_in_a_right_to_left_page_is_moved_both_ways():
    page = ('<!doctype html><html dir="rtl" lang="he"><body>'
            '<input type="range" id="s" min="1" max="8" value="1"><p id="o" aria-live="polite">1</p>'
            '<script>document.getElementById("s").oninput = e => document.getElementById("o").textContent = e.target.value;'
            '</script></body></html>')
    assert _real_check(page) == []

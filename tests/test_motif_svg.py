"""The SVG cleaner (output guard) and the motif: a 4-part contract drawn from bundled Lucide icons,
two candidates from two models, the owner's choice at gate 2, and the motif in lesson.json."""

import json

import pytest

from agent import motif
from agent.gates import ask_motif_choice, format_motifs
from agent.lesson_schema import PALETTES
from agent.messages import msg
from agent.svg_clean import UnsafeSvg, clean_svg, is_clean
from tests.conftest import make_result

NS = 'xmlns="http://www.w3.org/2000/svg"'


# ---- the cleaner -----------------------------------------------------------------------------------
def test_a_clean_drawing_is_kept():
    svg = f'<svg {NS} viewBox="0 0 24 24"><g fill="none" stroke="#1D63A0"><path d="M1 1h4"/><circle cx="7" cy="7" r="2"/></g></svg>'
    cleaned, removed = clean_svg(svg)
    assert removed == [] and "<path" in cleaned and "<circle" in cleaned


@pytest.mark.parametrize("bad, gone", [
    ("<script>alert(1)</script>", "<script"),
    ('<foreignObject><div>x</div></foreignObject>', "foreignObject"),
    ('<image href="data:image/webp;base64,AAAA"/>', "<image"),
    ('<a href="https://evil.example"><path d="M1 1"/></a>', "evil.example"),
    ('<path d="M1 1" onclick="steal()"/>', "onclick"),
    ('<use href="https://evil.example/x.svg#a"/>', "evil.example"),
    ('<rect width="1" height="1" fill="url(https://evil.example/p)"/>', "evil.example"),
    ('<path d="M1 1" style="fill:url(javascript:x)"/>', "javascript"),
])
def test_scripts_events_foreign_objects_and_outside_links_are_removed(bad, gone):
    cleaned, removed = clean_svg(f"<svg {NS}>{bad}</svg>")
    assert gone not in cleaned and removed


def test_an_inner_link_to_the_same_svg_is_kept():
    svg = f'<svg {NS}><defs><linearGradient id="g"/></defs><rect width="1" height="1" fill="url(#g)"/><use href="#g"/></svg>'
    assert is_clean(svg)


@pytest.mark.parametrize("bad", ['<!DOCTYPE svg [<!ENTITY x "y">]><svg/>', "<svg><unclosed></svg>", "<html/>"])
def test_entities_broken_xml_and_other_roots_are_refused(bad):
    with pytest.raises(UnsafeSvg):
        clean_svg(bad)


def test_every_bundled_lucide_icon_is_clean():
    for name, body in motif.icons().items():
        assert is_clean(f'<svg {NS} viewBox="0 0 24 24">{body}</svg>'), name


# ---- the motif -------------------------------------------------------------------------------------
def test_the_contract_has_four_parts_and_only_bundled_icons():
    c = motif.parse_contract('{"idea": "דרך בחורף", "icons": ["car", "route"], "accent": "primary", "layout": "pair"}')
    assert (c.idea, c.icons, c.accent, c.layout) == ("דרך בחורף", ["car", "route"], "primary", "pair")
    with pytest.raises(ValueError):
        motif.parse_contract('{"idea": "x x x x", "icons": ["no-such-icon"], "accent": "primary", "layout": "badge"}')
    single = motif.parse_contract('{"idea": "דרך בחורף", "icons": ["car", "route"], "layout": "badge"}')
    assert single.icons == ["car"]  # a badge draws one icon
    pair_of_one = motif.parse_contract('{"idea": "דרך בחורף", "icons": ["car"], "layout": "pair"}')
    assert pair_of_one.layout == "badge"


@pytest.mark.parametrize("layout, icons", [("single", ["car"]), ("badge", ["sun"]), ("pair", ["car", "route"])])
def test_the_drawing_uses_the_palette_and_passes_the_cleaner(layout, icons):
    for lesson_type, palette in PALETTES.items():
        c = motif.MotifContract(idea="רעיון קצר", icons=icons, accent="accent", layout=layout)
        svg = motif.render(c, palette)
        assert is_clean(svg) and palette["accent"] in svg
        assert all(colour.upper() in {v.upper() for v in palette.values()} | {"NONE"}
                   for colour in __import__("re").findall(r'(?:fill|stroke)="([^"]+)"', svg)), lesson_type


def test_two_candidates_come_from_two_model_families_in_parallel():
    asked = []

    def call(messages, model):
        asked.append(model)
        return make_result(text='{"idea": "דרך בחורף", "icons": ["car"], "accent": "primary", "layout": "badge"}', model=model)

    writers = ["baidu/a:free", "baidu/b:free", "nvidia/c:free"]
    found, failed = motif.candidates("נהיגה בחורף", ["רכב"], "training", PALETTES["training"], call, writers)
    assert sorted(asked) == ["baidu/a:free", "nvidia/c:free"] and failed == {}
    assert len(found) == 2 and all(f["svg"].startswith("<svg") for f in found)


def test_an_invalid_candidate_gets_one_retry_and_a_failed_one_does_not_stop_the_other():
    replies = {"x/a": ['{"idea": "x"}', '{"idea": "דרך בחורף", "icons": ["sun"], "layout": "single"}'],
               "y/b": ["nonsense", "still nonsense"]}

    def call(messages, model):
        return make_result(text=replies[model].pop(0), model=model)

    found, failed = motif.candidates("t", [], "training", PALETTES["training"], call, ["x/a", "y/b"])
    assert [f["model"] for f in found] == ["x/a"] and "y/b" in failed


def test_gate_2_lists_the_candidates_and_reads_the_choice():
    cands = [{"description": "א", "icons": ["car"]}, {"description": "ב", "icons": ["sun"]}]
    text = format_motifs(cands)
    assert msg("gate.structure.motif_candidate", index=2, text="ב", icons="sun") in text
    assert ask_motif_choice(2, lambda p: "2", lambda *a, **k: None) == 1
    assert ask_motif_choice(2, lambda p: "", lambda *a, **k: None) == 0
    assert ask_motif_choice(2, lambda p: "Y", lambda *a, **k: None) == 0
    assert ask_motif_choice(2, lambda p: "7", lambda *a, **k: None) == 0  # gives up after 3 unclear answers


def test_the_chosen_motif_reaches_the_exported_lesson_cleaned(tmp_path):
    from agent.lesson_export import export_lesson

    lesson = json.loads(open("examples/sample_lesson/lesson.json", encoding="utf-8").read())
    lesson["motif"] = {"description": "x", "svg": f'<svg {NS}><path d="M1 1"/><script>alert(1)</script></svg>'}
    export_lesson(lesson, tmp_path, placeholders=True)
    js = (tmp_path / "lesson.js").read_text(encoding="utf-8")
    assert "<path" in js and "alert(1)" not in js


def test_the_graph_puts_the_chosen_candidate_in_lesson_json(monkeypatch, output, tmp_path):
    import agent.nodes as nodes
    import agent.persistence as persistence
    from agent import cli
    from agent.graph import build_graph
    from tests.test_lesson_flow import FakeModel, scripted
    from tests.test_review_flow import deck_session  # noqa: F401

    from pathlib import Path

    from agent.session import Session, SourceDoc
    from tests.test_lesson_flow import deck_chunks

    session = Session()
    session.add_file(Path("deck.pptx"), SourceDoc("slide", 2, deck_chunks(2)))
    monkeypatch.setattr(cli, "open_preview", lambda path: None)
    monkeypatch.setattr(persistence, "output_dir", lambda: tmp_path / "output")
    model = FakeModel()
    monkeypatch.setattr(nodes, "call_llm", model)
    graph = build_graph()
    cli.run_question("build", session, graph, ask_fn=scripted(["Y", "Y", "Y"]))
    state = graph.get_state(cli.graph_config(session)).values
    assert len(state["motif_candidates"]) == 2
    assert state["lesson"]["motif"]["svg"].startswith("<svg") and state["lesson"]["motif"]["description"]
    assert {r.node for r in session.usage} >= {"motif"}


def test_weather_icons_are_bundled_from_the_npm_package():
    data = json.loads(open("agent/assets/lucide/icons.json", encoding="utf-8").read())
    assert data["source"].startswith("npm lucide-static@")
    for name in ("snowflake", "cloud-fog", "truck"):
        assert name in data["icons"] and "<image" not in data["icons"][name]

"""Lesson schema (JSON Schema + Pydantic), the exporter, and the short_winter fixture."""

import copy
import json
from pathlib import Path

import pytest

from agent.lesson_export import export_lesson, lesson_js, main as export_main
from agent.lesson_schema import (
    PALETTES,
    LessonValidationError,
    contrast_problems,
    load_lesson,
    model_errors,
    schema_errors,
    validate_lesson,
)

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = ROOT / "examples" / "sample_lesson"


@pytest.fixture
def sample() -> dict:
    return json.loads((SAMPLE_DIR / "lesson.json").read_text(encoding="utf-8"))


def screen_with(sample, block_type):
    for s in sample["screens"]:
        for b in s["blocks"]:
            if b["type"] == block_type:
                return s, b
    raise AssertionError(block_type)


# ---- the sample and the palettes ------------------------------------------------------------
def test_sample_lesson_is_valid_in_both_validators(sample):
    assert schema_errors(sample) == [] and model_errors(sample) == []
    lesson = load_lesson(sample)
    assert len(lesson.screens) == 4


def test_sample_uses_every_block_type(sample):
    types = {b["type"] for s in sample["screens"] for b in s["blocks"]}
    assert types == {"paragraph", "list", "did_you_know", "stop_and_think", "multiple_choice",
                     "illustration", "decision_diagram"}


@pytest.mark.parametrize("lesson_type", ["training", "compliance"])
def test_default_palettes_meet_wcag_aa(lesson_type):
    assert contrast_problems(PALETTES[lesson_type]) == []


def test_training_is_light_and_compliance_is_dark():
    from agent.lesson_schema import _luminance

    assert _luminance(PALETTES["training"]["background"]) > 0.9
    assert _luminance(PALETTES["compliance"]["background"]) < 0.05


# ---- invalid lessons -------------------------------------------------------------------------
def mutate(sample, change):
    data = copy.deepcopy(sample)
    change(data)
    return data


def _mc(data):
    return screen_with(data, "multiple_choice")[1]


def _diagram(data):
    return screen_with(data, "decision_diagram")[1]


CASES = [
    ("missing title", lambda d: d.pop("title"), "'title' is a required property"),
    ("unknown block type", lambda d: d["screens"][0]["blocks"].append({"type": "video"}), "'video' is not one of"),
    ("extra field", lambda d: d["screens"][0].update(color="red"), "Additional properties are not allowed"),
    ("bad color", lambda d: d["palette"].update(text="black"), "does not match"),
    ("two options", lambda d: _mc(d).update(options=_mc(d)["options"][:2]), "is too short"),
    ("no alternate", lambda d: _mc(d).pop("alternate"), "'alternate' is a required property"),
    ("no sources", lambda d: d["screens"][0]["blocks"][0].update(sources=[]), "should be non-empty"),
    ("six diagram nodes", lambda d: _diagram(d)["nodes"].extend(
        [{"id": f"x{i}", "label": "extra", "kind": "action"} for i in range(2)]), "is too long"),
    ("illustration outside its folder", lambda d: screen_with(d, "illustration")[1].update(
        file="../secret.html"), "does not match"),
    ("answer index out of range", lambda d: _mc(d).update(options=_mc(d)["options"][:3], correct_index=3),
     "out of range"),
    ("feedback reveals the answer", lambda d: _mc(d).update(
        feedback_incorrect="Wrong. The answer is: " + _mc(d)["options"][_mc(d)["correct_index"]]),
     "reveals the correct answer"),
    ("alternate index out of range", lambda d: _mc(d)["alternate"].update(
        options=_mc(d)["alternate"]["options"][:3], correct_index=3), "out of range"),
    ("edge to unknown node", lambda d: _diagram(d)["edges"].append({"from": "abs", "to": "nowhere"}),
     "unknown node 'nowhere'"),
    ("diagram without a start", lambda d: _diagram(d)["edges"].append({"from": "report", "to": "abs"}),
     "needs a start node"),
    ("duplicate screen ids", lambda d: d["screens"][1].update(id=d["screens"][0]["id"]), "screen ids must be unique"),
    ("low contrast palette", lambda d: d["palette"].update(text="#9AA5B1"), "below WCAG AA"),
    ("did you know in compliance", lambda d: d.update(lesson_type="compliance"), "no did_you_know"),
    ("compliance word limit", lambda d: (d.update(lesson_type="compliance"),
                                         d["screens"][0]["blocks"].pop(3),  # the did_you_know
                                         d["screens"][0]["blocks"][0].update(text="word " * 90)),
     "a compliance screen allows at most 80"),
]


@pytest.mark.parametrize("change, expected", [(c, e) for _, c, e in CASES], ids=[n for n, _, _ in CASES])
def test_invalid_lessons_are_rejected_with_clear_messages(sample, change, expected):
    errors = validate_lesson(mutate(sample, change))
    assert errors, "expected the lesson to be rejected"
    assert any(expected in e for e in errors), errors


def test_errors_name_the_exact_place(sample):
    errors = validate_lesson(mutate(sample, lambda d: _mc(d).update(options=_mc(d)["options"][:2])))
    assert any(e.startswith("screens[1].blocks[2].options: ") for e in errors), errors


def test_load_lesson_raises_with_all_errors(sample):
    with pytest.raises(LessonValidationError) as info:
        load_lesson(mutate(sample, lambda d: (d.pop("title"), d.pop("summary"))))
    assert len(info.value.errors) == 2


# ---- exporter --------------------------------------------------------------------------------
def test_export_writes_a_self_contained_folder(sample, tmp_path):
    out = tmp_path / "lesson"
    index = export_lesson(sample, out, source_dir=SAMPLE_DIR)
    assert index == out / "index.html"
    for rel in ("index.html", "lesson.js", "assets/player.css", "assets/player.js", "assets/strings.he.js",
                "assets/illustrations/stopping_distance.html"):
        assert (out / rel).is_file(), rel
    script = (out / "lesson.js").read_text(encoding="utf-8")
    assert script.startswith("window.LESSON = ") and script.rstrip().endswith(";")
    assert json.loads(script[len("window.LESSON = "):].rstrip().rstrip(";")) == sample
    html = (out / "index.html").read_text(encoding="utf-8")
    assert '<script src="lesson.js">' in html and "fetch(" not in (out / "assets/player.js").read_text(encoding="utf-8")


def test_export_refuses_an_invalid_lesson_and_writes_nothing(sample, tmp_path):
    out = tmp_path / "lesson"
    with pytest.raises(LessonValidationError):
        export_lesson(mutate(sample, lambda d: d.pop("title")), out, source_dir=SAMPLE_DIR)
    assert not out.exists()


def test_export_needs_the_illustration_files(sample, tmp_path):
    with pytest.raises(FileNotFoundError):
        export_lesson(sample, tmp_path / "a", source_dir=tmp_path)  # no illustrations there
    with pytest.raises(FileNotFoundError):
        export_lesson(sample, tmp_path / "b", source_dir=None)


def test_lesson_js_cannot_close_its_script_tag():
    assert "</script>" not in lesson_js({"title": "x</script><script>alert(1)</script>"})


def test_export_command_line(sample, tmp_path, capsys):
    assert export_main([str(SAMPLE_DIR / "lesson.json"), str(tmp_path / "cli")]) == 0
    assert "exported:" in capsys.readouterr().out
    assert export_main(["only-one-arg"]) == 2


# ---- the short_winter source fixture -------------------------------------------------------------
def test_short_winter_fixture_is_rebuilt_from_its_json(tmp_path):
    import sys

    sys.path.insert(0, str(ROOT / "evals" / "fixtures"))
    try:
        from make_short_winter import build
    finally:
        sys.path.pop(0)
    from agent.mcp_servers.source_reader import read_source_file

    built = build(output_path=tmp_path / "short_winter.pptx")
    result = read_source_file(str(built))
    assert result["ok"] and result["units"] == 5
    kinds = [c["kind"] for c in result["chunks"]]
    assert kinds.count("notes") == 5 and kinds.count("title") == 5
    committed = read_source_file(str(ROOT / "evals" / "fixtures" / "short_winter.pptx"))
    assert [c["text"] for c in committed["chunks"]] == [c["text"] for c in result["chunks"]]

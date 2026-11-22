"""/model: the profile and the models per role; switching asks [Y/N] after explaining the cost."""

from agent import llm
from agent.commands import CommandHandler
from agent.messages import msg
from agent.session import Session


def handler(answers, outputs):
    replies = iter(answers)
    return CommandHandler(Session(), out=lambda text, **k: outputs.append(text), ask_line=lambda p: next(replies))


def test_shows_the_profile_and_every_role(monkeypatch):
    llm.reset_model_state()
    monkeypatch.setenv("PROFILE", "free")
    out: list[str] = []
    handler([], out).handle("/model")
    assert out[0] == msg("commands.model.profile", profile="free")
    assert any(llm.load_models("writer")[0] in line for line in out)
    assert len(out) == 1 + len(llm.ROLES) + 1


def test_switching_explains_the_cost_and_needs_y(monkeypatch):
    llm.reset_model_state()
    monkeypatch.setenv("PROFILE", "free")
    out: list[str] = []
    assert not handler(["N"], out).model("balanced")
    assert llm.active_profile() == "free" and msg("commands.model.explain.balanced") in out
    assert handler(["Y"], out).model("balanced")
    assert llm.active_profile() == "balanced"
    assert handler(["Y"], out).model("free") and llm.active_profile() == "free"
    out.clear()
    assert not handler([], out).model("premium")
    assert "premium" in out[0]
    llm.reset_model_state()

"""Windows sleep prevention while the graph runs: set on entry, restored on every exit."""

import pytest

from agent import cli
from agent.keep_awake import ES_CONTINUOUS, ES_SYSTEM_REQUIRED, active, keep_awake


def test_sets_and_restores_the_state():
    calls = []
    with keep_awake(set_state=calls.append, enabled=True):
        assert calls == [ES_CONTINUOUS | ES_SYSTEM_REQUIRED]
    assert calls == [ES_CONTINUOUS | ES_SYSTEM_REQUIRED, ES_CONTINUOUS]


@pytest.mark.parametrize("error", [RuntimeError("node failed"), KeyboardInterrupt()])
def test_restores_on_an_error_and_on_ctrl_c(error):
    calls = []
    with pytest.raises(type(error)):
        with keep_awake(set_state=calls.append, enabled=True):
            raise error
    assert calls[-1] == ES_CONTINUOUS


def test_does_nothing_under_pytest_or_off_windows(monkeypatch):
    assert not active()  # PYTEST_CURRENT_TEST is set
    calls = []
    with keep_awake(set_state=calls.append):
        pass
    assert calls == []
    monkeypatch.delenv("PYTEST_CURRENT_TEST")
    monkeypatch.setattr("sys.platform", "linux")
    assert not active()


def test_a_failing_windows_call_never_stops_the_work():
    def broken(flags):
        raise OSError("no kernel32")

    with keep_awake(set_state=broken, enabled=True):
        done = True
    assert done


def test_the_cli_keeps_awake_only_while_nodes_run(monkeypatch):
    entered = []

    class Graph:
        def stream(self, payload, config=None, stream_mode=None, **kw):
            entered.append("stream")
            return iter(())

    import agent.keep_awake as ka

    monkeypatch.setattr(ka, "active", lambda: True)
    monkeypatch.setattr(ka, "_set_state", lambda flags: entered.append(flags))
    cli.stream_until_pause(Graph(), {}, {})
    assert entered == [ES_CONTINUOUS | ES_SYSTEM_REQUIRED, "stream", ES_CONTINUOUS]

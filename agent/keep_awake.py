"""Keep Windows awake while the agent works on a lesson.

A lesson takes tens of minutes of model calls; if Windows goes to sleep in the middle, the calls
time out and the run stalls (on day 9 a free run took 13 hours of wall time for 55 minutes of
work). While the graph runs, SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) asks
Windows not to sleep; the previous state comes back when the work ends, on an error and on
Ctrl+C (the finally block). The display may still turn off. Waiting at a gate for the user's
answer is not covered, so a computer left at a gate can sleep as usual.

On other systems, and under pytest, it does nothing.
"""

import os
import sys
from contextlib import contextmanager

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def _set_state(flags: int) -> int:
    import ctypes

    return ctypes.windll.kernel32.SetThreadExecutionState(flags)  # type: ignore[attr-defined]


def active() -> bool:
    return sys.platform == "win32" and not os.environ.get("PYTEST_CURRENT_TEST")


@contextmanager
def keep_awake(set_state=None, enabled: bool | None = None):
    """No sleep inside the block (Windows only). The state is restored on any exit."""
    enabled = active() if enabled is None else enabled
    if not enabled:
        yield
        return
    set_state = set_state or _set_state
    try:
        set_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    except Exception:  # noqa: BLE001 - never stop the work because of this
        yield
        return
    try:
        yield
    finally:
        try:
            set_state(ES_CONTINUOUS)
        except Exception:  # noqa: BLE001
            pass

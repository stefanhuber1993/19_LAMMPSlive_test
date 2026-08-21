"""Every way the app can end has to reach `scancel`.

The window closing is the easy one. The ones that cost an A100 are the others: an
exception on the way out of the loop, a `kill` from a script, a terminal being
closed, a second Ctrl-C on a teardown that is already running. All of them used to
share one `finally` of five bare statements, so the first thing to raise took every
step after it with it -- and the step that matters, the one that gives the GPU
back, was fourth of five.

These drive `App._shutdown` against a stub rather than a real window: what is under
test is the ORDER and the GUARDS, and neither needs pygame, a GPU or a cluster.
"""
import signal

import pytest

from lammps_live.app import App


class _Step:
    """One thing shutdown does, which records that it was asked and may refuse."""

    def __init__(self, calls, name, raises=None):
        self.calls, self.name, self.raises = calls, name, raises

    def __call__(self, *_args):
        self.calls.append(self.name)
        if self.raises is not None:
            raise self.raises


class _StubApp:
    """The five attributes `App._shutdown` touches, and nothing else."""

    def __init__(self, failing=()):
        self.calls = []
        self._shut_down = False
        for name in ("release", "_sim_idle", "close_source", "close_system",
                     "quit"):
            setattr(self, "step_" + name,
                    _Step(self.calls, name, failing.get(name)
                          if isinstance(failing, dict) else None))
        self.remote_panel = type("P", (), {"release": self.step_release})()
        self._sim_idle = self.step__sim_idle
        self.source = type("S", (), {"close": self.step_close_source})()
        self.system = type("Y", (), {"close": self.step_close_system})()

    _shutdown = App._shutdown
    _install_exit_signals = App._install_exit_signals
    EXIT_SIGNALS = App.EXIT_SIGNALS


@pytest.fixture(autouse=True)
def no_pygame_quit(monkeypatch):
    """`pygame.quit` is the last step and the only real one here."""
    import lammps_live.app as app_mod
    monkeypatch.setattr(app_mod.pygame, "quit", lambda: None)


def test_the_allocation_is_released_before_anything_that_can_hang():
    """Ordering is the point. Waiting for the simulation thread, closing the
    joystick and closing the system are all things that can block or raise; the
    scancel is the one whose failure is measured in GPU-hours, so it goes first."""
    app = _StubApp()
    app._shutdown()
    assert app.calls[0] == "release"
    assert app.calls == ["release", "_sim_idle", "close_source", "close_system"]


def test_a_step_that_raises_does_not_take_the_rest_with_it():
    """The real shape of this: `_sim_idle` re-raises whatever killed the simulation
    thread, and the joystick's close fails when the device has already gone. Either
    used to skip everything after it."""
    app = _StubApp(failing={"_sim_idle": RuntimeError("the worker died"),
                            "close_source": OSError("device gone")})
    app._shutdown()
    assert app.calls == ["release", "_sim_idle", "close_source", "close_system"]


def test_a_second_ctrl_c_during_the_teardown_does_not_abort_it():
    """A teardown that looks slow -- and it does, it is running an ssh -- invites a
    second interrupt. That must not be what stops the scancel from happening."""
    app = _StubApp(failing={"release": KeyboardInterrupt()})
    app._shutdown()          # must not propagate
    assert app.calls == ["release", "_sim_idle", "close_source", "close_system"]


def test_shutting_down_twice_releases_once():
    """It is reachable from the loop's `finally`, from an atexit hook, and from a
    signal that unwinds through both."""
    app = _StubApp()
    app._shutdown()
    app._shutdown()
    assert app.calls.count("release") == 1


def test_the_signals_that_would_skip_the_finally_are_all_caught():
    """SIGKILL is not here because it cannot be -- that one is what the server's own
    `--exit-when-idle` exists for. Every other way a process is asked to end is."""
    assert set(App.EXIT_SIGNALS) == {"SIGTERM", "SIGHUP", "SIGINT"}


def test_an_exit_signal_unwinds_rather_than_ending_the_process(monkeypatch):
    """A handler that called `os._exit` would be no better than the default action.
    Raising is what puts the teardown back on the thread that owns the window."""
    installed = {}
    monkeypatch.setattr(signal, "signal",
                        lambda sig, handler: installed.setdefault(sig, handler))
    app = _StubApp()
    App._install_exit_signals(app)

    assert signal.SIGTERM in installed and signal.SIGHUP in installed
    with pytest.raises(SystemExit) as caught:
        installed[signal.SIGTERM](signal.SIGTERM, None)
    assert caught.value.code == 128 + int(signal.SIGTERM)

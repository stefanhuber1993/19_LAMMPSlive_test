"""The app in front of a simulation that just died: card, rebuild, sliders.

The pieces are tested elsewhere (test_faults.py); this is the only place they are
wired together, and the wiring is where the interesting mistake lives. The app
pushes every slider INTO the system once per frame, so a rebuild that had to put
`zeta` back is undone one line later by a slider still sitting on the value that
killed it -- which is a reset loop, not a recovery.
"""
import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

pytest.importorskip("lammps")

from lammps_live.app import App

FRAME = 1.0 / 60
LOST_ATOMS = "ERROR: Lost atoms: original 1500 current 12 (src/thermo.cpp:483)"


@pytest.fixture
def app():
    a = App(input_mode="mouse", initial_system_key="mesomem_assembly")
    a.sim_playing = True
    a._tick(FRAME)
    yield a
    a.system.close()


def _break_next_run(system, error=LOST_ATOMS, times=1):
    """Make the next `times` chunks raise, the way a blown-up LAMMPS does."""
    real = system.lmp.command
    left = {"n": times}

    def flaky(cmd):
        if cmd.startswith("run ") and left["n"] > 0:
            left["n"] -= 1
            raise Exception(error)
        return real(cmd)

    system.lmp.command = flaky


def test_a_blow_up_shows_a_card_and_rebuilds_itself(app):
    when = app.system.get_sim_time()
    _break_next_run(app.system)

    app._tick(FRAME)                    # the chunk raises; step() latches it
    app._tick(FRAME)                    # the app shows it and rebuilds

    assert app.alert.visible
    assert "flew out of the box" in app.alert.summary
    assert "Lost atoms" in app.alert.detail, "the raw error is on the card too"
    # Back on its feet: a fresh state, stepping again, nothing latched.
    assert app.system.unstable is None
    assert app.system.lmp is not None
    assert app.system.get_sim_time() <= when
    app._tick(FRAME)


def test_a_value_the_build_refuses_moves_the_slider_back(app):
    """The zeta failure: legal to this build until a rebuild validates it."""
    real = app.system.force_field.pair_commands
    app.system.force_field.pair_commands = (
        lambda p: real(p) + ["pair_coeff 1 1 nonsense"] if float(p["zeta"]) < 1.0
        else real(p))
    index = app.extra_slider_keys.index("zeta")
    good = float(app.system.params["zeta"])

    app.extra_sliders[index].value = 0.4
    app._tick(FRAME)
    assert app.system.params["zeta"] == pytest.approx(0.4), "the drag took effect"

    app._playback_action("reset")       # the Reset button
    app._tick(FRAME)

    assert app.alert.visible
    assert "not valid for this build" in app.alert.summary
    assert "nonsense" in app.alert.detail
    # THE POINT: the slider follows the simulation, so the next frame does not
    # push the killer straight back in.
    assert app.extra_sliders[index].value == pytest.approx(good)
    assert app.system.params["zeta"] == pytest.approx(good)
    app._tick(FRAME)
    assert app.system.params["zeta"] == pytest.approx(good)
    assert app.system.unstable is None


def _break_every_run(system, error=LOST_ATOMS):
    """Make every chunk raise, rebuilds included.

    A rebuild makes a NEW LAMMPS instance, so patching the current one only breaks
    the run before the recovery -- which is how the first version of this test
    passed while proving nothing.
    """
    real_setup = system._setup

    def setup(seed):
        real_setup(seed)
        _break_next_run(system, error, times=99)

    system._setup = setup
    _break_next_run(system, error, times=99)


def test_settings_that_destroy_every_fresh_state_stop_the_retrying(app):
    """One free recovery. A value that blows up every rebuild must not put the app
    in a loop of card, rebuild, card -- it stops and says what to do."""
    _break_every_run(app.system)

    app._tick(FRAME)                    # blows up
    app._tick(FRAME)                    # first recovery: rebuild
    assert app.alert.visible
    app._tick(FRAME)                    # blows up again, inside the cooldown
    app._tick(FRAME)

    assert "dial them back" in app.alert.summary
    assert app.system.unstable, "stepping stays stopped rather than looping"

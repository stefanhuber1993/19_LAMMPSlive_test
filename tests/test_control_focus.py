"""Driving the demo from the joystick alone: what the stick moves, and how.

The stick has two axes, the demo has a camera plus a handful of live parameters,
and the hat switch is what says which of them is listening (see
lammps_live/control_focus.py). Three things have to hold for that to be usable
in front of an audience, and each is pinned below:

  * the cycle is the everyday controls, in panel order, starting on the viewport
    -- not every slider the playground declares;
  * a stop moves exactly one place per flick of the hat, in the direction flicked;
  * whatever holds the focus is the ONLY thing the stick moves. In particular,
    focusing a slider releases the puller, so a bead is never dragged across the
    box by a value change -- and coming back to the viewport picks it up again.
"""
import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from lammps_live.control_focus import (
    AXIS_DEADZONE, AXIS_FAST_RATE, AXIS_SLOW_END, AXIS_SLOW_RATE, Choice,
    ControlFocus, axis_rate,
)
from lammps_live.input.base import InputSource

pytest.importorskip("lammps")

from lammps_live import config
from lammps_live.app import App

FRAME = 1.0 / 60


# ---- the axis response, on its own ----------------------------------------

def test_the_deadzone_is_dead():
    """A hand resting on the stick, or coming back from a swing, changes nothing."""
    for x in (0.0, 0.1, 0.3, AXIS_DEADZONE):
        assert axis_rate(x) == 0.0
        assert axis_rate(-x) == 0.0


def test_the_slow_plateau_is_flat_and_covers_most_of_the_travel():
    """"Slow" has to be a place on the stick you can find by feel."""
    plateau = [axis_rate(x) for x in (AXIS_DEADZONE + 0.01, 0.5, 0.65, AXIS_SLOW_END)]
    assert plateau == [pytest.approx(AXIS_SLOW_RATE)] * 4
    assert AXIS_SLOW_END - AXIS_DEADZONE > 0.4, "the fine-control band is the big one"
    assert axis_rate(-0.5) == pytest.approx(-AXIS_SLOW_RATE)


def test_the_ramp_to_fast_is_smooth_and_accelerating():
    """No step in rate at the band edge -- a jump reads as the control snatching --
    and the speed is at the top of the travel, not spread evenly over the ramp."""
    ramp = [axis_rate(x) for x in (AXIS_SLOW_END, 0.85, 0.9, 0.95, 1.0)]
    assert ramp[0] == pytest.approx(AXIS_SLOW_RATE)
    assert ramp[-1] == pytest.approx(AXIS_FAST_RATE)
    assert all(b > a for a, b in zip(ramp, ramp[1:])), "monotonic"
    gaps = [b - a for a, b in zip(ramp, ramp[1:])]
    assert all(b > a for a, b in zip(gaps, gaps[1:])), "accelerating, not linear"
    # Barely off the plateau is still slow: the ramp does not begin with a leap.
    assert axis_rate(0.85) < 0.25 * AXIS_FAST_RATE
    assert axis_rate(-1.0) == pytest.approx(-AXIS_FAST_RATE)


def test_an_empty_cycle_cannot_leave_the_viewport():
    """A playground with no everyday sliders and no choices still has a valid
    focus."""
    focus = ControlFocus()
    focus.set_stops([])
    focus.cycle(1)
    assert focus.on_viewport
    assert focus.slider is None
    assert focus.choice is None


# ---- the choice stops ------------------------------------------------------

def test_a_choice_steps_once_per_push_however_long_it_is_held():
    """The bead colouring is a set of pictures, not a scale: one push, one change.
    Without the latch a single push scrolls the whole list in a few frames."""
    seen = []
    choice = Choice("bead colour", ("director", "energy", "third"),
                    on_change=seen.append)

    for _ in range(60):                      # held hard right for a whole second
        choice.push(1.0)
    assert choice.option == "energy"
    assert seen == [1]

    for _ in range(30):                      # back to centre re-arms it
        choice.push(0.0)
    choice.push(1.0)
    assert choice.option == "third"

    choice.push(-1.0)                        # still held: no second step
    assert choice.option == "third"


def test_a_choice_needs_a_firm_push_and_wraps_both_ways():
    choice = Choice("bead colour", ("director", "energy"))
    assert not choice.push(0.4), "a wobble is not a push"
    assert choice.option == "director"
    choice.push(-1.0)                        # left from the first wraps to the last
    assert choice.option == "energy"


# ---- driving a real app ----------------------------------------------------

class FakeStick(InputSource):
    """A joystick with no hardware behind it: the app only ever asks it for
    numbers, so the numbers can simply be set."""

    def __init__(self):
        self.xy = (0.0, 0.0)
        self.yaw = 0.0
        self.buttons = frozenset()
        self.hat = (0, 0)

    def poll(self):
        return self.xy

    def poll_yaw(self):
        return self.yaw

    def poll_buttons(self):
        return self.buttons

    def poll_hat(self):
        return self.hat


def _stick_app(key):
    """An app in joystick mode without a joystick.

    Built as `mouse` and switched afterwards: App.__init__ opens the real device
    for `joystick`, and there is none here. Everything downstream reads
    self.input_mode and self.source, which is exactly what is swapped.
    """
    app = App(input_mode="mouse", initial_system_key=key)
    app.input_mode = "joystick"
    app.source = FakeStick()
    return app


def _flick(app, dx):
    """One flick of the hat and back to centre -- an edge, not a held direction."""
    app.source.hat = (dx, 0)
    app._poll_device_buttons()
    app.source.hat = (0, 0)
    app._poll_device_buttons()


def _press(app, button):
    app.source.buttons = frozenset({button})
    app._poll_device_buttons()
    app.source.buttons = frozenset()
    app._poll_device_buttons()


@pytest.fixture(scope="module")
def sim_app():
    """The self-assembly box: a playback playground with a turntable camera and
    nothing to pull -- the scene this control scheme was asked for."""
    app = _stick_app("mesomem_assembly")
    yield app
    app.system.close()
    pygame.event.clear()


@pytest.fixture(scope="module")
def game_app():
    """The 7-bead patch: a game playground, where the viewport focus means the
    puller and the force feedback that comes with it.

    ORDER MATTERS, and it is why the game_app tests are all at the end: building a
    second App calls `set_mode` again, which invalidates the FIRST app's display
    surface. Anything that DRAWS with `sim_app` (i.e. calls `_tick`) therefore has
    to run before this fixture is first requested; everything after it must stay on
    the state-only paths (`_route_stick`, `_poll_device_buttons`).
    """
    app = _stick_app("mesomem_patch")
    yield app
    app.system.close()
    pygame.event.clear()


@pytest.fixture(autouse=True)
def _focus_on_the_viewport(request):
    """Every test starts where a fresh playground starts."""
    for name in ("sim_app", "game_app"):
        if name in request.fixturenames:
            app = request.getfixturevalue(name)
            app.focus.index = 0
            app._focus_released_puller = False
            app.source.xy = (0.0, 0.0)
            app.source.yaw = 0.0
            # Both ends of the edge detection, so a test that leaves the hat or a
            # button held cannot swallow the next test's first flick.
            app.source.hat = (0, 0)
            app.source.buttons = frozenset()
            app._prev_hat = (0, 0)
            app._prev_buttons = frozenset()


def test_the_cycle_is_the_viewport_the_everyday_sliders_and_the_colouring(sim_app):
    labels = ["viewport"]
    for _ in range(5):
        _flick(sim_app, 1)
        labels.append(sim_app.focus.label)
    assert labels == ["viewport", "Temperature", "k_tilt", "k_splay",
                      "zeta (attraction falloff, higher=shorter reach)",
                      "bead colour (director)"]
    # ... and it wraps back, rather than stopping at the end.
    _flick(sim_app, 1)
    assert sim_app.focus.on_viewport


def test_the_hat_moves_one_stop_per_flick_in_the_direction_flicked(sim_app):
    _flick(sim_app, 1)
    assert sim_app.focus.label == "Temperature"
    _flick(sim_app, -1)
    assert sim_app.focus.on_viewport
    # Left from the viewport wraps to the far end of the cycle.
    _flick(sim_app, -1)
    assert sim_app.focus.label.startswith("bead colour")


def test_a_held_hat_does_not_sweep_the_cycle(sim_app):
    """Holding the hat over is one move, not sixty a second."""
    sim_app.source.hat = (1, 0)
    for _ in range(10):
        sim_app._poll_device_buttons()
    assert sim_app.focus.index == 1


def test_the_focused_slider_is_what_the_stick_moves(sim_app):
    _flick(sim_app, 1)                       # -> Temperature
    slider = sim_app.focus.slider
    slider.value = 0.2
    span = slider.vmax - slider.vmin

    # Inside the deadzone: nothing at all, however long it is held.
    for _ in range(60):
        assert sim_app._route_stick(0.3, 0.9, 0.5, FRAME) == (0.0, 0.0, 0.0)
    assert slider.value == pytest.approx(0.2)

    # The slow plateau, one second's worth of frames -> ~2% of the range.
    for _ in range(60):
        sim_app._route_stick(0.6, 0.0, 0.0, FRAME)
    assert slider.value == pytest.approx(0.2 + AXIS_SLOW_RATE * span, rel=0.05)

    # Fast band, the other way, and it stops at the end of the track.
    for _ in range(120):
        sim_app._route_stick(-1.0, 0.0, 0.0, FRAME)
    assert slider.value == pytest.approx(slider.vmin)


def test_the_stick_flies_the_camera_while_the_viewport_holds_the_focus(sim_app):
    cam = sim_app.orbit_cam
    assert cam is not None
    cam.auto = True
    azimuth, elev, dist = cam.azimuth, cam.elev, cam.dist

    left = sim_app._route_stick(-0.9, 0.0, 0.0, FRAME)
    assert left == (0.0, 0.0, 0.0), "the camera took the stick"
    assert cam.azimuth < azimuth, "pushing left travels left around the box"
    assert not cam.auto, "taking the stick stops the automatic turn"

    sim_app._route_stick(0.0, 0.9, 0.0, FRAME)
    assert cam.elev > elev, "pushing forward climbs over the box"

    sim_app._route_stick(0.0, 0.0, 0.8, FRAME)
    assert cam.dist != dist, "the twist axis dollies"


def test_a_resting_stick_does_not_drift_the_camera(sim_app):
    cam = sim_app.orbit_cam
    cam.auto = False
    azimuth, elev, dist = cam.azimuth, cam.elev, cam.dist
    for _ in range(60):
        sim_app._route_stick(0.15, -0.18, 0.1, FRAME)
    assert (cam.azimuth, cam.elev, cam.dist) == (azimuth, elev, dist)


def test_the_camera_creeps_on_the_plateau_and_swings_at_the_stop(sim_app):
    """The same banded feel as a slider: most of the stick is fine placement."""
    cam = sim_app.orbit_cam
    spec = cam.spec

    cam.azimuth = 0.0
    for _ in range(60):
        sim_app._route_stick(0.5, 0.0, 0.0, FRAME)
    creep = cam.azimuth
    assert creep == pytest.approx(spec.stick_slow_speed, rel=0.05)

    cam.azimuth = 0.0
    for _ in range(60):
        sim_app._route_stick(1.0, 0.0, 0.0, FRAME)
    assert cam.azimuth == pytest.approx(spec.stick_speed, rel=0.05)
    assert cam.azimuth > 4 * creep


def test_twisting_away_pushes_the_scene_away(sim_app):
    """The first attempt had this inverted: the twist moves the SCENE, so twisting
    one way must make the box smaller, not reel the camera in."""
    cam = sim_app.orbit_cam
    cam.dist = cam.dist0
    sim_app._route_stick(0.0, 0.0, 1.0, FRAME)
    assert cam.dist > cam.dist0


def test_shift_drag_pans_the_scene_and_the_camera_follows_the_target(sim_app):
    """Panning moves what the turntable turns about, so an off-centre membrane can
    be brought to the middle and then orbited."""
    cam = sim_app.orbit_cam
    before = cam.target.copy()
    eye_before = cam.eye()
    cam.pan(40, 0)
    assert not (cam.target == before).all(), "the target moved across the view"
    assert not (cam.eye() == eye_before).all(), "and the eye came with it"
    # The scene follows the pointer: dragging right takes the camera the other way
    # along its own horizontal axis.
    import numpy as np
    right = np.array([np.cos(cam.azimuth), np.sin(cam.azimuth), 0.0])
    assert (cam.eye() - eye_before) @ right < 0
    # A pan is a hand on the camera, like a drag: the automatic turn stops.
    assert not cam.auto
    cam.target = before


def test_the_camera_stops_moving_once_a_slider_has_the_focus(sim_app):
    cam = sim_app.orbit_cam
    _flick(sim_app, 1)                       # -> Temperature
    cam.auto = False
    azimuth = cam.azimuth
    for _ in range(60):
        sim_app._route_stick(1.0, 1.0, 1.0, FRAME)
    assert cam.azimuth == azimuth


def test_the_trigger_is_the_run_switch_and_button_2_resets(sim_app):
    assert not sim_app.sim_playing, "a playback playground comes up paused"
    _press(sim_app, config.JOYSTICK_PLAY_PAUSE_BUTTON)
    assert sim_app.sim_playing
    _press(sim_app, config.JOYSTICK_PLAY_PAUSE_BUTTON)
    assert not sim_app.sim_playing

    sim_app.sim_playing = True
    sim_app.total_steps = 1234
    _press(sim_app, config.JOYSTICK_RESET_BUTTON)
    assert sim_app.total_steps == 0
    assert not sim_app.sim_playing, "Reset shows the fresh state before it moves"


def test_a_held_trigger_does_not_flutter_the_run(sim_app):
    sim_app.source.buttons = frozenset({config.JOYSTICK_PLAY_PAUSE_BUTTON})
    for _ in range(10):
        sim_app._poll_device_buttons()
    assert sim_app.sim_playing
    sim_app.source.buttons = frozenset()
    sim_app._poll_device_buttons()
    sim_app.sim_playing = False


def test_buttons_3_and_4_step_through_the_playgrounds(sim_app, monkeypatch):
    """The step, not the rebuild: switching playground for real costs a LAMMPS
    build, and what is being pinned here is which button means which direction."""
    steps = []
    monkeypatch.setattr(sim_app, "_cycle_system", steps.append)
    _press(sim_app, config.JOYSTICK_PREV_PLAYGROUND_BUTTON)
    _press(sim_app, config.JOYSTICK_NEXT_PLAYGROUND_BUTTON)
    assert steps == [-1, 1]


def test_nothing_fires_while_the_connect_panel_is_up(sim_app, monkeypatch):
    """The panel is modal and waiting for a login code; a button that switched
    playground mid-login would cancel the allocation it is asking for."""
    steps = []
    monkeypatch.setattr(sim_app, "_cycle_system", steps.append)
    monkeypatch.setattr(sim_app.remote_panel, "visible", True)

    _flick(sim_app, 1)
    _press(sim_app, config.JOYSTICK_PLAY_PAUSE_BUTTON)
    _press(sim_app, config.JOYSTICK_NEXT_PLAYGROUND_BUTTON)

    assert sim_app.focus.on_viewport
    assert not sim_app.sim_playing
    assert steps == []


def test_the_colouring_stop_drives_the_renderer_and_the_mouse_toggle_agrees(sim_app):
    """One state, two ways to move it: a click that changed the colouring behind
    the Choice's back would make the next stick push step from the wrong option."""
    for _ in range(5):                       # -> bead colour
        _flick(sim_app, 1)
    assert sim_app.focus.choice is not None
    assert not sim_app.renderer.bead_color_energy

    sim_app._route_stick(1.0, 0.0, 0.0, FRAME)
    assert sim_app.renderer.bead_color_energy
    assert sim_app.focus.label == "bead colour (energy)"

    # The mouse toggle moves the same Choice, so the stick carries on from there.
    sim_app.color_choice.step(1)
    assert not sim_app.renderer.bead_color_energy
    for _ in range(20):                      # re-arm, then push again
        sim_app._route_stick(0.0, 0.0, 0.0, FRAME)
    sim_app._route_stick(1.0, 0.0, 0.0, FRAME)
    assert sim_app.renderer.bead_color_energy
    sim_app.color_choice.step(1)             # leave it as it was found


def test_a_stick_that_holds_nothing_gets_a_strong_centring_spring(sim_app):
    """Both the camera and the sliders are rate controls read off the stick's own
    position, so it has to come back to true centre by itself -- and there is no
    contact force to render, because nothing is being held."""
    from lammps_live.input import SPRING_STIFFNESS_MAX

    sent = []
    sim_app.source.send_force = lambda fx, fy, k=None: sent.append((fx, fy, k))
    dt = FRAME
    for _ in range(2):                       # viewport: flying the camera
        dt = sim_app._tick(dt)
    assert sent[-1] == (0.0, 0.0, SPRING_STIFFNESS_MAX)

    _flick(sim_app, 1)                       # -> Temperature
    for _ in range(2):
        dt = sim_app._tick(dt)
    assert sent[-1] == (0.0, 0.0, SPRING_STIFFNESS_MAX)
    del sim_app.source.send_force


# ---- the force-feedback playgrounds ---------------------------------------

def test_the_colouring_is_a_stop_here_too(game_app):
    """Every 3D bead scene gets it, puller playgrounds included -- what the beads
    are coloured by is the viewer's question, not the mode's."""
    assert game_app.system.spec.render_3d
    labels = []
    for _ in range(6):
        _flick(game_app, 1)
        labels.append(game_app.focus.label)
    assert "bead colour (director)" in labels



def test_focusing_a_slider_releases_the_puller_and_the_viewport_takes_it_back(game_app):
    assert game_app.system.puller_attached(), "the patch comes up holding its bead"

    _flick(game_app, 1)                      # -> Temperature
    assert not game_app.system.puller_attached(), (
        "the stick cannot hold a bead and set a number at the same time")
    # ... and the stick drives the slider instead of the bead.
    slider = game_app.focus.slider
    before = slider.value
    game_app._route_stick(1.0, 0.0, 0.0, FRAME)
    assert slider.value != before

    _flick(game_app, -1)                     # back to the viewport
    assert game_app.system.puller_attached()


def test_a_bead_released_with_the_trigger_stays_released(game_app):
    """Only a puller the focus released is picked back up."""
    _press(game_app, config.JOYSTICK_ATTACH_BUTTON)
    assert not game_app.system.puller_attached()

    _flick(game_app, 1)
    _flick(game_app, -1)
    assert not game_app.system.puller_attached()

    _press(game_app, config.JOYSTICK_ATTACH_BUTTON)   # leave it as it was found
    assert game_app.system.puller_attached()


def test_the_viewport_still_means_the_puller_where_there_is_one(game_app):
    """No turntable here, and the bead is the point: the stick reaches it
    unchanged, twist included."""
    assert game_app.orbit_cam is None
    assert game_app._route_stick(0.8, -0.3, 0.5, FRAME) == (0.8, -0.3, 0.5)

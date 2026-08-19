"""Clicking into the window before it has finished starting up.

The window is on screen for a second or two -- joystick handshake, LAMMPS, the
shaders -- before anything reads its event queue, and clicking it then is the
natural thing to do. Two separate things have to hold for that to be harmless:
the events aimed at the half-built UI must not be replayed into the real one,
and a press whose release went missing must not wedge the pointer for the rest
of the session.
"""
import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pytest.importorskip("lammps")

from lammps_live.app import App

FRAME = 1.0 / 60


@pytest.fixture
def app():
    # An orbit-camera playground: the turntable is the drag that, stuck, eats
    # everything else's mouse events.
    a = App(input_mode="mouse", initial_system_key="mesomem_assembly")
    yield a
    a.system.close()
    pygame.event.clear()


def _down(pos):
    return pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos)


def _up(pos):
    return pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=pos)


def _motion(pos, rel=(4, 0), held=False):
    return pygame.event.Event(pygame.MOUSEMOTION, pos=pos, rel=rel,
                              buttons=(1 if held else 0, 0, 0))


def _slider_pos(slider, frac=0.5):
    return (slider.rect.x + int(slider.rect.width * frac), slider.rect.centery)


def test_startup_clicks_are_not_replayed_into_the_real_ui(app):
    """Whatever was clicked at the splash is gone by the time the loop runs."""
    pygame.event.post(_down((10, 10)))
    pygame.event.post(_motion((40, 10)))
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_2,
                                         mod=0, unicode="2"))

    app._drain_startup_input()

    assert pygame.event.get() == [], "the queue is empty before the first frame"


def test_a_quit_during_startup_still_quits(app):
    """Closing the window while it is still coming up is not a stray click."""
    pygame.event.post(_down((10, 10)))
    pygame.event.post(pygame.event.Event(pygame.QUIT))

    app._drain_startup_input()

    assert [e.type for e in pygame.event.get()] == [pygame.QUIT]


def test_a_press_that_lost_its_release_does_not_wedge_the_sliders(app):
    """The reported bug: click during startup, then the sliders stop moving.

    A left press in the sim view opens a turntable drag. If its release never
    arrives, `_handle_orbit_mouse` goes on eating MOUSEMOTION -- so a slider
    drag started afterwards never sees the motion that should move it, and
    never sees the release that should end it either.
    """
    app._handle_events(FRAME)          # lay nothing out yet, just prove it runs
    app._tick(FRAME)                   # first draw: sliders get their real rects
    assert app.orbit_cam is not None, "this playground has a turntable"

    # A press in the sim view whose MOUSEBUTTONUP is lost.
    pygame.event.post(_down((app.renderer.sim_width // 2, 200)))
    app._handle_events(FRAME)
    assert app._orbit_dragging

    # Now drag the temperature slider the way a user would.
    slider = app.temp_slider
    start = slider.value
    pygame.event.post(_down(_slider_pos(slider, 0.2)))
    pygame.event.post(_motion(_slider_pos(slider, 0.8), held=True))
    pygame.event.post(_up(_slider_pos(slider, 0.8)))
    app._handle_events(FRAME)

    assert slider.value != start, "the slider followed the drag"
    assert slider.value == pytest.approx(
        slider.vmin + 0.8 * (slider.vmax - slider.vmin), rel=0.05)
    assert not slider.dragging, "and the release ended it"
    assert not app._orbit_dragging


def test_a_plain_move_clears_a_drag_nobody_is_holding(app):
    """Moving the mouse with no button down proves every drag is over."""
    app._tick(FRAME)
    app.temp_slider.dragging = True
    app._orbit_dragging = True

    pygame.event.post(_motion((300, 300)))
    app._handle_events(FRAME)

    assert not app.temp_slider.dragging
    assert not app._orbit_dragging


def test_a_real_drag_still_works(app):
    """The recovery must not cut a drag that is genuinely in progress."""
    app._tick(FRAME)
    slider = app.temp_slider

    pygame.event.post(_down(_slider_pos(slider, 0.1)))
    app._handle_events(FRAME)
    assert slider.dragging

    for frac in (0.3, 0.5, 0.7):
        pygame.event.post(_motion(_slider_pos(slider, frac), held=True))
        app._handle_events(FRAME)
        assert slider.dragging, "held across frames"
    assert slider.value == pytest.approx(
        slider.vmin + 0.7 * (slider.vmax - slider.vmin), rel=0.05)

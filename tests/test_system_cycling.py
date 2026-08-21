"""Walking the playground picker from the keyboard.

Tab steps to the next playground, shift-Tab to the previous one, and the two
have to agree on the same order -- a step forward followed by a step back is
where you started.
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
    a = App(input_mode="mouse", initial_system_key="mesomem_assembly")
    yield a
    a.system.close()
    pygame.event.clear()


def _tab(shift=False):
    return pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB,
                              mod=pygame.KMOD_LSHIFT if shift else 0,
                              unicode="\t")


def test_shift_tab_steps_back_to_where_tab_came_from(app, monkeypatch):
    steps = []
    monkeypatch.setattr(app, "_cycle_system", steps.append)

    pygame.event.post(_tab())
    app._handle_events(FRAME)
    pygame.event.post(_tab(shift=True))
    app._handle_events(FRAME)

    assert steps == [1, -1]


def test_a_tab_and_a_shift_tab_land_on_the_playground_they_started_from(app):
    start = app.system_key

    pygame.event.post(_tab())
    app._handle_events(FRAME)
    assert app.system_key != start, "Tab actually moved"

    pygame.event.post(_tab(shift=True))
    app._handle_events(FRAME)
    assert app.system_key == start

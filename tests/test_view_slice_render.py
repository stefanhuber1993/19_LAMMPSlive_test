"""The slice and the per-species colours where they actually take effect: in the
renderer, and in the app's frame loop that feeds it.

tests/test_view_slice.py pins the state machine on its own. What is left, and what
this covers, is the wiring -- that a plane reaching the renderer really removes
beads from the scene, that it composes with the fixed section cut a playground may
already have declared, and that the app advances it from the device's lever rather
than from anything the stick is doing.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from lammps_live.render_style import DEFAULT_STYLE
from lammps_live.ui.renderer import Renderer
from lammps_live.view_slice import SlicePlane, ViewSlice

PTS = np.array([[0.0, y, 0.0] for y in (-9.0, -1.0, 0.0, 1.0, 9.0)])


@pytest.fixture(scope="module")
def renderer():
    import pygame

    pygame.display.init()
    pygame.font.init()
    yield Renderer((900, 700))
    pygame.display.quit()


def test_no_cut_is_no_mask(renderer):
    """None rather than an all-true array: the caller skips a whole indexing pass
    over every per-bead channel on every frame nothing is cut. A ViewSlice that
    has not engaged hands back exactly that None, which is how the app's frame
    loop and this pass agree without either testing a flag."""
    assert ViewSlice().plane is None
    assert renderer._cut_mask(DEFAULT_STYLE, None, PTS) is None


def test_a_slice_alone_keeps_its_slab(renderer):
    plane = SlicePlane(normal=(0.0, 1.0, 0.0), center=0.0, half=2.0)
    keep = renderer._cut_mask(DEFAULT_STYLE, plane, PTS)
    assert list(keep) == [False, True, True, True, False]


def test_a_section_alone_keeps_its_half_space(renderer):
    style = DEFAULT_STYLE.varied(section_axis=(0.0, 1.0, 0.0), section_min=0.0)
    keep = renderer._cut_mask(style, None, PTS)
    assert list(keep) == [False, False, True, True, True]


def test_the_two_cuts_intersect(renderer):
    """A playground drawn in section, sliced: the answer is a section of the slab,
    which is what both of them asked for."""
    style = DEFAULT_STYLE.varied(section_axis=(0.0, 1.0, 0.0), section_min=0.0)
    plane = SlicePlane(normal=(0.0, 1.0, 0.0), center=0.0, half=2.0)
    keep = renderer._cut_mask(style, plane, PTS)
    assert list(keep) == [False, False, True, True, False]


def test_static_tints_convert_once_and_are_cached(renderer):
    tints = np.zeros((4, 4))
    tints[:, :3] = 255.0
    tints[:, 3] = 1.0
    out = renderer._static_tints(tints, 4, DEFAULT_STYLE)
    assert out.shape == (4, 4)
    assert np.allclose(out[:, :3], 1.0)      # 255 display-space -> 1.0 linear
    assert np.allclose(out[:, 3], 1.0)
    assert renderer._static_tints(tints, 4, DEFAULT_STYLE) is out
    # A count that does not match the scene is refused rather than mis-indexed.
    assert renderer._static_tints(tints, 7, DEFAULT_STYLE) is None
    assert renderer._static_tints(None, 4, DEFAULT_STYLE) is None


# --- the app's frame loop -----------------------------------------------------

pytest.importorskip("lammps")


def test_the_app_drives_the_slice_from_the_lever(monkeypatch):
    """The lever is read every frame, whatever the stick is pointed at, and what
    it produces reaches the renderer through the 3D scene."""
    import pygame

    from lammps_live.app import App

    app = App(input_mode="mouse", initial_system_key="mesomem_assembly")
    try:
        lever = {"value": None}
        monkeypatch.setattr(type(app.source), "poll_throttle",
                            lambda self: lever["value"], raising=False)
        seen = {}

        def capture(*a, **kw):
            seen["slice"] = (kw.get("scene_3d") or {}).get("view_slice")
            seen["tints"] = (kw.get("scene_3d") or {}).get("bead_tints")

        monkeypatch.setattr(app.renderer, "draw", capture)

        # A lever nobody has touched leaves the box whole, whatever it reads.
        lever["value"] = 0.8
        for _ in range(10):
            app._tick(1.0 / 60)
        assert seen["slice"] is None

        # Moved, and held: it closes to the slab within the transition.
        lever["value"] = 0.35
        for _ in range(int(60 * app.view_slice.transition_seconds) + 4):
            app._tick(1.0 / 60)
        plane = seen["slice"]
        assert plane is not None
        box = app.system.get_box_bounds_3d()
        span = box[1] - box[0]
        assert plane.half == pytest.approx(
            0.5 * app.view_slice.thickness_fraction * span, rel=1e-3)
        assert plane.mask(app.system.get_positions_3d()[1]) is not None

        # Left alone, it opens back up.
        for _ in range(int(60 * (app.view_slice.hold_seconds
                                 + app.view_slice.transition_seconds)) + 8):
            app._tick(1.0 / 60)
        assert seen["slice"] is None
    finally:
        app.system.close()
        pygame.event.clear()


def test_switching_playground_forgets_the_lever(monkeypatch):
    import pygame

    from lammps_live.app import App

    app = App(input_mode="mouse", initial_system_key="mesomem_assembly")
    try:
        lever = {"value": 0.5}
        monkeypatch.setattr(type(app.source), "poll_throttle",
                            lambda self: lever["value"], raising=False)
        for _ in range(4):
            app._tick(1.0 / 60)
        lever["value"] = 0.9
        for _ in range(40):
            app._tick(1.0 / 60)
        assert app.view_slice.engaged
        app._build_system("mesomem_patch")
        assert not app.view_slice.engaged
        # The lever has not moved since, so the new scene comes up whole.
        for _ in range(40):
            app._tick(1.0 / 60)
        assert app.view_slice.plane is None
    finally:
        app.system.close()
        pygame.event.clear()

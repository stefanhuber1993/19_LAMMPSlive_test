"""The torque drive's rings: a torque drawn as the rotation it is.

What is pinned here is the thing that was wrong before, and it is a rendering
claim rather than a physics one. A torque has an axis, the axis has a direction
in the scene, and the plane the rotation happens in is seen by the camera at
whatever angle that plane is at -- face on, edge on, or an ellipse between the
two. The old drawing had a circle in SCREEN space and a straight arrow along the
axial vector, which between them said neither of those things: the circle was the
same circle whichever way the axis pointed, and the arrow was a picture of a push
along a direction nothing moves in, on the one drive that pushes nothing at all.

The geometry is a static method on the renderer and needs no display, so most of
this is arithmetic. The last two tests do open a (dummy) one, because "no straight
arrows on a torque drive" is a claim about what the draw pass calls.
"""
import math
import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from lammps_live.ui.camera import Camera3D
from lammps_live.ui.renderer import Renderer
from lammps_live.ui.theme import TORQUE_RING_MIN

CENTER = np.zeros(3)
RADIUS = 0.8


def _camera(eye=(0.0, -12.0, 0.0)):
    """Looking at the origin from -y, which is the angle the patch scenes use."""
    return Camera3D(eye=eye, target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0),
                    fov_deg=40.0, viewport_w=900, viewport_h=700)


def _ring(vec, radius=RADIUS, camera=None):
    return Renderer._torque_ring_points(camera or _camera(), CENTER,
                                        np.asarray(vec, dtype=float), radius)


def test_a_ring_is_a_circle_in_the_plane_the_rotation_happens_in():
    """Perpendicular to the axis, at the radius asked for, about the bead. That is
    the whole geometric claim -- everything else is where the camera puts it."""
    for axis in ((0, 0, 1), (0, 1, 0), (1, 0, 0), (0.3, -0.5, 0.81)):
        axis = np.asarray(axis, dtype=float)
        axis /= np.linalg.norm(axis)
        pts, _ = _ring(axis * 0.7)
        rel = pts - CENTER
        assert np.allclose(np.linalg.norm(rel, axis=1), RADIUS)
        assert np.allclose(rel @ axis, 0.0, atol=1e-12), f"{axis} is not the normal"


def test_the_sweep_grows_with_the_torque_and_stops_at_a_semicircle():
    """Length of the vector is how far round it goes -- a semicircle at full scale,
    and no further when the reaction runs past its ceiling, or a big enough torque
    would close the ring and stop reading as an arrow at all."""
    for mag, expect in ((0.25, 0.25 * math.pi), (1.0, math.pi), (3.0, math.pi)):
        _, sweep = _ring((0.0, 0.0, mag))
        assert sweep == pytest.approx(expect)


def test_a_negligible_torque_draws_nothing():
    assert _ring((0.0, 0.0, 0.9 * TORQUE_RING_MIN)) is None
    assert _ring((0.0, 0.0, 0.0)) is None
    assert _ring((0.0, 0.0, 1.1 * TORQUE_RING_MIN)) is not None


def test_the_ring_sweeps_the_way_the_right_hand_says():
    """Seen from +z, a torque about +z runs counter-clockwise. The sense is the
    only thing the arrowhead is there to say, and it is the thing a flat circle
    could not get right for an axis it did not know about."""
    pts, _ = _ring((0.0, 0.0, 0.8))
    start, end = pts[0], pts[-1]
    assert float(np.cross(start, end)[2]) > 0.0
    # ...and the other way about -z, which is the same rotation seen from behind.
    pts, _ = _ring((0.0, 0.0, -0.8))
    assert float(np.cross(pts[0], pts[-1])[2]) < 0.0


def test_the_arc_is_centred_on_the_point_nearest_the_camera():
    """So the drawn piece faces the viewer and the head at its end is never the
    part that goes behind the bead."""
    camera = _camera()
    pts, _ = _ring((0.0, 0.0, 0.9), camera=camera)
    d = np.linalg.norm(pts - camera.eye[None, :], axis=1)
    assert int(np.argmin(d)) == (len(pts) - 1) // 2
    # Both ends are the same distance away: the arc is symmetric about that point.
    assert d[0] == pytest.approx(d[-1])


def test_the_camera_is_what_makes_it_an_ellipse():
    """The test the old screen-space circle could not have passed. One torque, two
    axes, same magnitude: about the axis pointing AT the camera the ring projects
    to a round shape; about an axis lying across the view it is seen edge on and
    collapses to a line. Nothing here says "ellipse" -- the projection does."""
    camera = _camera()
    def extent(vec):
        pts, _ = _ring(vec, camera=camera)
        scr, _, _ = camera.project(pts)
        scr = scr - scr.mean(axis=0)
        # Singular values of the projected cloud: the two axes of the ellipse.
        s = np.linalg.svd(scr, compute_uv=False)
        return s[1] / s[0]
    face_on = extent((0.0, -1.0, 0.0))      # axis along the view direction
    edge_on = extent((0.0, 0.0, 1.0))       # axis across it
    assert face_on > 0.4, "a ring turned toward the camera should read as round"
    assert edge_on < 0.02, "a ring seen edge on should read as a line"
    assert face_on > 20.0 * edge_on


def test_only_the_beads_the_ring_reaches_can_hide_it():
    """The clip is against real spheres, so it has to look at them -- but only at
    the handful the ring physically passes through. On a 50k scene the alternative
    is fifty thousand sphere tests per sample."""
    r = Renderer.__new__(Renderer)          # geometry only; no display needed
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [9.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0]])
    depth = np.array([10.0, 10.0, 10.0, np.inf])
    occ = r._ring_occluders(pts, depth, CENTER, reach=1.5, shown=None)
    assert list(occ) == [0, 1], "a bead out of reach, or behind the lens, is not one"
    # A bead the slice cut away is not there to hide anything.
    shown = np.array([True, False, True, True])
    assert list(r._ring_occluders(pts, depth, CENTER, 1.5, shown)) == [0]


# --- what the draw pass actually calls ----------------------------------------

@pytest.fixture(scope="module")
def renderer():
    import pygame

    pygame.display.init()
    pygame.font.init()
    yield Renderer((900, 700))
    pygame.display.quit()


def _spec(drive):
    """A minimal SystemSpec -- the overlay pass reads the drive, the bead radius
    and a few labels off it, and nothing else here needs a simulation."""
    from lammps_live.mdsystem import SliderSpec, SystemSpec
    from lammps_live.playground.spec import REDUCED_UNIT_FEEDBACK

    t = SliderSpec("Temperature", 0.0, 0.5, 0.001)
    return SystemSpec(
        key=f"{drive}-test", name=f"{drive} test", description="",
        element_label="bead", lattice_spacing=1.0, timestep=0.005,
        temperature=t, damping=SliderSpec("Puller damping", 0.0, 1.0, 0.1),
        melt_temp=0.3, force_feedback=REDUCED_UNIT_FEEDBACK,
        puller_speed_cap=1.0, max_input_force=1.0,
        control_drive=drive, render_3d=True, atom_radius_A=0.5,
        reduced_units=True, director_arrows=False)


def _overlay(renderer, monkeypatch, drive):
    """Run one overlay pass and report which of the three drawings it reached
    for."""
    calls = {"arrow": 0, "arc": 0, "ring": 0}
    for name, key in (("_draw_arrow_3d", "arrow"), ("_draw_torque_arc", "arc"),
                      ("_draw_torque_ring", "ring")):
        monkeypatch.setattr(Renderer, name,
                            (lambda k: lambda *a, **kw: calls.__setitem__(
                                k, calls[k] + 1))(key))
    camera = _camera()
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    screen, depth, scale = camera.project(pts)
    radii = np.maximum(0.5 * scale, 2.0)
    renderer._scene_style = _spec(drive).render_style
    renderer._draw_3d_overlays(
        camera, pts, screen, depth, radii, 0.5,
        np.array([True, False]), _spec(drive),
        input_force=(1.0, 0.0), reaction_force=(-0.6, 0.0), fps=60.0,
        sim_time_ps=1.0, total_steps=10, steps_per_frame=1, potential_terms=None,
        torque_signals=(1.0, -0.6),
        torque_vectors=(np.array([0.0, 1.0, 0.0]), np.array([0.0, -0.6, 0.2])),
        hud_lines=None, debug_line=None)
    return calls


def test_a_torque_drive_draws_rings_and_no_straight_arrows(renderer, monkeypatch):
    """The point of the exercise. Nothing pushes the bead on this drive, so nothing
    at the bead is allowed to look like a push -- and the flat arc does not run
    alongside the rings either, which would be the same torque drawn twice, once
    truthfully and once flattened."""
    calls = _overlay(renderer, monkeypatch, "torque")
    assert calls == {"arrow": 0, "arc": 0, "ring": 2}


def test_a_force_drive_still_draws_its_arrows_and_its_flat_arcs(renderer,
                                                                monkeypatch):
    """Unchanged, and deliberately: a force IS a push along a line, and the twist
    there is one secondary axis, the normal of the plane those scenes are looked
    down -- the one axis a circle drawn flat is honest about."""
    calls = _overlay(renderer, monkeypatch, "force")
    assert calls == {"arrow": 2, "arc": 2, "ring": 0}

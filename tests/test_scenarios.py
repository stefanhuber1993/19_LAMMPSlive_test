"""Scenario, parameter and geometry tests -- all pure numpy, no LAMMPS.

Being able to test the geometry and the parameter rules without starting a
simulation is one of the concrete wins of splitting them out of the system class:
none of this was reachable before without building a LAMMPS instance.
"""
import math

import numpy as np
import pytest

from lammps_live.playground.params import Param, ParamSet, Tier, structural
from lammps_live.playground.scenario import (
    HexPatch, HexSheet, RandomFill, align_normal_rate, compose, hex_patch,
)
from lammps_live.playground.state import (
    Box, build_pairs, hex_lattice_2d, hex_ring_2d, principal_normal,
)


# --- geometry -----------------------------------------------------------------

def test_hex_ring_is_centre_first_and_angularly_ordered():
    """Scenarios build bond lists from these indices (spokes 0-k, then the closed
    ring k -> k+1), so centre-first and angle-ordered is a contract."""
    pts = hex_ring_2d(1, a=1.0)
    assert len(pts) == 7
    assert np.allclose(pts[0], (0.0, 0.0))
    # Ring at unit distance, angles 0, 60, ... 300 degrees in order.
    ring = pts[1:]
    assert np.allclose(np.linalg.norm(ring, axis=1), 1.0)
    angles = np.degrees(np.arctan2(ring[:, 1], ring[:, 0])) % 360.0
    assert np.allclose(angles, [0, 60, 120, 180, 240, 300], atol=1e-9)


def test_hex_ring_shell_counts():
    """Shell k of a triangular lattice holds 6k sites."""
    for n_rings in (1, 2, 3):
        pts = hex_ring_2d(n_rings, a=1.0)
        assert len(pts) == 1 + sum(6 * k for k in range(1, n_rings + 1))


def test_hex_lattice_is_centred_and_tiles_its_cell():
    a, n_cols, n_rows = 0.8, 30, 30
    pts = hex_lattice_2d(n_cols, n_rows, a)
    assert len(pts) == n_cols * n_rows
    assert np.allclose(pts.mean(axis=0), 0.0)
    # The cell the sheet scenario derives from these counts must contain them.
    lx, ly = n_cols * a, n_rows * a * math.sqrt(3.0) / 2.0
    assert np.ptp(pts[:, 0]) < lx
    assert np.ptp(pts[:, 1]) < ly


def test_principal_normal_of_a_tilted_plane():
    """The normal-up housekeeping and the tilt observable both rest on this."""
    rng = np.random.default_rng(0)
    pts2d = rng.uniform(-1, 1, size=(200, 2))
    # A plane tilted 30 degrees about x: z = tan(30) * y.
    t = math.radians(30.0)
    pts = np.column_stack([pts2d[:, 0], pts2d[:, 1], math.tan(t) * pts2d[:, 1]])
    n = principal_normal(pts)
    assert n[2] > 0.0                                  # sign-fixed upward
    assert np.degrees(math.acos(min(1.0, n[2]))) == pytest.approx(30.0, abs=1e-6)


def test_align_rate_vanishes_for_a_flat_cloud():
    rng = np.random.default_rng(1)
    flat = np.column_stack([rng.uniform(-1, 1, 100), rng.uniform(-1, 1, 100),
                            np.zeros(100)])
    assert np.allclose(align_normal_rate(flat, 10.0), 0.0, atol=1e-9)


# --- scenarios ----------------------------------------------------------------

def test_hex_patch_build():
    s = HexPatch(n_rings=1, a=1.0, box=6.0)
    build = s.build(s.new_params(), np.random.default_rng(0))
    assert len(build.positions) == 7
    assert np.allclose(build.positions[:, 2], 0.0)         # lies in the xy plane
    assert np.allclose(build.directors, (0.0, 0.0, 1.0))   # directors along +z
    assert build.box.periodic == (False, False, False)
    # 6 spokes + a closed 6-segment ring.
    assert len(build.bonds) == 12


def test_hex_sheet_box_matches_the_lattice():
    """The cell must be sized exactly to the lattice, or the periodic sheet does
    not tile seamlessly -- which is the whole reason it holds itself flat."""
    s = HexSheet(n_cols=10, n_rows=10, a=0.8, z_half=4.0)
    build = s.build(s.new_params(), np.random.default_rng(0))
    assert len(build.positions) == 100
    assert build.box.periodic == (True, True, False)
    assert build.box.lengths[0] == pytest.approx(10 * 0.8)
    assert build.box.lengths[1] == pytest.approx(10 * 0.8 * math.sqrt(3) / 2)


def test_sheet_tracer_marks_one_cluster_of_seven():
    s = HexSheet(n_cols=20, n_rows=20, a=0.8)
    build = s.build(s.new_params(), np.random.default_rng(0))
    b = build.brightness
    assert b is not None
    assert np.count_nonzero(b > 1.0) == 7      # a centre plus six neighbours
    assert b.max() == pytest.approx(2.1)


def test_random_fill_defers_placement_to_lammps():
    s = RandomFill(n=50, box=10.0)
    params = s.new_params()
    build = s.build(params, np.random.default_rng(0))
    assert len(build.positions) == 0            # LAMMPS places them
    assert build.box.periodic == (True, True, True)
    cmds = s.atom_creation_commands(params, seed=1234)
    assert any("create_atoms" in c and "random 50 1234" in c for c in cmds)


def test_housekeeping_excludes_the_controlled_particle():
    """Its position IS the user's input; a correction force there would fight it."""
    s = HexPatch()
    params = s.new_params()
    pts = s.build(params, np.random.default_rng(0)).positions
    f = s.housekeeping(pts, params, controlled=0)
    assert np.allclose(f[0], 0.0)
    assert not np.allclose(f[1:], 0.0)


def test_compose_stacks_geometry_and_reindexes_bonds():
    scenario = compose(hex_patch(at=(-6, 0, 0)), hex_patch(at=(+6, 0, 0)))
    build = scenario.build(scenario.new_params(), np.random.default_rng(0))
    assert len(build.positions) == 14
    # Each patch centred on its offset.
    assert build.positions[0][0] == pytest.approx(-6.0)
    assert build.positions[7][0] == pytest.approx(+6.0)
    # The second patch's bonds must point into its own atoms, not the first's.
    assert (7, 8) in build.bonds
    assert max(max(b) for b in build.bonds) == 13


def test_wall_commands_cover_only_non_periodic_faces():
    s = HexSheet()
    walls = s.wall_commands(Box.centered(10, 10, 8, periodic=(True, True, False)))
    assert len(walls) == 1
    assert "zlo EDGE" in walls[0] and "zhi EDGE" in walls[0]
    assert "xlo" not in walls[0]
    # A fully periodic cell needs none.
    assert s.wall_commands(Box.cube(10, (True, True, True))) == []


def test_scenario_kwargs_split_into_params_and_attributes():
    s = HexPatch(n_rings=2, timestep=0.001, sim_time_per_frame=0.02)
    assert s.timestep == 0.001              # attribute, not a parameter
    assert s.sim_time_per_frame == 0.02
    assert s.new_params()["n_rings"] == 2    # parameter


# --- parameters ---------------------------------------------------------------

def test_clamp_is_applied_wherever_the_value_is_read():
    """The wc <= rc rule was a helper each old system had to remember to call.
    Declared on the Param, it applies to every read."""
    params = ParamSet.build([
        Param("rc", 2.5, vmin=0.0, vmax=3.0),
        Param("wc", 2.0, vmin=0.0, vmax=3.0, clamp=lambda v, p: min(v, p["rc"])),
    ])
    assert params["wc"] == pytest.approx(2.0)
    params.set("rc", 1.2)
    assert params["wc"] == pytest.approx(1.2)      # clamped by the new rc
    assert params.raw("wc") == pytest.approx(2.0)  # raw value preserved
    assert params.as_dict()["wc"] == pytest.approx(1.2)


def test_set_reports_whether_the_value_changed():
    """The app pushes every slider every frame, so a cheap no-op matters."""
    params = ParamSet.build([Param("k", 1.0, vmin=0.0, vmax=5.0)])
    assert params.set("k", 2.0) is True
    assert params.set("k", 2.0) is False
    assert params.set("nonexistent", 1.0) is False


def test_unknown_override_raises_rather_than_being_ignored():
    """A typo in a preset would otherwise look like the preset having no effect."""
    with pytest.raises(KeyError, match="unknown parameter"):
        ParamSet.build([Param("k", 1.0)], {"kk": 2.0})


def test_structural_params_generate_no_sliders():
    params = ParamSet.build([
        structural("n_beads", 100),
        Param("k", 1.0, "k", 0.0, 5.0),
        Param("rc", 2.5, "rc", 0.0, 3.0, tier=Tier.HOT_RESTYLE),
    ])
    keys = [s.key for s in params.slider_specs()]
    assert keys == ["k", "rc"]          # n_beads is file-time only


def test_slider_range_override():
    params = ParamSet.build([Param("k_splay", 1.0, "k_splay", 0.0, 3.0)])
    wide = params.slider_specs({"k_splay": (0.0, 40.0)})
    assert (wide[0].vmin, wide[0].vmax) == (0.0, 40.0)
    assert params.spec("k_splay").vmax == 3.0     # declaration untouched


# --- pair list ----------------------------------------------------------------

def test_pair_list_respects_minimum_image():
    """Two particles either side of a periodic seam are neighbours."""
    box = Box((0.0, 0.0, 0.0), (10.0, 10.0, 10.0), periodic=(True, True, True))
    pos = np.array([[0.5, 5.0, 5.0], [9.5, 5.0, 5.0]])
    pairs = build_pairs(pos, cutoff=2.0, box=box)
    assert len(pairs) == 1
    assert pairs.r[0] == pytest.approx(1.0)
    # Non-periodic, the same two are 9 apart and not neighbours.
    assert len(build_pairs(pos, cutoff=2.0, box=None)) == 0


def test_pair_list_tolerates_coordinates_a_hair_below_the_lower_bound():
    """LAMMPS reports x = -9e-16 for an atom nominally at 0. The modulo sends that
    to exactly L, which cKDTree rejects outright -- a crash the 2D deposition port
    hit for real."""
    box = Box((0.0, 0.0, 0.0), (10.0, 10.0, 10.0), periodic=(True, False, True))
    pos = np.array([[-9.3e-16, 1.0, 0.0], [0.5, 1.0, 0.0]])
    pairs = build_pairs(pos, cutoff=2.0, box=box)
    assert len(pairs) == 1


def test_touching_selects_a_particles_pairs():
    pos = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [8.0, 0, 0]])
    pairs = build_pairs(pos, cutoff=1.5, box=None)
    assert pairs.touching(0).sum() == 1     # only (0,1)
    assert pairs.touching(1).sum() == 2     # (0,1) and (1,2)
    assert pairs.touching(3).sum() == 0     # isolated

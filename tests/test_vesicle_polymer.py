"""The vesicle-with-polymer scenario, and the force field that runs on it.

Split in two on purpose. The first half is pure geometry -- an icosphere and a
Hamiltonian cycle on a cubic lattice -- and it is where the guarantees the rest of
the setup RELIES on are pinned: every bond at the bond length, no two beads
overlapping, the ring closed. Those are what let the polymer be handed straight to
a FENE bond with no soft-core push-off, so if they stop holding the failure is a
simulation that explodes on the first step, and it should be caught here instead.

The second half builds a small one in LAMMPS, because the deck has a shape nothing
else in this codebase has: bonded particles arriving through a molecule template,
after the atoms the runtime placed itself.
"""
import builtins
import math

import numpy as np
import pytest

from lammps_live.playground import registry
from lammps_live.playground.scenario import VesiclePolymer
from lammps_live.playground.state import (icosphere_faces, icosphere_spacing,
                                          lattice_ring,
                                          nearest_neighbour_distances)

# Small enough to build in a second, and shaped like the real one: an icosphere
# with several rings inside it.
SMALL = dict(n_membrane=1280, n_polymer=400, ring_side=4, a=0.8,
             fill_fraction=0.90, settle_steps=20)


# --- the geometry -------------------------------------------------------------

@pytest.mark.parametrize("nu", [1, 2, 5, 12])
def test_icosphere_has_20_nu_squared_faces_on_the_unit_sphere(nu):
    centres, normals = icosphere_faces(nu)
    assert len(centres) == 20 * nu * nu
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
    # The normals point OUT, which is what makes the vesicle's directors right
    # without a per-face sign check.
    assert np.all(np.einsum("ij,ij->i", normals, centres) > 0)


def test_icosphere_spacing_is_the_real_nearest_neighbour_distance():
    """The radius of every vesicle is `a / icosphere_spacing(nu)`, so this number
    being right is the difference between a membrane at its relaxed packing and
    one that immediately buckles."""
    cKDTree = pytest.importorskip("scipy.spatial").cKDTree
    for nu in (4, 8, 16):
        centres, _ = icosphere_faces(nu)
        d, _ = cKDTree(centres).query(centres, k=2)
        measured = float(d[:, 1].mean())
        assert icosphere_spacing(nu) == pytest.approx(measured, rel=0.02)


def test_nearest_neighbour_distances_are_exact():
    """The pure-numpy sweep that replaced the KD-tree, against brute force. Small
    enough clouds that the O(N^2) oracle is the cheap way round."""
    rng = np.random.default_rng(3)
    clouds = [
        rng.normal(size=(2, 3)),
        rng.normal(size=(400, 3)),
        icosphere_faces(6)[0],
        # Coincident points, and a cloud flat in z -- both make the sweep's band
        # degenerate, and both must still come out exact.
        np.zeros((16, 3)),
        np.column_stack([rng.normal(size=(60, 2)), np.zeros(60)]),
        # Two clumps far apart on z: the first band is far wider than the spacing
        # inside a clump, which is the other side of the adaptive band.
        np.vstack([rng.normal(size=(80, 3)), rng.normal(size=(80, 3)) + (0, 0, 400)]),
    ]
    for pts in clouds:
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        assert np.allclose(nearest_neighbour_distances(pts), d.min(axis=1)), (
            f"{pts.shape} cloud")
    # One point has no neighbour to be near.
    assert nearest_neighbour_distances(np.zeros((1, 3))).tolist() == [0.0]


def test_icosphere_spacing_is_bounded():
    """The packing varies by about half end to end and STAYS bounded as the
    subdivision grows -- a lat/long grid's crowding at its poles is unbounded in
    exactly this limit, which is why it is not used (see icosphere_spacing). Half
    is a lot; the membrane is a fluid and relaxes it out, and this is a bound
    rather than a target."""
    cKDTree = pytest.importorskip("scipy.spatial").cKDTree
    for nu in (8, 16, 32):
        centres, _ = icosphere_faces(nu)
        d, _ = cKDTree(centres).query(centres, k=2)
        assert d[:, 1].max() / d[:, 1].min() < 1.6


def test_lattice_ring_is_a_closed_self_avoiding_walk():
    for dims in ((4, 4, 4), (8, 8, 8), (6, 4, 3), (2, 2, 2)):
        ring = lattice_ring(*dims)
        n = dims[0] * dims[1] * dims[2]
        assert len(ring) == n
        assert len(np.unique(ring, axis=0)) == n, f"{dims}: a site is visited twice"
        # Closed: the step from the last site back to the first counts.
        steps = np.diff(np.vstack([ring, ring[:1]]), axis=0)
        assert np.all(np.abs(steps).sum(axis=1) == 1), f"{dims}: a bond is not one step"


def test_lattice_ring_rejects_a_grid_it_cannot_close():
    with pytest.raises(ValueError, match="even ny"):
        lattice_ring(4, 5, 4)
    with pytest.raises(ValueError, match="at least 2"):
        lattice_ring(4, 4, 1)


# --- the scenario's build -----------------------------------------------------

def _built(**overrides):
    scenario = VesiclePolymer(**{**SMALL, **overrides})
    params = scenario.new_params()
    return scenario, params, scenario.build(params, np.random.default_rng(7))


def test_build_describes_the_whole_system_but_uploads_only_the_membrane():
    scenario, params, build = _built()
    _, n_mem = scenario.subdivision(params)
    assert build.n_uploaded == n_mem
    assert len(build.positions) == scenario.particle_count(params)
    assert np.count_nonzero(build.types == 1) == n_mem
    assert np.count_nonzero(build.types == 2) == len(build.positions) - n_mem


def test_the_membrane_is_a_sphere_with_radial_directors():
    scenario, params, build = _built()
    n_mem = build.n_uploaded
    p = build.positions[:n_mem]
    r = np.linalg.norm(p, axis=1)
    assert np.allclose(r, scenario.radius(params))
    assert np.allclose(build.directors[:n_mem], p / r[:, None])
    # And the polymer carries no orientation at all -- it has none to carry, and
    # the shader is told so by the zero (see gl3d's banded albedo).
    assert np.allclose(build.directors[n_mem:], 0.0)


def test_the_polymer_is_inside_the_vesicle_with_clearance():
    scenario, params, build = _built()
    radius = scenario.radius(params)
    r = np.linalg.norm(build.positions[build.n_uploaded:], axis=1)
    assert r.max() < radius - 1.0, "a chain starts inside the membrane"
    # ...and it actually fills the lumen rather than huddling in the middle.
    assert r.max() > 0.5 * radius


def test_every_polymer_bond_is_intact_and_no_two_beads_overlap():
    """The guarantee the whole no-push-off construction rests on. FENE breaks past
    r0 = 1.5 and a repulsive core explodes below about 0.8, so both ends matter."""
    cKDTree = pytest.importorskip("scipy.spatial").cKDTree
    scenario, params, build = _built()
    per_ring = int(params["ring_side"]) ** 3
    polymer = build.positions[build.n_uploaded:]
    rings = polymer.reshape(-1, per_ring, 3)
    for ring in rings:
        bonds = np.linalg.norm(np.diff(np.vstack([ring, ring[:1]]), axis=0), axis=1)
        assert bonds.max() < 1.4, "a bond starts beyond FENE's maximum extension"
    d, _ = cKDTree(polymer).query(polymer, k=2)
    assert d[:, 1].min() > 0.55, "two polymer beads start on top of each other"


def test_the_rings_do_not_all_face_the_same_way():
    """A melt of identically oriented rings is a lamellar phase, not a melt."""
    scenario, params, build = _built(n_polymer=1600, jitter=0.0)
    per_ring = int(params["ring_side"]) ** 3
    rings = build.positions[build.n_uploaded:].reshape(-1, per_ring, 3)
    assert len(rings) >= 4
    # The vector from a ring's first bead to its centre, per ring: identical
    # placements would make these all equal.
    spokes = rings[:, 0, :] - rings.mean(axis=1)
    assert len({tuple(np.round(v, 6)) for v in spokes} ) > 1


def test_a_target_the_lumen_cannot_hold_gives_a_smaller_centred_melt():
    scenario, params, build = _built(n_polymer=200000)
    radius = scenario.radius(params)
    r = np.linalg.norm(build.positions[build.n_uploaded:], axis=1)
    assert scenario.ring_count(params) >= 1
    assert r.max() < radius


def test_the_build_is_reproducible_from_the_seed():
    _, _, a = _built()
    _, _, b = _built()
    assert np.allclose(a.positions, b.positions)


def test_render_tints_leave_the_membrane_banded_and_paint_the_polymer():
    scenario, params, build = _built()
    tints = scenario.render_tints(params)
    n_mem = build.n_uploaded
    assert tints.shape == (len(build.positions), 4)
    assert np.all(tints[:n_mem, 3] == 0.0), "the membrane must keep its banding"
    assert np.all(tints[n_mem:, 3] == 1.0)
    # The ramp runs the full range within every ring, and comes back to where it
    # started so a closed ring shows no seam.
    per_ring = int(params["ring_side"]) ** 3
    first = tints[n_mem:n_mem + per_ring, :3]
    assert np.allclose(first[0], tints[n_mem + per_ring - 1: n_mem + per_ring, :3],
                       atol=30)
    assert np.ptp(first[:, 0]) > 100


def test_the_build_runs_with_no_scipy_installed(monkeypatch):
    """The remote server is given numpy and nothing else -- scipy is the client's,
    because the analysis is what needs it. A scipy import anywhere under `build`
    is a ModuleNotFoundError that only shows up once the cluster has allocated a
    node, which is how the last one was found, so pin it here instead."""
    real_import = builtins.__import__

    def no_scipy(name, *args, **kwargs):
        if name == "scipy" or name.startswith("scipy."):
            raise ModuleNotFoundError("No module named 'scipy'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_scipy)
    icosphere_spacing.cache_clear()   # or a warm cache hides the import
    scenario, params, build = _built()
    assert len(build.positions) == scenario.particle_count(params)


# --- the deck -----------------------------------------------------------------

pytest.importorskip("lammps")


@pytest.fixture(scope="module")
def small_system():
    """A miniature of the shipped playground, actually built in LAMMPS."""
    import dataclasses

    from lammps_live.playground.scenario import vesicle_polymer
    from lammps_live.playground.system import PlaygroundSystem

    playground = dataclasses.replace(
        registry.load("mesomem_polymer"),
        scenario=vesicle_polymer(**SMALL, timestep=0.005, sim_time_per_frame=0.05),
        remote=None, seed=99)
    system = PlaygroundSystem(playground, mode_name="sim")
    yield system
    system.close()


def test_the_deck_creates_both_species_and_the_topology(small_system):
    scenario = small_system.scenario
    params = small_system.scenario_params
    _, n_mem = scenario.subdivision(params)
    n_poly = scenario.ring_count(params) * int(params["ring_side"]) ** 3
    assert small_system.natoms == n_mem + n_poly
    types = np.asarray(small_system.frame_state().types)
    assert np.count_nonzero(types == 1) == n_mem
    assert np.count_nonzero(types == 2) == n_poly
    # A ring of L beads has L bonds and L angles -- the walk closes, so no free end.
    assert small_system.lmp.extract_global("nbonds") == n_poly
    assert small_system.lmp.extract_global("nangles") == n_poly


def test_the_membrane_directors_come_out_radial(small_system):
    state = small_system.frame_state()
    membrane = np.asarray(state.types) == 1
    p = state.positions[membrane]
    mu = state.directors[membrane]
    radial = np.einsum("ij,ij->i", mu, p / np.linalg.norm(p, axis=1)[:, None])
    assert radial.mean() > 0.9
    # The polymer's are left at zero: nothing integrates them and nothing reads
    # them, and a unit vector there would put a white pole cap on every chain bead.
    assert np.allclose(state.directors[~membrane], 0.0)


def test_it_runs_without_the_chains_blowing_up(small_system):
    small_system.step(400)
    state = small_system.frame_state()
    assert np.isfinite(state.positions).all()
    per_ring = int(small_system.scenario_params["ring_side"]) ** 3
    polymer = state.positions[np.asarray(state.types) == 2]
    rings = polymer.reshape(-1, per_ring, 3)
    closed = np.concatenate([rings, rings[:, :1]], axis=1)
    bonds = np.linalg.norm(np.diff(closed, axis=1), axis=2)
    assert bonds.max() < 1.5, "a FENE bond has been stretched past breaking"
    temp = small_system.get_thermo_state()[0]
    assert 0.0 < temp < 1.5


def test_the_polymer_stays_sealed_inside_the_vesicle(small_system):
    """Essentially all of it, not literally all of it.

    A monolayer is one bead thick and thermally rough, so a chain pressed against
    it sits partly WITHIN the shell -- and this test's vesicle is a tenth the
    shipped radius, so it is curved far past what the membrane's bending modulus
    is comfortable with and is rougher still. What would be a failure is the melt
    bursting out, which is a fraction, not a maximum. (Measured on the shipped
    size over 6000 steps: nothing gets out at all.)
    """
    state = small_system.frame_state()
    types = np.asarray(state.types)
    r = np.linalg.norm(state.positions, axis=1)
    outside = np.count_nonzero(r[types == 2] > r[types == 1].max())
    assert outside / np.count_nonzero(types == 2) < 0.01


def test_the_energy_panels_name_the_polymer_and_keep_it_off_the_membrane_terms(
        small_system):
    from lammps_live.forcefields.mesomem_polymer import EXCLUDED
    from lammps_live.forcefields.mesomem import ISO
    from lammps_live.playground.observables import analysis_pairs

    state = small_system.frame_state()
    ff, params = small_system.force_field, small_system.params
    pairs = analysis_pairs(ff, state, params)
    terms = ff.energy_terms(state, pairs, params)
    assert EXCLUDED in terms
    polymer = np.asarray(state.types) == 2
    touched = polymer[pairs.a] | polymer[pairs.b]
    # The membrane's own terms are evaluated only between membrane beads: on a
    # polymer pair they would be reading an absent director as an orientation.
    assert np.all(terms[ISO][touched] == 0.0)
    assert np.all(terms[EXCLUDED][~touched] == 0.0)
    # ...and the excluded-volume term is repulsive wherever it is anything at all.
    assert np.all(terms[EXCLUDED] >= 0.0)


def test_the_observables_report_a_filled_vesicle(small_system):
    values = dict(zip(("radius", "gyration", "contact"),
                      (float(line.split("=")[1]) for line in
                       small_system.get_hud_lines())))
    scenario, params = small_system.scenario, small_system.scenario_params
    assert values["radius"] == pytest.approx(scenario.radius(params), rel=0.1)
    # A uniformly filled sphere has Rg = sqrt(3/5) R; the melt starts short of
    # the wall, so it is below that and well clear of zero.
    assert 0.2 * values["radius"] < values["gyration"] < 0.8 * values["radius"]
    assert values["contact"] >= 0.0


def test_k_bend_reaches_the_angle_style(small_system):
    """The one live parameter the base class's pair-coefficient re-issue would
    silently drop."""
    assert small_system.force_field.live_commands(
        small_system.params, "k_bend") == ["angle_coeff 1 2.0"]
    small_system.set_extra_param("k_bend", 9.0)
    small_system.step(20)
    assert small_system.params["k_bend"] == pytest.approx(9.0)
    small_system.set_extra_param("k_bend", 2.0)


def test_the_cluster_profile_keeps_the_hybrid_sub_style_names():
    """`pair_coeff 1 1 mesomem <n values>` truncated to the cluster's shorter
    coefficient list must lose values, not the sub-style name -- and must lose the
    right number of them."""
    from lammps_live.remote import hosts

    from lammps_live.playground import forcefield

    ff = hosts.CLUSTER_GPU.with_coeff_values(8).adapt(
        forcefield.get("mesomem_polymer")())
    lines = ff.coeff_commands(ff.new_params())
    membrane = next(c for c in lines if c.startswith("pair_coeff 1 1"))
    assert membrane.split()[3] == "mesomem"
    assert len(membrane.split()) == 3 + 1 + 8
    for c in lines:
        if "lj/cut" in c:
            assert c.split()[3] == "lj/cut" and len(c.split()) == 3 + 1 + 3


# --- the remote pipeline ------------------------------------------------------
# The playground is a REMOTE one: the cluster integrates and this end draws. What
# is worth a loopback test of its own (tests/test_remote_loopback.py already
# covers the pipe itself) is the one thing this playground needs from it that no
# other does -- the client knowing WHICH SPECIES each bead is, over a wire that
# carries no such thing. It is recovered by building the same scenario at this
# end (see RemoteSystem.__init__), and if that stops lining up the failure is
# quiet: the energy panels start reporting membrane physics for the chains.

REMOTE_SOURCE = '''
"""A miniature of the vesicle playground, for the loopback test."""
from lammps_live.playground import Playground, vesicle_polymer
from lammps_live.remote import RemoteTarget

PLAYGROUND = Playground(
    name="loopback vesicle",
    description="a small vesicle with rings in it, served locally",
    force_field="mesomem_polymer",
    scenario=vesicle_polymer(n_membrane=1280, n_polymer=400, ring_side=4, a=0.8,
                             fill_fraction=0.90, settle_steps=20,
                             timestep=0.005, sim_time_per_frame=0.05),
    mode="sim",
    observables=["vesicle_radius", "polymer_gyration", "polymer_contact"],
    temperature_default=0.2,
    seed=4242,
    remote=RemoteTarget(host="localhost", profile="local", port=0, local_port=0),
)
'''


@pytest.fixture(scope="module")
def remote_pair(tmp_path_factory):
    """A local FrameServer for the miniature playground, and a client on it."""
    import socket
    import threading
    import time

    from lammps_live.remote import RemoteTarget
    from lammps_live.remote.server import FrameServer

    path = tmp_path_factory.mktemp("vesicle-remote") / "loopback_vesicle.py"
    path.write_text(REMOTE_SOURCE)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    srv = FrameServer(playground=str(path), profile="local", port=port,
                      bind="127.0.0.1", token="test-token-not-a-secret",
                      fps=0.0, verbose=False)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.05)
    else:
        pytest.fail("the server never started listening")

    client = registry.build(str(path),
                            remote_override=RemoteTarget(host="127.0.0.1",
                                                         local_port=port,
                                                         profile="local"))
    client.connect("127.0.0.1", port, "test-token-not-a-secret", timeout=60.0)
    client.set_playing(True)
    seen, deadline = 0, time.monotonic() + 60.0
    while seen < 6 and time.monotonic() < deadline:
        before = client._seq
        client.step(20)
        if client._seq != before:
            seen += 1
    assert seen >= 6, f"only {seen} frames arrived"
    yield client
    client.close()
    srv.stop()
    thread.join(timeout=30.0)
    srv.close()


def test_the_client_knows_the_species_without_being_told(remote_pair):
    state = remote_pair._state
    assert state is not None and state.types is not None
    assert len(state.types) == len(state.positions)
    n_mem = remote_pair.scenario.subdivision(remote_pair.scenario_params)[1]
    assert np.count_nonzero(state.types == 1) == n_mem
    # And the two species really are where the build said: the membrane is the
    # shell, so it is the one at large radius.
    r = np.linalg.norm(state.positions, axis=1)
    assert r[state.types == 1].mean() > r[state.types == 2].mean()


def test_the_client_paints_the_polymer_from_its_own_build(remote_pair):
    tints = remote_pair.get_bead_tints()
    assert tints is not None and len(tints) == remote_pair.natoms
    n_mem = remote_pair.scenario.subdivision(remote_pair.scenario_params)[1]
    assert np.all(tints[:n_mem, 3] == 0.0)
    assert np.all(tints[n_mem:, 3] == 1.0)


def test_the_polymer_observables_run_at_the_drawing_end(remote_pair):
    # ...plus the link's own line, which a remote system appends.
    lines = [line for line in remote_pair.get_hud_lines() if "=" in line]
    assert len(lines) == 3
    values = [float(line.split("=")[1]) for line in lines]
    assert all(np.isfinite(v) for v in values)
    radius, gyration, _contact = values
    assert radius == pytest.approx(
        remote_pair.scenario.radius(remote_pair.scenario_params), rel=0.15)
    assert 0.0 < gyration < radius


def test_the_energy_panel_separates_the_two_species_over_the_wire(remote_pair):
    from lammps_live.forcefields.mesomem_polymer import EXCLUDED

    terms = remote_pair.get_total_potential_terms()
    assert terms is not None
    labels = [label for label, _value, _color in terms] if isinstance(
        terms[0], tuple) else list(terms)
    assert any(EXCLUDED in str(entry) for entry in labels)

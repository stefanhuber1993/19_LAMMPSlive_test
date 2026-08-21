"""The analysis' pair-list budget: above Analysis.MAX_PAIR_BEADS the pair work
runs on a subsample, and what it feeds is corrected for the dilution.

The point of these tests is that the correction is UNBIASED, not merely present:
a sample of fraction f keeps each particle with probability f and each pair with
probability f^2, so a per-particle mean needs one factor back and a pair sum two.
Getting that wrong is invisible on the small playgrounds (which never sample) and
silently rescales every number on the big one.
"""
import numpy as np
import pytest

from lammps_live.playground import forcefield as ff_registry
from lammps_live.playground.observables import Analysis
from lammps_live.playground.state import Box, FrameState, build_pairs

N = 20_000
BUDGET = 4_000


def _state(n=N, seed=0, spacing=1.06):
    """A condensed but PHYSICAL configuration: n beads on distinct sites of a
    jittered cubic lattice at just over one sigma, in a fully periodic cube.

    Non-overlapping on purpose. A cloud with beads at r << sigma has an isotropic
    energy dominated by a handful of 1/r^12 spikes, and a sum over a heavy tail
    like that needs far more samples to converge than the estimator itself is at
    fault for -- which would make these tests measure the configuration rather
    than the correction. Real MD never has those overlaps; the repulsive core is
    what stops them.
    """
    rng = np.random.default_rng(seed)
    # A lattice with room for several times n, so the picked sites form dense
    # patches with gaps rather than one solid block.
    side = int(np.ceil((3.0 * n) ** (1.0 / 3.0)))
    box_side = side * spacing
    box = Box.cube(box_side, periodic=(True, True, True))
    grid = np.stack(np.meshgrid(*(np.arange(side),) * 3, indexing="ij"), axis=-1)
    sites = grid.reshape(-1, 3) * spacing - box_side / 2.0
    # Clustered rather than uniform: the n sites nearest a handful of seeds, so
    # the pair count per bead varies the way it does in an assembled run instead
    # of being the same everywhere.
    seeds = sites[rng.choice(len(sites), 12, replace=False)]
    to_nearest_seed = np.linalg.norm(sites[:, None, :] - seeds[None, :, :],
                                    axis=2).min(axis=1)
    pos = sites[np.argsort(to_nearest_seed)[:n]]
    pos = pos + rng.uniform(-0.02, 0.02, size=(n, 3))
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    return FrameState(positions=box.wrap(pos), directors=d, types=None,
                      ids=np.arange(1, n + 1), box=box)


def _analysis(budget, names=("coordination",), energy_every=1):
    ff = ff_registry.get("mesomem")()
    a = Analysis(ff, names, energy_every=energy_every)
    a.MAX_PAIR_BEADS = budget
    return a


def test_below_the_budget_nothing_is_sampled():
    """A playground under the budget must get exactly the analysis it got before
    this existed -- every local playground is in this case."""
    st = _state(n=1500)
    a = _analysis(BUDGET)
    a.update(st, a.force_field.new_params(), force=True)
    assert a.pairs.dilution == 1.0
    exact = build_pairs(st.positions, a.force_field.interaction_cutoff(
        a.force_field.new_params()), st.box)
    assert len(a.pairs) == len(exact)


def test_above_the_budget_samples_and_says_so():
    st = _state()
    a = _analysis(BUDGET)
    a.update(st, a.force_field.new_params(), force=True)
    assert a.pairs.dilution == pytest.approx(BUDGET / N)
    # f^2 of the pairs, give or take sampling noise -- the check that the
    # dilution being reported is the one that was actually applied.
    params = a.force_field.new_params()
    full = len(build_pairs(st.positions,
                           a.force_field.interaction_cutoff(params), st.box))
    expected = full * (BUDGET / N) ** 2
    assert len(a.pairs) == pytest.approx(expected, rel=0.15)


def test_coordination_survives_the_sampling():
    """The HUD number has to mean the same thing sampled or not: it is a
    per-particle mean, so it needs one factor of the dilution back."""
    st = _state()
    params = _analysis(BUDGET).force_field.new_params()

    exact = _analysis(10 ** 9)
    exact.update(st, params, force=True)
    truth = exact.values()["coordination"]
    assert truth > 5.0        # a condensed cloud, or the test proves nothing

    # Averaged over independent samples, the way the running app sees it: each
    # analysis frame redraws, so the noise averages away over a second.
    sampled = _analysis(BUDGET)
    seen = []
    for _ in range(12):
        sampled.update(st, params, force=True)
        seen.append(sampled.values()["coordination"])
    assert np.mean(seen) == pytest.approx(truth, rel=0.05)


def test_energy_panel_totals_survive_the_sampling():
    """The panel is a pair SUM, so it needs the square of the dilution back."""
    st = _state()
    params = _analysis(BUDGET).force_field.new_params()

    exact = _analysis(10 ** 9, names=())
    exact.update(st, params, force=True)
    truth = dict(exact.energy_panel("t", 1.0)[1])

    sampled = _analysis(BUDGET, names=())
    totals = {}
    for _ in range(12):
        sampled.update(st, params, force=True)
        for label, value in sampled.energy_panel("t", 1.0)[1]:
            totals.setdefault(label, []).append(value)

    assert set(totals) == set(truth)
    for label, values in totals.items():
        if abs(truth[label]) < 1e-9:
            continue
        assert np.mean(values) == pytest.approx(truth[label], rel=0.08), label


def test_the_sample_is_redrawn_every_analysis_frame():
    """A fixed sample would be a fixed wrong answer; a redrawn one is noise that
    averages away."""
    st = _state()
    a = _analysis(BUDGET)
    params = a.force_field.new_params()
    counts = set()
    for _ in range(6):
        a.update(st, params, force=True)
        counts.add(len(a.pairs))
    assert len(counts) > 1


def test_naming_a_particle_switches_sampling_off():
    """The single-particle energy panel cannot be built from a sample -- one
    bead's share of its own neighbours would be a handful of pairs -- so a
    playground with a puller keeps the exact pair list."""
    st = _state()
    a = _analysis(BUDGET)
    a.update(st, a.force_field.new_params(), force=True, keep_index=0)
    assert a.pairs.dilution == 1.0
    panel = a.energy_panel("one bead", 1.0, index=0)
    assert panel is not None and panel[1]


def test_a_sampled_frame_refuses_a_single_particle_panel():
    """Rather than drawing one particle's energy off pairs numbered within a
    sample it is probably not even in."""
    st = _state()
    a = _analysis(BUDGET)
    a.update(st, a.force_field.new_params(), force=True)
    assert a.energy_panel("one bead", 1.0, index=0) is None
    assert a.energy_panel("whole system", 1.0) is not None

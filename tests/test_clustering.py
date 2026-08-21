"""The cluster colouring: does it find the right aggregates, and does it hold
still?

Two halves, and the second is the one worth having. The geometry -- connected
components of a contact graph, across periodic walls -- is checked against a
brute-force O(N^2) reference on random configurations, because the fast path is a
cell list plus a label-propagation loop and both are exactly the kind of code that
is right on the cases you thought of.

The identity is checked against the failure it exists to prevent: a colouring
that changes when nothing has. So the tests below build the events that actually
happen in the assembly box -- a bead rattling across the cutoff, two aggregates
merging, one splitting -- and assert what the picture does, not what the
algorithm does.
"""
import numpy as np
import pytest

from lammps_live.playground import clustering as cl
from lammps_live.playground.clustering import ClusterTracker
from lammps_live.playground.state import Box


def _brute_labels(positions, box, cutoff):
    """Connected components the slow, obviously-correct way: every pair, then a
    flood fill. O(N^2), so only for the small cases here."""
    n = len(positions)
    d = box.minimum_image(positions[None, :, :] - positions[:, None, :])
    near = np.einsum("ijk,ijk->ij", d, d) < cutoff * cutoff
    np.fill_diagonal(near, False)
    label = np.full(n, -1)
    group = 0
    for seed in range(n):
        if label[seed] >= 0:
            continue
        stack = [seed]
        label[seed] = group
        while stack:
            node = stack.pop()
            for other in np.flatnonzero(near[node]):
                if label[other] < 0:
                    label[other] = group
                    stack.append(other)
        group += 1
    return label


def _same_partition(a, b):
    """Two labellings that group the beads identically, however they number
    them -- which is the only thing a component labelling promises."""
    a, b = np.asarray(a), np.asarray(b)
    pairs = set(zip(a.tolist(), b.tolist()))
    return len(pairs) == len(set(a.tolist())) == len(set(b.tolist()))


def _grid_of_blobs(nx, ny, nz, spacing, per_blob, spread=0.35, seed=0):
    """A lattice of compact blobs, and the bead indices of each. The shape a
    dilute cell actually has: many small aggregates, well separated.

    The lattice is JITTERED, which matters: on an exact lattice a blob has six
    neighbours at identical distances, "the four nearest" is not a well-defined
    set, and the code and the test would each pick a different four of the six
    and disagree about what a clash is."""
    rng = np.random.default_rng(seed)
    pts, blobs, at = [], [], 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cell = np.array([i, j, k], float) - [nx / 2, ny / 2, nz / 2]
                centre = cell * spacing + rng.uniform(-0.2, 0.2, 3) * spacing
                pts.append(centre + rng.normal(0, spread, (per_blob, 3)))
                blobs.append(np.arange(at, at + per_blob))
                at += per_blob
    return np.concatenate(pts), blobs


def _blobs(centres, per_blob, spread=0.35, seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([np.asarray(c, float) + rng.normal(0, spread,
                                                             (per_blob, 3))
                           for c in centres])


# ---- the geometry ------------------------------------------------------------

@pytest.mark.parametrize("periodic", [(False, False, False), (True, True, True),
                                      (True, True, False)])
def test_the_cell_list_finds_the_same_components_as_every_pair(periodic):
    """The fast path against the slow one, on configurations dense enough to have
    real clusters and sparse enough to have several."""
    rng = np.random.default_rng(7)
    for _ in range(12):
        box = Box.cube(float(rng.uniform(4.0, 9.0)), periodic)
        n = int(rng.integers(20, 150))
        pos = rng.uniform(box.lo[0], box.hi[0], (n, 3))
        cutoff = float(rng.uniform(0.6, 1.8))
        assert _same_partition(cl.cluster_labels(pos, box, cutoff),
                               _brute_labels(pos, box, cutoff))


def test_an_aggregate_straddling_a_periodic_wall_is_one_cluster():
    """The reason minimum images are in here at all. A blob centred on the box
    face is one object; in a cell list that forgot to wrap it is two, sitting at
    opposite edges of the picture in different colours -- which on a scene that
    draws its periodic images is the most visible bug this could have.
    """
    box = Box.cube(10.0, (True, True, True))
    pos = box.wrap(_blobs([(5.0, 0.0, 0.0)], 40, spread=0.3))   # astride x = +/-5
    assert pos[:, 0].min() < -4.0 and pos[:, 0].max() > 4.0     # really astride it
    assert len(set(cl.cluster_labels(pos, box, 1.25).tolist())) == 1
    # ... and it is genuinely the wrapping that does it, not the blob being small.
    free = Box.cube(10.0, (False, False, False))
    assert len(set(cl.cluster_labels(pos, free, 1.25).tolist())) == 2


def test_the_cutoff_is_where_a_cluster_stops():
    """Two beads either side of the contact distance: in, then out."""
    box = Box.cube(20.0, (False, False, False))
    for gap, groups in ((1.2, 1), (1.3, 2)):
        pos = np.array([[0.0, 0.0, 0.0], [gap, 0.0, 0.0]])
        assert len(set(cl.cluster_labels(pos, box, 1.25).tolist())) == groups


def test_a_long_chain_is_one_cluster():
    """Connectivity is transitive and the label has to travel the whole way. A
    200-bead chain is 200 hops, which is what separates hooking onto a tree's ROOT
    from hooking onto its neighbour -- the latter would need 200 rounds and is
    capped well below that."""
    box = Box.cube(400.0, (False, False, False))
    pos = np.zeros((200, 3))
    pos[:, 0] = np.arange(200) * 1.0
    assert len(set(cl.cluster_labels(pos, box, 1.25).tolist())) == 1


def test_a_cell_list_with_too_few_cells_still_sees_every_pair():
    """A box only a couple of cutoffs across cannot have a 3x3x3 stencil of
    distinct cells, and the wrapped offsets fold back onto the cell they came
    from. That has to cost duplicate pairs, not missing ones."""
    box = Box.cube(2.6, (True, True, True))
    rng = np.random.default_rng(1)
    pos = rng.uniform(-1.3, 1.3, (30, 3))
    assert _same_partition(cl.cluster_labels(pos, box, 1.25),
                           _brute_labels(pos, box, 1.25))


def test_a_scene_with_nothing_in_it_does_not_crash():
    box = Box.cube(10.0, (True, True, True))
    for n in (0, 1):
        labels = cl.cluster_labels(np.zeros((n, 3)), box, 1.25)
        assert len(labels) == n


# ---- the identity ------------------------------------------------------------

def _settle(tracker, pos, box, frames, start=0):
    """Run the tracker past its recompute cadence and its hysteresis, the way the
    app does: one call per frame, at increasing frame numbers."""
    slots = None
    for f in range(start, start + frames):
        slots = tracker.slots(pos, box, f)
    return slots, start + frames


BOX = Box.cube(20.0, (True, True, True))
CUTOFF = 1.25


def test_two_aggregates_get_two_colours_and_the_gas_gets_none():
    tracker = ClusterTracker(CUTOFF)
    pos = np.concatenate([_blobs([(-6, 0, 0), (6, 0, 0)], 40, seed=1),
                          np.array([[0.0, 0.0, 0.0], [0.0, 8.0, 0.0]])])
    slots, _ = _settle(tracker, pos, BOX, 20)
    assert sorted(set(slots[:80].tolist())) == [0, 1]     # two clusters, two slots
    assert set(slots[80:].tolist()) == {-1}               # the two loners are gas

def test_a_still_scene_never_changes_colour():
    """The whole point. Nothing has moved, so nothing may be repainted -- over
    far more frames than the recompute cadence."""
    tracker = ClusterTracker(CUTOFF)
    pos = _blobs([(-6, 0, 0), (6, 0, 0), (0, 6, 0)], 40, seed=2)
    first, frame = _settle(tracker, pos, BOX, 12)
    for _ in range(200):
        later = tracker.slots(pos, BOX, frame)
        frame += 1
        assert np.array_equal(later, first)


def test_a_bead_rattling_across_the_cutoff_never_reaches_the_screen():
    """A bead just outside a cluster, stepping in and out every other labelling.
    Its slot is genuinely ambiguous; the picture must pick one and keep it, which
    is what the commit hold buys."""
    tracker = ClusterTracker(CUTOFF)
    core = _blobs([(0, 0, 0)], 40, seed=3)
    inside = np.concatenate([core, [[1.6, 0.0, 0.0]]])
    outside = np.concatenate([core, [[3.0, 0.0, 0.0]]])
    settled, frame = _settle(tracker, inside, BOX, 12)
    seen = [settled]
    for k in range(60):
        seen.append(tracker.slots(outside if k % 2 else inside, BOX, frame))
        frame += tracker.RECLUSTER_EVERY      # one labelling per step
    assert all(np.array_equal(s, seen[0]) for s in seen), "the flicker got through"


def test_a_merge_recolours_the_smaller_aggregate():
    """Two clusters run into each other. One colour has to go, and it must be the
    small one's: repainting the large aggregate would change most of the screen to
    report an event that happened at its edge."""
    tracker = ClusterTracker(CUTOFF)
    big, small = _blobs([(-4, 0, 0)], 60, seed=4), _blobs([(4, 0, 0)], 12, seed=5)
    apart = np.concatenate([big, small])
    together = np.concatenate([big, small - (7.0, 0.0, 0.0)])
    before, frame = _settle(tracker, apart, BOX, 12)
    assert len(set(before.tolist())) == 2
    after, _ = _settle(tracker, together, BOX, 40, start=frame)
    assert len(set(after.tolist())) == 1, "the two did not become one"
    assert np.array_equal(after[:60], before[:60]), "the large one was repainted"


def test_a_split_leaves_the_larger_fragment_alone():
    """The same rule read backwards: the bigger piece is the continuation of the
    thing that was there, and the smaller one is what is new."""
    tracker = ClusterTracker(CUTOFF)
    joined = _blobs([(0, 0, 0)], 70, spread=0.5, seed=6)
    joined = joined[np.argsort(joined[:, 0])]        # split cleanly in x
    before, frame = _settle(tracker, joined, BOX, 12)
    assert len(set(before.tolist())) == 1
    torn = joined.copy()
    torn[:20, 0] -= 6.0                              # tear off the smaller end
    after, _ = _settle(tracker, torn, BOX, 40, start=frame)
    assert len(set(after.tolist())) == 2
    assert np.array_equal(after[20:], before[20:]), "the larger fragment moved colour"
    assert after[0] != after[-1]


def test_far_more_clusters_than_colours_still_all_get_one():
    """A dilute cell holds hundreds of aggregates and the palette holds ten.
    Naming ten and greying the rest answers the question for a twentieth of the
    picture, so colours repeat -- every cluster gets one."""
    tracker = ClusterTracker(CUTOFF)
    pos, blobs = _grid_of_blobs(6, 5, 4, spacing=7.0, per_blob=12)
    box = Box.cube(6 * 7.0 * 1.4, (True, True, True))
    slots, _ = _settle(tracker, pos, box, 20)
    assert len(blobs) == 120
    assert -1 not in set(slots.tolist()), "aggregates were left grey"
    assert len(set(slots.tolist())) == cl.PALETTE_SLOTS


def test_no_cluster_wears_a_near_neighbour_s_colour():
    """What repeating a colour is allowed to cost, and what it is not. Two reds on
    opposite sides of the cell are never confused; two reds touching each other
    say the wrong thing about the one question this colouring answers. So the
    guarantee is local: a cluster differs from each of its nearest neighbours."""
    tracker = ClusterTracker(CUTOFF)
    pos, blobs = _grid_of_blobs(5, 5, 3, spacing=7.0, per_blob=12)
    box = Box.cube(5 * 7.0 * 1.4, (True, True, True))
    slots, _ = _settle(tracker, pos, box, 20)

    of_blob = np.array([slots[b[0]] for b in blobs])
    assert all(len(set(slots[b].tolist())) == 1 for b in blobs)   # one per blob
    centres = np.array([pos[b].mean(axis=0) for b in blobs])
    d = np.linalg.norm(box.minimum_image(centres[None] - centres[:, None]), axis=-1)
    np.fill_diagonal(d, np.inf)
    near = np.argsort(d, axis=1)[:, :cl.NEIGHBOUR_COUNT]
    clashes = [(i, j) for i in range(len(blobs)) for j in near[i]
               if of_blob[i] == of_blob[j]]
    assert not clashes, f"{len(clashes)} neighbouring pairs share a colour"


def test_the_palette_is_spread_evenly_rather_than_first_come_first_served():
    """Left to itself, "any colour a neighbour is not using" hands most of the box
    to whichever colour comes first in the palette. Balancing the load is what
    keeps a hundred clusters looking like ten colours rather than one colour and
    nine accents."""
    tracker = ClusterTracker(CUTOFF)
    pos, blobs = _grid_of_blobs(6, 5, 4, spacing=7.0, per_blob=12)
    box = Box.cube(6 * 7.0 * 1.4, (True, True, True))
    slots, _ = _settle(tracker, pos, box, 20)
    used = np.bincount(np.array([slots[b[0]] for b in blobs]),
                       minlength=cl.PALETTE_SLOTS)
    fair = len(blobs) / cl.PALETTE_SLOTS
    assert used.min() >= 0.5 * fair and used.max() <= 2.0 * fair, used.tolist()


def test_the_size_threshold_follows_the_density_it_has_to_beat():
    """What a cluster's SIZE means depends on the density it formed in: at the
    contact cutoff a random gas glues beads together for free, and the bigger
    those free clumps get, the bigger a real one has to be to stand out. An
    absolute threshold tuned on the paper's dense box greys a dilute one
    entirely; tuned on the dilute one it turns the dense one into confetti."""
    tracker = ClusterTracker(CUTOFF)
    dense = Box.cube(20.0, (True, True, True))          # phi ~ 0.1
    dilute = Box.cube(20.0 * 5 ** (1 / 3), (True,) * 3)  # phi ~ 0.02, same count
    assert tracker._min_size(1500, dense) > 4 * tracker._min_size(1500, dilute)
    assert tracker._min_size(1500, dilute) == cl.MIN_CLUSTER_SIZE
    # It cannot climb past a chance clump either: the dense box's threshold has
    # to sit above the ~14-bead clumps a fresh phi = 0.1 cell comes up with.
    assert 15 <= tracker._min_size(1500, dense) <= 40


def test_a_handful_of_beads_is_still_an_aggregate():
    """The seven-bead patch. Four loose beads need something to be loose in, and
    a threshold that outruns the whole scene paints the entire playground the gas
    colour -- a colouring that shows nothing at all. So it is capped at a third
    of the scene, which binds only when the scene is that small."""
    tracker = ClusterTracker(CUTOFF)
    tiny = Box.cube(4.2, (False, False, False))
    assert tracker._min_size(7, tiny) <= 7
    patch = _blobs([(0, 0, 0)], 7, spread=0.3, seed=11)
    slots, _ = _settle(tracker, patch, tiny, 12)
    assert set(slots.tolist()) == {0}, "the whole patch was called gas"


def test_the_labelling_is_paced_by_the_bead_count_not_by_the_frame():
    """A labelling costs O(N), so the cadence has to fall as the scene grows or
    the big remote box would spend its whole frame budget here."""
    tracker = ClusterTracker(CUTOFF)
    assert tracker._period(900) == tracker.RECLUSTER_EVERY
    assert tracker._period(1500) == tracker.RECLUSTER_EVERY
    assert tracker._period(50_000) > 4 * tracker.RECLUSTER_EVERY


def test_the_palette_offers_exactly_the_slots_the_tracker_hands_out():
    """The two halves live in different layers on purpose -- the clustering knows
    no colours and the theme knows no clusters -- so the one number they share is
    asserted here rather than imported across the boundary."""
    from lammps_live.ui.theme import CLUSTER_COLORS
    assert len(CLUSTER_COLORS) == cl.PALETTE_SLOTS


# ---- the seam between the two layers -----------------------------------------

def test_a_periodic_copy_of_a_bead_wears_that_bead_s_colour():
    """The colours are resolved on the real beads and then COPIED with them --
    a wrapped ghost and a periodic image are the same bead seen through a wall,
    so they cannot be allowed to fade on their own or, worse, be dropped and
    leave the copies black. This is the one place the (N, 3) tint has to survive
    the same reshuffling the positions do.
    """
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame
    pygame.init()
    from lammps_live.ui.renderer import Renderer
    from lammps_live.render_style import DEFAULT_STYLE

    class _Spec:                     # the two fields the two helpers read
        atom_radius_A = 0.5
        wrap_fade_fraction = 0.2

    n = 24
    rng = np.random.default_rng(0)
    pts = rng.uniform(-4.0, 4.0, (n, 3))
    dips = np.tile([0.0, 0.0, 1.0], (n, 1))
    tints = rng.uniform(0.0, 1.0, (n, 3))
    style = DEFAULT_STYLE.varied(periodic_images=(1, 1, 0))
    renderer = Renderer.__new__(Renderer)      # no window: only the two helpers
    renderer.box_x = renderer.box_y = 8.0

    _, _, _, _, ghost = renderer._wrap_ghost_instances(pts, dips, None, _Spec(),
                                                       None, tints)
    assert len(ghost) > n, "no seam ghosts were made, so nothing was tested"
    assert np.array_equal(ghost[:n], tints)     # the real beads, untouched

    box = (-4.0, 4.0, -4.0, 4.0, -4.0, 4.0)
    P, _, _, _, T, _ = renderer._periodic_image_instances(
        pts, dips, np.ones(n), np.zeros(n), tints, style, box,
        (True, True, False))
    assert len(T) == len(P) > n
    # Every copy's colour is one of the real beads' colours, and each real bead's
    # colour appears once per image drawn -- i.e. the tiling repeated the array
    # rather than reindexing or truncating it.
    assert np.array_equal(np.unique(T, axis=0), np.unique(tints, axis=0))


def test_hundreds_of_clusters_stay_separated_and_evenly_coloured():
    """The scale the dilute remote cell actually runs at. Not a duplicate of the
    two tests above: a regular grid is the easy case for a greedy colouring, and
    what a real cell hands it is irregular neighbourhoods where some clusters have
    eight near others and some have one. That is where a colouring runs out of
    palette locally, and where the graceful-degradation path in _claim is either
    doing its job or quietly clashing.
    """
    rng = np.random.default_rng(4)
    n_blob, per = 300, 12
    box = Box.cube(70.0, (True, True, True))
    centres = rng.uniform(-35.0, 35.0, (n_blob, 3))
    pos = box.wrap((centres[:, None, :]
                    + rng.normal(0, 0.4, (n_blob, per, 3))).reshape(-1, 3))
    tracker = ClusterTracker(CUTOFF)
    slots, _ = _settle(tracker, pos, box, 20)

    # Work from the CLUSTERS the tracker actually saw, not from the blobs: two
    # blobs that landed on top of each other are one cluster and one colour, and
    # counting that as a clash would be the test's mistake, not the code's.
    labels = cl.cluster_labels(pos, box, CUTOFF)
    sizes = np.bincount(labels)
    rows = np.flatnonzero(sizes >= tracker._min_size(len(pos), box))
    of_row = np.array([slots[np.flatnonzero(labels == r)[0]] for r in rows])
    assert len(rows) > 250, "the fixture did not produce many clusters"
    assert (of_row >= 0).all(), "clusters were left grey"

    centre, radius = cl.cluster_geometry(pos, labels, rows, box)
    starts, other, _ = cl.neighbour_graph(centre, radius, box, cl.NEIGHBOUR_COUNT)
    clashes = sum(1 for i in range(len(rows))
                  for e in range(starts[i], starts[i + 1])
                  if other[e] > i and of_row[other[e]] == of_row[i])
    assert clashes == 0, f"{clashes} neighbouring clusters share a colour"

    used = np.bincount(of_row, minlength=cl.PALETTE_SLOTS)
    fair = len(rows) / cl.PALETTE_SLOTS
    assert used.min() >= 0.6 * fair and used.max() <= 1.5 * fair, used.tolist()


def test_a_cluster_straddling_a_wall_is_not_placed_in_the_middle_of_the_box():
    """cluster_geometry decides what is near what, so its centre has to be a
    circular mean: an ordinary one puts a blob sitting on a periodic face in the
    empty centre of the cell, where it would be declared the neighbour of
    everything there and give away colours it has no business forbidding."""
    box = Box.cube(20.0, (True, True, True))
    pos = box.wrap(_blobs([(10.0, 0.0, 0.0)], 30, spread=0.4, seed=5))
    labels = cl.cluster_labels(pos, box, CUTOFF)
    centre, radius = cl.cluster_geometry(pos, labels, np.array([labels[0]]), box)
    assert abs(abs(centre[0, 0]) - 10.0) < 1.0, centre[0]
    assert radius[0] < 2.0, "the radius was measured across the box, not the blob"

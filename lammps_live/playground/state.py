"""Geometry and per-frame state primitives shared by force fields, scenarios,
observables and the verifier.

Everything here is pure numpy -- no LAMMPS import -- so scenarios, energy
decompositions and observables are all unit-testable without spinning up a
simulation. That is deliberate: the whole point of splitting the force field's
energy expression out of the system class is to be able to evaluate it on a
handmade configuration and compare it against what LAMMPS computed.
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Box:
    """A simulation cell: bounds plus which axes are periodic.

    `periodic` drives both the LAMMPS `boundary` command and the minimum-image
    convention used by the pair list / energy decomposition, so the two can
    never disagree -- a class of bug the old per-system code was wide open to
    (each system hand-wrote both its `boundary p p f` string and its own
    `d[:, 0] -= Lx * round(...)` wrapping).
    """
    lo: tuple
    hi: tuple
    periodic: tuple = (False, False, False)

    @classmethod
    def cube(cls, side, periodic=(False, False, False)):
        h = side / 2.0
        return cls((-h, -h, -h), (h, h, h), periodic)

    @classmethod
    def centered(cls, lx, ly, lz, periodic=(False, False, False)):
        return cls((-lx / 2.0, -ly / 2.0, -lz / 2.0),
                   (lx / 2.0, ly / 2.0, lz / 2.0), periodic)

    @property
    def lengths(self):
        return tuple(h - l for l, h in zip(self.lo, self.hi))

    @property
    def center(self):
        return tuple(0.5 * (l + h) for l, h in zip(self.lo, self.hi))

    def boundary_command(self):
        return "boundary " + " ".join("p" if p else "f" for p in self.periodic)

    def region_command(self, name="box"):
        (xlo, ylo, zlo), (xhi, yhi, zhi) = self.lo, self.hi
        return (f"region {name} block {xlo} {xhi} {ylo} {yhi} {zlo} {zhi} "
                f"units box")

    def bounds_3d(self):
        """(xlo, xhi, ylo, yhi, zlo, zhi) -- the renderer's box-outline form."""
        return (self.lo[0], self.hi[0], self.lo[1], self.hi[1],
                self.lo[2], self.hi[2])

    def corners(self):
        """The eight corner points, for camera framing."""
        return np.array([(x, y, z) for x in (self.lo[0], self.hi[0])
                         for y in (self.lo[1], self.hi[1])
                         for z in (self.lo[2], self.hi[2])], dtype=float)

    def minimum_image(self, d):
        """Wrap separation vectors (..., 3) into the minimum image on every
        periodic axis. Returns a new array; non-periodic axes pass through."""
        d = np.asarray(d, dtype=float).copy()
        for axis, (per, length) in enumerate(zip(self.periodic, self.lengths)):
            if per and length > 0.0:
                d[..., axis] -= length * np.round(d[..., axis] / length)
        return d

    def wrap(self, positions):
        """Fold positions into [lo, hi) on every periodic axis."""
        p = np.asarray(positions, dtype=float).copy()
        for axis, (per, length) in enumerate(zip(self.periodic, self.lengths)):
            if per and length > 0.0:
                p[:, axis] = (p[:, axis] - self.lo[axis]) % length + self.lo[axis]
        return p


@dataclass
class FrameState:
    """One frame's particle state, as plain arrays.

    `directors` is the per-particle unit orientation vector (LAMMPS `mu` for
    dipole atom styles), or None for force fields whose particles have no
    orientation. Already normalized on construction by from_lammps.
    """
    positions: np.ndarray            # (N, 3)
    directors: np.ndarray = None     # (N, 3) unit vectors, or None
    types: np.ndarray = None         # (N,) int
    ids: np.ndarray = None           # (N,) LAMMPS atom ids
    box: Box = None

    def __len__(self):
        return len(self.positions)


@dataclass
class PairData:
    """The particle pairs within a cutoff, with their separation vectors.

    Built once per analysis frame and handed to every consumer (the force
    field's energy decomposition, observables), because building it is the
    expensive part -- the old code built a KD-tree twice per frame in
    mesomem_hex (once for the pulled bead, once for the whole-patch total) and
    once more per system for the RDF.

    Because energies come back per-pair, one pass serves both the "energy of
    this one bead" panel (select pairs touching it) and the "energy of the whole
    system" panel (sum everything). Previously those were two separate O(pairs)
    passes over the same configuration.
    """
    a: np.ndarray      # (M,) first index
    b: np.ndarray      # (M,) second index
    d: np.ndarray      # (M, 3) minimum-image r_a - r_b
    r: np.ndarray      # (M,) |d|

    def __len__(self):
        return len(self.r)

    def touching(self, index):
        """Boolean mask of the pairs that involve particle `index`."""
        return (self.a == index) | (self.b == index)

    @classmethod
    def empty(cls):
        z = np.zeros(0)
        return cls(np.zeros(0, dtype=int), np.zeros(0, dtype=int),
                   np.zeros((0, 3)), z)


def build_pairs(positions, cutoff, box=None):
    """Unique particle pairs separated by less than `cutoff`, minimum-imaged.

    Uses a KD-tree over the PERIODIC subspace when the box has periodic axes,
    then filters on the exact minimum-image 3D distance. Restricting the tree to
    the periodic axes is what makes one code path cover all three MesoMem
    geometries: a separation measured in a subspace is never larger than the
    full 3D separation, so the tree's answer is a superset of the true within-
    cutoff pairs and the exact filter afterwards makes it precise. (This is the
    trick the old sheet system used by hand -- a periodic 2D tree on (x, y) for
    a 3D monolayer -- generalized.)

    With no periodic axes the tree runs on all three coordinates directly.
    """
    positions = np.asarray(positions, dtype=float)
    if len(positions) < 2 or cutoff <= 0.0:
        return PairData.empty()

    from scipy.spatial import cKDTree

    per_axes = []
    if box is not None:
        per_axes = [i for i, p in enumerate(box.periodic)
                    if p and box.lengths[i] > 0.0]

    if per_axes:
        lengths = [box.lengths[i] for i in per_axes]
        # cKDTree's toroidal topology needs coordinates strictly inside [0, L).
        # The modulo alone is not enough: a coordinate a hair BELOW the lower
        # bound (LAMMPS hands back x = -9e-16 for an atom nominally at 0) wraps to
        # a value that rounds to exactly L, which cKDTree rejects. Under
        # periodicity L and 0 are the same point, so fold it back.
        cols = []
        for i in per_axes:
            w = (positions[:, i] - box.lo[i]) % box.lengths[i]
            w[w >= box.lengths[i]] = 0.0
            cols.append(w)
        pts = np.column_stack(cols)
        tree = cKDTree(pts, boxsize=lengths)
    else:
        tree = cKDTree(positions)

    pairs = tree.query_pairs(cutoff, output_type="ndarray")
    if not len(pairs):
        return PairData.empty()

    a, b = pairs[:, 0], pairs[:, 1]
    d = positions[a] - positions[b]
    if box is not None:
        d = box.minimum_image(d)
    r = np.linalg.norm(d, axis=1)
    keep = (r < cutoff) & (r > 1e-9)
    return PairData(a[keep], b[keep], d[keep], r[keep])


def normalize_rows(v, eps=1e-9):
    """Row-wise unit vectors, leaving (near-)zero rows untouched rather than
    dividing by ~0 -- LAMMPS hands back a zero `mu` for a particle whose dipole
    was never set."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n < eps, 1.0, n)
    return v / n


def principal_normal(points):
    """Unit normal of the best-fit plane through `points` (the smallest
    principal axis of the centred second-moment matrix), sign-fixed to point
    along +z.

    Shared by the sheet/patch "keep the membrane face-on" housekeeping forces
    and the nematic-order observable, which independently derived it before.
    """
    p = np.asarray(points, dtype=float)
    q = p - p.mean(axis=0)
    _evals, evecs = np.linalg.eigh(q.T @ q)   # ascending
    n = evecs[:, 0]
    return -n if n[2] < 0.0 else n


def hex_lattice_2d(n_cols, n_rows, a):
    """(N, 2) hexagonal lattice sites, centred on the origin.

    Rows run along x at spacing `a`; successive rows step a*sqrt(3)/2 in y and
    offset a/2 in x -- the arrangement that tiles a periodic rectangular cell of
    size (n_cols*a) x (n_rows*a*sqrt(3)/2), which is why the sheet scenario can
    size its periodic box straight from these counts.
    """
    dy = a * math.sqrt(3.0) / 2.0
    xs = np.arange(n_cols, dtype=float) * a
    pts = []
    for j in range(n_rows):
        xoff = (a / 2.0) if (j % 2) else 0.0
        pts.append(np.column_stack([xs + xoff, np.full(n_cols, j * dy)]))
    pts = np.vstack(pts)
    return pts - pts.mean(axis=0)


def hex_ring_2d(n_rings, a):
    """(N, 2) points of a hexagonal patch: a centre site plus `n_rings` closed
    rings of neighbours around it, at nearest-neighbour spacing `a`. Ring 1 is
    the classic six-neighbour hexagon (the 7-bead patch).

    Ordered centre-first, then ring by ring, and WITHIN each ring by increasing
    polar angle starting from +x. Scenarios build their bond lists from these
    indices (spokes 0-k, then the closed ring k -> k+1), so the angular ordering
    is part of the contract, not an accident of the loop.
    """
    e1 = np.array([a, 0.0])
    e2 = np.array([a * 0.5, a * math.sqrt(3.0) / 2.0])
    shells = {}
    for i in range(-n_rings, n_rings + 1):
        for j in range(-n_rings, n_rings + 1):
            if i == 0 and j == 0:
                continue
            # Hexagonal (graph) distance on the triangular lattice: sites at
            # distance k form the k-th closed ring.
            dist = max(abs(i), abs(j), abs(i + j))
            if dist <= n_rings:
                shells.setdefault(dist, []).append(i * e1 + j * e2)
    pts = [np.zeros(2)]
    for k in sorted(shells):
        ring = shells[k]
        ring.sort(key=lambda p: math.atan2(p[1], p[0]) % (2.0 * math.pi))
        pts.extend(ring)
    return np.array(pts, dtype=float)

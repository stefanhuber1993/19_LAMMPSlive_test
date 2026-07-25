"""Radial distribution functions g(r), time-averaged over a rolling window.

Two flavours, sharing the same rolling-window / subsampling / throttling design:
`InPlaneRDF` for a monolayer (the in-plane pair correlation) and `RadialRDF3D`
for a bulk periodic cell. The 3D one was a private `_RDF3D` class inside the
self-assembly system; it lives here next to its 2D twin because the runtime picks
between them from the box's periodicity.

The in-plane case, in the original author's words:

The MesoMem systems are a bead monolayer lying in the xy plane, driven by a
custom pair style. LAMMPS' own `compute rdf` is a 3D radial average and is
awkward to normalize for a single-layer sheet in a thick, mostly-empty box, so
the in-plane g(r) is computed here directly from the bead (x, y) positions. That
gives a clean read of the membrane's lateral order -- sharp hexagonal shells
when the sheet is a gel/solid, washing out to one broad liquid-like hump when it
melts -- which is exactly the "what state of matter is this" question the RDF
panel is meant to answer.

A single instantaneous g(r) over a few hundred beads is noisy, so samples are
accumulated over a short rolling window (like the crystals' LAMMPS ave/time RDF)
and only reported once enough have piled up -- hence the "warming up..." period.
"""
from collections import deque

import numpy as np


class InPlaneRDF:
    """Rolling, time-averaged in-plane pair correlation g(r).

    box=(lx, ly) applies the minimum-image convention (periodic sheet); box=None
    treats the beads as a finite non-periodic cluster and normalizes against the
    density of its own in-plane bounding box.
    """

    def __init__(self, r_max, nbins=80, box=None, min_samples=12, window=45,
                 sample_every=3, max_atoms=300):
        self.box = box
        self.edges = np.linspace(0.0, float(r_max), int(nbins) + 1)
        self.r = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.ring_area = np.pi * (self.edges[1:] ** 2 - self.edges[:-1] ** 2)
        self.min_samples = min_samples
        self.sample_every = max(1, int(sample_every))
        self.max_atoms = max_atoms
        self._rng = np.random.default_rng()
        self._hist = deque(maxlen=window)   # per-sample raw pair histograms
        self._ideal = deque(maxlen=window)  # per-sample ideal-gas expectation
        self._counter = 0

    def add(self, xy):
        """Feed one frame's in-plane bead positions (N, 2). Cheap to call every
        frame; only every `sample_every`-th call is actually histogrammed, which
        bounds the O(N^2) pair cost on the large sheet without hurting the
        time-average.

        Above max_atoms beads a uniform random subsample is used each sample:
        g(r) is an intensive quantity (invariant to uniform subsampling in
        expectation), so a subset gives an unbiased, slightly noisier estimate
        that the rolling time-average cleans up -- turning the O(N^2) pair pass
        on the ~900-bead sheet from frame-dominating into cheap."""
        self._counter += 1
        if self._counter % self.sample_every != 0:
            return
        xy = np.asarray(xy, dtype=float)
        n = len(xy)
        if n < 2:
            return
        if self.max_atoms is not None and n > self.max_atoms:
            xy = xy[self._rng.choice(n, self.max_atoms, replace=False)]
            n = self.max_atoms
        iu, ju = np.triu_indices(n, k=1)
        dx = xy[iu, 0] - xy[ju, 0]
        dy = xy[iu, 1] - xy[ju, 1]
        if self.box is not None:
            lx, ly = self.box
            dx -= lx * np.round(dx / lx)
            dy -= ly * np.round(dy / ly)
            area = lx * ly
        else:
            span = xy.max(axis=0) - xy.min(axis=0)
            area = max(float(span[0] * span[1]), 1e-9)
        dist = np.hypot(dx, dy)
        hist, _ = np.histogram(dist, bins=self.edges)
        rho = n / area
        # Expected unordered pair count per shell for an ideal gas of the same
        # density: 0.5 * N * rho * (shell area). Dividing the real histogram by
        # this yields g(r) -> 1 at large r for a structureless liquid.
        self._hist.append(hist)
        self._ideal.append(0.5 * n * rho * self.ring_area)

    def get(self):
        """(r, g(r)) once enough samples have accumulated, else None (warming up)."""
        if len(self._hist) < self.min_samples:
            return None
        hist = np.sum(self._hist, axis=0)
        ideal = np.sum(self._ideal, axis=0)
        g = np.divide(hist, ideal, out=np.zeros(len(hist)), where=ideal > 0)
        return self.r, g

    def reset(self):
        self._hist.clear()
        self._ideal.clear()
        self._counter = 0


class NativeRDF:
    """LAMMPS' own time-averaged `compute rdf`, driven through `fix ave/time`.

    Used where the Python RDFs above would misreport: a crystal slab with vacuum
    above it has no meaningful in-plane bounding-box density, and normalizing
    against one puts a spurious offset on g(r). LAMMPS computes it over a named
    group with the correct cell volume instead.
    """

    def __init__(self, lmp, group="all", nbins=100, cutoff=10.0,
                 ave_every=5, ave_repeat=40):
        self.lmp = lmp
        self.nbins = int(nbins)
        freq = ave_every * ave_repeat
        lmp.command(f"compute rdf_raw {group} rdf {self.nbins} 1 1 cutoff {cutoff}")
        lmp.command(f"fix rdf_avg {group} ave/time {ave_every} {ave_repeat} {freq} "
                    f"c_rdf_raw[*] mode vector")
        # The averaging window has to fill before the fix has anything to report.
        self._ready_step = lmp.extract_global("ntimestep") + freq + ave_every
        self._r = None

    def add(self, _positions):
        """No-op: LAMMPS accumulates this itself. Present so the runtime can treat
        every RDF the same way."""

    def get(self):
        if self.lmp.extract_global("ntimestep") < self._ready_step:
            return None
        if self._r is None:
            self._r = np.array([self.lmp.extract_fix("rdf_avg", 0, 2, nrow=i, ncol=0)
                                for i in range(self.nbins)])
        g = np.array([self.lmp.extract_fix("rdf_avg", 0, 2, nrow=i, ncol=1)
                      for i in range(self.nbins)])
        return self._r, g

    def reset(self):
        self._r = None


class RadialRDF3D:
    """Rolling, time-averaged 3D radial distribution g(r), minimum-image in a
    periodic cubic cell.

    The spatial analog of InPlaneRDF, with a spherical-shell (4/3 pi dr^3)
    ideal-gas normalization, so g(r) -> 1 for a structureless gas and grows sharp
    shells as particles condense into ordered membranes. Subsampled and throttled
    like the 2D version to bound the O(N^2) pair pass on ~1500 particles.
    """

    def __init__(self, r_max, box_l, nbins=60, min_samples=10, window=40,
                 sample_every=3, max_atoms=400):
        self.box_l = box_l
        self.edges = np.linspace(0.0, float(r_max), int(nbins) + 1)
        self.r = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.shell_vol = (4.0 / 3.0) * np.pi * (self.edges[1:] ** 3
                                                - self.edges[:-1] ** 3)
        self.min_samples = min_samples
        self.sample_every = max(1, int(sample_every))
        self.max_atoms = max_atoms
        self._rng = np.random.default_rng()
        self._hist = deque(maxlen=window)
        self._ideal = deque(maxlen=window)
        self._counter = 0

    def add(self, xyz):
        self._counter += 1
        if self._counter % self.sample_every != 0:
            return
        xyz = np.asarray(xyz, dtype=float)
        n = len(xyz)
        if n < 2:
            return
        if self.max_atoms is not None and n > self.max_atoms:
            xyz = xyz[self._rng.choice(n, self.max_atoms, replace=False)]
            n = self.max_atoms
        iu, ju = np.triu_indices(n, k=1)
        d = xyz[iu] - xyz[ju]
        d -= self.box_l * np.round(d / self.box_l)   # minimum image, cubic cell
        dist = np.linalg.norm(d, axis=1)
        hist, _ = np.histogram(dist, bins=self.edges)
        rho = n / (self.box_l ** 3)
        self._hist.append(hist)
        self._ideal.append(0.5 * n * rho * self.shell_vol)

    def get(self):
        if len(self._hist) < self.min_samples:
            return None
        hist = np.sum(self._hist, axis=0)
        ideal = np.sum(self._ideal, axis=0)
        g = np.divide(hist, ideal, out=np.zeros(len(hist)), where=ideal > 0)
        return self.r, g

    def reset(self):
        self._hist.clear()
        self._ideal.clear()
        self._counter = 0

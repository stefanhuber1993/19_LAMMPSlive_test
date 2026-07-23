"""In-plane 2D radial distribution g(r), time-averaged over a rolling window.

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

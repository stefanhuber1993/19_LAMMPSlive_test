"""Temporal smoothing of the DRAWN particle state. Visuals only.

WHAT IT IS FOR: at any temperature the bath keeps every bead rattling, and on a
900-6000 bead scene that thermal jitter is most of what the eye sees -- the whole
screen wiggles while the thing actually worth watching (patches nucleating,
lamellae flattening, a membrane healing) happens an order of magnitude slower.
Low-passing each bead's drawn position separates the two: the wiggle averages
away, the rearrangement survives, because the first is fast and zero-mean and the
second is slow and directed. Nothing here touches the simulation -- the physics,
the observables, the energy panels and the haptics all keep reading LAMMPS' real
coordinates. This is a camera trick, and it is off by default.

WHY ON THE CPU, NOT IN THE VERTEX SHADER: a shader filter would need each bead's
previous smoothed position in a GPU buffer and a ping-pong to update it, and --
the real objection -- the smoothed positions would then exist only on the GPU,
while everything else in the frame is built on the CPU from the same coordinates:
the bond sticks, the wrap ghosts and periodic image copies, the depth sort, the
control net, the motion trails. Smoothing only the bead centres would leave the
sticks connecting where the beads no longer are. Filtering once here, upstream of
every consumer, keeps the picture consistent -- and costs an O(N) numpy pass
(three vector ops on an (N, 3) array, ~0.05 ms at 6000 beads), which is nothing
next to the 65 ms chunk it rides along with.

THE ONE SUBTLETY: an exponential average across a periodic seam is a disaster.
A bead that leaves at x = +L/2 and re-enters at -L/2 has, as far as the filter is
concerned, just teleported across the box, and averaging that draws it smearing
back through the middle. So the average is taken over the MINIMUM-IMAGE
displacement (Box.minimum_image) and the result folded back into the cell
(Box.wrap) -- the filter then follows the bead through the wall the same way the
physics does.
"""
from dataclasses import replace

import numpy as np

from .state import normalize_rows


class TrajectorySmoother:
    """A per-particle exponential low-pass on positions and directors.

    The time constant is given in SIMULATED time, not frames, and the per-frame
    weight is derived from the frame's own sim-time slice
    (alpha = 1 - exp(-dt/tau)). So the amount of smoothing is a property of the
    physics being averaged over rather than of the frame rate -- which matters
    here, because the frame rate falls from 60 to 15 Hz across the bead counts
    these playgrounds run at, and a fixed per-frame weight would quietly mean
    something different on every one of them.

    Particles are tracked in stable id order, the order every readout already
    hands back; a change in particle count reseeds.
    """

    def __init__(self):
        self._pos = None
        self._dirs = None

    def reset(self):
        """Forget the history. Next frame reseeds from the live coordinates, so
        re-enabling smoothing fades in from where things actually are instead of
        from a stale ghost."""
        self._pos = None
        self._dirs = None

    @property
    def active(self):
        return self._pos is not None

    def apply(self, state, tau, dt, keep_exact=None):
        """Return `state` with positions/directors low-passed over `tau`.

        `dt` is the sim time this frame advanced. `keep_exact` is an optional
        index (in the state's own id order) to leave untouched -- the controlled
        particle, whose motion is the user's own input rather than noise, and
        which must keep matching the puller marker and force arrows drawn from
        the unsmoothed physics.

        tau <= 0 (the default), a zero-length state, or a non-finite frame
        returns the state unchanged.
        """
        if tau <= 0.0 or dt <= 0.0 or not len(state.positions):
            self.reset()
            return state
        pos = np.asarray(state.positions, dtype=float)
        dirs = None if state.directors is None else np.asarray(state.directors,
                                                               dtype=float)
        # An unstable simulation hands back NaN coordinates; filtering those would
        # poison the history permanently (NaN * anything is NaN), so bail and let
        # the HUD's "unstable" latch do the talking.
        if not np.isfinite(pos).all():
            self.reset()
            return state

        if self._pos is None or len(self._pos) != len(pos):
            self._pos = pos.copy()
            self._dirs = None if dirs is None else dirs.copy()
            return state

        alpha = 1.0 - float(np.exp(-dt / tau))
        alpha = min(1.0, max(1e-6, alpha))
        box = state.box
        delta = pos - self._pos
        if box is not None:
            delta = box.minimum_image(delta)
        self._pos = self._pos + alpha * delta
        if box is not None:
            self._pos = box.wrap(self._pos)

        if dirs is not None:
            if self._dirs is None or len(self._dirs) != len(dirs):
                self._dirs = dirs.copy()
            else:
                # Kept normalized between frames rather than accumulating a raw
                # mean: a bead whose director genuinely flips would otherwise
                # drive the running vector through ~zero, where its direction is
                # meaningless. normalize_rows leaves a zero row alone.
                self._dirs = normalize_rows(self._dirs + alpha * (dirs - self._dirs))

        out_pos = self._pos.copy()
        out_dirs = None if self._dirs is None else self._dirs.copy()
        if keep_exact is not None and 0 <= keep_exact < len(out_pos):
            out_pos[keep_exact] = pos[keep_exact]
            # The history is left as it is for that particle -- it is being
            # written straight through, so it has no lag to accumulate.
            if out_dirs is not None and dirs is not None:
                out_dirs[keep_exact] = dirs[keep_exact]

        return replace(state, positions=out_pos, directors=out_dirs)

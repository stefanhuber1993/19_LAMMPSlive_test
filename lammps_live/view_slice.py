"""Cutting the scene open with the joystick's thrust lever.

An opaque scene hides its own inside. A vesicle with a polymer coiled in it is
the extreme case -- the membrane is a closed, densely packed shell, so from the
outside the thing the playground exists to show is simply not visible -- but the
same is true of any packed box of beads, which is why this lives here and not in
one playground file.

The cut is a SLAB: keep only the beads within `THICKNESS_FRACTION` of the box's
width of a plane, and slide that plane through the box with the lever. The slab's
normal is the CARDINAL world axis the camera is closest to looking along, so the
face you are left looking at is a clean section through the cell rather than an
oblique one, and it is re-picked as the camera moves (see `_cardinal`).

    RenderStyle.section_min is the other cut in this codebase and a different
    thing: a fixed half-space in a fixed world axis, declared by a playground
    because that scene is ALWAYS drawn in section (the rod, edge-on). This one is
    live, camera-relative, and belongs to the person at the controls. They
    compose -- the renderer draws the intersection.

THE LEVER IS A LEVER, NOT A BUTTON, and that is the whole design problem here.
It is an absolute control with no detent: wherever it was left at the end of the
last session is where it is at the start of this one, and a demo that came up
sliced in half because nobody had touched the throttle would be a bug. So:

  * UNTOUCHED SINCE STARTUP -> no slicing, whatever the lever reads. The first
    value seen is only recorded, never acted on.
  * MOVED -> the slab engages and follows the lever, over `TRANSITION_SECONDS`.
  * STILL FOR `HOLD_SECONDS` -> it opens back up to the whole box, over the same
    transition. The lever keeps its position; touching it again picks the slice
    straight back up.

That last rule is what makes it usable one-handed mid-demo: cut in, look, let go,
and the scene puts itself back together without anything to remember to reset.
"""
import math
from dataclasses import dataclass

import numpy as np

# The slab's thickness once fully engaged, as a fraction of the box's width along
# the cutting axis. 5% of a 100-sigma cell is five bead diameters -- thin enough
# to be a section rather than a slice of the whole thing, thick enough that the
# section of a one-bead-thick membrane is a continuous ring rather than a dotted
# one.
THICKNESS_FRACTION = 0.05
# How long the slab takes to close from "whole box" to that thickness, and to
# open back up again. Short enough to feel like a response to the lever, long
# enough to read as one object opening rather than beads vanishing.
TRANSITION_SECONDS = 0.5
# How long the lever may sit still before the box opens back up.
HOLD_SECONDS = 3.0
# How much the lever has to move to count as touched. The device reports it as 7
# bits, so one notch is 1/127 ~ 0.008; this is a few notches -- past the last
# bit's dither, well inside a deliberate nudge.
TOUCH_EPSILON = 0.02
# The slab's half-thickness at the fully-open end of the transition, as a multiple
# of the box width. Not simply "infinity": the transition interpolates between
# this and the target, so it has to be a real number, and it has to be wide enough
# that nothing is cut at the open end -- including a periodically tiled scene,
# whose copies reach 1.5 box widths out from the centre (see
# RenderStyle.periodic_images).
OPEN_FACTOR = 3.0
# How far off the view direction the cutting axis is allowed to drift before it is
# re-picked, in degrees. Pure "nearest cardinal" would flip the cut at exactly 45
# degrees and so chatter there; this is the hysteresis that stops it, and it also
# means a camera nudged a few degrees does not re-aim the cut under you. Past it,
# the axis really is no longer the one you are looking along -- the section would
# be foreshortening away to an edge-on line -- so it flips.
REAIM_DEGREES = 55.0

_AXES = (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
         np.array([0.0, 0.0, 1.0]))


def _smoothstep(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclass(frozen=True)
class SlicePlane:
    """One frame's cut: keep the beads within `half` of `center` along `normal`.

    A value object rather than a live handle, so the renderer is handed a fact
    about this frame and cannot accidentally advance the animation by drawing.
    """

    normal: tuple
    center: float
    half: float

    def mask(self, points):
        """Boolean keep-mask over world points (N, 3), or None if it cuts nothing.

        None rather than an all-true array on purpose: the caller can then skip
        the whole indexing pass, which at 50k beads and a dozen per-bead channels
        is worth having on every frame the slice is not engaged.
        """
        if not np.isfinite(self.half):
            return None
        pts = np.asarray(points, dtype=float)
        if not len(pts):
            return None
        d = pts @ np.asarray(self.normal, dtype=float)
        keep = np.abs(d - self.center) <= self.half
        return None if keep.all() else keep


class ViewSlice:
    """The lever's state, and the slab it currently asks for.

    Device-free and geometry-free: it is fed a lever reading, a view direction
    and the box, and hands back a SlicePlane. That keeps the two questions worth
    testing -- when does it engage, and where does the plane land -- answerable
    without a joystick or a renderer (tests/test_view_slice.py).
    """

    def __init__(self, thickness_fraction=THICKNESS_FRACTION,
                 transition_seconds=TRANSITION_SECONDS,
                 hold_seconds=HOLD_SECONDS, touch_epsilon=TOUCH_EPSILON,
                 open_factor=OPEN_FACTOR, reaim_degrees=REAIM_DEGREES):
        self.thickness_fraction = thickness_fraction
        self.transition_seconds = transition_seconds
        self.hold_seconds = hold_seconds
        self.touch_epsilon = touch_epsilon
        self.open_factor = open_factor
        self.reaim_cos = math.cos(math.radians(reaim_degrees))
        # None until the first reading arrives -- which is exactly "the lever has
        # not been touched since this session started", since the device only
        # reports on change.
        self._lever = None
        self._idle = 0.0
        self._engaged = False
        self._progress = 0.0      # 0 = whole box, 1 = fully sliced
        self._axis = None         # the cutting normal, a signed cardinal axis
        self._plane = None

    # ---- state ---------------------------------------------------------------

    @property
    def engaged(self):
        """Whether the lever is currently asking for a cut (before easing)."""
        return self._engaged

    @property
    def progress(self):
        """0 = whole box on screen, 1 = the slab at its full thinness."""
        return self._progress

    @property
    def plane(self):
        """This frame's SlicePlane, or None while the whole box is shown."""
        return self._plane

    def reset(self):
        """Forget the lever, as if the session had just started. Called on a
        playground switch: the box it was cutting is gone, and coming up sliced
        into a scene you have not touched the lever for is the same surprise the
        untouched-at-startup rule exists to avoid."""
        self._lever = None
        self._idle = 0.0
        self._engaged = False
        self._progress = 0.0
        self._axis = None
        self._plane = None

    # ---- per frame -----------------------------------------------------------

    def update(self, lever, dt, forward=None, box_bounds=None):
        """One frame. `lever` is 0..1 (or None on a device with no lever),
        `forward` the camera's view direction, `box_bounds` the cell as
        (xlo, xhi, ylo, yhi, zlo, zhi). Returns this frame's plane, or None."""
        if lever is None:
            # No lever on this input device -- open back up rather than freezing
            # whatever the last joystick session left behind.
            self._engaged = False
        elif self._lever is None:
            self._lever = float(lever)          # recorded, deliberately not acted on
        else:
            if abs(float(lever) - self._lever) > self.touch_epsilon:
                self._lever = float(lever)
                self._idle = 0.0
                self._engaged = True
            else:
                self._idle += dt
                if self._idle > self.hold_seconds:
                    self._engaged = False

        step = dt / max(self.transition_seconds, 1e-6)
        target = 1.0 if self._engaged else 0.0
        if self._progress < target:
            self._progress = min(target, self._progress + step)
        elif self._progress > target:
            self._progress = max(target, self._progress - step)

        self._plane = self._resolve(forward, box_bounds)
        return self._plane

    # ---- geometry ------------------------------------------------------------

    def _resolve(self, forward, box_bounds):
        if self._progress <= 0.0 or box_bounds is None or forward is None:
            self._axis = None
            return None
        axis = self._cardinal(np.asarray(forward, dtype=float))
        self._axis = axis
        lo, hi = _extent(box_bounds, axis)
        span = max(hi - lo, 1e-9)
        # The lever's travel maps to the whole of the box along the cutting axis,
        # near face to far: pushed forward, the plane moves away from the eye,
        # which is the direction the axis was signed to point in.
        center = lo + float(self._lever or 0.0) * span
        target_half = 0.5 * self.thickness_fraction * span
        open_half = self.open_factor * span
        # Geometric, not linear, between the open and closed thicknesses: the slab
        # then thins at a constant PROPORTIONAL rate, which is what reads as a
        # steady closing. A linear ramp spends most of the half second in a slab
        # far wider than the cell and then snaps shut at the end.
        s = _smoothstep(self._progress)
        half = open_half * (target_half / open_half) ** s
        return SlicePlane(normal=tuple(axis), center=center, half=half)

    def _cardinal(self, forward):
        """The signed cardinal axis to cut along, pointing away from the eye.

        Sticky: the axis already in use is kept until the view has swung more
        than REAIM_DEGREES off it (see the constant), so an orbiting camera
        re-aims the cut once rather than flickering between two axes as it
        crosses the halfway angle.
        """
        n = np.linalg.norm(forward)
        if n < 1e-9:
            return self._axis if self._axis is not None else _AXES[1].copy()
        f = forward / n
        if self._axis is not None and float(self._axis @ f) >= self.reaim_cos:
            return self._axis
        k = int(np.argmax(np.abs(f)))
        return _AXES[k] * (1.0 if f[k] >= 0.0 else -1.0)


def _extent(box_bounds, axis):
    """(min, max) of the box's corners projected on `axis`."""
    xlo, xhi, ylo, yhi, zlo, zhi = (float(v) for v in box_bounds)
    lo = hi = None
    for x in (xlo, xhi):
        for y in (ylo, yhi):
            for z in (zlo, zhi):
                d = float(axis @ np.array([x, y, z]))
                lo = d if lo is None else min(lo, d)
                hi = d if hi is None else max(hi, d)
    return lo, hi

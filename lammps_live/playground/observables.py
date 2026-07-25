"""Declared observables, and the scheduler that keeps them inside the frame
budget.

A playground names the observables it wants; they are plotted live and are the
output columns of a headless sweep. This replaces the previous approach of
formatting ad-hoc strings in a per-system `get_hud_lines`, which could not be
plotted, could not be logged, and had to be rewritten for each new system.

Cost control matters here -- this is Python running between LAMMPS steps at 60
fps. Three things keep it cheap:

  * one pair list per analysis frame, shared by every consumer (the old code
    built a KD-tree twice per frame in the patch system, and again for the RDF);
  * a declared cadence per observable, so a heavy whole-system quantity runs
    every Nth frame and returns its cached value in between;
  * staggered phases, so several every-4-frames observables don't all land on
    the same frame and spike it.
"""
import numpy as np

from .state import build_pairs, principal_normal

_REGISTRY = {}


def observable(name, label=None, unit="", every=4, needs_directors=False):
    """Register a function(state, pairs, params) -> float as a named observable."""
    def decorate(fn):
        _REGISTRY[name] = Observable(
            name=name, label=label or name, unit=unit, every=every,
            needs_directors=needs_directors, fn=fn,
        )
        return fn
    return decorate


class Observable:
    def __init__(self, name, label, unit, every, needs_directors, fn):
        self.name = name
        self.label = label
        self.unit = unit
        self.every = max(1, int(every))
        self.needs_directors = needs_directors
        self.fn = fn

    def __call__(self, state, pairs, params):
        return self.fn(state, pairs, params)


def get(name):
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown observable {name!r}. Available: {known}") from None


def available():
    return dict(_REGISTRY)


# --- the bundled observables --------------------------------------------------

@observable("nematic_S", "nematic order S", every=4, needs_directors=True)
def _nematic_order(state, pairs, params):
    """Nematic order parameter S = (3<cos^2 theta> - 1)/2 of the directors about
    their own mean axis, from the largest eigenvalue of the ordering tensor
    Q = <3 n n - I>/2.

    S = 1 is perfectly aligned directors (a flat, well-ordered membrane), S = 0
    is isotropic (a disordered gas). This is the single number that says whether
    the self-assembly run has actually formed membranes, and it responds sharply
    to k_tilt crossing the compact-vs-planar transition -- the thing a slider
    session is trying to find by eye.
    """
    n = state.directors
    if n is None or len(n) < 2:
        return float("nan")
    q = 1.5 * (n[:, :, None] * n[:, None, :]).mean(axis=0) - 0.5 * np.eye(3)
    return float(np.linalg.eigvalsh(q)[-1])


@observable("thickness", "membrane thickness", every=4)
def _thickness(state, pairs, params):
    """RMS spread of the particles along their own best-fit plane normal.

    For a monolayer this reads out buckling/roughening: near zero when the sheet
    is flat, growing as it corrugates or breaks up.
    """
    p = state.positions
    if len(p) < 3:
        return float("nan")
    n = principal_normal(p)
    return float(np.sqrt(np.mean(((p - p.mean(axis=0)) @ n) ** 2)))


@observable("area_per_particle", "area per particle", every=8)
def _area_per_particle(state, pairs, params):
    """In-plane area divided by particle count.

    Uses the periodic cell's cross-section when the box is periodic in-plane
    (the meaningful definition for a tension-free sheet), and the projected
    bounding box otherwise.
    """
    p = state.positions
    if not len(p):
        return float("nan")
    box = state.box
    if box is not None and box.periodic[0] and box.periodic[1]:
        area = box.lengths[0] * box.lengths[1]
    else:
        span = p[:, :2].max(axis=0) - p[:, :2].min(axis=0)
        area = float(span[0] * span[1])
    return area / len(p)


@observable("coordination", "mean neighbours within rc", every=4)
def _coordination(state, pairs, params):
    """Mean number of neighbours inside the interaction cutoff -- the simplest
    read on condensation: a dilute gas sits near zero, a packed membrane near the
    lattice's coordination number."""
    n = len(state.positions)
    return (2.0 * len(pairs) / n) if n else float("nan")


@observable("mean_tilt_deg", "mean director tilt", unit=" deg", every=4,
            needs_directors=True)
def _mean_tilt(state, pairs, params):
    """Mean angle between each director and the cloud's best-fit plane normal.
    Zero for a perfectly flat, untilted membrane; this is what the tilt modulus
    is holding down."""
    d = state.directors
    if d is None or len(d) < 3:
        return float("nan")
    normal = principal_normal(state.positions)
    # Directors are bidirectional (both +n and -n are minima of the tilt term),
    # so fold onto [0, 90] via |cos|.
    c = np.clip(np.abs(d @ normal), 0.0, 1.0)
    return float(np.degrees(np.arccos(c)).mean())


# --- scheduling ---------------------------------------------------------------

class Analysis:
    """Runs the energy decomposition and the requested observables on a budget.

    Holds the shared pair list, the cached energy terms and the cached observable
    values. The runtime calls `update` once per frame; only the work that is due
    on that frame actually runs.
    """

    # The energy panels are the most expensive consumer (a full pass over every
    # pair) and the aggregate barely changes frame to frame.
    ENERGY_EVERY = 4

    def __init__(self, force_field, names=(), energy_every=None):
        self.force_field = force_field
        self.observables = tuple(get(n) for n in names)
        if energy_every is not None:
            self.ENERGY_EVERY = max(1, int(energy_every))
        self._frame = 0
        self._pairs = None
        self._energy = None          # {label: per-pair array}
        # The pair list the cached energy was computed WITH. It has to be kept
        # alongside: the energy runs on its own (slower) cadence, while the pair
        # list is rebuilt whenever any observable is due, so the live list can be a
        # different length than the cached energy arrays. Masking one with the
        # other is then an index error -- which is exactly what happened.
        self._energy_pairs = None
        self._values = {}            # observable name -> last value
        # Stagger phases so several every-N observables don't share a frame.
        self._phase = {ob.name: i for i, ob in enumerate(self.observables)}

    @property
    def pairs(self):
        return self._pairs

    @property
    def energy(self):
        return self._energy

    def values(self):
        return dict(self._values)

    def _due(self, every, phase=0):
        return (self._frame + phase) % every == 0

    def update(self, state, params, force=False):
        """Advance one frame. Rebuilds the pair list only when something that
        needs it is due, so a frame where nothing is scheduled costs nothing."""
        self._frame += 1
        energy_due = force or self._due(self.ENERGY_EVERY)
        due_obs = [ob for ob in self.observables
                   if force or self._due(ob.every, self._phase[ob.name])]
        if not energy_due and not due_obs:
            return

        cutoff = self.force_field.interaction_cutoff(params)
        self._pairs = build_pairs(state.positions, cutoff, state.box)

        if energy_due:
            self._energy = self.force_field.energy_terms(state, self._pairs, params)
            self._energy_pairs = self._pairs
        for ob in due_obs:
            if ob.needs_directors and state.directors is None:
                continue
            self._values[ob.name] = ob(state, self._pairs, params)

    def energy_panel(self, title, scale, index=None):
        """(title, [(label, value), ...], scale) for the renderer's signed-bar
        panel, or None.

        `index` selects one particle's share: the pairs touching it. Whole-system
        and single-particle panels therefore come from ONE evaluation of the
        energy expression, which is why per-pair arrays are the interface.
        """
        if self._energy is None or self._energy_pairs is None:
            return None
        # Mask with the pair list the energy was computed with, NOT the live one.
        mask = None if index is None else self._energy_pairs.touching(index)
        terms = []
        for label in self.force_field.energy_terms_labels:
            arr = self._energy.get(label)
            if arr is None:
                continue
            terms.append((label, float(arr.sum() if mask is None else arr[mask].sum())))
        return (title, terms, scale) if terms else None

    def hud_lines(self):
        """The requested observables as short display strings, in declared
        order."""
        out = []
        for ob in self.observables:
            v = self._values.get(ob.name)
            if v is None:
                continue
            out.append(f"{ob.label} = {v:.3f}{ob.unit}")
        return out

"""Scenarios: geometry, box, relaxation and per-frame housekeeping.

A Scenario answers "what am I simulating, and in what cell" -- independent of
which force field runs on it and of whether the user is playing with it or
watching it run. Its core contract is pure numpy:

    build(params, rng) -> ScenarioBuild(positions, directors, types, box, bonds)

so scenarios are unit-testable with no LAMMPS instance, and the particles reach
LAMMPS through a single `create_atoms` call instead of one command per particle
(the old sheet system issued 900 of them).

Housekeeping forces are also pure numpy: a scenario returns an (N, 3) force array
and the runtime applies it as a momentum kick. These are NOT force-field physics
-- they are the soft corrections that keep a small free membrane well-posed for
interactive play (centred, face-on, not drifting out of frame).

Relaxation has two hooks because the two shipped patterns genuinely differ:
`pre_control_settle` runs before the thermostat and any control fixes exist (the
sheet's barostat relaxation to zero lateral tension, which must not fight a
puller); `post_control_settle` runs with everything already in place (the patch's
short silent settle).
"""
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .params import ParamSet, structural
from .state import Box, hex_lattice_2d, hex_ring_2d, principal_normal


@dataclass
class ScenarioBuild:
    """The initial configuration, as arrays."""
    positions: np.ndarray                 # (N, 3); empty if LAMMPS places them
    box: Box
    directors: np.ndarray = None          # (N, 3) or None
    types: np.ndarray = None              # (N,) int, defaults to all type 1
    bonds: tuple = ()                     # index pairs to draw as sticks
    # Per-particle render brightness multiplier (1.0 = normal), or None. Used to
    # spotlight a tracked cluster so it can be followed as it diffuses.
    brightness: np.ndarray = None

    def __post_init__(self):
        if self.types is None:
            self.types = np.ones(len(self.positions), dtype=int)


class Scenario(ABC):
    """Base class for a scenario."""

    name = ""
    # Declared structural parameters (counts, spacings, box size, stiffnesses).
    # Structural by definition: changing any of them means rebuilding, so they
    # live in the playground file and never become sliders.
    params = ()
    # MD timestep and how much simulated time one rendered frame advances. Both
    # live here rather than on the force field because they are properties of the
    # scenario's size and dynamics: the 1500-particle assembly wants a coarser
    # step and a much bigger per-frame slice than a 7-particle patch being probed.
    timestep = 0.005
    sim_time_per_frame = 0.05
    langevin_damp = 0.5     # implicit-solvent relaxation time
    # Neighbour-list skin. A denser scenario wants a smaller one (less wasted
    # pair checking); a dilute gas of fast-diffusing particles wants enough that
    # the list does not need rebuilding every step.
    neighbor_skin = 1.0

    # Rendering hints, surfaced on the generated SystemSpec.
    director_arrows = True     # per-particle director spikes; clutter above ~50
    wrap_fade_fraction = 0.0   # periodic-seam crossfade, as a fraction of the side

    # The thermostat this scenario wants. Langevin (implicit solvent) suits a
    # coarse-grained membrane in water; a crystal slab wants CSVR, which leaves the
    # dynamics deterministic. See thermostat.py.
    thermostat = None          # None -> Langevin, set in __init_subclass__ terms

    def get_thermostat(self):
        from .thermostat import Langevin
        return self.thermostat or Langevin()

    def thermostat_damp(self, params):
        """Relaxation time for the thermostat. A scenario that declares its own
        `thermostat_damp` parameter uses it; otherwise the Langevin time."""
        if params.has("thermostat_damp"):
            return params["thermostat_damp"]
        return self.langevin_damp

    # Constructor keywords that set an attribute rather than a parameter: they
    # describe the scenario's dynamics rather than its geometry, and nothing reads
    # them through the ParamSet.
    _ATTR_KWARGS = ("timestep", "sim_time_per_frame", "langevin_damp",
                    "neighbor_skin", "director_arrows", "wrap_fade_fraction")

    def __init__(self, **overrides):
        """Keyword arguments are structural-parameter overrides, so a playground
        file reads `HexPatch(n_rings=2)` instead of assembling a ParamSet."""
        for name in self._ATTR_KWARGS:
            if name in overrides:
                setattr(self, name, overrides.pop(name))
        self.defaults = dict(overrides)

    def new_params(self, overrides=None):
        """Declared defaults, then this scenario's construction overrides, then
        anything the playground passes -- in that precedence order."""
        merged = dict(self.defaults)
        merged.update(overrides or {})
        return ParamSet.build(self.params, merged)

    @abstractmethod
    def build(self, params, rng):
        """Return a ScenarioBuild. `rng` is a seeded numpy Generator, so a
        scenario's randomness is reproducible from the playground's seed."""

    def atom_creation_commands(self, params, seed):
        """LAMMPS-side particle creation, for scenarios that let LAMMPS place
        particles (e.g. `create_atoms random`, which does its own overlap
        rejection). None -> the runtime uploads build().positions instead."""
        return None

    def create_commands(self, params, build, seed):
        """Commands issued right after the atoms exist -- per-particle attributes
        the scenario (not the force field) owns, e.g. the initial directors."""
        return []

    def wall_commands(self, box):
        """Reflecting walls on every non-periodic face.

        A particle flung out during violent play, or one that thermally
        evaporates out of a monolayer, feels no pair force past the cutoff and
        would otherwise random-walk to the box edge and be lost. An elastic
        reflection injects no energy and keeps it close enough for the membrane's
        own attraction to recapture it -- the implicit-solvent confinement the
        model otherwise omits.
        """
        faces = []
        for axis, per in zip("xyz", box.periodic):
            if not per:
                faces += [f"{axis}lo EDGE", f"{axis}hi EDGE"]
        return ["fix walls all wall/reflect " + " ".join(faces)] if faces else []

    def group_commands(self, params, controlled_id):
        """Groups this scenario needs beyond the mode's own (a frozen floor, a
        mobile subset the thermostat acts on). Issued after the mode's groups, so
        `controlled` already exists."""
        return []

    def thermostat_group(self):
        """Group the thermostat acts on, or None to let the mode decide (which is
        normally "everything except the controlled particle")."""
        return None

    def integrator_commands(self, params):
        """Time integrators this scenario installs itself, when a single global
        one will not do -- a frozen floor plus a mobile crystal, say. Empty ->
        the force field's global integrator is used."""
        return []

    def extra_setup_commands(self, params):
        """Anything else the deck needs: LAMMPS variables read per frame,
        comm_modify tweaks, native computes."""
        return []

    def make_rdf(self, params, lmp, box):
        """Override the runtime's RDF choice. None -> let it pick from the box's
        periodicity."""
        return None

    def frame_commands(self, params, lmp):
        """Commands to issue once per frame, when they depend on measured state
        and so cannot be a fix (e.g. a drag proportional to this frame's centre-of-
        mass velocity). `lmp` is the live instance, for reading variables."""
        return []

    def pre_control_settle(self, params, seed):
        """Relaxation run BEFORE the thermostat and control fixes are defined.
        Owns its own bath/barostat and its own `run`; anything it defines must be
        listed in settle_cleanup_commands."""
        return []

    def settle_cleanup_commands(self):
        return []

    def post_control_settle(self, params):
        """Relaxation run with the thermostat and control fixes already in
        place."""
        return []

    def housekeeping(self, positions, params, controlled=None, box=None):
        """Per-frame soft correction forces (N, 3), or None.

        `controlled` is the index of the interactively-controlled particle (or
        None), which must be excluded: its position IS the user's input, and a
        correction force would fight it. `box` is the live cell, which a periodic
        scenario needs to know where "the middle" is.
        """
        return None

    def director_housekeeping(self, positions, directors, params, controlled=None,
                              box=None):
        """Per-frame soft correction ANGULAR velocities (N, 3) on the particles'
        directors, or None.

        The counterpart of `housekeeping` for orientation. It exists separately
        because in a periodic cell it is the only one of the two that can steer
        which way a structure faces: translating everything is fine on a torus,
        rotating everything is not.
        """
        return None

    def camera(self, box):
        """dict(eye=, target=, up=, fov_deg=) for the 3D scene. The runtime then
        zooms to fit_points, so this only sets a good angle and rough distance."""
        span = max(box.lengths)
        return dict(eye=(0.0, -0.9 * span, 0.6 * span), target=(0.0, 0.0, 0.0),
                    up=(0.0, 0.0, 1.0), fov_deg=34.0)

    def fit_points(self, params, box):
        """World points the camera should frame. The box corners keep the drawn
        box outline in view at any aspect ratio."""
        return box.corners()


# --- housekeeping building blocks ---------------------------------------------
# Shared by the patch and the sheet, which independently derived the same
# "rotate the smallest principal axis to +z" correction.

def _ramp(x, lo, hi):
    """Smoothstep gate: 0 below lo, 1 above hi, no kink in between. Every
    whole-system correction here is gated by one, so it fades in with the
    structure it is correcting instead of switching on."""
    t = min(max((x - lo) / max(hi - lo, 1e-9), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def align_normal_rate(points, strength):
    """Rigid-rotation rate turning the cloud's best-fit plane normal toward +z,
    as omega = strength * (n x z). A flat cloud gets no rotation; a tilted one is
    turned back flat, with magnitude ~sin(tilt)."""
    n = principal_normal(points)
    return strength * np.cross(n, np.array([0.0, 0.0, 1.0]))


def align_normal_forces(points, strength):
    """Per-point forces producing a net normal-up alignment torque and ZERO net
    force, via F_i = a x r_i' with a = Iang^-1 T about the centroid. Used where
    the cluster is small enough that a genuine torque (rather than a rotation
    rate) is the right correction."""
    com = points.mean(axis=0)
    q = points - com
    torque = align_normal_rate(points, strength)
    # Inertia-like tensor Iang = sum(|r'|^2 I - r' r'^T).
    iang = np.eye(3) * np.sum(q * q) - q.T @ q
    a = np.linalg.solve(iang + 1e-6 * np.eye(3), torque)
    return np.cross(a[None, :], q)


def _exclude(n, controlled):
    mask = np.ones(n, dtype=bool)
    if controlled is not None:
        mask[controlled] = False
    return np.nonzero(mask)[0]


# --- concrete scenarios -------------------------------------------------------

class HexPatch(Scenario):
    """A small hexagonal patch of particles in the xy-plane: a centre site ringed
    by closed hexagonal shells, directors along +z.

    The smallest geometry that already shows tilt/splay physics -- pull the middle
    particle out of plane and its neighbours' directors splay to follow. The
    surrounding rings are held in place by soft forces (not a geometric
    imposition), standing in for continuation into a larger membrane, so they
    still move, jitter and respond dynamically.
    """

    name = "hex_patch"

    params = (
        structural("n_rings", 1, "closed hexagonal shells around the centre site"),
        structural("a", 1.0, "nearest-neighbour spacing"),
        structural("box", 6.0, "cubic container side"),
        structural("settle_steps", 300, "silent relaxation before control begins"),
        # Strong, but still forces -- the patch can dome, tilt and recover
        # realistically. The homing term is a safety net so no single outer
        # particle can be flung out of the box during violent play.
        structural("k_center", 7.0, "centre-of-mass centering stiffness"),
        structural("k_align", 7.0, "normal-up alignment torque strength"),
        structural("k_home", 0.5, "per-particle homing stiffness toward the origin"),
        # What the camera looks at and frames -- a rectangle in the control
        # plane, centred a little above the patch (see camera / fit_points).
        # Smaller half-extents = closer in.
        structural("view_center_z", 0.7, "camera framing: centre height, in z"),
        structural("view_half_width", 2.0, "camera framing: half-width, in x"),
        structural("view_half_height", 1.6, "camera framing: half-height, in z"),
    )

    def build(self, params, rng):
        pts2d = hex_ring_2d(int(params["n_rings"]), params["a"])
        pos = np.column_stack([pts2d, np.zeros(len(pts2d))])
        dirs = np.tile([0.0, 0.0, 1.0], (len(pos), 1))
        # Spokes from the centre plus the closed first ring. hex_ring_2d orders
        # each shell by angle, so the ring closes correctly.
        n1 = min(6, len(pos) - 1)
        bonds = [(0, k) for k in range(1, n1 + 1)]
        bonds += [(k, k % n1 + 1) for k in range(1, n1 + 1)]
        return ScenarioBuild(positions=pos, directors=dirs,
                             box=Box.cube(params["box"]), bonds=tuple(bonds))

    def create_commands(self, params, build, seed):
        return ["set group all dipole 0.0 0.0 1.0"]

    def post_control_settle(self, params):
        return [f"run {int(params['settle_steps'])}"]

    def housekeeping(self, positions, params, controlled=None, box=None):
        """Centring + normal-up alignment + homing on every particle except the
        controlled one, as per-frame momentum kicks the Langevin bath then damps
        (so the patch settles flat and centred instead of oscillating)."""
        sel = _exclude(len(positions), controlled)
        if len(sel) < 3:
            return None
        p = positions[sel]
        f = np.zeros((len(positions), 3))
        # Same force on each -> moves the centre of mass without distorting shape.
        f[sel] = -params["k_center"] * p.mean(axis=0)
        f[sel] += align_normal_forces(p, params["k_align"])
        f[sel] -= params["k_home"] * p
        return f

    def camera(self, box):
        """Aimed a little ABOVE the patch, not at it.

        Everything interesting here happens upward: the controlled particle is
        pulled out of the plane and the ring domes after it. Aiming at the patch
        itself spends half the frame on the empty space below it, so the target
        is lifted by `view_center_z` and the vertical framing budget goes where
        the motion is."""
        span = max(box.lengths)
        return dict(eye=(0.0, -0.9 * span, 0.6 * span),
                    target=(0.0, 0.0, self.new_params()["view_center_z"]),
                    up=(0.0, 0.0, 1.0), fov_deg=34.0)

    def fit_points(self, params, box):
        """Frame a rectangle in the CONTROL PLANE, not the container.

        Framing the box corners -- the obvious thing, and what this did -- puts
        the camera far too far back: the box's near face sits several units
        closer to the eye than the patch does, subtends a much larger angle, and
        the fit backs off until THAT fits. On the default 6-sigma box the seven
        beads came out filling about a fifth of the frame width, adrift in an
        empty cell.

        Framing a rectangle at y = 0 instead -- the plane the beads and the
        control net live in -- puts the patch where it belongs, at a bit over
        half the frame width. What that costs is the outer corners of the net and
        the box outline, which now fall outside the view; both are context for a
        scene whose subject is seven beads, and the beads win.
        """
        w, h = params["view_half_width"], params["view_half_height"]
        z = params["view_center_z"]
        return np.array([(-w, 0.0, z - h), (w, 0.0, z - h),
                         (-w, 0.0, z + h), (w, 0.0, z + h)])


class HexSheet(Scenario):
    """A periodic hexagonal monolayer -- the paper's planar-stability test.

    Particles on a hexagonal lattice at spacing a, periodic in-plane, relaxed
    under Langevin dynamics with a barostat driving the lateral pressure to zero
    so the sheet equilibrates tension-free. The barostat is then removed and the
    cell frozen, so interactive pulling happens at a fixed, relaxed lattice. The
    periodic cell means no artificial tether is needed -- the sheet holds itself
    flat.
    """

    name = "hex_sheet"
    sim_time_per_frame = 0.1
    director_arrows = False     # hundreds of spikes are clutter and cost
    wrap_fade_fraction = 0.03   # slide across the seam instead of popping

    params = (
        structural("n_cols", 30, "particles per row (x)"),
        structural("n_rows", 30, "rows (y)"),
        structural("a", 0.8, "hexagonal spacing (the paper's benchmark value)"),
        # Kept shallow on purpose: an evaporated particle stays right by the
        # sheet, so the membrane's own attraction recaptures it.
        structural("z_half", 4.0, "out-of-plane half-height of the container"),
        structural("settle_steps", 1000, "barostat relaxation to zero lateral tension"),
        # How much of the (possibly tiled) membrane the camera frames, as a
        # multiple of the cell's in-plane half-extent. Pair it with
        # RenderStyle.periodic_images: framing 1.0 with a 3x3 tiling wastes the
        # tiling off-screen, framing 1.8 with no tiling wastes the frame.
        structural("view_span", 1.0, "camera framing: multiples of the cell half-width"),
        # How far past the cell's centre the camera aims, in multiples of its
        # in-plane half-width. 0 looks straight at the middle of the cell, which
        # leaves the near edge floating in the middle of the frame; aiming
        # further in drops that edge toward the bottom, which is what you want
        # when the cell is the FRONT of a tiled block and the images recede
        # behind it (see RenderStyle.periodic_images).
        structural("view_aim_ahead", 0.0, "camera framing: aim this far past the centre"),
        structural("baro_press", 0.0, "target lateral pressure (tension-free)"),
        structural("baro_damp", 2.0, "barostat relaxation time"),
        structural("k_plane", 0.1, "pull toward the central z-plane"),
        structural("k_align", 10.0, "normal-up alignment rate"),
        structural("tracer_fraction", 0.3,
                   "where to place the highlighted diffusion tracer, as a "
                   "fraction in from the front-left corner (None -> no tracer)"),
    )

    def build(self, params, rng):
        n_cols, n_rows, a = int(params["n_cols"]), int(params["n_rows"]), params["a"]
        pts2d = hex_lattice_2d(n_cols, n_rows, a)
        pos = np.column_stack([pts2d, np.zeros(len(pts2d))])
        dirs = np.tile([0.0, 0.0, 1.0], (len(pos), 1))
        # Sized exactly to the lattice so the sheet tiles seamlessly.
        lx = n_cols * a
        ly = n_rows * a * math.sqrt(3.0) / 2.0
        box = Box((-lx / 2.0, -ly / 2.0, -params["z_half"]),
                  (lx / 2.0, ly / 2.0, params["z_half"]),
                  periodic=(True, True, False))
        # Particles at a = 0.8 with diameter 1.0 overlap into a continuous sheet,
        # so no bond sticks (and none would tile across the periodic seam).
        return ScenarioBuild(positions=pos, directors=dirs, box=box,
                             brightness=self._tracer_brightness(pts2d, params))

    def _tracer_brightness(self, pts2d, params):
        """Brighten a small cluster -- one particle and its six nearest
        neighbours -- so it can be followed as it diffuses through the membrane.
        Chosen once from the initial lattice; because the runtime orders
        rendering by atom id, the highlight tracks the same particles as they
        wander."""
        frac = params["tracer_fraction"]
        if frac is None:
            return None
        lo, hi = pts2d.min(axis=0), pts2d.max(axis=0)
        target = lo + frac * (hi - lo)
        centre = int(np.argmin(np.linalg.norm(pts2d - target, axis=1)))
        d = np.linalg.norm(pts2d - pts2d[centre], axis=1)
        bright = np.ones(len(pts2d))
        bright[np.argsort(d)[1:7]] = 1.5
        bright[centre] = 2.1
        return bright

    def create_commands(self, params, build, seed):
        return ["set group all dipole 0.0 0.0 1.0"]

    def pre_control_settle(self, params, seed):
        """Langevin + a Berendsen barostat on x,y to reach the tension-free
        equilibrium spacing. The barostat is not an integrator (it rides on top
        of nve/sphere), so it only relaxes the in-plane spacing. Run before any
        control fixes exist so it relaxes a free sheet."""
        p, damp = params["baro_press"], params["baro_damp"]
        return [
            f"fix settle_bath all langevin 0.0 0.0 {self.langevin_damp} {seed} omega yes",
            f"fix settle_baro all press/berendsen x {p} {p} {damp} "
            f"y {p} {p} {damp} couple xy",
            f"run {int(params['settle_steps'])}",
        ]

    def settle_cleanup_commands(self):
        return ["unfix settle_baro", "unfix settle_bath"]

    def housekeeping(self, positions, params, controlled=None, box=None):
        """Plane centring toward z = 0, plus a rigid normal-up rotation of the
        whole sheet -- the same "smallest principal component upward" idea as the
        patch's ring torque, made size-independent by applying it as a rotation
        rate rather than a torque."""
        sel = _exclude(len(positions), controlled)
        if len(sel) < 3:
            return None
        p = positions[sel]
        f = np.zeros((len(positions), 3))
        f[sel, 2] += -params["k_plane"] * p[:, 2]
        omega = align_normal_rate(p, params["k_align"])
        f[sel] += np.cross(omega[None, :], p - p.mean(axis=0))
        return f

    def camera(self, box):
        """The eye pulls back with `view_span`, not just the zoom.

        Re-framing alone cannot show a wider membrane: at 1.7x the cell the
        sheet's near edge reaches the camera's own position, and fitting THAT in
        sends the zoom to infinity. Scaling the distance keeps the geometry
        similar, so the picture is the same one seen from further away."""
        params = self.new_params()
        span = max(box.lengths[0], box.lengths[1]) * params["view_span"]
        aim = params["view_aim_ahead"] * 0.5 * box.lengths[1]
        return dict(eye=(0.0, -0.85 * span + aim, 0.6 * span),
                    target=(0.0, aim, 0.0), up=(0.0, 0.0, 1.0), fov_deg=34.0)

    def fit_points(self, params, box):
        """Frame the cell, or more of it when the renderer is drawing periodic
        images: `view_span` multiplies the in-plane extent the camera pulls back
        to cover, so a 3x3 tiling can actually be seen rather than sitting mostly
        off the edges of the frame. 1.0 frames the real cell exactly."""
        span = params["view_span"]
        aim = params["view_aim_ahead"] * 0.5 * box.lengths[1]
        corners = box.corners().astype(float)
        corners[:, :2] *= span
        corners[:, 1] += aim
        return np.vstack([corners, [(0.0, aim, 2.6), (0.0, aim, -2.2)]])


class RandomFill(Scenario):
    """N particles at random positions and orientations in a fully periodic cell
    -- the paper's spontaneous-assembly test.

    Under Langevin dynamics the disordered gas coarsens: small patches by
    t ~ 500 tau, coalescing into large planar membranes by t ~ 2000 tau. There is
    nothing to steer, so this is normally run in sim mode -- though game mode
    works too (it will pick a controllable particle), which is exactly the
    orthogonality the mode split buys.
    """

    name = "random_fill"
    timestep = 0.01
    sim_time_per_frame = 0.2
    # Weaker friction than the probing scenarios (larger damp = weaker drag) lets
    # particles diffuse and find each other, so assembly proceeds watchably.
    langevin_damp = 1.0
    neighbor_skin = 0.6
    director_arrows = False

    params = (
        structural("n", 1500, "particle count"),
        structural("box", 20.0,
                   "cubic cell side (phi = N*Vp/L^3 ~ 0.1 at the defaults)"),
        # 0.9 sigma sits just outside the 4-2 core's minimum at r = sigma, where
        # the pair force is small and attractive, so no two particles are dropped
        # inside each other's hard core and blow up the first step.
        structural("overlap", 0.9, "minimum centre-to-centre separation when seeding"),
        structural("maxtry", 200, "placement attempts per particle"),
        # --- two corrections that keep the run watchable, both OFF at 0 ------
        # Neither is part of the paper's experiment. Measured over 40k steps
        # against unforced controls: the field takes the membranes' common normal
        # to within 4-8 degrees of vertical (unforced, it wanders 35-64) while
        # the nematic order rises just as it does without it, so what is being
        # steered is the orientation, not the assembly. The centring only acts on
        # an axis the particles are actually concentrated along, so it does
        # nothing at all until something has formed.
        structural("k_upright", 0.6,
                   "field aligning directors with z, so sheets form flat (0 = off)"),
        structural("center_accel", 0.05,
                   "drift acceleration re-centring the aggregate (0 = off)"),
        structural("center_softness", 2.0,
                   "distance over which the centring eases off near the middle, sigma"),
        structural("center_concentration_min", 0.15,
                   "circular concentration below which an axis reads as uniform"),
        structural("center_concentration_full", 0.35,
                   "circular concentration at which centring reaches full strength"),
    )

    def housekeeping(self, positions, params, controlled=None, box=None):
        """Nudge whatever has assembled back to the middle of the cell.

        "The middle" needs care on a torus: an ordinary mean is meaningless
        there (a blob straddling the seam averages to the far side of the box).
        The circular mean is not -- map each coordinate onto the unit circle,
        theta = 2*pi*x/L, average the unit vectors, and read the angle back. That
        answer is translation-covariant, exactly as a centre of mass should be.
        The length R of the averaged vector comes free and is the useful part: it
        is the CONCENTRATION of the distribution along that axis, 1 for a tight
        blob and ~1/sqrt(N) for a uniform spread. So the correction is applied
        per axis and gated on R, which gives the right behaviour for free:
        a droplet is concentrated on all three axes and gets centred in all
        three, while a lamella is concentrated only along its normal and gets
        centred only in that direction -- exactly right, since it is uniform
        along the other two and has no centre there to find.
        Uniform on every particle, so it translates the system without shearing
        it, and eased off (tanh) near the target so it settles instead of
        oscillating. The Langevin bath turns the sustained acceleration into a
        slow terminal drift.
        """
        accel = params["center_accel"]
        if accel <= 0.0 or box is None:
            return None
        lo = np.asarray(box.lo, dtype=float)
        L = np.asarray(box.lengths, dtype=float)
        theta = 2.0 * np.pi * (positions - lo) / L
        c, sn = np.cos(theta).mean(axis=0), np.sin(theta).mean(axis=0)
        concentration = np.hypot(c, sn)
        mean = lo + L * (np.arctan2(sn, c) % (2.0 * np.pi)) / (2.0 * np.pi)
        d = np.asarray(box.center, dtype=float) - mean
        d -= L * np.round(d / L)                       # minimum image
        gate = np.array([_ramp(r, params["center_concentration_min"],
                               params["center_concentration_full"])
                         for r in concentration])
        a = accel * gate * np.tanh(d / max(params["center_softness"], 1e-6))
        if not np.any(a):
            return None
        f = np.tile(a, (len(positions), 1))
        if controlled is not None:
            f[controlled] = 0.0
        return f

    def director_housekeeping(self, positions, directors, params, controlled=None,
                              box=None):
        """A weak field that prefers membranes lying flat, normal along z.

        WHY A FIELD ON EACH DIRECTOR, AND NOT A TORQUE ON THE WHOLE SYSTEM.
        The obvious construction -- read the common normal off the directors as a
        tensor, and apply the one rigid rotation that brings it upright -- was
        tried first and is worse than useless. It rotates every director away
        from ITS OWN local membrane normal, which is precisely what the force
        field's tilt term exists to punish, so the membrane wins and the only
        result is energy pumped into the aggregates: measured over 60k steps
        against an unforced control, it did not flatten anything and pulled the
        nematic order back down from 0.52 to 0.11 as the sheets it was wrestling
        came apart.
        And that is the honest answer for a PERIODIC cell, not a bug. A membrane
        spanning one has to be commensurate with it -- it lies parallel to a pair
        of box faces -- so its available orientations are three discrete choices,
        not a continuum to be rotated through. Getting from one to another means
        dissolving and re-forming, which no gentle torque can drive.
        What CAN be biased is which one it forms in. Each director is pulled
        toward the nearer of +/-z, exactly as an external field aligns a nematic:
        in the disordered gas, where the particles are free to turn, that tilts
        the odds toward horizontal patches nucleating, and they then grow the way
        they always would. It is self-extinguishing -- the torque is proportional
        to sin(angle to z), so a membrane that IS flat feels nothing at all --
        which is why it needs no gate and never fights a finished sheet.
        """
        k = params["k_upright"]
        if k <= 0.0 or not len(directors):
            return None
        # Toward the NEARER pole: +n and -n are the same physical orientation, so
        # a director in the lower half turns toward -z, not the long way round.
        sense = np.where(directors[:, 2] < 0.0, -1.0, 1.0)
        up = np.zeros_like(directors)
        up[:, 2] = sense
        w = k * np.cross(directors, up)
        if controlled is not None:
            w[controlled] = 0.0
        return w

    def build(self, params, rng):
        """Positions are placed by LAMMPS (`create_atoms random` does the overlap
        rejection), so the positions array is empty and the box is what
        matters."""
        return ScenarioBuild(positions=np.zeros((0, 3)), directors=None,
                             box=Box.cube(params["box"], (True, True, True)))

    def atom_creation_commands(self, params, seed):
        return [
            f"create_atoms 1 random {int(params['n'])} {seed} box "
            f"overlap {params['overlap']} maxtry {int(params['maxtry'])} units box"
        ]

    def create_commands(self, params, build, seed):
        # Random initial director orientations -- the disordered orientational
        # start the paper's self-assembly begins from.
        return [f"set group all dipole/random {seed} 1.0"]

    def camera(self, box):
        s = max(box.lengths)
        return dict(eye=(0.6 * s, -1.1 * s, 0.7 * s), target=(0.0, 0.0, 0.0),
                    up=(0.0, 0.0, 1.0), fov_deg=34.0)


class Composite(Scenario):
    """Several sub-scenarios' geometries placed at offsets in one shared cell.

    This is what makes "two patches colliding" a three-line playground instead of
    a new 800-line module. The first part's timing, housekeeping and rendering
    settings apply to the whole assembly.
    """

    name = "composite"

    def __init__(self, parts, box=None, periodic=(False, False, False),
                 margin=2.0):
        super().__init__()
        self.parts = tuple(parts)            # ((scenario, offset), ...)
        self._box_side = box
        self._periodic = periodic
        self._margin = margin
        # Each part is configured at its own call site -- compose(hex_patch(
        # n_rings=2), ...) -- so it keeps its own ParamSet rather than having its
        # parameters re-exported under a prefix here.
        self._part_params_cache = tuple(s.new_params() for s, _ in self.parts)
        base = self.parts[0][0]
        self.timestep = base.timestep
        self.sim_time_per_frame = base.sim_time_per_frame
        self.langevin_damp = base.langevin_damp
        self.director_arrows = base.director_arrows
        self.wrap_fade_fraction = base.wrap_fade_fraction

    def build(self, params, rng):
        chunks, dirs, bonds = [], [], []
        offset = 0
        for i, (scenario, at) in enumerate(self.parts):
            sub = scenario.build(self._part_params_cache[i], rng)
            chunks.append(sub.positions + np.asarray(at, dtype=float))
            dirs.append(sub.directors if sub.directors is not None
                        else np.tile([0.0, 0.0, 1.0], (len(sub.positions), 1)))
            bonds += [(a + offset, b + offset) for a, b in sub.bonds]
            offset += len(sub.positions)
        pos = np.vstack(chunks)
        if self._box_side is not None:
            box = Box.cube(self._box_side, self._periodic)
        else:
            # Snug box around everything, with a margin for the pull reach.
            half = np.abs(pos).max(axis=0) + self._margin
            box = Box(tuple(-half), tuple(half), self._periodic)
        return ScenarioBuild(positions=pos, directors=np.vstack(dirs), box=box,
                             bonds=tuple(bonds))

    def create_commands(self, params, build, seed):
        return ["set group all dipole 0.0 0.0 1.0"]

    def post_control_settle(self, params):
        return self.parts[0][0].post_control_settle(self._part_params_cache[0])

    def housekeeping(self, positions, params, controlled=None):
        return self.parts[0][0].housekeeping(
            positions, self._part_params_cache[0], controlled)


def compose(*parts, box=None, periodic=(False, False, False), margin=2.0):
    """compose(hex_patch(at=(-6, 0, 0)), hex_patch(at=(+6, 0, 0)))

    Each argument is a Scenario (placed at the origin) or a (Scenario, offset)
    pair in either order.
    """
    normalized = []
    for part in parts:
        if isinstance(part, Scenario):
            normalized.append((part, (0.0, 0.0, 0.0)))
        else:
            a, b = part
            normalized.append((a, b) if isinstance(a, Scenario) else (b, a))
    return Composite(normalized, box=box, periodic=periodic, margin=margin)


# --- convenience constructors used in playground files ------------------------
# Lower-case aliases so a playground file reads like a description of the setup.
# `at=` returns the (scenario, offset) pair compose() expects.

def _configured(cls, at, overrides):
    scenario = cls(**overrides)
    return (scenario, at) if at is not None else scenario


def hex_patch(at=None, **overrides):
    return _configured(HexPatch, at, overrides)


def hex_sheet(at=None, **overrides):
    return _configured(HexSheet, at, overrides)


def random_fill(at=None, **overrides):
    return _configured(RandomFill, at, overrides)

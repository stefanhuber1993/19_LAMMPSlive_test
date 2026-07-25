"""MesoMem membrane sheet -- the paper's planar-stability test, made interactive.

Same real MesoMem force field as the 7-bead patch (see mesomem_hex.py), but at
the scale the paper (Sillano, Marrink & Idema 2026, Sec. IV) actually uses to
check that the interaction scheme supports a *stable planar membrane*: particles
on a hexagonal lattice at spacing a = 0.8 sigma, periodic in-plane, relaxed under
Langevin dynamics with a barostat that drives the lateral pressure to zero
(Pxx = Pyy = 0) so the sheet equilibrates tension-free. The paper's benchmark is
50x50 sites; here we use a smaller sheet that still limits finite-size effects
while staying real-time under interactive control.

Differences from the 7-bead patch:
  - Periodic (p p f) box sized exactly to the hex lattice, so the sheet tiles
    seamlessly and holds itself flat -- no artificial ring tether is needed
    (that was the 7-bead demo's stand-in for "the rest of the membrane").
  - A short settle phase runs Langevin + a Berendsen barostat on x,y to reach
    the tension-free equilibrium spacing; the barostat is then removed and the
    box frozen, so interactive pulling happens at a fixed, relaxed lattice.
  - The center bead (nearest the box center) is the puller, exactly as before:
    the joystick/mouse slides it in the world xz control plane and Q/E twist its
    director. Everything the haptics/readouts need is the same 2D projection.

Units are the paper's reduced LJ units (sigma = eps = m = 1).
"""
import math
import random

import numpy as np
from lammps import lammps
from scipy.spatial import cKDTree

from .base import ForceFeedbackProfile, MDSystem, SliderSpec, SystemSpec
from .mesomem_ff import ensure_plugin_loaded
from .rdf2d import InPlaneRDF

# --- MesoMem potential parameters (identical to the 7-bead patch) -------------
SIGMA = 1.0
EPS = 1.0
K_TILT = 12.0
K_SPLAY = 1.0
RC = 2.5
WC = 2.0
ZETA = 5.0
C0 = 0.0

# Live-tunable ranges for the MesoMem coefficient sliders, each with the paper's
# recommended value marked as the slider "optimum" (see the 7-bead patch for the
# per-parameter rationale; same standard-condition centers and spans).
K_TILT_MIN, K_TILT_MAX, K_TILT_OPT = 0.0, 50.0, 12.0
K_SPLAY_MIN, K_SPLAY_MAX, K_SPLAY_OPT = 0.0, 40.0, 1.0
ZETA_MIN, ZETA_MAX, ZETA_OPT = 0.0, 12.0, 5.0
# Both cutoffs are allowed all the way down to 0 (interactions can be switched
# fully off) rather than starting at the paper's operating values.
RC_MIN, RC_MAX, RC_OPT = 0.0, 3.0, 2.5
WC_MIN, WC_MAX, WC_OPT = 0.0, 3.0, 2.0

# Splay symmetry (advanced): 0..1 weight blending the splay term's SIGNED
# director dot product (0, the paper's original form -- antiparallel neighbours
# punished) toward its ABSOLUTE value (1 -- parallel and antiparallel treated
# identically, i.e. the term cares only about the axis, not the sense). See the
# pair style's compute() and _apply_pair_coeff below.
SPLAY_SYM = 0.0
SPLAY_SYM_MIN, SPLAY_SYM_MAX = 0.0, 1.0

# --- Sheet "cleanup" forces (housekeeping, not MesoMem physics) --------------
# Two soft per-frame corrections keep the free periodic monolayer well-posed for
# interactive play, applied as momentum kicks (delta v = F*dt) and damped by the
# Langevin bath so they settle rather than ring:
#   - a restoring pull of every bead toward the box's central z-plane (z = 0),
#     so the sheet can't drift/buckle out of frame or slowly rotate out of plane;
#   - an alignment torque on the whole sheet that rotates its smallest principal
#     axis (the membrane normal) toward +z, exactly like the 7-bead patch's ring
#     torque, so the membrane stays face-on to the camera.
# K_PLANE is the tunable "pull toward the central plane" stiffness the demo asks
# to expose; K_ALIGN is the normal-up torque strength.
K_PLANE = 0.1
K_ALIGN = 10.0

# --- Sheet geometry -----------------------------------------------------------
A_LATTICE = 0.8      # hexagonal nearest-neighbor spacing (paper's benchmark value)
N_COLS = 30          # beads per row (x)
N_ROWS = 30          # rows (y); ~900 beads -- large enough to look like a membrane,
                     # small enough to stay interactive in real time
BEAD_DIAMETER = 1.0  # sphere diameter = sigma (rendered radius 0.5 sigma)
# Out-of-plane half-height of the box container, and a reflecting wall at each z
# face (see _build). Kept shallow (a few sigma above the puller's z reach of 2.5)
# on purpose: a monolayer bead that thermally evaporates out of plane at high T
# feels no pair force past rc=2.5 and would otherwise random-walk to the box edge
# and be lost. The wall reflects it (elastic -> no heating) and, because the box
# is shallow, it stays right by the sheet so the membrane's own attraction
# recaptures it -- the implicit-solvent confinement the model otherwise lacks.
Z_HALF = 4.0

TIMESTEP = 0.005     # tau_LJ (paper uses 0.01; halved for stability while pulling)

# Settle: relax the sheet from the perfect lattice to a tension-free equilibrium.
# The paper runs 5e4 steps at dt=0.01; a=0.8 is already close to equilibrium, so
# a shorter relaxation suffices to remove residual lateral stress here.
SETTLE_STEPS = 1000
BARO_PRESS = 0.0     # target lateral pressure (tension-free)
BARO_DAMP = 2.0      # barostat relaxation time (tau_LJ)

# Langevin (implicit solvent), same spirit as the patch: calm enough to probe.
LANGEVIN_DAMP = 0.5

# Puller extra drag (damping dial) and its own plane/leash constraints.
PULLER_DAMPING_DEFAULT = 4.0
PULLER_DAMPING_MIN = 0.0
PULLER_DAMPING_MAX = 8.0

T_MIN = 0.0
T_MAX = 0.5
T_MELT = 0.3

# Yaw steering of the puller's director (identical mechanism to the 7-bead
# patch, but at 2x authority -- the sheet is a large, cohesive membrane and the
# patch's forces barely tented it, so pull and twist are both doubled here to
# actually let the user manipulate the sheet).
YAW_TORQUE = 1.0
ROT_DAMP = 0.88
REACTION_TORQUE_DISPLAY_MAX = 6.0

FORCE_FEEDBACK = ForceFeedbackProfile(
    ff_exaggeration=1.3,
    ff_knee=4.0,
    ff_max_mag=120.0,
    stiffness_threshold=0.3,
    stiffness_knee=2.5,
    damper_min_fraction=0.10,
    damper_max_fraction=0.55,
    vel_damp_max_fraction=0.5,
)

SPEC = SystemSpec(
    key="membrane",
    name="MesoMem membrane sheet (3D)",
    description="Paper-scale hexagonal MesoMem sheet (periodic, barostat-relaxed): pull one bead out and watch tilt/splay propagate.",
    element_label="membrane bead (director)",
    lattice_spacing=A_LATTICE,
    timestep=TIMESTEP,
    temperature=SliderSpec("Temperature", T_MIN, T_MAX, 0.001, fmt="{:.3f}", unit=" T*"),
    damping=SliderSpec("Puller damping", PULLER_DAMPING_MIN, PULLER_DAMPING_MAX,
                       PULLER_DAMPING_DEFAULT, fmt="{:.2f}", advanced=True),
    melt_temp=T_MELT,
    force_feedback=FORCE_FEEDBACK,
    max_input_force=12.0,   # reduced units at full deflection, shared by joystick/WASD/mouse
    puller_speed_cap=0.06 * SIGMA / TIMESTEP,
    crystal_color=None,
    atom_radius_A=0.5 * SIGMA,
    # 0.1 tau_LJ per frame (20 steps at dt=0.005) -- double the patch's slice so
    # the sheet actually evolves/heals/melts at a watchable rate rather than
    # inching along. The renderer optimizations (capped bead-sprite generation,
    # net-local z-buffer) buy back the extra per-frame sim cost.
    sim_time_per_frame=0.1,
    bond_overlay=False,
    render_3d=True,
    reduced_units=True,
    # No per-bead director spikes on the sheet: hundreds of arrows are clutter,
    # and the white-pole bead coloring already shows each director's sense/tilt.
    director_arrows=False,
    # Periodic in x,y: crossfade beads over the seam (last 3% of each side) so
    # they slide across instead of popping.
    wrap_fade_fraction=0.03,
    extra_sliders=(
        SliderSpec("k_tilt", K_TILT_MIN, K_TILT_MAX, K_TILT, fmt="{:.1f}",
                   key="k_tilt", optimum=K_TILT_OPT),
        SliderSpec("k_splay", K_SPLAY_MIN, K_SPLAY_MAX, K_SPLAY, fmt="{:.2f}",
                   key="k_splay", optimum=K_SPLAY_OPT),
        SliderSpec("zeta (attraction falloff, higher=shorter reach)",
                   ZETA_MIN, ZETA_MAX, ZETA, fmt="{:.1f}", key="eta",
                   optimum=ZETA_OPT),
        # Advanced controls, hidden behind the panel's "Advanced" toggle.
        SliderSpec("splay symmetry (0=signed, 1=|dot|)", SPLAY_SYM_MIN,
                   SPLAY_SYM_MAX, SPLAY_SYM, fmt="{:.2f}", key="splay_symmetry",
                   advanced=True),
        SliderSpec("rc (interaction cutoff)", RC_MIN, RC_MAX, RC, fmt="{:.2f}",
                   key="rc", optimum=RC_OPT, advanced=True),
        SliderSpec("wc (orientation cutoff)", WC_MIN, WC_MAX, WC, fmt="{:.2f}",
                   key="wc", optimum=WC_OPT, advanced=True),
    ),
)


class MesoMemSheetSystem(MDSystem):
    spec = SPEC

    def __init__(self):
        self.lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
        self._input_fx = 0.0
        self._input_fz = 0.0
        self._yaw = 0.0
        self._target_temp = SPEC.temperature.default
        self._puller_damping = PULLER_DAMPING_DEFAULT
        self._interactive_t = 0.0
        self._seed = random.randint(1, 900_000_000)
        # Live-tunable MesoMem coefficients (start at the paper's standard values).
        self._ktilt = K_TILT
        self._ksplay = K_SPLAY
        self._zeta = ZETA
        self._rc = RC
        self._wc = WC
        self._splay_sym = SPLAY_SYM
        # Throttled cache for the whole-sheet potential total (O(pairs) each time,
        # so it's recomputed every few frames rather than every frame).
        self._total_terms_cache = None
        self._total_terms_ctr = 0

        self._lattice_xy = self._hex_lattice()      # (N,2) intended lattice sites
        self._n = len(self._lattice_xy)
        # Puller = the bead nearest the box center.
        self.center_id = int(np.argmin(np.hypot(*self._lattice_xy.T))) + 1
        self.all_ids = np.arange(1, self._n + 1)

        # Diffusion tracer: a bead ~30% in from the front-left corner of the
        # rendered sheet (small x = left, small y = toward the camera) plus its
        # six nearest neighbours, drawn brighter so the cluster can be followed as
        # it diffuses through the membrane. Chosen once from the initial lattice;
        # the ids are stable, so the highlight tracks the same beads as they wander.
        self._tracer_ids, self._tracer_brightness = self._pick_tracer_cluster()

        self._build()

    # Tracer cluster placement, as a fraction of the sheet's span measured from
    # the front-left corner (0 = corner, 1 = far/right edge). Kept off the corner
    # so the highlighted beads sit inside the membrane, not crammed into the edge.
    _TRACER_FRAC = 0.3

    def _pick_tracer_cluster(self):
        """(ids, per-bead brightness) for the diffusion tracer: the lattice bead
        nearest a point _TRACER_FRAC of the way in from the front-left corner and
        its six nearest neighbours, brightened 1.5x (the centre bead 2.1x).
        Returns the brightness aligned to all_ids (1.0 everywhere else)."""
        xy = self._lattice_xy
        f = self._TRACER_FRAC
        target = np.array([
            xy[:, 0].min() + f * (xy[:, 0].max() - xy[:, 0].min()),
            xy[:, 1].min() + f * (xy[:, 1].max() - xy[:, 1].min()),
        ])                                      # 30% in from the front-left corner
        center = int(np.argmin(np.hypot(*(xy - target).T)))
        d = np.hypot(*(xy - xy[center]).T)
        neighbours = np.argsort(d)[1:7]         # six nearest (exclude self)
        bright = np.ones(self._n)
        bright[neighbours] = 1.5
        bright[center] = 2.1
        ids = np.array([center + 1] + [int(k) + 1 for k in neighbours])
        return ids, bright

    # ---- construction -------------------------------------------------------

    def _hex_lattice(self):
        """(N,2) hexagonal lattice sites centered on the origin. Rows run along x
        at spacing a; successive rows step by a*sqrt(3)/2 in y and offset by a/2
        in x, which is the arrangement that tiles a periodic rectangular cell of
        size (N_COLS*a) x (N_ROWS*a*sqrt(3)/2)."""
        dy = A_LATTICE * math.sqrt(3.0) / 2.0
        pts = []
        for j in range(N_ROWS):
            xoff = (A_LATTICE / 2.0) if (j % 2) else 0.0
            y = j * dy
            for i in range(N_COLS):
                pts.append((i * A_LATTICE + xoff, y))
        pts = np.array(pts, dtype=float)
        pts -= pts.mean(axis=0)   # center on origin
        return pts

    def _build(self):
        lmp = self.lmp
        c = lmp.command
        ensure_plugin_loaded(lmp)

        c("units lj")
        c("dimension 3")
        c("atom_style hybrid sphere dipole")
        c("boundary p p f")           # periodic in-plane, fixed out-of-plane
        c("atom_modify map array")

        lx = N_COLS * A_LATTICE
        ly = N_ROWS * A_LATTICE * math.sqrt(3.0) / 2.0
        hx, hy = lx / 2.0, ly / 2.0
        c(f"region box block {-hx} {hx} {-hy} {hy} {-Z_HALF} {Z_HALF} units box")
        c("create_box 1 box")
        for x, y in self._lattice_xy:
            c(f"create_atoms 1 single {x:.6f} {y:.6f} 0.0 units box")

        c("mass 1 1.0")
        c(f"set group all diameter {BEAD_DIAMETER}")
        c("set group all dipole 0.0 0.0 1.0")

        c(f"pair_style mesomem {self._rc}")
        self._apply_pair_coeff()

        c("neighbor 1.0 bin")
        c("neigh_modify every 1 delay 0 check yes")

        c(f"group center id {self.center_id}")
        c(f"group sheet subtract all center")

        c("fix integrate all nve/sphere update dipole")
        # Reflecting z walls: a bead that thermally pops out of the monolayer at
        # high T can't leave the (shallow) box and be lost -- it bounces back
        # (elastic, so no energy is injected) and the membrane recaptures it.
        # Per-step and essentially free; stands in for the implicit solvent's
        # hydrophobic confinement, which the model otherwise omits.
        c("fix zwall all wall/reflect zlo EDGE zhi EDGE")
        c(f"timestep {TIMESTEP}")
        c("thermo 100000")

        # --- settle: Langevin + barostat -> tension-free equilibrium ---------
        c(f"fix settle_bath all langevin {self._target_temp} {self._target_temp} "
          f"{LANGEVIN_DAMP} {self._seed} omega yes")
        # Berendsen barostat rescales the box toward zero lateral pressure; it is
        # not an integrator (it rides on top of nve/sphere), so it just relaxes
        # the in-plane spacing during equilibration and is removed afterwards.
        c(f"fix settle_baro all press/berendsen x {BARO_PRESS} {BARO_PRESS} {BARO_DAMP} "
          f"y {BARO_PRESS} {BARO_PRESS} {BARO_DAMP} couple xy")
        c(f"run {SETTLE_STEPS}")
        c("unfix settle_baro")
        c("unfix settle_bath")

        # Freeze the relaxed box and switch to interactive configuration:
        # thermostat the sheet (not the puller, so its haptics stay clean and the
        # membrane reaction force can be recovered), drive/damp/plane the puller.
        c(f"fix bath sheet langevin {self._target_temp} {self._target_temp} "
          f"{LANGEVIN_DAMP} {self._seed} omega yes")
        c("fix drive center addforce 0.0 0.0 0.0")
        c(f"fix damp center viscous {self._puller_damping}")
        c("fix plane center setforce NULL 0.0 NULL")

        c("compute sheet_temp sheet temp")
        c("compute ke_atom all ke/atom")
        c("compute pe_atom all pe/atom")
        c("run 0")

        # Record the frozen box for rendering / camera framing.
        self.box_lx = self.lmp.extract_global("boxxhi") - self.lmp.extract_global("boxxlo")
        self.box_ly = self.lmp.extract_global("boxyhi") - self.lmp.extract_global("boxylo")
        self._puller_y = self._puller_pos3()[1]
        # In-plane g(r) of the whole sheet, minimum-image in the periodic box.
        # r_max spans a handful of lattice shells (but never more than half the
        # box, the minimum-image limit) so the hexagonal peaks -- and their
        # collapse on melting -- are all visible.
        r_max = min(0.5 * min(self.box_lx, self.box_ly), 6.0 * A_LATTICE)
        self._rdf = InPlaneRDF(r_max, box=(self.box_lx, self.box_ly))
        self._constrain_center()
        self._interactive_t = 0.0

    # ---- id <-> local-index helpers ----------------------------------------

    def _id_index(self):
        n = self.lmp.get_natoms()
        ids = self.lmp.numpy.extract_atom("id")[:n]
        return {int(i): k for k, i in enumerate(ids)}, n

    def _center_local(self):
        idx, n = self._id_index()
        return idx.get(self.center_id), n

    def _puller_pos3(self):
        ic, n = self._center_local()
        x = self.lmp.numpy.extract_atom("x")[:n]
        return np.array([x[ic][0], x[ic][1], x[ic][2]])

    # ---- controls (identical interface to the 7-bead patch) -----------------

    def set_input_force(self, fx, fy):
        self._input_fx = fx
        self._input_fz = fy
        self.lmp.command(f"fix drive center addforce {fx} 0.0 {fy}")

    def set_target_temp(self, T):
        T = max(T_MIN, min(T_MAX, T))
        if T == self._target_temp:
            return
        self._target_temp = T
        self.lmp.command(
            f"fix bath sheet langevin {T} {T} {LANGEVIN_DAMP} {self._seed} omega yes"
        )

    def steer_orientation(self, rate, dt):
        self._yaw = -rate

    def set_puller_damping(self, gamma):
        gamma = max(PULLER_DAMPING_MIN, min(PULLER_DAMPING_MAX, gamma))
        if gamma == self._puller_damping:
            return
        self._puller_damping = gamma
        self.lmp.command(f"fix damp center viscous {gamma}")

    def _effective_wc(self):
        """Orientational cutoff actually used: capped at rc (the paper's upper
        bound), so a wc slider dragged past rc is clamped rather than fed an
        ill-posed wc > rc."""
        return min(self._wc, self._rc)

    def _apply_pair_coeff(self):
        """(Re)issue the mesomem pair_coeff from the current live coefficients.

        The trailing 11th argument is `splay_symmetry` (0..1): 0 gives the paper's
        original splay term built on the signed director dot product ni.nj; 1
        makes it use |ni.nj|, so parallel and antiparallel neighbour directors are
        penalised identically (the term then cares only about the shared axis, not
        the sense). Intermediate values blend the two continuously -- see the pair
        style's compute()."""
        self.lmp.command(
            f"pair_coeff 1 1 {SIGMA} {EPS} {self._ktilt} {self._ksplay} "
            f"{self._rc} {self._effective_wc()} {self._zeta} {C0} {self._splay_sym}"
        )

    def set_extra_param(self, key, value):
        """Live k_tilt / k_splay / zeta / rc / wc / splay_symmetry dials; re-issue
        pair_coeff on change (and the pair_style global cutoff too when rc moves)."""
        attr = {"k_tilt": "_ktilt", "k_splay": "_ksplay", "eta": "_zeta",
                "rc": "_rc", "wc": "_wc", "splay_symmetry": "_splay_sym"}.get(key)
        if attr is None or getattr(self, attr) == value:
            return
        setattr(self, attr, value)
        if key == "rc":
            self.lmp.command(f"pair_style mesomem {self._rc}")
        self._apply_pair_coeff()

    def get_bead_brightness(self):
        """Per-bead albedo brightness (aligned to all_ids / get_positions_3d):
        the front-left diffusion tracer cluster is brightened, everything else
        drawn normally."""
        return self._tracer_brightness

    # Leash: keep the puller near its home column and below a runaway speed. The
    # in-plane (x) range is a few lattice spacings; z is the out-of-plane pull.
    _CTRL_X = (-5.0, 5.0)
    _CTRL_Z = (-3.5, 3.5)
    _SPEED_CAP = 6.0

    def _constrain_center(self):
        ic, n = self._center_local()
        if ic is None:
            return
        x = self.lmp.numpy.extract_atom("x")
        v = self.lmp.numpy.extract_atom("v")
        # Pin the puller to the control plane through its home row (y fixed).
        x[ic][1] = self._puller_y
        v[ic][1] = 0.0
        cx0 = 0.0
        for axis, (lo, hi), home in ((0, self._CTRL_X, cx0), (2, self._CTRL_Z, 0.0)):
            if x[ic][axis] < home + lo:
                x[ic][axis] = home + lo
                if v[ic][axis] < 0.0:
                    v[ic][axis] = 0.0
            elif x[ic][axis] > home + hi:
                x[ic][axis] = home + hi
                if v[ic][axis] > 0.0:
                    v[ic][axis] = 0.0
        speed = math.sqrt(v[ic][0] ** 2 + v[ic][1] ** 2 + v[ic][2] ** 2)
        if speed > self._SPEED_CAP:
            s = self._SPEED_CAP / speed
            v[ic][0] *= s
            v[ic][1] *= s
            v[ic][2] *= s

        # Director dynamics: constrain to the xz plane, add the yaw kick, damp.
        mu = self.lmp.numpy.extract_atom("mu")
        omega = self.lmp.numpy.extract_atom("omega")
        omega[ic][0] = 0.0
        omega[ic][2] = 0.0
        omega[ic][1] = omega[ic][1] * ROT_DAMP + self._yaw * YAW_TORQUE
        nx, nz = mu[ic][0], mu[ic][2]
        m = math.hypot(nx, nz)
        if m > 1e-9:
            mu[ic][0] = nx / m
            mu[ic][1] = 0.0
            mu[ic][2] = nz / m

    def _apply_cleanup_forces(self, dt):
        """Two soft per-frame housekeeping corrections on the non-puller beads
        (applied as momentum kicks, delta v; the Langevin bath damps them so they
        settle rather than ring):

          - Plane centering: pull every bead toward the box's central z-plane
            (z = 0) with stiffness K_PLANE, so the free monolayer can't drift or
            buckle out of the frame. The puller is excluded -- its z IS the
            user's out-of-plane pull, which this would otherwise fight.
          - Normal-up alignment: rotate the whole sheet, as a rigid body, so its
            smallest principal axis (the membrane normal) points toward +z. The
            correction rate is proportional to the current tilt (K_ALIGN), applied
            as the rigid-rotation velocity field omega x r about the centre of
            mass -- the same "smallest principal component upward" idea as the
            7-bead patch's ring torque, made size-independent for the big sheet."""
        idx, n = self._id_index()
        ic = idx.get(self.center_id)
        x = self.lmp.numpy.extract_atom("x")
        v = self.lmp.numpy.extract_atom("v")
        P = np.array(x[:n], dtype=float)
        mask = np.ones(n, dtype=bool)
        if ic is not None:
            mask[ic] = False
        sel = np.nonzero(mask)[0]
        if len(sel) < 3:
            return
        Ps = P[sel]

        kick = np.zeros((len(sel), 3))
        # Plane centering toward z = 0.
        kick[:, 2] += -K_PLANE * Ps[:, 2] * dt

        # Alignment: smallest principal axis (membrane normal) -> +z.
        com = Ps.mean(axis=0)
        Q = Ps - com
        _evals, evecs = np.linalg.eigh(Q.T @ Q)   # ascending eigenvalues
        e_min = evecs[:, 0]
        if e_min[2] < 0.0:                        # pick the upward-facing normal
            e_min = -e_min
        # omega ~ K_ALIGN * (e_min x up): axis of the tilt, magnitude ~ sin(tilt),
        # so a flat sheet gets no rotation and a tilted one is turned back flat.
        omega = K_ALIGN * np.cross(e_min, np.array([0.0, 0.0, 1.0]))
        kick += np.cross(omega[None, :], Q) * dt

        v[sel] += kick

    def step(self, n=10):
        self.lmp.command(f"run {n}")
        self._apply_cleanup_forces(n * TIMESTEP)
        self._constrain_center()
        self._interactive_t += n * TIMESTEP

    # ---- readouts (2D control-plane projection) -----------------------------

    def get_puller_state(self):
        ic, n = self._center_local()
        if ic is None:
            return None, None
        x = self.lmp.numpy.extract_atom("x")[:n]
        v = self.lmp.numpy.extract_atom("v")[:n]
        return (np.array([x[ic][0], x[ic][2]]),
                np.array([v[ic][0], v[ic][2]]))

    def get_puller_energy(self):
        ic, n = self._center_local()
        if ic is None:
            return None, None
        ke = self.lmp.numpy.extract_compute("ke_atom", 1, 1)[:n]
        pe = self.lmp.numpy.extract_compute("pe_atom", 1, 1)[:n]
        return float(ke[ic]), float(pe[ic])

    def get_potential_terms(self):
        """Live additive decomposition of the puller's MesoMem energy over its
        neighbours within RC (vectorized over the whole sheet, minimum-image in
        the periodic x,y). Same three terms as the 7-bead patch. The splay term
        mirrors the pair style's splay_symmetry blend (signed ni.nj -> |ni.nj|)
        so the panel matches the force actually being applied."""
        idx, n = self._id_index()
        ic = idx.get(self.center_id)
        if ic is None:
            return None
        x = np.array(self.lmp.numpy.extract_atom("x")[:n], dtype=float)
        mu = np.array(self.lmp.numpy.extract_atom("mu")[:n], dtype=float)[:, :3]
        ni = mu[ic] / (np.linalg.norm(mu[ic]) or 1.0)

        rc, wc = self._rc, self._effective_wc()
        d = x - x[ic]
        d[:, 0] -= self.box_lx * np.round(d[:, 0] / self.box_lx)
        d[:, 1] -= self.box_ly * np.round(d[:, 1] / self.box_ly)
        r = np.linalg.norm(d, axis=1)
        sel = (r < rc) & (r > 1e-9)
        sel[ic] = False

        u_iso = u_tilt = u_splay = 0.0
        for k in np.nonzero(sel)[0]:
            rk = float(r[k])
            rhat = d[k] / rk
            if rk < SIGMA:
                t2 = (SIGMA / rk) ** 2
                u_iso += EPS * (t2 * t2 - 2.0 * t2)
            else:
                g = math.pi * 0.5 * (rk - SIGMA) / (rc - SIGMA)
                u_iso += -EPS * math.cos(g) ** (2.0 * self._zeta)
            w = 0.0
            if rk < wc:
                rga = 0.5 * wc
                denom = (rk / wc) ** 4 - 1.0
                if denom < -1e-14:
                    w = math.exp((rk * rk) / (rga * rga * denom))
            if w > 0.0:
                nj = mu[k] / (np.linalg.norm(mu[k]) or 1.0)
                nir = float(ni @ rhat)
                njr = float(nj @ rhat)
                ninj = float(ni @ nj)
                ninj_eff = (1.0 - self._splay_sym) * ninj + self._splay_sym * abs(ninj)
                u_tilt += 0.5 * self._ktilt * (nir * nir + njr * njr) * w
                u_splay += 0.5 * self._ksplay * (ninj_eff - 1.0) ** 2 * w
        terms = [
            ("isotropic  (repel + attract)", u_iso),
            ("tilt  (directors normal to bonds)", u_tilt),
            ("splay  (neighbour directors align)", u_splay),
        ]
        return ("Pulled bead energy -- additive (reduced units)", terms, 6.0)

    # Recompute the whole-sheet total only every few frames -- it is an O(pairs)
    # pass over the entire membrane, far heavier than the single puller bead, and
    # the aggregate barely changes frame to frame.
    _TOTAL_EVERY = 4

    def get_total_potential_terms(self):
        """Whole-sheet additive energy: the three MesoMem terms summed over every
        unique bead pair within rc across the entire membrane (minimum-image in
        the periodic x,y), so the panel shows the total attraction / tilt / splay
        the sheet is storing -- not just the puller's local share.

        Candidate pairs come from a periodic 2D KD-tree on the in-plane (x,y):
        the in-plane separation never exceeds the 3D separation, so query_pairs(rc)
        is a superset of the true within-rc pairs, which are then filtered on the
        exact minimum-image 3D distance. Vectorized over the ~1e4 candidate pairs
        and throttled to every _TOTAL_EVERY frames."""
        self._total_terms_ctr += 1
        if self._total_terms_cache is not None and self._total_terms_ctr % self._TOTAL_EVERY:
            return self._total_terms_cache

        idx, n = self._id_index()
        x = np.array(self.lmp.numpy.extract_atom("x")[:n], dtype=float)
        mu = np.array(self.lmp.numpy.extract_atom("mu")[:n], dtype=float)[:, :3]
        order = np.array([idx[int(i)] for i in self.all_ids])
        P = x[order]
        D = mu[order]
        D /= np.clip(np.linalg.norm(D, axis=1, keepdims=True), 1e-9, None)

        rc, wc = self._rc, self._effective_wc()
        Lx, Ly = self.box_lx, self.box_ly
        xy = np.column_stack([(P[:, 0] + Lx / 2.0) % Lx, (P[:, 1] + Ly / 2.0) % Ly])
        tree = cKDTree(xy, boxsize=[Lx, Ly])
        pairs = tree.query_pairs(rc, output_type="ndarray")

        u_iso = u_tilt = u_splay = 0.0
        if len(pairs):
            a, b = pairs[:, 0], pairs[:, 1]
            d = P[a] - P[b]
            d[:, 0] -= Lx * np.round(d[:, 0] / Lx)
            d[:, 1] -= Ly * np.round(d[:, 1] / Ly)
            r = np.linalg.norm(d, axis=1)
            m = (r < rc) & (r > 1e-9)
            a, b, d, r = a[m], b[m], d[m], r[m]
            rhat = d / r[:, None]
            core = r < SIGMA
            iso = np.empty_like(r)
            t2 = (SIGMA / r[core]) ** 2
            iso[core] = EPS * (t2 * t2 - 2.0 * t2)
            att = ~core
            g = math.pi * 0.5 * (r[att] - SIGMA) / (rc - SIGMA)
            iso[att] = -EPS * np.cos(g) ** (2.0 * self._zeta)
            u_iso = float(iso.sum())

            w = np.zeros_like(r)
            inw = r < wc
            rga = 0.5 * wc
            denom = (r / wc) ** 4 - 1.0
            valid = inw & (denom < -1e-14)
            w[valid] = np.exp((r[valid] * r[valid]) / (rga * rga * denom[valid]))
            ni, nj = D[a], D[b]
            nir = np.einsum("ij,ij->i", ni, rhat)
            njr = np.einsum("ij,ij->i", nj, rhat)
            ninj = np.einsum("ij,ij->i", ni, nj)
            ninj_eff = (1.0 - self._splay_sym) * ninj + self._splay_sym * np.abs(ninj)
            u_tilt = float((0.5 * self._ktilt * (nir * nir + njr * njr) * w).sum())
            u_splay = float((0.5 * self._ksplay * (ninj_eff - 1.0) ** 2 * w).sum())

        terms = [
            ("isotropic  (repel + attract)", u_iso),
            ("tilt  (directors normal to bonds)", u_tilt),
            ("splay  (neighbour directors align)", u_splay),
        ]
        # Bar half-range scaled to the sheet's dominant (attraction) term, which
        # runs to a few thousand across the ~900 beads -- so its bar reads without
        # pegging. As in the single-bead panel the far smaller tilt/splay totals
        # then show mainly through their printed values, not bar length.
        self._total_terms_cache = (
            "Whole-sheet energy -- additive (reduced units)", terms, 4000.0)
        return self._total_terms_cache

    def get_torque_signals(self):
        ic, n = self._center_local()
        if ic is None:
            return None
        applied = max(-1.0, min(1.0, self._yaw))
        tau = self.lmp.numpy.extract_atom("torque")
        reaction = 0.0
        if tau is not None:
            reaction = max(-1.0, min(1.0, float(tau[:n][ic][1]) / REACTION_TORQUE_DISPLAY_MAX))
        return applied, reaction

    def get_interaction_force(self):
        ic, n = self._center_local()
        if ic is None:
            return np.array([0.0, 0.0])
        f = self.lmp.numpy.extract_atom("f")[:n]
        v = self.lmp.numpy.extract_atom("v")[:n]
        g = self._puller_damping
        pair_fx = f[ic][0] - self._input_fx + g * v[ic][0]
        pair_fz = f[ic][2] - self._input_fz + g * v[ic][2]
        return np.array([pair_fx, pair_fz])

    def get_thermo_state(self):
        temp = self.lmp.extract_compute("sheet_temp", 0, 0)
        press = self.lmp.get_thermo("press")
        ke = self.lmp.get_thermo("ke")
        pe = self.lmp.get_thermo("pe")
        etotal = self.lmp.get_thermo("etotal")
        return temp, press, ke, pe, etotal

    def get_sim_time(self):
        return self._interactive_t

    def get_rdf(self):
        idx, n = self._id_index()
        x = self.lmp.numpy.extract_atom("x")[:n]
        order = [idx[int(i)] for i in self.all_ids]
        self._rdf.add(np.array([[x[k][0], x[k][1]] for k in order]))
        return self._rdf.get()

    def get_all_positions(self):
        idx, n = self._id_index()
        x = self.lmp.numpy.extract_atom("x")[:n]
        order = [idx[int(i)] for i in self.all_ids]
        pos2d = np.array([[x[k][0], x[k][1]] for k in order])
        is_puller = self.all_ids == self.center_id
        return np.array(self.all_ids), pos2d, is_puller, None

    # ---- 3D rendering data --------------------------------------------------

    def get_positions_3d(self):
        idx, n = self._id_index()
        x = self.lmp.numpy.extract_atom("x")[:n]
        order = [idx[int(i)] for i in self.all_ids]
        pos = np.array([[x[k][0], x[k][1], x[k][2]] for k in order])
        is_puller = self.all_ids == self.center_id
        return np.array(self.all_ids), pos, is_puller

    def get_dipoles_3d(self):
        idx, n = self._id_index()
        mu = self.lmp.numpy.extract_atom("mu")[:n]
        order = [idx[int(i)] for i in self.all_ids]
        dirs = np.array([mu[k][:3] for k in order], dtype=float)
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        return dirs / norms

    def get_bonds_3d(self):
        # Beads at a=0.8 with diameter 1.0 overlap into a continuous sheet, so no
        # bond sticks are drawn (and none would tile cleanly across the periodic
        # seam anyway).
        return []

    def get_camera_params(self):
        # Three-quarter view sized to the sheet: from below in y and above in z,
        # far enough back to see the whole membrane. fit_to_points then zooms to
        # fill the viewport, so this only needs to set a good angle and distance.
        span = max(self.box_lx, self.box_ly)
        return dict(
            eye=(0.0, -0.85 * span, 0.6 * span),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
            fov_deg=34.0,
        )

    def get_control_grid(self):
        # Local control plane (world xz) around the puller: a net marking exactly
        # where the puller can be dragged -- its extents are the movement limits
        # (_CTRL_X / _CTRL_Z), not an arbitrary smaller patch.
        return dict(
            origin=(0.0, float(self._puller_y), 0.0),
            u_axis=(1.0, 0.0, 0.0),
            v_axis=(0.0, 0.0, 1.0),
            u_range=self._CTRL_X,
            v_range=self._CTRL_Z,
            step=0.8,
        )

    def get_box_bounds_3d(self):
        """The periodic simulation cell (frozen after settle), for the renderer to
        outline in white: full x,y extent and the shallow +/-Z_HALF z container."""
        return (-self.box_lx / 2.0, self.box_lx / 2.0,
                -self.box_ly / 2.0, self.box_ly / 2.0,
                -Z_HALF, Z_HALF)

    def get_scene_fit_points(self):
        """Frame the whole sheet: its in-plane bounding box (at z=0) plus a bit
        of out-of-plane headroom for a pulled bead / directors."""
        hx, hy = self.box_lx / 2.0, self.box_ly / 2.0
        # Full box corners (including the +/-Z_HALF container height) so the white
        # box outline stays framed, plus a bit of headroom for a pulled bead.
        pts = [(sx * hx, sy * hy, sz * Z_HALF)
               for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        pts.append((0.0, 0.0, 2.6))
        pts.append((0.0, 0.0, -2.2))
        return np.array(pts)

    def get_box_size(self):
        return self.box_lx, self.box_ly

    def close(self):
        self.lmp.close()

"""MesoMem membrane patch -- a 3D reference demo built on the *real* MesoMem
force field (Sillano, Marrink & Idema 2026), not a stand-in.

Unlike the `lipid` system (which reproduces MesoMem's physics "in spirit" with
stock LAMMPS styles because the authors' custom pair-style isn't in a normal
build), this system loads the authors' actual `mesomem` pair-style -- compiled
straight from their `cpp_files/` as a runtime LAMMPS plugin (see
systems/mesomem_ff/). Each bead is a one-particle-thick patch of bilayer
carrying an orientation vector (director) n_i = the local membrane normal, and
the pair-style supplies the paper's additive potential:

    U = U_rep(r) + U_attr(r) + [ U_tilt(n_i,n_j,r_hat) + U_splay(n_i,n_j) ] w(r)

with a soft 4-2 repulsive core, a cosine-squared attraction, a tilt term that
penalizes directors tipping away from the local surface normal, and a splay
term that penalizes neighboring directors misaligning. Forces AND torques are
evaluated pairwise; beads translate and rotate under Langevin (implicit-solvent)
dynamics.

Geometry. Seven beads: one central bead ringed by six at hexagonal spacing, all
in the world xy-plane with directors initially along +z -- the smallest patch
that already shows the tilt/splay physics (pull the middle bead out of plane and
its neighbors' directors splay to follow, resisting through force feedback). The
six-bead ring is tethered to its lattice sites (fix spring/self) to stand in for
continuation into a larger membrane, so the patch stays put while you probe it.

Control. The central bead is the puller. The joystick's two axes slide it in the
world xz-plane -- a plane perpendicular to the membrane and facing the camera,
drawn in the scene as a faint "net". Axis x -> world x (screen-horizontal),
axis y -> world z (screen-vertical, i.e. out of the membrane). The bead is held
in that plane (fix setforce zeroes its y-force), so the 2-axis stick fully
controls its 3D position. Everything the app's force-feedback loop needs
(puller position/velocity, membrane reaction force) is reported as the 2D
projection onto this control plane, so haptics work unchanged.

Units are the paper's LJ-reduced units (sigma = eps = m = 1); the temperature
dial and readouts are therefore in reduced units, not Kelvin/eV.
"""
import math
import random

import numpy as np
from lammps import lammps

from .base import ForceFeedbackProfile, MDSystem, SliderSpec, SystemSpec
from .mesomem_ff import ensure_plugin_loaded
from .rdf2d import InPlaneRDF

# --- MesoMem potential parameters (paper "standard conditions", Sec. II-III) --
SIGMA = 1.0        # bead diameter / length unit
EPS = 1.0          # energy unit (LJ well depth)
K_TILT = 12.0      # tilt modulus -- above ~10 the patch stays planar (paper)
K_SPLAY = 1.0      # splay modulus
RC = 2.5           # isotropic interaction cutoff (>= 2.5 sigma needed for cohesion)
WC = 2.0           # orientational (tilt/splay) interaction cutoff, must be <= RC
ZETA = 5.0         # steepness/width of the cosine-squared attractive branch
C0 = 0.0           # spontaneous curvature (0 -> flat preferred)

# Live-tunable ranges for the MesoMem coefficient sliders, each with the paper's
# recommended value marked as the slider "optimum". The spans bracket the regimes
# the MesoMem preprint explores:
#   - k_tilt: floppy membrane up through the stiff-planar regime (planar above
#     ~10); optimum 12.
#   - k_splay: around its soft default; optimum 1.0.
#   - zeta: steepness of the cosine-squared attractive branch; optimum 5.
#   - rc: isotropic interaction cutoff. The paper needs rc >= 2.5 sigma to sustain
#     aggregation; beyond that, stability is largely insensitive -> optimum 2.5.
#   - wc: orientational (tilt/splay) cutoff, effectively upper-bounded by rc.
#     Below that bound it barely affects structure but strongly tunes stiffness
#     (paper Sec. III D 4) -> optimum 2.0.
K_TILT_MIN, K_TILT_MAX, K_TILT_OPT = 0.0, 50.0, 12.0
K_SPLAY_MIN, K_SPLAY_MAX, K_SPLAY_OPT = 0.0, 3.0, 1.0
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

A_LATTICE = 1.0    # in-plane nearest-neighbor spacing (near the isotropic min at r=sigma)
BEAD_DIAMETER = 2.0  # sphere radius = sigma -> moment of inertia I = (2/5) m sigma^2 (paper)

# Cubic box half-extent is BOX/2; non-periodic, just a container with reflecting
# walls on all six faces (see _build). Sized snugly around the patch and its pull
# reach (leash +/-1.4 in x and z, bead radius 0.5 -> a bead edge reaches 1.9 <
# BOX/2 = 2.0) rather than large and arbitrary: because the camera frames the box
# outline (get_scene_fit_points), a tight box makes the beads fill the view
# instead of floating small inside a cavernous cell.
BOX = 6.0

TIMESTEP = 0.005   # tau_LJ (paper uses 0.01; halved for stability while pulling)
SETTLE_STEPS = 300

# Langevin (implicit solvent). The ring tether + this drag keep the patch quasi-
# static so you probe it rather than knock it across the box.
LANGEVIN_DAMP = 0.5     # tau_LJ, translational relaxation time (stronger friction -> calmer)
# The ring is the demo's orientation reference, but it's held there by real
# (soft) forces rather than any geometric imposition, so the beads still move,
# jitter and respond dynamically (see _apply_ring_forces). Two forces act on the
# outer beads each frame:
#  - a centering force driving their centre of mass to the box centre (origin),
#  - an alignment torque (applied as per-bead forces via the cluster's inertia
#    tensor) rotating the patch's smallest principal axis -- its normal -- up.
# Both are applied as per-frame momentum kicks (F*dt); the Langevin bath damps
# them so the patch settles flat and centred instead of oscillating. Strong, but
# still a force -- the ring can dome, tilt and recover realistically.
K_CENTER = 7.0    # centre-of-mass centering stiffness
K_ALIGN = 7.0     # normal-up alignment torque strength
# A tiny per-bead spring toward the origin, ON TOP of the COM centering above
# (which only moves the centre of mass, not individual beads). This is a safety
# net so no single outer bead can be flung out of the box during violent play;
# kept small so it barely perturbs the resting patch.
K_HOME = 0.5       # per-bead homing stiffness toward (0,0,0)

# Puller's own extra drag (the damping dial), on top of the Langevin bath.
PULLER_DAMPING_DEFAULT = 4.0
PULLER_DAMPING_MIN = 0.0
PULLER_DAMPING_MAX = 8.0

# Temperature dial, in reduced units. Cold -> rigid flat patch; hot -> directors
# disorder and the patch frays. The paper's fluid membrane sits at intermediate
# reduced T; a small patch stays recognizable up to ~0.6 before fraying.
# Reduced-temperature dial, rescaled low: even a small reduced T looked too
# active before, so the default starts near-frozen and the whole range is
# gentler (paired with the stiff ring tether and stronger friction above).
T_MIN = 0.0
T_MAX = 0.5
T_MELT = 0.3

# Yaw steering applies a TORQUE to the central bead's director about the control
# plane's normal (world y), then the membrane's own tilt stiffness swings it
# back: you push the spike over and watch k_tilt spring it toward the local
# normal. The restoring torque is genuine physics -- the mesomem pair style's
# tilt term, integrated by LAMMPS. We only add two things each frame: the yaw
# torque itself (as an angular-momentum kick about y -- a torque is dL/dt) and a
# rotational drag standing in for the puller's share of the implicit-solvent
# rotational friction (the paper's gamma_r), since the central bead sits outside
# the Langevin group; without it an undamped director would oscillate forever.
# Strong enough that a firm twist drives the director over the tilt barrier at
# 45 deg and flips it to the OPPOSITE normal -- the tilt term is bistable, both
# +n and -n are energy minima, so the bead is equally happy pointing "up" or
# "down". That flip (visible via the in-bead arrow) is the pedagogical point.
# A gentle twist still just deflects and springs back. Re-tune for stick feel.
YAW_TORQUE = 1.0   # angular-momentum kick about y, per unit yaw, per frame
ROT_DAMP = 0.88    # per-frame rotational-velocity retention (how fast swings settle)

# Reaction-torque magnitude (about the control-plane normal) that maps to a full
# semicircle of the red torque arc. The membrane's tilt term reaches O(k_tilt/2)
# at large deflection; this is picked so a firm twist against it fills the arc.
REACTION_TORQUE_DISPLAY_MAX = 6.0

# MesoMem contact/restoring forces on the pulled bead run O(1-10) in reduced
# units; profile is scaled to that, with enough stick authority to tent the
# membrane and pop the bead out against tilt/splay resistance.
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
    key="mesomem",
    name="MesoMem membrane patch (3D)",
    description="Real MesoMem force field: pull the center bead out of a 7-bead patch, feel tilt/splay resist.",
    element_label="membrane bead (director)",
    lattice_spacing=A_LATTICE,
    timestep=TIMESTEP,
    temperature=SliderSpec("Temperature", T_MIN, T_MAX, 0.001, fmt="{:.3f}", unit=" T*"),
    damping=SliderSpec("Puller damping", PULLER_DAMPING_MIN, PULLER_DAMPING_MAX,
                       PULLER_DAMPING_DEFAULT, fmt="{:.2f}", advanced=True),
    melt_temp=T_MELT,
    force_feedback=FORCE_FEEDBACK,
    max_input_force=9.0,   # reduced units at full deflection, shared by joystick/WASD/mouse
    puller_speed_cap=0.06 * SIGMA / TIMESTEP,
    crystal_color=None,   # 3D path uses theme.MEMBRANE_BEAD_COLOR
    atom_radius_A=0.5 * SIGMA,   # physical bead radius used for perspective sphere sizing
    sim_time_per_frame=0.05,     # tau_LJ per frame (10 steps at 0.005)
    bond_overlay=False,
    render_3d=True,
    reduced_units=True,
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

# Camera for the three-quarter view: looking at the patch from below in y and
# above in z, so the membrane (xy-plane) tilts and the directors (+z) stand up.
# Pulled further back than the scene is wide (eye ~12.6 from the patch) so the
# perspective foreshortening stays gentle; fit_to_points then zooms in to fill
# the viewport, so the extra distance costs no apparent size, only a flatter,
# more telephoto look.
CAMERA_PARAMS = dict(
    eye=(0.0, -10.5, 7.0),
    target=(0.0, 0.0, 0.1),
    up=(0.0, 0.0, 1.0),
    fov_deg=30.0,
)


class MesoMemHexSystem(MDSystem):
    spec = SPEC

    def __init__(self):
        self.lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
        self._input_fx = 0.0   # world-x drive (joystick axis x)
        self._input_fz = 0.0   # world-z drive (joystick axis y)
        self._yaw = 0.0   # current yaw command in [-1, 1] (steering torque on the director)
        self._target_temp = SPEC.temperature.default
        self._puller_damping = PULLER_DAMPING_DEFAULT
        self._interactive_t = 0.0
        self._seed = random.randint(1, 900_000_000)
        # Live-tunable MesoMem coefficients (start at the paper's standard values;
        # the k_tilt / k_splay / zeta / rc / wc sliders drive these via
        # set_extra_param).
        self._ktilt = K_TILT
        self._ksplay = K_SPLAY
        self._zeta = ZETA
        self._rc = RC
        self._wc = WC
        self._splay_sym = SPLAY_SYM

        # id 1 = central puller; ids 2..7 = hexagonal ring.
        self.center_id = 1
        self.ring_ids = tuple(range(2, 8))
        self.all_ids = (self.center_id,) + self.ring_ids
        # Bonds to draw (index pairs into the id-ordered array, center first):
        # 6 spokes + the 6-segment ring.
        self._bond_index_pairs = [(0, k) for k in range(1, 7)]
        self._bond_index_pairs += [(k, k % 6 + 1) for k in range(1, 7)]

        # In-plane g(r) of the patch (non-periodic, only 7 beads, so it resolves
        # just the first couple of neighbour shells -- the spoke/ring spacing at
        # ~a and the second-shell distances) rather than a bulk phase, but it
        # populates the RDF panel instead of leaving it stuck on "warming up".
        self._rdf = InPlaneRDF(3.0 * A_LATTICE, nbins=48, box=None, sample_every=1)

        self._build()

    # ---- construction -------------------------------------------------------

    def _build(self):
        lmp = self.lmp
        c = lmp.command
        ensure_plugin_loaded(lmp)

        c("units lj")
        c("dimension 3")
        c("atom_style hybrid sphere dipole")
        c("boundary f f f")
        c("atom_modify map array")

        h = BOX / 2.0
        c(f"region box block {-h} {h} {-h} {h} {-h} {h}")
        c("create_box 1 box")

        # Central bead at the origin, six around it at hexagonal spacing.
        c(f"create_atoms 1 single 0.0 0.0 0.0 units box")
        for k in range(6):
            ang = math.radians(60 * k)
            x = A_LATTICE * math.cos(ang)
            y = A_LATTICE * math.sin(ang)
            c(f"create_atoms 1 single {x:.6f} {y:.6f} 0.0 units box")

        c(f"mass 1 1.0")
        c(f"set group all diameter {BEAD_DIAMETER}")
        c("set group all dipole 0.0 0.0 1.0")   # directors along +z (membrane normal)

        c(f"pair_style mesomem {self._rc}")
        # sigma eps ktilt ksplay cut weight_rcut zeta c0
        self._apply_pair_coeff()

        c("neighbor 1.0 bin")
        c("neigh_modify every 1 delay 0 check yes")

        c(f"group center id {self.center_id}")
        ring = " ".join(str(i) for i in self.ring_ids)
        c(f"group ring id {ring}")

        # Integrate translation + dipole rotation for everything. The Langevin
        # bath (implicit solvent) thermostats only the RING, not the puller:
        # keeping the central bead noise-free makes its stick control clean AND
        # lets us recover the membrane's reaction force on it exactly (its total
        # force minus the two forces we apply -- drive and viscous -- see
        # get_interaction_force), since the mesomem pair style has no single()
        # method and so can't feed compute group/group.
        c("fix integrate all nve/sphere update dipole")
        # Reflecting walls on all six box faces: a ring bead flung out during
        # violent play (past where the soft centering/homing forces can recover it)
        # bounces back elastically instead of escaping the box and being lost. The
        # box is sized so resting beads sit well inside, so the walls only act in
        # those rare runaway spots.
        c("fix wall all wall/reflect xlo EDGE xhi EDGE ylo EDGE yhi EDGE "
          "zlo EDGE zhi EDGE")
        c(f"fix bath ring langevin {self._target_temp} {self._target_temp} "
          f"{LANGEVIN_DAMP} {self._seed} omega yes")
        # The ring's centering + normal-up alignment are applied each frame in
        # Python as soft forces (_apply_ring_forces), not by a LAMMPS fix; its
        # directors are left to the tilt physics, which keeps them up.
        # Puller drive + its extra drag (damping dial).
        c("fix drive center addforce 0.0 0.0 0.0")
        c(f"fix damp center viscous {self._puller_damping}")
        # Constrain the puller to the world xz control plane: zero its y-force
        # every step (defined last so it also cancels any y interaction force).
        c("fix plane center setforce NULL 0.0 NULL")

        c("compute ring_temp ring temp")
        c("compute ke_atom all ke/atom")
        c("compute pe_atom all pe/atom")

        c(f"timestep {TIMESTEP}")
        c("thermo 100000")

        c(f"run {SETTLE_STEPS}")
        self._constrain_center()
        self._interactive_t = 0.0

    # ---- id <-> local-index helpers ----------------------------------------

    def _id_index(self):
        """Map atom id -> local array index (LAMMPS may reorder the 7 atoms)."""
        n = self.lmp.get_natoms()
        ids = self.lmp.numpy.extract_atom("id")[:n]
        return {int(i): k for k, i in enumerate(ids)}, n

    def _center_local(self):
        idx, n = self._id_index()
        return idx.get(self.center_id), n

    # ---- controls -----------------------------------------------------------

    def set_input_force(self, fx, fy):
        # Joystick axis x -> world x; axis y (up) -> world z (out of membrane).
        self._input_fx = fx
        self._input_fz = fy
        self.lmp.command(f"fix drive center addforce {fx} 0.0 {fy}")

    def set_target_temp(self, T):
        T = max(T_MIN, min(T_MAX, T))
        if T == self._target_temp:
            return
        self._target_temp = T
        # Thermostat the RING only, never `all` -- redefining this fix on `all`
        # (an earlier bug) folds the central puller bead into the Langevin bath,
        # so once the temperature slider was touched the bead's director picked up
        # thermal rotational noise that never went away (the torque arrow then
        # jiggled wildly even back at T=0.001, and get_interaction_force's
        # force-recovery silently broke). The bead must stay noise-free.
        self.lmp.command(
            f"fix bath ring langevin {T} {T} {LANGEVIN_DAMP} {self._seed} omega yes"
        )

    def steer_orientation(self, rate, dt):
        # Yaw commands a steering torque on the director (applied in the next
        # _constrain_center). Sign flipped so the twist turns the spike the way
        # the hand expects on screen.
        self._yaw = -rate

    def set_puller_damping(self, gamma):
        gamma = max(PULLER_DAMPING_MIN, min(PULLER_DAMPING_MAX, gamma))
        if gamma == self._puller_damping:
            return
        self._puller_damping = gamma
        self.lmp.command(f"fix damp center viscous {gamma}")

    def _effective_wc(self):
        """Orientational cutoff actually used: the paper caps wc at rc (it is
        'effectively upper-bounded by rc'), so a wc slider dragged past the
        current rc is clamped rather than fed an ill-posed wc > rc to the pair
        style."""
        return min(self._wc, self._rc)

    def _apply_pair_coeff(self):
        """(Re)issue the mesomem pair_coeff from the current live coefficients.
        LAMMPS overwrites the stored per-type coefficients in place and re-inits
        the pair style on the next run, so this is safe to call between steps.

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
        """Live k_tilt / k_splay / zeta / rc / wc / splay_symmetry dials. Re-issues
        pair_coeff only when a value actually changes so it's a cheap no-op most
        frames. Changing rc also resizes the pair_style's global cutoff (and thus
        the neighbour list), so the pair_style is re-declared before the coeffs in
        that case."""
        attr = {"k_tilt": "_ktilt", "k_splay": "_ksplay", "eta": "_zeta",
                "rc": "_rc", "wc": "_wc", "splay_symmetry": "_splay_sym"}.get(key)
        if attr is None or getattr(self, attr) == value:
            return
        setattr(self, attr, value)
        if key == "rc":
            self.lmp.command(f"pair_style mesomem {self._rc}")
        self._apply_pair_coeff()

    # Keep the puller on the control plane, inside the visible net, and below a
    # runaway speed. Without this a sustained max pull would accelerate the
    # (undamped-by-thermostat, un-limited) central bead straight out of the box.
    # Leash kept inside the ring's interaction range (a bead at height z sits
    # sqrt(1 + z^2) from its neighbors; past the rc = 2.5 cutoff it would detach
    # and float free), so the membrane always exerts a restoring pull and the
    # bead snaps back on release instead of sticking at the ceiling.
    _CTRL_X = (-2.8, 2.8)
    _CTRL_Z = (-2.8, 2.8)
    _SPEED_CAP = 6.0

    def _constrain_center(self):
        ic, n = self._center_local()
        if ic is None:
            return
        x = self.lmp.numpy.extract_atom("x")
        v = self.lmp.numpy.extract_atom("v")
        # Exact plane constraint (belt-and-braces with the y setforce).
        x[ic][1] = 0.0
        v[ic][1] = 0.0
        for axis, (lo, hi) in ((0, self._CTRL_X), (2, self._CTRL_Z)):
            if x[ic][axis] < lo:
                x[ic][axis] = lo
                if v[ic][axis] < 0.0:
                    v[ic][axis] = 0.0
            elif x[ic][axis] > hi:
                x[ic][axis] = hi
                if v[ic][axis] > 0.0:
                    v[ic][axis] = 0.0
        speed = math.sqrt(v[ic][0] ** 2 + v[ic][1] ** 2 + v[ic][2] ** 2)
        if speed > self._SPEED_CAP:
            s = self._SPEED_CAP / speed
            v[ic][0] *= s
            v[ic][1] *= s
            v[ic][2] *= s

        # Director dynamics. The membrane's tilt term (integrated by LAMMPS
        # during the step) already applied its restoring torque, swinging the
        # director toward the local normal. Here we (a) constrain the director's
        # rotation to the xz control plane -- spin only about y, so it stays an
        # in-plane swing -- (b) add the yaw steering torque as an angular-
        # momentum kick about y, and (c) apply a rotational drag so the swing
        # settles. The spring-back the user sees is genuine k_tilt physics; only
        # the push and the drag are added here.
        mu = self.lmp.numpy.extract_atom("mu")
        omega = self.lmp.numpy.extract_atom("omega")
        omega[ic][0] = 0.0
        omega[ic][2] = 0.0
        omega[ic][1] = omega[ic][1] * ROT_DAMP + self._yaw * YAW_TORQUE
        # Keep the director exactly in the xz plane (kill any out-of-plane drift).
        nx, nz = mu[ic][0], mu[ic][2]
        m = math.hypot(nx, nz)
        if m > 1e-9:
            mu[ic][0] = nx / m
            mu[ic][1] = 0.0
            mu[ic][2] = nz / m

    def _apply_ring_forces(self, dt):
        """Apply the two soft ring forces as per-frame momentum kicks (delta v =
        F*dt/m, m=1): a centering force that drives the outer beads' centre of
        mass to the origin, and an alignment torque that rotates the patch's
        smallest-principal-axis (its normal) toward +z. The torque is realized
        as per-bead forces F_i = a x r_i' (r_i' relative to the COM) with a =
        Iang^-1 * T, which yields exactly the wanted net torque T and zero net
        force. Nothing is positioned by hand -- the beads then evolve under these
        forces plus the MesoMem pair forces and the Langevin bath (which damps
        the kicks so the patch settles flat and centred)."""
        idx, _ = self._id_index()
        v = self.lmp.numpy.extract_atom("v")
        ring_local = [idx[i] for i in self.ring_ids]
        P = np.array([[self.lmp.numpy.extract_atom("x")[k][j] for j in range(3)]
                      for k in ring_local])
        com = P.mean(axis=0)

        # Centering force: same force on each bead -> pushes the COM to origin
        # without distorting the cluster's shape.
        f_center = -K_CENTER * com

        # Alignment torque: rotate the smallest principal axis (patch normal) up.
        Q = P - com
        cov = Q.T @ Q
        evals, evecs = np.linalg.eigh(cov)     # ascending eigenvalues
        e_min = evecs[:, 0]
        if e_min[2] < 0.0:                      # pick the upward-facing normal
            e_min = -e_min
        torque = K_ALIGN * np.cross(e_min, np.array([0.0, 0.0, 1.0]))
        # Inertia-like tensor Iang = sum(|r'|^2 I - r' r'^T); a = Iang^-1 T.
        iang = np.eye(3) * np.sum(Q * Q) - Q.T @ Q
        a = np.linalg.solve(iang + 1e-6 * np.eye(3), torque)

        for m, k in enumerate(ring_local):
            # centering (COM) + alignment torque + a tiny per-bead pull home so a
            # single bead can't escape when the puller is thrown around hard.
            f = f_center + np.cross(a, Q[m]) - K_HOME * P[m]
            v[k][0] += f[0] * dt
            v[k][1] += f[1] * dt
            v[k][2] += f[2] * dt

    def step(self, n=10):
        self.lmp.command(f"run {n}")
        self._apply_ring_forces(n * TIMESTEP)
        self._constrain_center()
        self._interactive_t += n * TIMESTEP

    # ---- readouts (2D control-plane projection for the app/haptics) ---------

    def get_puller_state(self):
        ic, n = self._center_local()
        if ic is None:
            return None, None
        x = self.lmp.numpy.extract_atom("x")[:n]
        v = self.lmp.numpy.extract_atom("v")[:n]
        # Control plane basis = (world x, world z).
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
        """Live decomposition of the central bead's MesoMem interaction energy
        into the paper's additive terms, in reduced (eps) units. Computed here
        directly from the paper's formulas (Eqs. 2-6, with c0 = 0) over the
        bead's neighbours, because the pair style only exposes total per-atom
        energy, not the separate terms. Summed over the bead's pair bonds:
          - isotropic  U_iso  : 4-2 soft core (r < sigma) + cosine-squared
                                 attraction (sigma < r < rc)  -- packing/cohesion
          - tilt       U_tilt : (k_tilt/2)[(n_i.rhat)^2 + (n_j.rhat)^2] w(r)
                                 -- penalizes directors tipping off the bond normal
          - splay      U_splay: (k_splay/2)(n_i.n_j - 1)^2 w(r)
                                 -- penalizes neighbouring directors misaligning
                                 (NB: as published this is polar -- parallel is
                                 favoured over antiparallel; the splay_symmetry
                                 dial blends n_i.n_j toward |n_i.n_j| to remove
                                 that polarity, mirrored here so the panel matches
                                 the applied force)
        These three add up to the bead's total interaction energy -- the whole
        point of the demo's display."""
        idx, _ = self._id_index()
        ic = idx.get(self.center_id)
        if ic is None:
            return None
        x = self.lmp.numpy.extract_atom("x")
        mu = self.lmp.numpy.extract_atom("mu")
        ri = np.array([x[ic][0], x[ic][1], x[ic][2]])
        ni = np.array([mu[ic][0], mu[ic][1], mu[ic][2]])
        ni = ni / (np.linalg.norm(ni) or 1.0)

        rc, wc = self._rc, self._effective_wc()
        u_iso = u_tilt = u_splay = 0.0
        for jid in self.ring_ids:
            jl = idx.get(jid)
            if jl is None:
                continue
            rj = np.array([x[jl][0], x[jl][1], x[jl][2]])
            d = ri - rj
            r = float(np.linalg.norm(d))
            if r >= rc or r < 1e-9:
                continue
            rhat = d / r
            # Isotropic branch: 4-2 core below rmin=sigma, cosine^2 attraction above.
            if r < SIGMA:
                t2 = (SIGMA / r) ** 2
                u_iso += EPS * (t2 * t2 - 2.0 * t2)
            else:
                g = math.pi * 0.5 * (r - SIGMA) / (rc - SIGMA)
                u_iso += -EPS * math.cos(g) ** (2.0 * self._zeta)
            # Orientational weight w(r), nonzero only within wc.
            w = 0.0
            if r < wc:
                rga = 0.5 * wc
                denom = (r / wc) ** 4 - 1.0
                if denom < -1e-14:
                    w = math.exp((r * r) / (rga * rga * denom))
            if w > 0.0:
                nj = np.array([mu[jl][0], mu[jl][1], mu[jl][2]])
                nj = nj / (np.linalg.norm(nj) or 1.0)
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

    def _pair_terms(self, d, ni, nj):
        """The three additive MesoMem energies for one ordered pair separated by
        d = r_i - r_j, with unit directors ni, nj -- the same formulas as
        get_potential_terms, factored out so the whole-system total can reuse
        them. Returns (u_iso, u_tilt, u_splay); all zero past the cutoff."""
        rc, wc = self._rc, self._effective_wc()
        r = float(np.linalg.norm(d))
        if r >= rc or r < 1e-9:
            return 0.0, 0.0, 0.0
        rhat = d / r
        if r < SIGMA:
            t2 = (SIGMA / r) ** 2
            u_iso = EPS * (t2 * t2 - 2.0 * t2)
        else:
            g = math.pi * 0.5 * (r - SIGMA) / (rc - SIGMA)
            u_iso = -EPS * math.cos(g) ** (2.0 * self._zeta)
        u_tilt = u_splay = 0.0
        if r < wc:
            rga = 0.5 * wc
            denom = (r / wc) ** 4 - 1.0
            if denom < -1e-14:
                w = math.exp((r * r) / (rga * rga * denom))
                nir = float(ni @ rhat)
                njr = float(nj @ rhat)
                ninj = float(ni @ nj)
                ninj_eff = (1.0 - self._splay_sym) * ninj + self._splay_sym * abs(ninj)
                u_tilt = 0.5 * self._ktilt * (nir * nir + njr * njr) * w
                u_splay = 0.5 * self._ksplay * (ninj_eff - 1.0) ** 2 * w
        return u_iso, u_tilt, u_splay

    def get_total_potential_terms(self):
        """Whole-patch additive energy: the same three MesoMem terms summed over
        every unique bead pair (each counted once), so the panel shows the total
        attraction / tilt / splay stored in the membrane, not just the puller's
        share. Only 7 beads here, so the full O(N^2) pair loop is trivial."""
        idx, n = self._id_index()
        order = [idx.get(i) for i in self.all_ids]
        if any(k is None for k in order):
            return None
        x = np.array(self.lmp.numpy.extract_atom("x")[:n], dtype=float)
        mu = np.array(self.lmp.numpy.extract_atom("mu")[:n], dtype=float)[:, :3]
        P = x[order]
        D = mu[order]
        D /= np.clip(np.linalg.norm(D, axis=1, keepdims=True), 1e-9, None)
        u_iso = u_tilt = u_splay = 0.0
        m = len(order)
        for a in range(m):
            for b in range(a + 1, m):
                ui, ut, us = self._pair_terms(P[a] - P[b], D[a], D[b])
                u_iso += ui
                u_tilt += ut
                u_splay += us
        terms = [
            ("isotropic  (repel + attract)", u_iso),
            ("tilt  (directors normal to bonds)", u_tilt),
            ("splay  (neighbour directors align)", u_splay),
        ]
        return ("Whole-patch energy -- additive (reduced units)", terms, 18.0)

    def get_interaction_force(self):
        """Membrane's (pair) reaction force on the central bead, projected onto
        the (x, z) control plane. Recovered exactly as total force minus the two
        forces we apply ourselves: the joystick drive (addforce) and the viscous
        damping (-gamma*v). The setforce on y is irrelevant here (we only read
        x, z). This substitutes for compute group/group, which the mesomem pair
        style can't support (no single())."""
        ic, n = self._center_local()
        if ic is None:
            return np.array([0.0, 0.0])
        f = self.lmp.numpy.extract_atom("f")[:n]
        v = self.lmp.numpy.extract_atom("v")[:n]
        g = self._puller_damping
        pair_fx = f[ic][0] - self._input_fx + g * v[ic][0]
        pair_fz = f[ic][2] - self._input_fz + g * v[ic][2]
        return np.array([pair_fx, pair_fz])

    def get_torque_signals(self):
        """(applied, reaction) torques about the control-plane normal (world y),
        normalized to [-1, 1] for the circular torque arrows.

          - applied  : the user's yaw steering torque. `self._yaw` is already the
            per-frame angular kick about y in [-1, 1], so it IS the fraction.
          - reaction : the membrane's restoring torque on the pulled bead about y,
            read straight from the pair-style's per-atom torque (the y component
            is the part that rotates the director within the control plane -- the
            projection onto the net we draw), scaled to the display max.

        Positive = the +y sense, which tips the director toward +x (screen-right)
        -- the same handedness a positive yaw command produces."""
        ic, n = self._center_local()
        if ic is None:
            return None
        applied = max(-1.0, min(1.0, self._yaw))
        tau = self.lmp.numpy.extract_atom("torque")
        reaction = 0.0
        if tau is not None:
            reaction = max(-1.0, min(1.0, float(tau[:n][ic][1]) / REACTION_TORQUE_DISPLAY_MAX))
        return applied, reaction

    def get_thermo_state(self):
        temp = self.lmp.extract_compute("ring_temp", 0, 0)
        press = self.lmp.get_thermo("press")
        ke = self.lmp.get_thermo("ke")
        pe = self.lmp.get_thermo("pe")
        etotal = self.lmp.get_thermo("etotal")
        return temp, press, ke, pe, etotal

    def get_sim_time(self):
        return self._interactive_t

    def get_rdf(self):
        idx, _ = self._id_index()
        x = self.lmp.numpy.extract_atom("x")
        self._rdf.add(np.array([[x[idx[i]][0], x[idx[i]][1]] for i in self.all_ids]))
        return self._rdf.get()

    def get_all_positions(self):
        """2D (membrane-plane) fallback required by the interface. The 3D
        renderer uses get_positions_3d instead; this returns the xy shadow so
        trails and any 2D consumer stay valid."""
        idx, n = self._id_index()
        x = self.lmp.numpy.extract_atom("x")[:n]
        ids = self.lmp.numpy.extract_atom("id")[:n]
        order = [idx[i] for i in self.all_ids]
        pos2d = np.array([[x[k][0], x[k][1]] for k in order])
        is_puller = np.array([i == self.center_id for i in self.all_ids])
        return np.array(self.all_ids), pos2d, is_puller, None

    # ---- 3D rendering data --------------------------------------------------

    def get_positions_3d(self):
        """(ids, positions (7,3), is_puller (7,)), ordered center-first."""
        idx, n = self._id_index()
        x = self.lmp.numpy.extract_atom("x")[:n]
        order = [idx[i] for i in self.all_ids]
        pos = np.array([[x[k][0], x[k][1], x[k][2]] for k in order])
        is_puller = np.array([i == self.center_id for i in self.all_ids])
        return np.array(self.all_ids), pos, is_puller

    def get_dipoles_3d(self):
        """Unit director n_i per bead (7,3), ordered center-first."""
        idx, n = self._id_index()
        mu = self.lmp.numpy.extract_atom("mu")[:n]   # columns 0..2 = vector, 3 = magnitude
        order = [idx[i] for i in self.all_ids]
        dirs = np.array([mu[k][:3] for k in order], dtype=float)
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        return dirs / norms

    def get_bonds_3d(self):
        return list(self._bond_index_pairs)

    def get_camera_params(self):
        return dict(CAMERA_PARAMS)

    def get_control_grid(self):
        """The joystick's control plane (world xz through the patch center), as
        basis + extents. Its extents are exactly the puller's movement limits
        (_CTRL_X / _CTRL_Z), so the drawn net marks precisely where the bead can
        be dragged."""
        return dict(
            origin=(0.0, 0.0, 0.0),
            u_axis=(1.0, 0.0, 0.0),   # world x  (joystick axis x, screen-horizontal)
            v_axis=(0.0, 0.0, 1.0),   # world z  (joystick axis y, screen-vertical)
            u_range=self._CTRL_X,
            v_range=self._CTRL_Z,
            step=0.5,
        )

    def get_box_bounds_3d(self):
        """The container cube, for the renderer to outline in white."""
        h = BOX / 2.0
        return (-h, h, -h, h, -h, h)

    def get_scene_fit_points(self):
        """World extent the camera should frame: the container box corners (so the
        white box outline stays in view) plus the control-plane net corners and a
        little vertical headroom for the puller/director spikes."""
        g = self.get_control_grid()
        origin = np.asarray(g["origin"], dtype=float)
        u = np.asarray(g["u_axis"], dtype=float)
        v = np.asarray(g["v_axis"], dtype=float)
        (u0, u1), (v0, v1) = g["u_range"], g["v_range"]
        pts = [origin + uu * u + vv * v for uu in (u0, u1) for vv in (v0, v1)]
        # Headroom above/below for the puller bead and its director spike, which
        # can rise a little past the net's top edge.
        pts.append(origin + (v1 + 0.6) * v)
        pts.append(origin + (v0 - 0.2) * v)
        h = BOX / 2.0
        pts += [(sx * h, sy * h, sz * h)
                for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        return np.array(pts)

    def get_box_size(self):
        return BOX, BOX

    def close(self):
        self.lmp.close()

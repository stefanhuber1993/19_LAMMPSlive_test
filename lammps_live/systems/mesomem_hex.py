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

# --- MesoMem potential parameters (paper "standard conditions", Sec. II-III) --
SIGMA = 1.0        # bead diameter / length unit
EPS = 1.0          # energy unit (LJ well depth)
K_TILT = 12.0      # tilt modulus -- above ~10 the patch stays planar (paper)
K_SPLAY = 1.0      # splay modulus
RC = 2.5           # isotropic interaction cutoff (>= 2.5 sigma needed for cohesion)
WC = 2.0           # orientational (tilt/splay) interaction cutoff, must be <= RC
ZETA = 5.0         # steepness/width of the cosine-squared attractive branch
C0 = 0.0           # spontaneous curvature (0 -> flat preferred)

A_LATTICE = 1.0    # in-plane nearest-neighbor spacing (near the isotropic min at r=sigma)
BEAD_DIAMETER = 2.0  # sphere radius = sigma -> moment of inertia I = (2/5) m sigma^2 (paper)

BOX = 16.0         # cubic box half-extent is BOX/2; non-periodic, just a container

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
K_CENTER = 45.0    # centre-of-mass centering stiffness
K_ALIGN = 10.0     # normal-up alignment torque strength
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

# MesoMem contact/restoring forces on the pulled bead run O(1-10) in reduced
# units; profile is scaled to that, with enough stick authority to tent the
# membrane and pop the bead out against tilt/splay resistance.
FORCE_FEEDBACK = ForceFeedbackProfile(
    input_force_scale=9.0,
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
                       PULLER_DAMPING_DEFAULT, fmt="{:.2f}"),
    melt_temp=T_MELT,
    force_feedback=FORCE_FEEDBACK,
    puller_speed_cap=0.06 * SIGMA / TIMESTEP,
    crystal_color=None,   # 3D path uses theme.MEMBRANE_BEAD_COLOR
    atom_radius_A=0.5 * SIGMA,   # physical bead radius used for perspective sphere sizing
    sim_time_per_frame=0.05,     # tau_LJ per frame (10 steps at 0.005)
    bond_overlay=False,
    render_3d=True,
)

# Camera for the three-quarter view: looking at the patch from below in y and
# above in z, so the membrane (xy-plane) tilts and the directors (+z) stand up.
CAMERA_PARAMS = dict(
    eye=(0.0, -7.5, 5.2),
    target=(0.0, 0.0, 0.2),
    up=(0.0, 0.0, 1.0),
    fov_deg=34.0,
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

        # id 1 = central puller; ids 2..7 = hexagonal ring.
        self.center_id = 1
        self.ring_ids = tuple(range(2, 8))
        self.all_ids = (self.center_id,) + self.ring_ids
        # Bonds to draw (index pairs into the id-ordered array, center first):
        # 6 spokes + the 6-segment ring.
        self._bond_index_pairs = [(0, k) for k in range(1, 7)]
        self._bond_index_pairs += [(k, k % 6 + 1) for k in range(1, 7)]

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

        c(f"pair_style mesomem {RC}")
        # sigma eps ktilt ksplay cut weight_rcut zeta c0
        c(f"pair_coeff 1 1 {SIGMA} {EPS} {K_TILT} {K_SPLAY} {RC} {WC} {ZETA} {C0}")

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
        self.lmp.command(
            f"fix bath all langevin {T} {T} {LANGEVIN_DAMP} {self._seed} omega yes"
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

    # Keep the puller on the control plane, inside the visible net, and below a
    # runaway speed. Without this a sustained max pull would accelerate the
    # (undamped-by-thermostat, un-limited) central bead straight out of the box.
    # Leash kept inside the ring's interaction range (a bead at height z sits
    # sqrt(1 + z^2) from its neighbors; past the rc = 2.5 cutoff it would detach
    # and float free), so the membrane always exerts a restoring pull and the
    # bead snaps back on release instead of sticking at the ceiling.
    _CTRL_X = (-2.3, 2.3)
    _CTRL_Z = (-1.7, 1.7)
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
                                 favoured over antiparallel; see the question
                                 raised with the authors before changing it)
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

        u_iso = u_tilt = u_splay = 0.0
        for jid in self.ring_ids:
            jl = idx.get(jid)
            if jl is None:
                continue
            rj = np.array([x[jl][0], x[jl][1], x[jl][2]])
            d = ri - rj
            r = float(np.linalg.norm(d))
            if r >= RC or r < 1e-9:
                continue
            rhat = d / r
            # Isotropic branch: 4-2 core below rmin=sigma, cosine^2 attraction above.
            if r < SIGMA:
                t2 = (SIGMA / r) ** 2
                u_iso += EPS * (t2 * t2 - 2.0 * t2)
            else:
                g = math.pi * 0.5 * (r - SIGMA) / (RC - SIGMA)
                u_iso += -EPS * math.cos(g) ** (2.0 * ZETA)
            # Orientational weight w(r), nonzero only within wc.
            w = 0.0
            if r < WC:
                rga = 0.5 * WC
                denom = (r / WC) ** 4 - 1.0
                if denom < -1e-14:
                    w = math.exp((r * r) / (rga * rga * denom))
            if w > 0.0:
                nj = np.array([mu[jl][0], mu[jl][1], mu[jl][2]])
                nj = nj / (np.linalg.norm(nj) or 1.0)
                nir = float(ni @ rhat)
                njr = float(nj @ rhat)
                ninj = float(ni @ nj)
                u_tilt += 0.5 * K_TILT * (nir * nir + njr * njr) * w
                u_splay += 0.5 * K_SPLAY * (ninj - 1.0) ** 2 * w
        terms = [
            ("isotropic  (repel + attract)", u_iso),
            ("tilt  (directors normal to bonds)", u_tilt),
            ("splay  (neighbour directors align)", u_splay),
        ]
        return ("Pulled bead energy -- additive (reduced units)", terms, 6.0)

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
        return None   # 7 beads: no meaningful radial distribution

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
        """The joystick's control plane (world xz through the patch center),
        as basis + extents, for the renderer to draw as a net."""
        return dict(
            origin=(0.0, 0.0, 0.0),
            u_axis=(1.0, 0.0, 0.0),   # world x  (joystick axis x, screen-horizontal)
            v_axis=(0.0, 0.0, 1.0),   # world z  (joystick axis y, screen-vertical)
            u_range=(-2.6, 2.6),
            v_range=(-2.0, 2.0),
            step=0.5,
        )

    def get_box_size(self):
        return BOX, BOX

    def close(self):
        self.lmp.close()

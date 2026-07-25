"""MesoMem self-assembly -- the paper's spontaneous-lamellae test, made live.

Same real MesoMem force field as the 7-bead patch and the relaxed sheet (see
mesomem_hex.py / mesomem_sheet.py), but set up exactly as the paper's first
validation experiment (Sillano, Marrink & Idema 2026, "Self-assembly"): N = 1500
directored particles dropped at RANDOM positions and orientations into a cubic,
fully periodic box of side L = 20 sigma. That is a reduced volume fraction
phi = N * Vp / Vbox ~ 0.1 with Vp = (pi/6) sigma^3. Under Langevin (implicit-
solvent) dynamics the disordered gas coarsens: small patches by t ~ 500 tau_LJ,
coalescing into large planar membranes by t ~ 2000 tau_LJ. tilt modulus k_tilt is
the critical control parameter -- below ~10 the aggregates stay compact and
isotropic, above it they flatten into membranes (standard value 12.0); secondary
parameters are held at k_splay = 1, zeta = 5.0, wc = 2.0 sigma, rc = 2.5 sigma.

Unlike the sheet and patch systems there is NO puller and NO joystick control:
the point is to watch assembly happen, not to poke a membrane. Instead the app
draws Play / Pause / Reset buttons (see SystemSpec.playback_controls); Reset
re-randomizes the box so a new run can start from a fresh disordered state with
whatever coefficients the sliders currently hold. The live sliders are the same
MesoMem coefficients the sheet exposes (temperature + k_tilt / k_splay / zeta and
the advanced splay-symmetry / rc / wc), so the user can dial k_tilt through the
compact-vs-planar transition and watch the morphology change.

Units are the paper's reduced LJ units (sigma = eps = m = 1).
"""
import math
import random
from collections import deque

import numpy as np
from lammps import lammps
from scipy.spatial import cKDTree

from .base import ForceFeedbackProfile, MDSystem, SliderSpec, SystemSpec
from .mesomem_ff import ensure_plugin_loaded

# --- MesoMem potential parameters (paper "standard conditions") ---------------
SIGMA = 1.0
EPS = 1.0
K_TILT = 12.0       # standard value; below ~10 -> compact aggregates, above -> membranes
K_SPLAY = 1.0
RC = 2.5
WC = 2.0
ZETA = 5.0
C0 = 0.0

# Live-tunable ranges for the MesoMem coefficient sliders -- identical to the
# membrane sheet, each with the paper's recommended value marked as the slider
# "optimum" (see mesomem_sheet.py for the per-parameter rationale).
K_TILT_MIN, K_TILT_MAX, K_TILT_OPT = 0.0, 50.0, 12.0
K_SPLAY_MIN, K_SPLAY_MAX, K_SPLAY_OPT = 0.0, 40.0, 1.0
ZETA_MIN, ZETA_MAX, ZETA_OPT = 0.0, 12.0, 5.0
RC_MIN, RC_MAX, RC_OPT = 0.0, 3.0, 2.5
WC_MIN, WC_MAX, WC_OPT = 0.0, 3.0, 2.0

# Splay symmetry (advanced): 0..1 weight blending the splay term's SIGNED
# director dot product (0, the paper's original polar form) toward its ABSOLUTE
# value (1 -- parallel and antiparallel treated identically). See the pair
# style's compute() and _apply_pair_coeff.
SPLAY_SYM = 0.0
SPLAY_SYM_MIN, SPLAY_SYM_MAX = 0.0, 1.0

# --- Self-assembly geometry (paper's exact setup) ----------------------------
N_PARTICLES = 1500     # paper's particle count
BOX_L = 20.0           # cubic box side, sigma (phi = N*Vp/L^3 ~ 0.1)
BEAD_DIAMETER = 1.0    # sphere diameter = sigma (rendered radius 0.5 sigma)
# Minimum center-to-center separation enforced when the disordered start is
# randomly seeded, so no two beads are dropped inside each other's hard core and
# blow the first step up. 0.9 sigma sits just outside the 4-2 core's minimum at
# r = sigma (the pair force there is small and attractive), and at phi ~ 0.1 the
# box is dilute enough that placing all 1500 with this spacing is easy.
SEED_OVERLAP = 0.9
SEED_MAXTRY = 200

TIMESTEP = 0.01        # tau_LJ (the paper's value; the soft 4-2 core + the
                       #  overlap-free start keep this stable with no puller)
# tau_LJ advanced per rendered frame. 20 steps/frame: fast enough that the
# coarsening (patches by ~500 tau, membranes by ~2000 tau) is watchable in a few
# minutes of real time, cheap enough to stay interactive at 1500 beads.
SIM_TIME_PER_FRAME = 0.2

# Langevin (implicit solvent). A slightly weaker friction than the sheet's
# probing bath (larger damp = weaker drag) lets the beads diffuse and find each
# other, so assembly proceeds at a watchable rate.
LANGEVIN_DAMP = 1.0

# Reduced-temperature dial. Assembly needs finite T so beads can diffuse and
# anneal into flat membranes, but below the ~eps attraction well so they stay
# condensed rather than boiling back into a gas. The default sits in that fluid-
# membrane window; the same 0..0.5 range and melt marker as the sheet.
T_MIN = 0.0
T_MAX = 0.5
T_DEFAULT = 0.2
T_MELT = 0.3

# The instrumentation/haptics layer is scaled for the interactive systems; this
# one has no puller, so the numbers below are only ever fed zeros. Kept present
# (the app always reads a profile) but never actually exercised.
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
    key="assembly",
    name="MesoMem self-assembly (3D)",
    description="Paper's spontaneous-assembly run: 1500 random beads in a periodic 20-sigma box coarsen into membranes. Play / Pause / Reset.",
    element_label="membrane bead (director)",
    lattice_spacing=SIGMA,
    timestep=TIMESTEP,
    temperature=SliderSpec("Temperature", T_MIN, T_MAX, T_DEFAULT, fmt="{:.3f}", unit=" T*"),
    # No puller, so no "puller damping"; the slider is required by the spec but
    # kept advanced (hidden) and wired to a no-op setter -- see set_puller_damping.
    damping=SliderSpec("Puller damping (unused)", 0.0, 8.0, 0.0, fmt="{:.2f}",
                       advanced=True),
    melt_temp=T_MELT,
    force_feedback=FORCE_FEEDBACK,
    max_input_force=0.0,        # no interactive force -- nothing to push
    puller_speed_cap=0.06 * SIGMA / TIMESTEP,
    crystal_color=None,
    atom_radius_A=0.5 * SIGMA,
    sim_time_per_frame=SIM_TIME_PER_FRAME,
    bond_overlay=False,
    render_3d=True,
    reduced_units=True,
    # Play / Pause / Reset buttons instead of joystick/mouse control.
    playback_controls=True,
    # Hundreds of per-bead director spikes would be clutter; the banded
    # pole/equator coloring already shows each director's tilt.
    director_arrows=False,
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


class _RDF3D:
    """Rolling, time-averaged 3D radial distribution g(r), minimum-image in the
    periodic cubic box -- the spatial analog of rdf2d.InPlaneRDF but with a
    spherical-shell (4/3 pi dr^3) ideal-gas normalization, so g(r) -> 1 for a
    structureless gas and grows sharp shells as the beads condense into ordered
    membranes. Subsampled and throttled like the 2D version to bound the O(N^2)
    pair pass on 1500 beads."""

    def __init__(self, r_max, box_l, nbins=60, min_samples=10, window=40,
                 sample_every=3, max_atoms=400):
        self.box_l = box_l
        self.edges = np.linspace(0.0, float(r_max), int(nbins) + 1)
        self.r = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.shell_vol = (4.0 / 3.0) * np.pi * (self.edges[1:] ** 3 - self.edges[:-1] ** 3)
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
        d -= self.box_l * np.round(d / self.box_l)   # minimum image, cubic box
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


class MesoMemAssemblySystem(MDSystem):
    spec = SPEC

    def __init__(self):
        self._target_temp = T_DEFAULT
        # Live-tunable MesoMem coefficients (start at the paper's standard values;
        # preserved across Reset so a re-randomized run keeps the user's dials).
        self._ktilt = K_TILT
        self._ksplay = K_SPLAY
        self._zeta = ZETA
        self._rc = RC
        self._wc = WC
        self._splay_sym = SPLAY_SYM
        # Throttled cache for the whole-box potential total (O(pairs) each time).
        self._total_terms_cache = None
        self._total_terms_ctr = 0

        self.all_ids = np.arange(1, N_PARTICLES + 1)
        r_max = min(0.5 * BOX_L, 6.0 * SIGMA)
        self._rdf = _RDF3D(r_max, BOX_L)

        self.lmp = None
        self._setup(seed=random.randint(1, 900_000_000))

    # ---- construction -------------------------------------------------------

    def _setup(self, seed):
        """Build a fresh LAMMPS instance seeded with a new random disordered
        configuration, using the current (slider-held) coefficients. Called from
        __init__ and again from reset()."""
        self._seed = seed
        self.lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
        lmp = self.lmp
        c = lmp.command
        ensure_plugin_loaded(lmp)

        c("units lj")
        c("dimension 3")
        c("atom_style hybrid sphere dipole")
        c("boundary p p p")          # fully periodic cubic cell
        c("atom_modify map array")

        h = BOX_L / 2.0
        c(f"region box block {-h} {h} {-h} {h} {-h} {h} units box")
        c("create_box 1 box")
        # N beads at random positions with a minimum separation (overlap-free), so
        # the disordered start doesn't contain overlapping hard cores.
        c(f"create_atoms 1 random {N_PARTICLES} {self._seed} box "
          f"overlap {SEED_OVERLAP} maxtry {SEED_MAXTRY} units box")

        c("mass 1 1.0")
        c(f"set group all diameter {BEAD_DIAMETER}")
        # Random initial director orientations (unit dipoles) -- the disordered
        # orientational start the paper's self-assembly begins from.
        c(f"set group all dipole/random {self._seed} 1.0")

        c(f"pair_style mesomem {self._rc}")
        self._apply_pair_coeff()

        c("neighbor 0.6 bin")
        c("neigh_modify every 1 delay 0 check yes")

        # Translation + dipole rotation under a Langevin implicit-solvent bath on
        # every bead (no puller to exclude here).
        c("fix integrate all nve/sphere update dipole")
        c(f"fix bath all langevin {self._target_temp} {self._target_temp} "
          f"{LANGEVIN_DAMP} {self._seed} omega yes")

        c("compute ke_atom all ke/atom")
        c("compute pe_atom all pe/atom")
        c(f"timestep {TIMESTEP}")
        c("thermo 100000")
        c("run 0")

        self._interactive_t = 0.0
        self._total_terms_cache = None
        self._total_terms_ctr = 0

    def reset(self):
        """Re-randomize the box for a fresh assembly run, keeping the current
        coefficients/temperature. Rebuilds the LAMMPS instance with a new seed."""
        if self.lmp is not None:
            self.lmp.close()
        self._rdf.reset()
        self._setup(seed=random.randint(1, 900_000_000))

    # ---- id <-> local-index helpers ----------------------------------------

    def _id_index(self):
        n = self.lmp.get_natoms()
        ids = self.lmp.numpy.extract_atom("id")[:n]
        return {int(i): k for k, i in enumerate(ids)}, n

    def _apply_pair_coeff(self):
        """(Re)issue the mesomem pair_coeff from the current live coefficients.
        The trailing splay_symmetry argument blends the splay term's signed
        director dot product toward |dot| (see mesomem_sheet.py)."""
        self.lmp.command(
            f"pair_coeff 1 1 {SIGMA} {EPS} {self._ktilt} {self._ksplay} "
            f"{self._rc} {self._effective_wc()} {self._zeta} {C0} {self._splay_sym}"
        )

    def _effective_wc(self):
        """Orientational cutoff actually used: capped at rc (the paper's upper
        bound)."""
        return min(self._wc, self._rc)

    # ---- controls -----------------------------------------------------------
    # No puller: the force/orientation inputs are no-ops; only the temperature and
    # the MesoMem coefficient dials do anything.

    def set_input_force(self, fx, fy):
        pass

    def set_puller_damping(self, gamma):
        pass

    def set_target_temp(self, T):
        T = max(T_MIN, min(T_MAX, T))
        if T == self._target_temp:
            return
        self._target_temp = T
        self.lmp.command(
            f"fix bath all langevin {T} {T} {LANGEVIN_DAMP} {self._seed} omega yes"
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

    def step(self, n=20):
        self.lmp.command(f"run {n}")
        self._interactive_t += n * TIMESTEP

    # ---- readouts -----------------------------------------------------------
    # No puller -- the puller-shaped hooks the app/haptics call return neutral
    # values (None / zeros), which the app already handles gracefully.

    def get_puller_state(self):
        return None, None

    def get_puller_energy(self):
        return None, None

    def get_interaction_force(self):
        return np.array([0.0, 0.0])

    def get_total_potential_terms(self):
        """Whole-box additive energy: the three MesoMem terms summed over every
        unique bead pair within rc across the entire box (minimum-image in the
        periodic cell), so the panel shows the total attraction / tilt / splay the
        system is storing -- which climbs (in magnitude) as the gas condenses into
        cohesive membranes. Candidate pairs come from a 3D periodic KD-tree,
        filtered on the exact minimum-image distance; throttled to every few
        frames since it is far heavier than a single-bead readout."""
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
        L = BOX_L
        # Wrap into [0, L) so the periodic KD-tree's boxsize is well-posed.
        wrapped = (P + L / 2.0) % L
        tree = cKDTree(wrapped, boxsize=L)
        pairs = tree.query_pairs(rc, output_type="ndarray")

        u_iso = u_tilt = u_splay = 0.0
        if len(pairs):
            a, b = pairs[:, 0], pairs[:, 1]
            d = P[a] - P[b]
            d -= L * np.round(d / L)
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
        self._total_terms_cache = (
            "Whole-box energy -- additive (reduced units)", terms, 5000.0)
        return self._total_terms_cache

    _TOTAL_EVERY = 4

    def get_thermo_state(self):
        temp = self.lmp.get_thermo("temp")
        press = self.lmp.get_thermo("press")
        ke = self.lmp.get_thermo("ke")
        pe = self.lmp.get_thermo("pe")
        etotal = self.lmp.get_thermo("etotal")
        return temp, press, ke, pe, etotal

    def get_sim_time(self):
        return self._interactive_t

    def get_hud_lines(self):
        """A short live status line naming the coarsening stage the paper's clock
        is in, so the disordered->patches->membranes progression reads at a glance
        (t is reduced tau_LJ)."""
        t = self._interactive_t
        if t < 1e-6:
            stage = "disordered start -- press Play"
        elif t < 500.0:
            stage = "nucleating small patches"
        elif t < 2000.0:
            stage = "patches growing / coalescing"
        else:
            stage = "large planar membranes"
        return [f"t = {t:,.0f} tau_LJ   |   {stage}",
                f"N = {N_PARTICLES} beads   box = {BOX_L:.0f} sigma (periodic)   k_tilt = {self._ktilt:.1f}"]

    def get_rdf(self):
        idx, n = self._id_index()
        x = self.lmp.numpy.extract_atom("x")[:n]
        order = [idx[int(i)] for i in self.all_ids]
        self._rdf.add(np.array([[x[k][0], x[k][1], x[k][2]] for k in order]))
        return self._rdf.get()

    def get_all_positions(self):
        idx, n = self._id_index()
        x = self.lmp.numpy.extract_atom("x")[:n]
        order = [idx[int(i)] for i in self.all_ids]
        pos2d = np.array([[x[k][0], x[k][1]] for k in order])
        is_puller = np.zeros(len(self.all_ids), dtype=bool)
        return np.array(self.all_ids), pos2d, is_puller, None

    # ---- 3D rendering data --------------------------------------------------

    def get_positions_3d(self):
        idx, n = self._id_index()
        x = self.lmp.numpy.extract_atom("x")[:n]
        order = [idx[int(i)] for i in self.all_ids]
        pos = np.array([[x[k][0], x[k][1], x[k][2]] for k in order])
        is_puller = np.zeros(len(self.all_ids), dtype=bool)
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
        return []

    def get_control_grid(self):
        # No puller, so no control-plane net to draw.
        return None

    def get_camera_params(self):
        # Three-quarter view of the cubic box; fit_to_points then zooms to fill
        # the viewport, so this only sets a good angle and distance.
        s = BOX_L
        return dict(
            eye=(0.6 * s, -1.1 * s, 0.7 * s),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
            fov_deg=34.0,
        )

    def get_box_bounds_3d(self):
        h = BOX_L / 2.0
        return (-h, h, -h, h, -h, h)

    def get_scene_fit_points(self):
        h = BOX_L / 2.0
        return np.array([(sx * h, sy * h, sz * h)
                         for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])

    def get_box_size(self):
        return BOX_L, BOX_L

    def close(self):
        if self.lmp is not None:
            self.lmp.close()
            self.lmp = None

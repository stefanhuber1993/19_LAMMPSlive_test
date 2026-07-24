"""2D "Mercedes-Benz" water -- the model that explains why ice floats.

Water's headline anomaly is that it EXPANDS when it freezes: solid ice is less
dense than the liquid, so ice floats and lakes freeze top-down. The cause is
the hydrogen bond, which is strongly *directional* -- each water molecule wants
to make four (in 2D: three) hydrogen bonds pointing at specific angles. In the
liquid, molecules give up some of that geometry to pack closely (dense); on
freezing they must satisfy all their hydrogen bonds at once, which forces them
onto an *open*, low-density lattice with holes in it. Fewer molecules per unit
area -> lower density -> ice floats.

The Mercedes-Benz (MB) model (Ben-Naim 1971) is the classic minimal cartoon of
this: a 2D disk with three hydrogen-bonding arms at 120 degrees, like the
Mercedes star. Two molecules hydrogen-bond when an arm of one points straight
at an arm of the other. Because a satisfied bond holds the two disks a full
arm-span apart, the fully hydrogen-bonded solid is an OPEN honeycomb -- more
spread out than the jumbled, close-touching liquid. That is the whole anomaly,
visible on screen.

How it is built here (all standard LAMMPS, metal units eV/A/ps):
- each molecule is a rigid 4-bead body: a central core (the O) plus three arm
  tips at 120 degrees (the hydrogen-bonding directions), integrated as a rigid
  body with `fix rigid/small` and thermostatted by its built-in Langevin bath
  (the temperature dial drives it directly).
- interactions use `cosine/squared` (the same soft, bounded, well-behaved style
  the lipid demo uses): a stiff repulsive core gives each molecule its size,
  and a short, narrow attractive well between arm TIPS is the hydrogen bond --
  short-ranged enough that a bond only forms when two arms point almost exactly
  at each other, which is what makes it directional and forces the open lattice.
- the system starts from the ideal open hydrogen-bonded honeycomb "ice" (a
  cluster in vacuum, free to shrink or spread), so the beautiful three-fold
  network is there from frame 0; heat it and watch it COLLAPSE into a denser,
  jumbled liquid -- the freezing-expansion anomaly, run in reverse.

Constants were tuned here the same way the other systems' were (a temperature
sweep of the mean O-O spacing): the open ice sits near ~3.35 A O-O spacing and
collapses to a denser ~3.0 A liquid on melting, with the density maximum
landing in a water-like few-hundred-kelvin range (the dial marks a nominal
freezing point; the live O-O spacing / density readout is the honest signal).

The interactive puller is itself a water molecule you steer by position AND, as
in the lipid demo, by ORIENTATION (joystick twist / Q-E keys rotate it): line
its arms up with the network's dangling arms and feel the hydrogen bonds catch.
"""
import math
import os
import tempfile

import numpy as np
from lammps import lammps

from .base import ForceFeedbackProfile, MDSystem, SliderSpec, SystemSpec

ARM_LEN = 1.85                 # Angstrom, core->tip (the hydrogen-bond reach)
ICE_BOND = 2 * ARM_LEN - 0.35  # ~3.35 A open-ice O-O spacing (opposing tips nearly meet)
TIP_HB_MIN = 0.35              # cosine/squared minimum (bonded tip-tip distance)
TIP_HB_CUT = 1.00             # narrow HB range -> a bond needs arms nearly collinear
CORE_SIGMA = 2.60             # core excluded-volume diameter (sets the dense-liquid spacing)
CORE_EPS = 0.40               # eV -- stiff core repulsion (must beat the HB pull, or cores collapse)
HB_EPS = 0.15                 # eV -- hydrogen-bond well depth
CORETIP_SIGMA = 1.15          # core-tip repulsion range (small, so a bonded tip isn't pried off)
CORETIP_EPS = 0.08

CORE_MASS = 14.0              # amu (the O)
TIP_MASS = (18.0 - CORE_MASS) / 3.0   # so a whole molecule masses ~18 (water)

NX_CELLS = 6                  # honeycomb cluster size (cells per side; ~2*NX^2 molecules)
CLUSTER_PAD = 14.0           # vacuum margin around the cluster: kept just wide enough to
                              # expand and maneuver the puller (~19 A clear above it), so the
                              # ice fills ~60% of the box rather than looking lost in it
PULLER_GAP = 6.0             # start height of the control molecule above the cluster

SETTLE_STEPS = 800
TIMESTEP = 0.001             # ps
LANGEVIN_DAMP = 1.0          # ps -- rigid-body Langevin relaxation (the implicit bath)

PULLER_DAMPING_DEFAULT = 0.02   # eV*ps/Angstrom^2
PULLER_DAMPING_MIN = 0.0
PULLER_DAMPING_MAX = 0.2

# Temperature dial. The open ice is stable at low T and collapses to a denser
# liquid by a few hundred K; the mark sits at water's nominal 273 K freezing
# point (the model's transition is gradual, as 2D transitions are -- the live
# O-O spacing readout is the trustworthy signal, like the RDF marks elsewhere).
T_MIN = 1.0
T_MAX = 600.0
T_MELT = 273.0
# Default: solid, so you first see the gorgeous open hydrogen-bonded ice.
T_DEFAULT = 160.0

# Above this mean O-O spacing the network reads as open ice; below it, as dense
# liquid (used only for the phase word in the HUD).
ICE_SPACING_THRESHOLD = 3.18

YAW_RATE = 3.0     # rad/s at full twist deflection

RDF_NBINS = 100
RDF_CUTOFF = 3.5 * ICE_BOND
RDF_AVE_EVERY = 5
RDF_AVE_REPEAT = 40
RDF_AVE_FREQ = RDF_AVE_EVERY * RDF_AVE_REPEAT

# Species (get_all_positions -> spec.species_colors/radii): 0 = core (O),
# 1 = arm tip (the hydrogen-bonding H direction).
CORE_COLOR = (95, 155, 235)     # water blue
ARM_COLOR = (205, 214, 230)     # pale hydrogen arms
SP_CORE, SP_TIP = 0, 1

# Hydrogen-bond forces are soft (like the lipid membrane's), so an argon-like
# gentle force-feedback profile with enough authority to push a molecule in.
FORCE_FEEDBACK = ForceFeedbackProfile(
    ff_exaggeration=4.0,
    ff_knee=0.3,
    ff_max_mag=120.0,
    stiffness_threshold=0.03,
    stiffness_knee=0.2,
    damper_min_fraction=0.10,
    damper_max_fraction=0.50,
    vel_damp_max_fraction=0.5,
)

SPEC = SystemSpec(
    key="mb_water",
    name="Mercedes-Benz water (ice floats)",
    description="2D hydrogen-bonded water -- heat the open ice and watch it COLLAPSE to denser liquid (why ice floats). Q/E or twist rotates your molecule.",
    element_label="water (O + H-arms)",
    lattice_spacing=TIP_HB_MIN,   # not used for a bond overlay (off); kept informational
    timestep=TIMESTEP,
    temperature=SliderSpec("Temperature", T_MIN, T_MAX, T_DEFAULT, fmt="{:.0f}", unit=" K"),
    damping=SliderSpec("Puller damping", PULLER_DAMPING_MIN, PULLER_DAMPING_MAX,
                        PULLER_DAMPING_DEFAULT, fmt="{:.3f}"),
    melt_temp=T_MELT,
    force_feedback=FORCE_FEEDBACK,
    max_input_force=2.0,   # eV/A at full deflection, shared by joystick/WASD/mouse
    puller_speed_cap=0.1 * ICE_BOND / TIMESTEP,
    species_colors=(CORE_COLOR, ARM_COLOR),
    species_radii_A=(1.05, 0.5),
    bond_overlay=False,   # draws its own arms (get_bond_pairs) + H-bonds (get_hbond_pairs)
)


def _honeycomb_cores(nx, ny, bond):
    """Open-honeycomb O positions with nearest-neighbor (H-bonded) spacing
    `bond` -- the ideal ice lattice."""
    a = bond * math.sqrt(3.0)
    a1 = np.array([a, 0.0])
    a2 = np.array([a / 2.0, a * math.sqrt(3.0) / 2.0])
    basis = [np.zeros(2), (a1 + a2) / 3.0]
    return np.array([i * a1 + j * a2 + b for j in range(ny) for i in range(nx) for b in basis])


class MBWaterSystem(MDSystem):
    spec = SPEC

    def __init__(self):
        self.lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
        self._input_fx = 0.0
        self._input_fy = 0.0
        self._target_angle = -math.pi / 2   # start pointing down, toward the cluster
        self._target_temp = None
        self._last_id_to_index = {}
        self._build()
        self.set_input_force(0.0, 0.0)
        self._interactive_ps = 0.0

    # ---- construction -------------------------------------------------------

    def _molecule_atoms(self, cx, cy, angles, mol, aid):
        """One molecule's 4 atoms (core + 3 arm tips) as data-file tuples."""
        atoms = [(aid + 1, mol, 1, cx, cy)]
        for k, th in enumerate(angles):
            atoms.append((aid + 2 + k, mol, 2, cx + ARM_LEN * math.cos(th),
                          cy + ARM_LEN * math.sin(th)))
        return atoms

    def _write_ice_data(self, path):
        """Ideal open hydrogen-bonded honeycomb ice cluster (each molecule's
        arms aimed at its neighbors), plus one control molecule (the puller)
        parked in the vacuum above. Returns (Lx, Ly, n_water)."""
        cores = _honeycomb_cores(NX_CELLS, NX_CELLS, ICE_BOND)
        cores -= cores.mean(axis=0)   # centre the cluster on the origin for now
        atoms = []
        mol = 0
        aid = 0
        for c in cores:
            mol += 1
            d = np.hypot(cores[:, 0] - c[0], cores[:, 1] - c[1])
            order = np.argsort(d)
            nb = [k for k in order[1:] if d[k] < 1.4 * ICE_BOND][:3]
            angs = [math.atan2(cores[k][1] - c[1], cores[k][0] - c[0]) for k in nb]
            while len(angs) < 3:   # edge molecule: fill remaining arms 120 apart
                base = angs[0] if angs else math.pi / 2
                angs.append(base + len(angs) * 2 * math.pi / 3)
            atoms += self._molecule_atoms(c[0], c[1], angs[:3], mol, aid)
            aid += 4
        n_water = mol

        # Control molecule (puller), above the cluster.
        top = cores[:, 1].max()
        self._puller_mol = mol + 1
        pcx, pcy = 0.0, top + PULLER_GAP
        base = self._target_angle
        atoms += self._molecule_atoms(pcx, pcy, [base + k * 2 * math.pi / 3 for k in range(3)],
                                      self._puller_mol, aid)
        self.puller_ids = tuple(range(aid + 1, aid + 5))
        self.puller_core_id = aid + 1

        xs = [a[3] for a in atoms]
        ys = [a[4] for a in atoms]
        lo = min(min(xs), min(ys)) - CLUSTER_PAD
        hi = max(max(xs), max(ys)) + CLUSTER_PAD
        # Shift everything into a 0-based square box: the renderer maps sim
        # coordinates assuming the box's lower-left corner is the origin.
        shift = -lo
        size = hi - lo
        with open(path, "w") as f:
            f.write("2D Mercedes-Benz water\n\n")
            f.write(f"{len(atoms)} atoms\n2 atom types\n\n")
            f.write(f"0.0 {size} xlo xhi\n0.0 {size} ylo yhi\n-0.1 0.1 zlo zhi\n\n")
            f.write(f"Masses\n\n1 {CORE_MASS}\n2 {TIP_MASS}\n\n")
            f.write("Atoms # molecular\n\n")
            for a in atoms:
                f.write(f"{a[0]} {a[1]} {a[2]} {a[3] + shift:.4f} {a[4] + shift:.4f} 0.0\n")
        self._rest_core = (pcx + shift, pcy + shift)
        return 0.0, size, n_water

    def _build(self):
        lmp = self.lmp
        fd, datafile = tempfile.mkstemp(suffix=".data", prefix="mbwater_")
        os.close(fd)
        try:
            lo, hi, n_water = self._write_ice_data(datafile)
            lmp.command("dimension 2")
            lmp.command("units metal")
            lmp.command("atom_style molecular")
            lmp.command("boundary p p p")
            lmp.command(f"read_data {datafile}")
        finally:
            os.remove(datafile)

        self.xlo = self.ylo = lo
        self.xhi = self.yhi = hi
        self.n_water = n_water
        self.n_water_atoms = 4 * n_water

        lmp.command(f"pair_style cosine/squared {CORE_SIGMA}")
        # core-core excluded volume; tip-tip hydrogen bond; core-tip repulsion.
        lmp.command(f"pair_coeff 1 1 {CORE_EPS} {CORE_SIGMA} {CORE_SIGMA} wca")
        lmp.command(f"pair_coeff 2 2 {HB_EPS} {TIP_HB_MIN} {TIP_HB_CUT}")
        lmp.command(f"pair_coeff 1 2 {CORETIP_EPS} {CORETIP_SIGMA} {CORETIP_SIGMA} wca")
        # A molecule's own beads are held rigid, not by pair forces -- exclude
        # intramolecular pairs so they neither cost time nor pollute energies.
        lmp.command("neigh_modify exclude molecule/intra all")
        lmp.command("neighbor 2.0 bin")
        lmp.command("neigh_modify every 1 delay 0 check yes")
        lmp.command(f"comm_modify cutoff {RDF_CUTOFF + 4.0}")

        # Groups: the real water (rigid bodies + Langevin bath) vs the control
        # molecule (the puller, integrated + rigidified by hand each frame).
        plist = " ".join(str(i) for i in self.puller_ids)
        lmp.command(f"group puller id {plist}")
        lmp.command("group water subtract all puller")

        lmp.command("fix twod all enforce2d")
        self._seed = 573921
        self._target_temp = T_DEFAULT
        # Rigid bodies, one per molecule, with their own Langevin thermostat ==
        # the temperature dial (this IS the implicit-solvent bath).
        lmp.command(
            f"fix integ_w water rigid/small molecule langevin {T_DEFAULT} {T_DEFAULT} "
            f"{LANGEVIN_DAMP} {self._seed}"
        )
        # The cluster floats free in vacuum, and the per-body Langevin bath gives
        # every molecule an independent random kick -- whose sum is a slow random
        # walk of the whole cluster's momentum, so left alone the ice block drifts
        # (and spins) bodily across the box, most visibly right after a
        # temperature change re-seeds the bath. Periodically zeroing the water's
        # net linear and angular momentum pins the cluster in place -- it still
        # freely expands, collapses and tumbles internally (that's the physics on
        # display), it just no longer sails away. (This does take on rigid bodies;
        # verified the drift drops from several Angstrom to a fraction of one.)
        lmp.command("fix recenter_w water momentum 10 linear 1 1 0 angular")
        # Puller: capped-velocity nve on its beads + the viscous damping dial;
        # its rigid shape/orientation are re-imposed every frame.
        lmp.command(f"fix integ_p puller nve/limit {0.1 * ICE_BOND}")
        self._puller_damping = PULLER_DAMPING_DEFAULT
        lmp.command(f"fix damp_p puller viscous {PULLER_DAMPING_DEFAULT}")

        # Net force from the water on the control molecule (reaction arrow / FF).
        lmp.command("compute iff puller group/group water")
        lmp.command("compute w_temp water temp")

        lmp.command(f"timestep {TIMESTEP}")
        lmp.command("thermo 100000")
        self._rigidify_puller()
        lmp.command(f"run {SETTLE_STEPS}")
        self._rigidify_puller()

        lmp.command("compute ke_atom all ke/atom")
        lmp.command("compute pe_atom all pe/atom")
        # Core-core g(r): the sharp open-ice shells broadening as it melts.
        lmp.command(f"compute rdf_raw water rdf {RDF_NBINS} 1 1 cutoff {RDF_CUTOFF}")
        lmp.command(
            f"fix rdf_avg water ave/time {RDF_AVE_EVERY} {RDF_AVE_REPEAT} {RDF_AVE_FREQ} "
            f"c_rdf_raw[*] mode vector"
        )
        self._rdf_bins_ready = False
        self._rdf_r = None
        self._rdf_ready_step = lmp.extract_global("ntimestep") + RDF_AVE_FREQ + RDF_AVE_EVERY

    # ---- puller orientation (rigid control molecule) ------------------------

    def _puller_local_indices(self):
        nlocal = self.lmp.get_natoms()
        ids = self.lmp.numpy.extract_atom("id")[:nlocal]
        idx = {int(i): k for k, i in enumerate(ids)}
        return [idx.get(i) for i in self.puller_ids], nlocal

    def _rigidify_puller(self):
        """Keep the control molecule a rigid 3-arm star: core at the beads' COM,
        the three arm tips at ARM_LEN and 120 degrees apart, aligned to the
        user-steered director angle. The COM is left free (so it still responds
        to input force, the water's reaction and the damping dial); only the
        internal shape/orientation are overwritten, and bead velocities are set
        to the common COM velocity so no spurious spin is injected."""
        idxs, nlocal = self._puller_local_indices()
        if any(i is None for i in idxs):
            return
        x = self.lmp.numpy.extract_atom("x")
        v = self.lmp.numpy.extract_atom("v")
        pts = np.array([x[i][:2] for i in idxs])
        com = pts.mean(axis=0)
        comv = np.array([v[i][:2] for i in idxs]).mean(axis=0)
        ic, i1, i2, i3 = idxs
        x[ic][0], x[ic][1] = com[0], com[1]
        for k, i in enumerate((i1, i2, i3)):
            th = self._target_angle + k * 2 * math.pi / 3
            x[i][0] = com[0] + ARM_LEN * math.cos(th)
            x[i][1] = com[1] + ARM_LEN * math.sin(th)
        for i in idxs:
            v[i][0], v[i][1] = comv[0], comv[1]

    def steer_orientation(self, rate, dt):
        self._target_angle += rate * YAW_RATE * dt

    def puller_bead_count(self):
        return 4   # core + 3 arm tips

    # ---- controls -----------------------------------------------------------

    def set_input_force(self, fx, fy):
        # Applied to each of the 4 beads; net force over total mass accelerates
        # the molecule's COM as a single body would (same feel as the others).
        self._input_fx = fx
        self._input_fy = fy
        self.lmp.command(f"fix input_force puller addforce {fx} {fy} 0.0")

    def set_target_temp(self, T):
        T = max(T_MIN, min(T_MAX, T))
        if T == self._target_temp:
            return
        self._target_temp = T
        self.lmp.command(
            f"fix integ_w water rigid/small molecule langevin {T} {T} "
            f"{LANGEVIN_DAMP} {self._seed}"
        )

    def set_puller_damping(self, gamma):
        gamma = max(PULLER_DAMPING_MIN, min(PULLER_DAMPING_MAX, gamma))
        if gamma == self._puller_damping:
            return
        self._puller_damping = gamma
        self.lmp.command(f"fix damp_p puller viscous {gamma}")

    def step(self, n=4):
        self.lmp.command(f"run {n}")
        self._interactive_ps += n * TIMESTEP
        self._rigidify_puller()

    # ---- readouts -----------------------------------------------------------

    def get_sim_time(self):
        return self._interactive_ps

    def _puller_com(self):
        idxs, nlocal = self._puller_local_indices()
        if any(i is None for i in idxs):
            return None, None, None
        x = self.lmp.numpy.extract_atom("x")[:nlocal]
        v = self.lmp.numpy.extract_atom("v")[:nlocal]
        pts = np.array([x[i][:2] for i in idxs])
        vel = np.array([v[i][:2] for i in idxs])
        return pts.mean(axis=0).copy(), vel.mean(axis=0).copy(), idxs

    def get_puller_state(self):
        com, comv, _ = self._puller_com()
        if com is None:
            return None, None
        return com, comv

    def get_puller_energy(self):
        com, comv, idxs = self._puller_com()
        if com is None:
            return None, None
        nlocal = self.lmp.get_natoms()
        ke = self.lmp.numpy.extract_compute("ke_atom", 1, 1)[:nlocal]
        pe = self.lmp.numpy.extract_compute("pe_atom", 1, 1)[:nlocal]
        return float(sum(ke[i] for i in idxs)), float(sum(pe[i] for i in idxs))

    def get_interaction_force(self):
        vec = self.lmp.extract_compute("iff", 0, 1)
        return np.array([vec[0], vec[1]])

    def get_thermo_state(self):
        temp = self.lmp.extract_compute("w_temp", 0, 0)
        press = self.lmp.get_thermo("press")
        ke = self.lmp.get_thermo("ke")
        pe = self.lmp.get_thermo("pe")
        etotal = self.lmp.get_thermo("etotal")
        return temp, press, ke, pe, etotal

    def get_rdf(self):
        step = self.lmp.extract_global("ntimestep")
        if step < self._rdf_ready_step:
            return None
        if not self._rdf_bins_ready:
            self._rdf_r = np.array(
                [self.lmp.extract_fix("rdf_avg", 0, 2, nrow=i, ncol=0) for i in range(RDF_NBINS)]
            )
            self._rdf_bins_ready = True
        g = np.array(
            [self.lmp.extract_fix("rdf_avg", 0, 2, nrow=i, ncol=1) for i in range(RDF_NBINS)]
        )
        return self._rdf_r, g

    def get_all_positions(self):
        nlocal = self.lmp.get_natoms()
        xs = self.lmp.numpy.extract_atom("x")[:nlocal]
        ids = self.lmp.numpy.extract_atom("id")[:nlocal].copy()
        types = self.lmp.numpy.extract_atom("type")[:nlocal]
        species = (types - 1).astype(int)   # 0 = core, 1 = tip
        is_puller = np.isin(ids, self.puller_ids)
        self._last_id_to_index = {int(i): k for k, i in enumerate(ids)}
        return ids, xs[:, :2].copy(), is_puller, species

    def _water_core_positions(self):
        """Current O positions of the real (non-puller) water molecules."""
        nlocal = self.lmp.get_natoms()
        xs = self.lmp.numpy.extract_atom("x")[:nlocal]
        ids = self.lmp.numpy.extract_atom("id")[:nlocal]
        types = self.lmp.numpy.extract_atom("type")[:nlocal]
        mask = (types == 1) & ~np.isin(ids, self.puller_ids)
        return xs[mask][:, :2]

    def _mean_oo_spacing(self):
        cores = self._water_core_positions()
        if len(cores) < 2:
            return ICE_BOND
        d = np.hypot(cores[:, 0, None] - cores[None, :, 0],
                     cores[:, 1, None] - cores[None, :, 1])
        np.fill_diagonal(d, np.inf)
        return float(d.min(axis=1).mean())

    def get_bond_pairs(self):
        """Every molecule's three core->tip arms, as sticks (the Mercedes star).
        Indices into the current get_all_positions ordering."""
        idx = self._last_id_to_index
        if not idx:
            return None
        pairs = []
        # Water molecules: ids 1..4*n_water in (core, t1, t2, t3) groups of 4.
        for m in range(self.n_water):
            base = 4 * m + 1
            ic = idx.get(base)
            if ic is None:
                continue
            for t in range(1, 4):
                it = idx.get(base + t)
                if it is not None:
                    pairs.append((ic, it))
        pc = idx.get(self.puller_core_id)
        if pc is not None:
            for t in range(1, 4):
                it = idx.get(self.puller_core_id + t)
                if it is not None:
                    pairs.append((pc, it))
        return np.array(pairs, dtype=int) if pairs else None

    def get_hbond_pairs(self):
        """Hydrogen bonds: tip-tip pairs from DIFFERENT molecules whose tips have
        come together (arms pointing at each other). Indices into the current
        get_all_positions ordering."""
        nlocal = self.lmp.get_natoms()
        xs = self.lmp.numpy.extract_atom("x")[:nlocal]
        ids = self.lmp.numpy.extract_atom("id")[:nlocal]
        types = self.lmp.numpy.extract_atom("type")[:nlocal]
        tips = np.where(types == 2)[0]
        if len(tips) < 2:
            return None
        tp = xs[tips][:, :2]
        # molecule id of each tip = (id-1)//4  (4 atoms per molecule, ids from 1)
        tmol = ((ids[tips] - 1) // 4).astype(int)
        d = np.hypot(tp[:, 0, None] - tp[None, :, 0], tp[:, 1, None] - tp[None, :, 1])
        iu, ju = np.triu_indices(len(tips), k=1)
        close = d[iu, ju] < (TIP_HB_MIN + 0.6)
        diffmol = tmol[iu] != tmol[ju]
        sel = close & diffmol
        pairs = [(int(tips[a]), int(tips[b])) for a, b in zip(iu[sel], ju[sel])]
        return np.array(pairs, dtype=int) if pairs else None

    def get_hud_lines(self):
        oo = self._mean_oo_spacing()
        # density relative to the open ice reference (area ~ spacing^2)
        rel = (ICE_BOND / oo) ** 2
        phase = "open, ICE-like network (low density)" if oo > ICE_SPACING_THRESHOLD \
            else "dense, LIQUID-like (collapsed)"
        return [
            "Rotate your molecule (Q/E or twist) to line its arms up and catch hydrogen bonds.",
            f"mean O-O spacing: {oo:.2f} A   density: {rel:4.2f}x the open ice   -> {phase}",
            "Heat the ice: the open hydrogen-bond network COLLAPSES to denser liquid -- why ice floats.",
        ]

    def get_box_size(self):
        return self.xhi - self.xlo, self.yhi - self.ylo

    def close(self):
        self.lmp.close()

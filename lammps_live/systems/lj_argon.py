"""2D Lennard-Jones argon: the other canonical MD teaching system, alongside
metallic Cu (see cu_deposition.py) -- soft van-der-Waals bonding instead of
EAM's coordination-hungry metallic bonding, so the same "pull an atom onto a
cold crystal and watch it stick" interaction feels noticeably different:
weaker, softer contact forces (~0.01-0.5 eV/A vs. copper's ~0.1-6 eV/A) and
a much lower melting point. No external potential file is needed (`lj/cut`
is a builtin LAMMPS pair style), which is what makes this a good second
system to demonstrate the codebase's system/ plugin structure with.

epsilon/sigma/mass are literally argon's textbook LJ parameters (e.g.
Rahman 1964): sigma = 3.40 A, epsilon/kB = 120.7 K, mass = 39.948 amu -- so
T_MELT below isn't an arbitrary dial mark the way copper's is, it's just
real argon's melting point (~84 K), which these parameters were originally
fit to reproduce in 3D.

Unlike copper's constants (see that module's docstring), the constants
below *were* empirically checked here, the same way -- not just guessed:
- lattice spacing: pressure-swept a periodic 2D-hex bulk (bisection on
  `get_thermo("press")`) and found the zero-pressure spacing directly,
  same method as copper's. Came out to ~1.113*sigma (compressed from the
  bare pair-potential minimum of 2^(1/6)*sigma ~= 1.122*sigma, same
  lattice-sum effect that compresses 3D FCC LJ crystals).
- timestep: LJ's bare r^-12 repulsive core is numerically stiffer at hard
  contact than EAM's smoother embedding-function repulsion, so copper's
  timestep (0.001 ps) is NOT safe here -- a deposition impact integrated at
  that timestep or even 0.0005 ps visibly *gains* energy over a few hundred
  steps after contact (checked by watching puller speed for several ps
  post-impact) and the puller never actually sticks, even with the
  Langevin bath draining energy from the rest of the crystal. 0.0002 ps
  (5x smaller) was the largest tested value that reliably stayed stable
  through a real deposition impact.
- puller damping default: swept viscous gamma at the above timestep,
  depositing a real 1 eV atom (same calibration energy as copper's),
  and picked a value from the range that reliably settles into a stuck,
  non-oscillating state.
- Langevin measured-vs-setpoint calibration: same DOF-accounting mismatch
  copper's has (see LANGEVIN_TARGET_CALIB there) -- measured directly here
  rather than assumed equal, and came out close (~1.45 vs copper's ~1.5),
  consistent with it being a geometry effect rather than a potential-
  specific one.
"""
import math
import random

import numpy as np
from lammps import lammps

from .base import ForceFeedbackProfile, MDSystem, SliderSpec, SystemSpec

EPSILON = 0.0104   # eV -- argon's LJ well depth (epsilon/kB ~= 120.7 K)
SIGMA = 3.40       # Angstrom -- argon's LJ length scale
CUTOFF = 2.5 * SIGMA
AR_MASS = 39.948   # amu

LATTICE_SPACING = 3.784884  # Angstrom; empirically-found 2D-hex zero-pressure spacing (see module docstring)
LATTICE_N = 16
ROW_HEIGHT = LATTICE_SPACING * math.sqrt(3) / 2
ROW_EPS = 0.1 * ROW_HEIGHT
CRYSTAL_ROWS = 10
FLOOR_ROWS = 2
PULLER_GAP = 3 * LATTICE_SPACING
SETTLE_STEPS = 600
TIMESTEP = 0.0002  # ps -- 5x smaller than copper's; see module docstring

PULLER_DAMPING_DEFAULT = 0.0015  # eV*ps/Angstrom^2
PULLER_DAMPING_MIN = 0.0
PULLER_DAMPING_MAX = 0.005

T_MIN = 0.0
T_MAX = 400.0     # K -- well past melting, into a clearly gas-like RDF
T_MELT = 84.0     # K -- real argon's melting point (these are its actual LJ parameters, not a dial guess)
LANGEVIN_TARGET_CALIB = 1.45  # measured directly for this geometry, see module docstring
LANGEVIN_DAMP = 0.1  # ps

RDF_NBINS = 100
RDF_CUTOFF = 4.0 * LATTICE_SPACING
RDF_AVE_EVERY = 5
RDF_AVE_REPEAT = 40
RDF_AVE_FREQ = RDF_AVE_EVERY * RDF_AVE_REPEAT

# Forces here run roughly two orders of magnitude weaker than EAM copper's
# (~0.01-0.5 eV/A vs. ~0.1-6 eV/A), so every force-feedback knob is scaled
# down to match -- reusing copper's would read as permanently "numb".
FORCE_FEEDBACK = ForceFeedbackProfile(
    input_force_scale=0.3,
    ff_exaggeration=4.0,
    ff_knee=0.08,
    ff_max_mag=120.0,
    stiffness_threshold=0.005,
    stiffness_knee=0.05,
    damper_min_fraction=0.10,
    damper_max_fraction=0.50,
    vel_damp_max_fraction=0.5,
)

SPEC = SystemSpec(
    key="lj_argon",
    name="Argon melting (Lennard-Jones)",
    description="A softer, weaker-bonded 2D crystal -- same deposition interaction, real argon LJ parameters.",
    element_label="Ar (LJ)",
    lattice_spacing=LATTICE_SPACING,
    timestep=TIMESTEP,
    temperature=SliderSpec("Temperature", T_MIN, T_MAX, T_MIN, fmt="{:.0f}", unit=" K"),
    damping=SliderSpec("Puller damping", PULLER_DAMPING_MIN, PULLER_DAMPING_MAX,
                        PULLER_DAMPING_DEFAULT, fmt="{:.5f}"),
    melt_temp=T_MELT,
    force_feedback=FORCE_FEEDBACK,
    puller_speed_cap=0.05 * LATTICE_SPACING / TIMESTEP,
)


class LJArgonSystem(MDSystem):
    """Structurally identical to CopperEAMSystem (same region/group/fix
    layout) -- only the pair style, lattice constant, timestep, and tuned
    constants differ. See cu_deposition.py for the layout's rationale."""

    spec = SPEC

    def __init__(self):
        self.lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
        self._build()
        self.set_input_force(0.0, 0.0)
        self._interactive_ps = 0.0  # elapsed sim time since settle finished, see get_sim_time

    def _build(self):
        lmp = self.lmp
        lmp.command("dimension 2")
        lmp.command("units metal")
        lmp.command("atom_style atomic")
        lmp.command("boundary p f p")
        lmp.command(f"lattice hex {LATTICE_SPACING}")
        box_size = LATTICE_N * LATTICE_SPACING
        lmp.command(
            f"region simbox block 0 {box_size} 0 {box_size} "
            f"{-0.25 * LATTICE_SPACING} {0.25 * LATTICE_SPACING} units box"
        )
        lmp.command("create_box 1 simbox")

        boxlo, boxhi, *_ = lmp.extract_box()
        self.xlo, self.ylo = boxlo[0], boxlo[1]
        self.xhi, self.yhi = boxhi[0], boxhi[1]

        crystal_top = self.ylo + CRYSTAL_ROWS * ROW_HEIGHT + ROW_EPS
        lmp.command(
            f"region crystal block {self.xlo} {self.xhi} {self.ylo} {crystal_top} -0.25 0.25 units box"
        )
        lmp.command("create_atoms 1 region crystal")
        self.n_crystal = lmp.get_natoms()

        puller_x = (self.xlo + self.xhi) / 2
        puller_y = crystal_top + PULLER_GAP
        self.rest_pos = (puller_x, puller_y)
        self.puller_id = self.n_crystal + 1
        lmp.command(f"create_atoms 1 single {puller_x} {puller_y} 0.0 units box")

        lmp.command(f"mass 1 {AR_MASS}")
        lmp.command(f"pair_style lj/cut {CUTOFF}")
        lmp.command(f"pair_coeff 1 1 {EPSILON} {SIGMA}")

        lmp.command("neighbor 1.0 bin")
        lmp.command("neigh_modify every 1 delay 0 check yes")
        lmp.command(f"comm_modify cutoff {RDF_CUTOFF + 2.0}")

        lmp.command(f"group puller id {self.puller_id}")
        lmp.command("group crystal subtract all puller")
        lmp.command(
            f"region floor_region block INF INF {self.ylo} {self.ylo + FLOOR_ROWS * ROW_HEIGHT + ROW_EPS} "
            f"INF INF units box"
        )
        lmp.command("group floor region floor_region")
        lmp.command("group crystal_mobile subtract crystal floor")
        lmp.command("group mobile union puller crystal_mobile")

        lmp.command("compute ljforce puller group/group crystal")

        lmp.command("fix freeze floor setforce 0.0 0.0 0.0")
        lmp.command("fix integ_crystal crystal_mobile nve")
        self._seed = random.randint(1, 900_000_000)
        self._target_temp = T_MIN
        lmp.command(
            f"fix damp_crystal crystal_mobile langevin {T_MIN / LANGEVIN_TARGET_CALIB} "
            f"{T_MIN / LANGEVIN_TARGET_CALIB} {LANGEVIN_DAMP} {self._seed} zero yes"
        )
        # Tighter displacement cap than copper's (0.05*a vs 0.1*a): LJ's
        # stiffer repulsive core needs it, same reasoning as the smaller
        # timestep above (see module docstring).
        lmp.command(f"fix integ_puller puller nve/limit {0.05 * LATTICE_SPACING}")
        self._puller_damping = PULLER_DAMPING_DEFAULT
        lmp.command(f"fix damp_puller puller viscous {PULLER_DAMPING_DEFAULT}")
        lmp.command(
            f"fix walls mobile wall/reflect ylo {self.ylo + 0.5 * LATTICE_SPACING} "
            f"yhi {self.yhi - 0.5 * LATTICE_SPACING} units box"
        )

        lmp.command(f"timestep {TIMESTEP}")
        lmp.command("thermo 100000")

        lmp.command(f"run {SETTLE_STEPS}")
        lmp.command("velocity mobile set 0.0 0.0 0.0")

        lmp.command("compute crystal_temp crystal_mobile temp")
        lmp.command("compute ke_atom all ke/atom")
        lmp.command("compute pe_atom all pe/atom")

        lmp.command(f"compute rdf_raw crystal rdf {RDF_NBINS} 1 1 cutoff {RDF_CUTOFF}")
        lmp.command(
            f"fix rdf_avg crystal ave/time {RDF_AVE_EVERY} {RDF_AVE_REPEAT} {RDF_AVE_FREQ} "
            f"c_rdf_raw[*] mode vector"
        )
        self._rdf_bins_ready = False
        self._rdf_r = None
        self._rdf_ready_step = lmp.extract_global("ntimestep") + RDF_AVE_FREQ + RDF_AVE_EVERY

    def set_input_force(self, fx, fy):
        self.lmp.command(f"fix input_force puller addforce {fx} {fy} 0.0")

    def set_target_temp(self, T):
        T = max(T_MIN, min(T_MAX, T))
        if T == self._target_temp:
            return
        self._target_temp = T
        setpoint = T / LANGEVIN_TARGET_CALIB
        self.lmp.command(
            f"fix damp_crystal crystal_mobile langevin {setpoint} {setpoint} "
            f"{LANGEVIN_DAMP} {self._seed} zero yes"
        )

    def set_puller_damping(self, gamma):
        gamma = max(PULLER_DAMPING_MIN, min(PULLER_DAMPING_MAX, gamma))
        if gamma == self._puller_damping:
            return
        self._puller_damping = gamma
        self.lmp.command(f"fix damp_puller puller viscous {gamma}")

    def step(self, n=4):
        self.lmp.command(f"run {n}")
        self._interactive_ps += n * TIMESTEP

    def get_sim_time(self):
        return self._interactive_ps

    def get_puller_state(self):
        nlocal = self.lmp.get_natoms()
        ids = self.lmp.numpy.extract_atom("id")[:nlocal]
        idx = np.where(ids == self.puller_id)[0]
        if len(idx) == 0:
            return None, None
        i = int(idx[0])
        xs = self.lmp.numpy.extract_atom("x")[:nlocal]
        vs = self.lmp.numpy.extract_atom("v")[:nlocal]
        return xs[i][:2].copy(), vs[i][:2].copy()

    def get_puller_energy(self):
        nlocal = self.lmp.get_natoms()
        ids = self.lmp.numpy.extract_atom("id")[:nlocal]
        idx = np.where(ids == self.puller_id)[0]
        if len(idx) == 0:
            return None, None
        i = int(idx[0])
        ke = self.lmp.numpy.extract_compute("ke_atom", 1, 1)[:nlocal]
        pe = self.lmp.numpy.extract_compute("pe_atom", 1, 1)[:nlocal]
        return float(ke[i]), float(pe[i])

    def get_interaction_force(self):
        vec = self.lmp.extract_compute("ljforce", 0, 1)
        return np.array([vec[0], vec[1]])

    def get_thermo_state(self):
        temp = self.lmp.extract_compute("crystal_temp", 0, 0)
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
        ids = self.lmp.numpy.extract_atom("id")[:nlocal]
        return xs[:, :2].copy(), (ids == self.puller_id)

    def get_box_size(self):
        return self.xhi - self.xlo, self.yhi - self.ylo

    def close(self):
        self.lmp.close()

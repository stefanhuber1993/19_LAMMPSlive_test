"""2D copper deposition, modeled on the classic MD demo image: a single Cu
atom deposited on a cold Cu(001) surface sticks rather than bounces off,
because its kinetic energy redistributes into the crystal via attractive
interatomic forces. Reduced to 2D (a cross-section, as in the reference
figure) using the real EAM potential for copper (Cu_u3.eam, bundled with
LAMMPS) rather than a toy pair potential.

The puller atom is the same element as the crystal (identified by atom ID,
not a separate LAMMPS atom type), matching the real "Cu on Cu" scenario, and
is under continuous interactive control instead of a single ballistic shot.

Units are LAMMPS "metal" (eV, Angstrom, ps, amu) -- real physical scales,
not reduced LJ units. All physical constants below (lattice spacing, viscous
damping) were empirically tuned in isolated tests, not guessed:
- lattice spacing: swept density in a periodic bulk (no free surface) and
  found the true zero-pressure equilibrium for 2D EAM Cu is a ~= 2.4605 A.
  The lattice is hexagonal (triangular, 6 in-plane neighbors), not the
  naive square cross-section of 3D FCC(001): a true one-atom-thick 2D layer
  has no out-of-plane bonds to brace a square arrangement the way the bulk
  crystal does, and EAM's isotropic, coordination-hungry bonding has no
  preference for 90 degree bond angles -- it just maximizes neighbor count,
  so the 2D energy minimum is the close-packed hexagonal lattice. Starting
  from a square lattice is only a local, metastable configuration: it
  visibly reorganizes into hexagonal within seconds once perturbed by
  deposition impacts. Starting from hex directly avoids that transient and
  matches the true 2D ground state.
- viscous damping: swept coefficients depositing a real 1 eV atom (the
  energy quoted in the reference image) and picked the smallest value that
  reliably dissipates the impact into a stuck, non-oscillating state.
"""
import math
import os
import random

import numpy as np
from lammps import lammps

from .base import ForceFeedbackProfile, MDSystem, SliderSpec, SystemSpec

LATTICE_SPACING = 2.4605  # Angstrom; empirically-found 2D-hex EAM Cu equilibrium (see module docstring)
LATTICE_N = 20             # box size, in units of LATTICE_SPACING (real Angstrom, not lattice-command units)
# hex atom rows are spaced a*sqrt(3)/2 apart (see module docstring). Cutting
# the crystal/floor regions at an arbitrary fraction of the box height lands
# mid-row for hex, leaving a ragged, partially-populated top/bottom row that
# then has to visibly relax into place over the first several frames. Sizing
# both cuts in whole rows keeps every row fully populated from frame 0.
ROW_HEIGHT = LATTICE_SPACING * math.sqrt(3) / 2
ROW_EPS = 0.1 * ROW_HEIGHT   # margin so a region bound lands cleanly between rows, not exactly on one
CRYSTAL_ROWS = 12          # bottom rows of the box filled with crystal (~half, row-aligned)
FLOOR_ROWS = 2              # bottom rows of the crystal frozen as a floor
PULLER_GAP = 3 * LATTICE_SPACING       # real units above the crystal surface where puller starts
SETTLE_STEPS = 600          # pre-roll steps run silently in __init__ before the render loop starts (see _build)
CU_MASS = 63.55            # amu
TIMESTEP = 0.001           # ps

# Puller's own velocity-proportional drag -- unrelated to the crystal's
# thermostat below, this is what makes dragging the puller around feel
# "heavy" or "slippery", exposed live as the game's damping slider. Default
# matches the old single VISCOUS_GAMMA value, empirically tuned by
# depositing a real 1 eV atom (see module docstring); min/max bracket a
# comfortable slider range around it.
PULLER_DAMPING_DEFAULT = 0.01   # eV*ps/Angstrom^2
PULLER_DAMPING_MIN = 0.0        # frictionless -- fix viscous with gamma=0 is a legal no-op
PULLER_DAMPING_MAX = 0.05

# The crystal is thermostatted with a Langevin bath (friction + random
# noise) instead of plain viscous drag, so it can be heated/cooled on demand
# and so atoms visibly jitter around their lattice sites at T>0 -- both a
# physically real effect of the Langevin noise term, not a cosmetic add-on.
#
# T_MIN/T_MAX/T_MELT are deliberately round, approximate numbers, not a
# rigorously-derived phase boundary: a melting-point scan run standalone
# (periodic-bulk RDF heating sweep + Langevin calibration, see repo history)
# showed 2D crystals don't have a clean melting knee the way 3D ones do
# (Mermin-Wagner long-wavelength fluctuations smear it out), and a small
# defect-free periodic crystal superheats well past any realistic transition
# anyway. T_MELT is kept as a rough, clearly-labeled dial mark -- the live
# RDF panel heating from sharp peaks to broad humps as you push past it is
# the actual, trustworthy signal, not this constant.
T_MIN = 0.0         # K -- a Langevin fix at Tstart=Tstop=0 is just pure friction (no noise term), a legal quench
T_MAX = 6000.0      # K
T_MELT = 1000.0     # K -- approximate dial marker, see note above
# Empirically, this group's measured (compute temp) temperature runs about
# 1.5x the Langevin fix's Tstart/Tstop setpoint in this geometry (a mobile
# group temp compute alongside frozen floor/puller groups counts degrees of
# freedom slightly differently than the fix's own internal accounting) --
# corrected here so T_MIN..T_MAX above are what actually gets measured and
# displayed, not what's silently fed to the fix.
LANGEVIN_TARGET_CALIB = 1.5
LANGEVIN_DAMP = 0.1  # ps -- thermostat relaxation time, fixed (not user-adjustable)

# RDF: time-averaged (not a single noisy snapshot -- ~250 atoms is too few
# for that alone) over a short rolling window so it still reads as "live".
RDF_NBINS = 100
RDF_CUTOFF = 4.0 * LATTICE_SPACING
RDF_AVE_EVERY = 5     # sample every N steps
RDF_AVE_REPEAT = 40   # samples per average
RDF_AVE_FREQ = RDF_AVE_EVERY * RDF_AVE_REPEAT  # window length in steps; must output on this cadence

POTENTIAL_FILE = os.path.join(os.path.dirname(__file__), "data", "Cu_u3.eam")

# Force-feedback tuning: EAM Cu contact forces run ~0.1-6 eV/A. The knee is
# deliberately small so a light touch already reads as "pulled toward the
# crystal", not just a hard push (see main app's force-shaping module).
FORCE_FEEDBACK = ForceFeedbackProfile(
    input_force_scale=2.0,
    ff_exaggeration=4.0,
    ff_knee=1.5,
    ff_max_mag=120.0,
    stiffness_threshold=0.05,
    stiffness_knee=0.5,
    damper_min_fraction=0.10,
    damper_max_fraction=0.50,
    vel_damp_max_fraction=0.5,
)

SPEC = SystemSpec(
    key="cu_eam",
    name="Copper deposition (EAM)",
    description="A Cu atom pulled onto a cold Cu(001)-like 2D crystal -- sticks via real metallic bonding (EAM).",
    element_label="Cu (EAM)",
    lattice_spacing=LATTICE_SPACING,
    timestep=TIMESTEP,
    temperature=SliderSpec("Temperature", T_MIN, T_MAX, T_MIN, fmt="{:.0f}", unit=" K"),
    damping=SliderSpec("Puller damping", PULLER_DAMPING_MIN, PULLER_DAMPING_MAX,
                        PULLER_DAMPING_DEFAULT, fmt="{:.4f}"),
    melt_temp=T_MELT,
    force_feedback=FORCE_FEEDBACK,
    puller_speed_cap=0.1 * LATTICE_SPACING / TIMESTEP,
)


class CopperEAMSystem(MDSystem):
    spec = SPEC

    def __init__(self):
        self.lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
        self._build()
        self.set_input_force(0.0, 0.0)

    def _build(self):
        lmp = self.lmp
        lmp.command("dimension 2")
        lmp.command("units metal")
        lmp.command("atom_style atomic")
        lmp.command("boundary p f p")
        # hex: 2D close-packed (triangular) lattice, 6 in-plane neighbors --
        # see module docstring for why this (not a square cross-section) is
        # the true 2D ground state. Its unit cell is rectangular (width a,
        # height a*sqrt(3)), so the box is sized explicitly in real Angstrom
        # ("units box") rather than in lattice-command units, to keep the
        # simulation box itself square regardless of that ratio.
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

        lmp.command(f"mass 1 {CU_MASS}")
        lmp.command("pair_style eam")
        lmp.command(f"pair_coeff 1 1 {POTENTIAL_FILE}")

        lmp.command("neighbor 1.0 bin")
        lmp.command("neigh_modify every 1 delay 0 check yes")
        # compute rdf's cutoff (see near end of _build) needs ghost atoms
        # communicated out further than the EAM pair cutoff + skin alone
        # provides, or it errors out ("cutoff plus skin exceeds ghost atom
        # range") the first time it's invoked.
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

        # Isolated puller<->crystal interaction force, independent of
        # whatever else is pushing the puller (input force, damping).
        lmp.command("compute ljforce puller group/group crystal")

        lmp.command("fix freeze floor setforce 0.0 0.0 0.0")
        lmp.command("fix integ_crystal crystal_mobile nve")
        # Langevin thermostat (friction + noise) instead of plain viscous
        # drag: lets the crystal be heated/cooled on demand and gives it
        # real thermal jitter at T>0. Tstart/Tstop don't accept a variable
        # reference in this fix ("Expected floating point parameter instead
        # of 'v_Tsetpoint'"), so set_target_temp redefines the fix with a
        # literal value instead, same pattern as set_puller_damping.
        self._seed = random.randint(1, 900_000_000)
        self._target_temp = T_MIN
        lmp.command(
            f"fix damp_crystal crystal_mobile langevin {T_MIN / LANGEVIN_TARGET_CALIB} "
            f"{T_MIN / LANGEVIN_TARGET_CALIB} {LANGEVIN_DAMP} {self._seed} zero yes"
        )
        # nve/limit (not plain nve) for the puller: under a sustained user
        # input force with the deliberately-weak, realistic viscous damping
        # below, terminal velocity (F/gamma) can reach ~300 A/ps -- enough
        # to tunnel through the wall/reflect boundary and the neighbor skin
        # in one step (observed: "Lost atoms" crash under a steady push).
        # Capping per-step displacement bounds velocity without touching the
        # physically-tuned damping that makes deposition look right.
        lmp.command(f"fix integ_puller puller nve/limit {0.1 * LATTICE_SPACING}")
        self._puller_damping = PULLER_DAMPING_DEFAULT
        lmp.command(f"fix damp_puller puller viscous {PULLER_DAMPING_DEFAULT}")
        # Both the puller AND ordinary crystal atoms need this: y is a
        # non-periodic "f" boundary, which does not wrap or reflect on its
        # own -- an atom that drifts past it is simply lost, which is a
        # fatal "Lost atoms" error that crashes the whole run. A hard puller
        # impact can eject a surface atom fast enough to punch through the
        # (unwalled) top before viscous damping catches it, so the wall has
        # to cover all mobile atoms, not just the puller.
        lmp.command(
            f"fix walls mobile wall/reflect ylo {self.ylo + 0.5 * LATTICE_SPACING} "
            f"yhi {self.yhi - 0.5 * LATTICE_SPACING} units box"
        )

        lmp.command(f"timestep {TIMESTEP}")
        lmp.command("thermo 100000")

        # Even starting from the true 2D-hex ground state, a freshly-cut
        # free surface is a small perturbation -- close-packed lattices have
        # a very low barrier to rigid interlayer shear (the same
        # low-Peierls-stress property that makes real FCC metals ductile),
        # so the crystal takes one quick collective "hop" to its true
        # relaxed registry (measured: done within ~200 steps, coordination
        # numbers unchanged throughout -- it's a shear, not melting) before
        # settling for good. Run that relaxation here, before the render
        # loop's first frame, so it happens silently instead of visibly.
        lmp.command(f"run {SETTLE_STEPS}")
        lmp.command("velocity mobile set 0.0 0.0 0.0")

        # Instantaneous temperature of the thermalized crystal only --
        # excludes the frozen floor (permanently at rest, would deflate an
        # "all group" reading) and the puller (user-driven, not part of the
        # thermostatted bath).
        lmp.command("compute crystal_temp crystal_mobile temp")

        # Per-atom KE/PE so the puller's own speed/energy can be read off
        # individually -- LAMMPS handles the amu*(Angstrom/ps)^2 -> eV
        # conversion internally, more reliable than reimplementing the
        # mvv2e factor by hand.
        lmp.command("compute ke_atom all ke/atom")
        lmp.command("compute pe_atom all pe/atom")

        # RDF g(r), time-averaged over a short rolling window since a single
        # ~250-atom snapshot alone is too noisy to read as a phase
        # fingerprint. "crystal" (not "mobile") excludes the puller so a
        # single passing atom doesn't skew the histogram.
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

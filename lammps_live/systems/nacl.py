"""2D sodium chloride: an ionic crystal, alongside metallic Cu (EAM, see
cu_deposition.py) and van-der-Waals Ar (Lennard-Jones, lj_argon.py). Instead
of neutral atoms bound by embedding energy or dispersion, this is a lattice of
alternating Na(+) cations and Cl(-) anions held together by long-range
Coulomb (Madelung) attraction balanced against a short-range Born-Mayer
repulsion -- the classic "rigid-ion" model of an alkali halide.

Starting configuration -- why a SQUARE (checkerboard) lattice, not hexagonal.
The other two systems crystallize on a 2D-hex (triangular, close-packed)
lattice because their bonding just maximizes neighbor count. Ionic bonding
does the opposite: every ion wants its *nearest* neighbors to be the opposite
charge and its like-charge neighbors pushed to the next shell. That requires a
lattice you can two-color so that no nearest-neighbor bond joins like charges
-- i.e. a bipartite lattice. The square lattice is bipartite: color it like a
checkerboard and every ion is surrounded by 4 nearest neighbors of the
opposite sign (attraction) with the 4 like-charge neighbors held farther out
on the diagonal. This is exactly the 2D analog of rock-salt and is a genuine
Madelung energy minimum. A triangular/hex lattice, by contrast, is NOT
bipartite -- its odd (3-membered) rings are geometrically frustrated, so an
alternating +/- assignment is impossible and any arrangement leaves like
charges in nearest contact. So hex, the stable choice for the neutral systems,
is precisely the wrong choice here; the checkerboard square lattice is the one
that is actually stable in 2D. It is built explicitly below (a custom 4-site
lattice, two sublattices per species) so the alternation -- and overall charge
neutrality, which the Coulomb solver needs -- is exact from frame 0.

Units are LAMMPS "metal" (eV, Angstrom, ps, amu, charge in electrons). All
constants below were checked here the same empirical way the argon ones were
(see that module), not guessed:
- Coulomb: damped shifted-force (Fennell-Gezelter 2006, "coul/dsf"), NOT an
  Ewald/PPPM k-space sum. DSF gives a smooth, energy-conserving, real-space-
  only Coulomb that needs no reciprocal-space part -- which matters because
  this is a 2D, non-periodic-in-y slab, exactly the geometry k-space solvers
  handle worst (they need the slab correction and a periodic 3rd dimension).
- spacing: pressure-swept a periodic 2D checkerboard bulk (no free surface)
  and took the zero-pressure nearest-neighbor distance, same bisection method
  as argon. Came out to ~2.89 A with the Born parameters below -- close to
  real NaCl's 2.82 A nearest-neighbor distance, and giving a ~-3.5 eV/ion
  cohesive energy in the right ballpark for an alkali halide.
- Born-Mayer repulsion A*exp((sigma-r)/rho): rho = 0.32 A is the standard
  alkali-halide hardness length; A and sigma were then set to place that
  zero-pressure spacing at the value above (a bare Coulomb + point repulsion
  otherwise collapses the lattice, since with sigma=0 the exponential is
  negligible at contact -- checked).
- timestep: 0.0002 ps, the same 5x-smaller-than-copper value argon needs --
  the strong, stiff ionic contact forces (interaction forces peak ~6 eV/A,
  comparable to EAM copper) were verified to integrate a real deposition
  impact stably at this step without gaining energy or losing atoms.
- vacuum gap below the crystal: unlike the neutral systems, the ionic crystal
  is placed a couple of rows ABOVE the fixed lower box boundary rather than
  sitting on it. A bare ionic (001) surface relaxes outward strongly, and the
  bottom row sitting exactly on the non-periodic y=0 boundary would relax
  straight out of the box (observed: instant "Lost atoms"). A small vacuum gap
  gives it room to relax in place.

The puller is a Na(+) ion (same as one sublattice), so it is pulled onto the
Cl(-) sites electrostatically -- the ionic analog of the Cu-on-Cu deposition.
"""
import math
import random

import numpy as np
from lammps import lammps

from .base import ForceFeedbackProfile, MDSystem, SliderSpec, SystemSpec

# Born-Mayer repulsion V_rep = A*exp((sigma-r)/rho); C=D=0 -> pure exponential
# core, added to the DSF Coulomb. See module docstring for how these were set.
BORN_A = 1.0        # eV prefactor
BORN_RHO = 0.32     # Angstrom -- standard alkali-halide hardness length
BORN_SIGMA = 2.4    # Angstrom -- repulsion onset (roughly the ion contact size)
DSF_ALPHA = 0.25    # 1/Angstrom -- damped shifted-force screening parameter
COUL_CUTOFF = 12.0  # Angstrom -- real-space Coulomb/pair cutoff

NA_CHARGE = 1.0     # e -- cation (type 1)
CL_CHARGE = -1.0    # e -- anion  (type 2)
NA_MASS = 22.99     # amu
CL_MASS = 35.45     # amu

# Per-species render colors (see SystemSpec.species_colors): 0 = Na+ cation,
# 1 = Cl- anion. Warm for the cation, cool for the anion.
NACL_CATION_COLOR = (235, 205, 90)
NACL_ANION_COLOR = (90, 190, 235)

# Nearest-neighbor (opposite-charge) spacing -- the empirically-found 2D
# checkerboard zero-pressure distance (see module docstring). This is the
# lattice's optimal bonding distance, reported to the UI (bond-line overlay).
LATTICE_SPACING = 2.892     # Angstrom
# Rows of the checkerboard are spaced one nearest-neighbor distance apart in y,
# each row alternating +-+- along x (so every full row is charge-neutral).
ROW_HEIGHT = LATTICE_SPACING
ROW_EPS = 0.1 * ROW_HEIGHT
# The alternating pattern's repeat cell is 2x2 nearest-neighbor spacings; the
# box is sized in whole cells so periodic-x wrapping preserves the checkerboard
# (and thus exact neutrality) across the seam.
CELL = 2 * LATTICE_SPACING
LATTICE_N = 8              # box size, in whole checkerboard cells
GAP_ROWS = 2               # vacuum rows between the box's fixed bottom and the crystal (see docstring)
CRYSTAL_ROWS = 6          # rows of the box filled with crystal
FLOOR_ROWS = 0             # bottom crystal rows frozen as a floor
PULLER_GAP = 3 * LATTICE_SPACING       # start height above the crystal surface
SETTLE_STEPS = 600
TIMESTEP = 0.0002           # ps -- see module docstring

PULLER_DAMPING_DEFAULT = 0.02   # eV*ps/Angstrom^2
PULLER_DAMPING_MIN = 0.0
PULLER_DAMPING_MAX = 0.1

# Thermostat: canonical velocity rescaling (Bussi et al. 2007, "temp/csvr"),
# identical in spirit to the other two systems -- see cu_deposition.py's
# thermostat note for the full rationale (no per-atom random forcing, cools
# cleanly to 0 K, bound to the displayed temperature compute so no
# setpoint-vs-measured fudge factor is needed).
T_MIN = 1.0        # K
T_MAX = 3000.0     # K -- well past melting, into a clearly disordered/ionic-melt RDF
T_MELT = 1074.0    # K -- real NaCl's melting point (dial marker; the 2D model's own
                    # disordering, visible in the live RDF, is the trustworthy signal)
THERMOSTAT_DAMP = 0.1  # ps -- relaxation time for total KE toward target
COLD_SEED_TEMP = 10.0   # K -- below this the lattice is at rest; heating from here is seeded, not rescaled
# Bulk-drift handling (see cu_deposition.py for the full rationale).
DRIFT_DAMP_PER_FRAME = 0.03
QUENCH_ZERO_DROP_FRAC = 0.3

RDF_NBINS = 100
RDF_CUTOFF = 4.0 * LATTICE_SPACING
RDF_AVE_EVERY = 5
RDF_AVE_REPEAT = 40
RDF_AVE_FREQ = RDF_AVE_EVERY * RDF_AVE_REPEAT

# Ionic contact forces run at a scale comparable to EAM copper's (~0.1-6 eV/A,
# measured), so the force-feedback tuning is copper-like rather than argon's
# much softer profile -- with a touch more input authority to work against the
# strong electrostatic pull.
FORCE_FEEDBACK = ForceFeedbackProfile(
    input_force_scale=3.0,
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
    key="nacl",
    name="Salt crystal (ionic, NaCl)",
    description="A 2D Na(+)/Cl(-) checkerboard held by Coulomb (Madelung) bonding -- pull an ion onto the lattice.",
    element_label="NaCl (ionic)",
    lattice_spacing=LATTICE_SPACING,
    timestep=TIMESTEP,
    temperature=SliderSpec("Temperature", T_MIN, T_MAX, T_MIN, fmt="{:.0f}", unit=" K"),
    damping=SliderSpec("Puller damping", PULLER_DAMPING_MIN, PULLER_DAMPING_MAX,
                        PULLER_DAMPING_DEFAULT, fmt="{:.4f}"),
    melt_temp=T_MELT,
    force_feedback=FORCE_FEEDBACK,
    puller_speed_cap=0.05 * LATTICE_SPACING / TIMESTEP,
    species_colors=(NACL_CATION_COLOR, NACL_ANION_COLOR),  # 0=Na+, 1=Cl-
    species_labels=("+", "-"),
)


class NaClSystem(MDSystem):
    """Ionic-crystal system. Same region/group/fix skeleton as the other two
    systems (see cu_deposition.py for the layout rationale); the differences
    are all ionic: a charged atom_style, two atom types on a bipartite
    checkerboard lattice, a Born-Mayer + damped-shifted-force Coulomb pair
    style, and -- because that pair style has no compute group/group support --
    an interaction force reconstructed from the puller's total force rather
    than measured by a group/group compute.
    """

    spec = SPEC

    def __init__(self):
        self.lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
        self._input_fx = 0.0
        self._input_fy = 0.0
        self._build()
        self.set_input_force(0.0, 0.0)
        self._interactive_ps = 0.0  # elapsed sim time since settle finished, see get_sim_time

    def _build(self):
        lmp = self.lmp
        lmp.command("dimension 2")
        lmp.command("units metal")
        # charge atom_style: ions carry a per-atom electrostatic charge, which
        # the Coulomb pair style reads. (The neutral systems use "atomic".)
        lmp.command("atom_style charge")
        lmp.command("boundary p f p")
        # Explicit 4-site checkerboard lattice: a 2x2-spacing cell whose two
        # even-parity sites become type 1 (Na+) and two odd-parity sites type 2
        # (Cl-). Building the alternation into the lattice itself (rather than a
        # simple square lattice recolored afterwards) guarantees exact +/-
        # alternation and exact charge neutrality per cell. Coordinates are in
        # units of LATTICE_SPACING (the nearest-neighbor distance).
        lmp.command(
            f"lattice custom {LATTICE_SPACING} a1 2 0 0 a2 0 2 0 "
            f"basis 0 0 0 basis 0.5 0.5 0 basis 0.5 0 0 basis 0 0.5 0"
        )
        box_size = LATTICE_N * CELL
        lmp.command(
            f"region simbox block 0 {box_size} 0 {box_size} "
            f"{-0.25 * LATTICE_SPACING} {0.25 * LATTICE_SPACING} units box"
        )
        lmp.command("create_box 2 simbox")

        boxlo, boxhi, *_ = lmp.extract_box()
        self.xlo, self.ylo = boxlo[0], boxlo[1]
        self.xhi, self.yhi = boxhi[0], boxhi[1]

        # Crystal sits a couple of rows above the fixed bottom boundary (see the
        # "vacuum gap" note in the module docstring) -- not on it, as the
        # neutral crystals do.
        crystal_bot = self.ylo + GAP_ROWS * ROW_HEIGHT + ROW_EPS
        crystal_top = crystal_bot + CRYSTAL_ROWS * ROW_HEIGHT + ROW_EPS
        lmp.command(
            f"region crystal block {self.xlo} {self.xhi} {crystal_bot} {crystal_top} "
            f"-0.25 0.25 units box"
        )
        # basis-index -> atom-type map: the two even-parity basis atoms (1,2)
        # are Na+ (type 1), the two odd-parity ones (3,4) are Cl- (type 2).
        lmp.command("create_atoms 1 region crystal basis 1 1 basis 2 1 basis 3 2 basis 4 2")
        self.n_crystal = lmp.get_natoms()

        # Puller is a Na+ ion (type 1), started above the surface. Placed on a
        # column that sits over a Cl- site so it is drawn straight down onto the
        # lattice electrostatically.
        puller_x = (self.xlo + self.xhi) / 2
        puller_y = crystal_top + PULLER_GAP
        self.rest_pos = (puller_x, puller_y)
        self.puller_id = self.n_crystal + 1
        lmp.command(f"create_atoms 1 single {puller_x} {puller_y} 0.0 units box")

        lmp.command(f"mass 1 {NA_MASS}")
        lmp.command(f"mass 2 {CL_MASS}")
        lmp.command(f"set type 1 charge {NA_CHARGE}")
        lmp.command(f"set type 2 charge {CL_CHARGE}")
        # Born-Mayer repulsion + damped shifted-force Coulomb, one pair-coeff
        # for all type combinations (the C/r^6, D/r^8 dispersion terms are
        # zeroed -- pure exponential repulsion). See module docstring.
        lmp.command(f"pair_style born/coul/dsf {DSF_ALPHA} {COUL_CUTOFF}")
        lmp.command(f"pair_coeff * * {BORN_A} {BORN_RHO} {BORN_SIGMA} 0.0 0.0")

        lmp.command("neighbor 2.0 bin")
        lmp.command("neigh_modify every 1 delay 0 check yes")
        lmp.command(f"comm_modify cutoff {max(RDF_CUTOFF, COUL_CUTOFF) + 2.0}")

        lmp.command(f"group puller id {self.puller_id}")
        lmp.command("group crystal subtract all puller")
        lmp.command(
            f"region floor_region block INF INF {self.ylo} "
            f"{crystal_bot + FLOOR_ROWS * ROW_HEIGHT} INF INF units box"
        )
        lmp.command("group floor region floor_region")
        lmp.command("group crystal_mobile subtract crystal floor")
        lmp.command("group mobile union puller crystal_mobile")

        lmp.command("fix freeze floor setforce 0.0 0.0 0.0")
        lmp.command("fix integ_crystal crystal_mobile nve")
        # Crystal temperature (COM-subtracted), excluding puller and floor --
        # see cu_deposition.py. Defined before the thermostat so csvr can be
        # bound to this exact compute.
        lmp.command("compute crystal_temp crystal_mobile temp/com")
        lmp.command("variable vcmx equal vcm(crystal_mobile,x)")
        lmp.command("variable vcmy equal vcm(crystal_mobile,y)")
        self._seed = random.randint(1, 900_000_000)
        self._target_temp = T_MIN
        lmp.command(
            f"fix damp_crystal crystal_mobile temp/csvr {T_MIN} {T_MIN} "
            f"{THERMOSTAT_DAMP} {self._seed}"
        )
        lmp.command("fix_modify damp_crystal temp crystal_temp")
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

        lmp.command("compute ke_atom all ke/atom")
        lmp.command("compute pe_atom all pe/atom")

        # Total (all-pairs) g(r) of the crystal ions -- the alternating shell
        # structure of the ionic lattice broadening into a liquid-like hump as
        # it melts. "crystal" excludes the puller.
        lmp.command(f"compute rdf_raw crystal rdf {RDF_NBINS} cutoff {RDF_CUTOFF}")
        lmp.command(
            f"fix rdf_avg crystal ave/time {RDF_AVE_EVERY} {RDF_AVE_REPEAT} {RDF_AVE_FREQ} "
            f"c_rdf_raw[*] mode vector"
        )
        self._rdf_bins_ready = False
        self._rdf_r = None
        self._rdf_ready_step = lmp.extract_global("ntimestep") + RDF_AVE_FREQ + RDF_AVE_EVERY

    def set_input_force(self, fx, fy):
        # Stored as well as applied: get_interaction_force reconstructs the
        # crystal's force on the puller by subtracting these applied forces from
        # the puller's total force (this pair style has no group/group compute).
        self._input_fx = fx
        self._input_fy = fy
        self.lmp.command(f"fix input_force puller addforce {fx} {fy} 0.0")

    def set_target_temp(self, T):
        T = max(T_MIN, min(T_MAX, T))
        if T == self._target_temp:
            return
        self._target_temp = T
        # Seed a Maxwell-Boltzmann distribution when heating from an
        # effectively-frozen lattice; otherwise let csvr rescale existing
        # motion. See cu_deposition.py for the full explanation.
        current = self.lmp.extract_compute("crystal_temp", 0, 0)
        if T > current and current < COLD_SEED_TEMP:
            self._seed = random.randint(1, 900_000_000)
            self.lmp.command(
                f"velocity crystal_mobile create {T} {self._seed} "
                f"mom yes rot yes dist gaussian"
            )
        elif current > COLD_SEED_TEMP and T < current - QUENCH_ZERO_DROP_FRAC * current:
            self.lmp.command("velocity crystal_mobile zero linear")
        self.lmp.command(
            f"fix damp_crystal crystal_mobile temp/csvr {T} {T} "
            f"{THERMOSTAT_DAMP} {self._seed}"
        )
        self.lmp.command("fix_modify damp_crystal temp crystal_temp")

    def set_puller_damping(self, gamma):
        gamma = max(PULLER_DAMPING_MIN, min(PULLER_DAMPING_MAX, gamma))
        if gamma == self._puller_damping:
            return
        self._puller_damping = gamma
        self.lmp.command(f"fix damp_puller puller viscous {gamma}")

    def step(self, n=4):
        self.lmp.command(f"run {n}")
        self._interactive_ps += n * TIMESTEP
        self._damp_drift()

    def _damp_drift(self):
        if DRIFT_DAMP_PER_FRAME <= 0.0:
            return
        vx = self.lmp.extract_variable("vcmx")
        vy = self.lmp.extract_variable("vcmy")
        f = DRIFT_DAMP_PER_FRAME
        self.lmp.command(
            f"velocity crystal_mobile set {-f * vx} {-f * vy} 0.0 sum yes units box"
        )

    def get_sim_time(self):
        return self._interactive_ps

    def _puller_index(self):
        nlocal = self.lmp.get_natoms()
        ids = self.lmp.numpy.extract_atom("id")[:nlocal]
        idx = np.where(ids == self.puller_id)[0]
        return (int(idx[0]) if len(idx) else None), nlocal

    def get_puller_state(self):
        i, nlocal = self._puller_index()
        if i is None:
            return None, None
        xs = self.lmp.numpy.extract_atom("x")[:nlocal]
        vs = self.lmp.numpy.extract_atom("v")[:nlocal]
        return xs[i][:2].copy(), vs[i][:2].copy()

    def get_puller_energy(self):
        i, nlocal = self._puller_index()
        if i is None:
            return None, None
        ke = self.lmp.numpy.extract_compute("ke_atom", 1, 1)[:nlocal]
        pe = self.lmp.numpy.extract_compute("pe_atom", 1, 1)[:nlocal]
        return float(ke[i]), float(pe[i])

    def get_interaction_force(self):
        # born/coul/dsf does not support compute group/group, so the crystal's
        # force on the puller is reconstructed instead: the puller's only
        # interactions are pairwise with the crystal, so its total per-atom
        # force is (pair force from crystal) + (the input force we add via
        # addforce) + (its own viscous drag, -gamma*v). Subtract the two forces
        # we applied to recover the pure interaction force:
        #     f_pair = f_total - f_input + gamma * v
        i, nlocal = self._puller_index()
        if i is None:
            return np.zeros(2)
        f = self.lmp.numpy.extract_atom("f")[:nlocal]
        v = self.lmp.numpy.extract_atom("v")[:nlocal]
        g = self._puller_damping
        fx = f[i][0] - self._input_fx + g * v[i][0]
        fy = f[i][1] - self._input_fy + g * v[i][1]
        return np.array([fx, fy])

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
        # species 0 = Na+ (type 1), 1 = Cl- (type 2); see SPEC.species_colors.
        types = self.lmp.numpy.extract_atom("type")[:nlocal]
        species = (types - 1).astype(int)
        return ids.copy(), xs[:, :2].copy(), (ids == self.puller_id), species

    def get_box_size(self):
        return self.xhi - self.xlo, self.yhi - self.ylo

    def close(self):
        self.lmp.close()

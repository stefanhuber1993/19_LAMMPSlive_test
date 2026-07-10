"""
LAMMPS-backed 2D deposition simulation, matching the classic MD demo: a
single Cu atom deposited on a cold Cu(001) surface (Knordlun-style image) --
kinetic energy redistributes into the crystal and the atom sticks rather
than bouncing off, due to attractive interatomic forces. Reduced to 2D (a
cross-section, as in the reference figure) using the real EAM potential for
copper (Cu_u3.eam, bundled with LAMMPS) rather than a toy pair potential.

The puller atom is controlled interactively (mouse/joystick) instead of
being a single ballistic 1 eV shot -- same underlying physics, continuous
control. It's the same element as the crystal (identified by atom ID, not a
separate type), matching the real "Cu on Cu" scenario.

Units are LAMMPS "metal" (eV, Angstrom, ps, amu) -- real physical scales,
not reduced LJ units. All physical constants below (lattice spacing,
viscous damping) were empirically tuned in isolated tests, not guessed:
- lattice spacing: swept density in a periodic bulk (no free surface) and
  found the true zero-pressure equilibrium for 2D EAM Cu is a ~= 2.4605 A.
  The lattice is hexagonal (triangular, 6 in-plane neighbors), not the
  naive square cross-section of 3D FCC(001): a true one-atom-thick 2D
  layer has no out-of-plane bonds to brace a square arrangement the way
  the bulk crystal does, and EAM's isotropic, coordination-hungry bonding
  has no preference for 90 degree bond angles -- it just maximizes
  neighbor count, so the 2D energy minimum is the close-packed hexagonal
  lattice. Starting from a square lattice (as this demo used to) is only
  a local, metastable configuration: it visibly reorganizes into hexagonal
  within seconds once perturbed by deposition impacts. Starting from hex
  directly avoids that transient and matches the true 2D ground state.
- viscous damping: swept coefficients depositing a real 1 eV atom (the
  energy quoted in the reference image) and picked the smallest value that
  reliably dissipates the impact into a stuck, non-oscillating state.
"""
import os
import math
import numpy as np
from lammps import lammps

LATTICE_SPACING = 2.4605  # Angstrom; empirically-found 2D-hex EAM Cu equilibrium (see module docstring)
LATTICE_N = 20             # box size, in units of LATTICE_SPACING (real Angstrom, not lattice-command units)
# hex atom rows are spaced a*sqrt(3)/2 apart (see module docstring). Cutting
# the crystal/floor regions at an arbitrary fraction of the box height (as
# the old square-lattice version did) lands mid-row for hex, leaving a
# ragged, partially-populated top/bottom row that then has to visibly
# relax into place over the first several frames. Sizing both cuts in
# whole rows keeps every row fully populated from frame 0.
ROW_HEIGHT = LATTICE_SPACING * math.sqrt(3) / 2
ROW_EPS = 0.1 * ROW_HEIGHT   # margin so a region bound lands cleanly between rows, not exactly on one
CRYSTAL_ROWS = 12          # bottom rows of the box filled with crystal (~half, row-aligned)
FLOOR_ROWS = 2              # bottom rows of the crystal frozen as a floor
PULLER_GAP = 3 * LATTICE_SPACING       # real units above the crystal surface where puller starts
SETTLE_STEPS = 600          # pre-roll steps run silently in __init__ before the render loop starts (see _build)
CU_MASS = 63.55            # amu
VISCOUS_GAMMA = 0.01       # eV*ps/Angstrom^2 -- see module docstring
TIMESTEP = 0.001           # ps

POTENTIAL_FILE = os.path.join(os.path.dirname(__file__), "Cu_u3.eam")


class CrystalSim:
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
        # height a*sqrt(3)), so the box is sized explicitly in real
        # Angstrom ("units box") rather than in lattice-command units, to
        # keep the simulation box itself square regardless of that ratio.
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
        lmp.command(f"fix damp_crystal crystal_mobile viscous {VISCOUS_GAMMA}")
        # nve/limit (not plain nve) for the puller: under a sustained user
        # input force with the deliberately-weak, realistic viscous damping
        # above, terminal velocity (F/gamma) can reach ~300 A/ps -- enough
        # to tunnel through the wall/reflect boundary and the neighbor skin
        # in one step (observed: "Lost atoms" crash under a steady push).
        # Capping per-step displacement bounds velocity without touching the
        # physically-tuned damping that makes deposition look right.
        lmp.command(f"fix integ_puller puller nve/limit {0.1 * LATTICE_SPACING}")
        lmp.command(f"fix damp_puller puller viscous {VISCOUS_GAMMA}")
        # Both the puller AND ordinary crystal atoms need this: y is a
        # non-periodic "f" boundary, which does not wrap or reflect on its
        # own -- an atom that drifts past it is simply lost, which is a
        # fatal "Lost atoms" error that crashes the whole run. A hard
        # puller impact can eject a surface atom fast enough to punch
        # through the (unwalled) top before viscous damping catches it, so
        # the wall has to cover all mobile atoms, not just the puller.
        lmp.command(
            f"fix walls mobile wall/reflect ylo {self.ylo + 0.5 * LATTICE_SPACING} "
            f"yhi {self.yhi - 0.5 * LATTICE_SPACING} units box"
        )

        lmp.command(f"timestep {TIMESTEP}")
        lmp.command("thermo 100000")

        # Even starting from the true 2D-hex ground state, a freshly-cut
        # free surface is a small perturbation -- close-packed lattices
        # have a very low barrier to rigid interlayer shear (the same
        # low-Peierls-stress property that makes real FCC metals ductile),
        # so the crystal takes one quick collective "hop" to its true
        # relaxed registry (measured: done within ~200 steps, coordination
        # numbers unchanged throughout -- it's a shear, not melting) before
        # settling for good. Run that relaxation here, before the render
        # loop's first frame, so it happens silently instead of visibly.
        lmp.command(f"run {SETTLE_STEPS}")
        lmp.command("velocity mobile set 0.0 0.0 0.0")

    def set_input_force(self, fx, fy):
        """Joystick/mouse-commanded force on the puller, in eV/Angstrom (0 at center)."""
        self.lmp.command(f"fix input_force puller addforce {fx} {fy} 0.0")

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

    def get_interaction_force(self):
        """Net EAM force on the puller from the crystal (excludes input/damping)."""
        vec = self.lmp.extract_compute("ljforce", 0, 1)
        return np.array([vec[0], vec[1]])

    def get_all_positions(self):
        """Returns (positions Nx2, is_puller boolarray N) for rendering."""
        nlocal = self.lmp.get_natoms()
        xs = self.lmp.numpy.extract_atom("x")[:nlocal]
        ids = self.lmp.numpy.extract_atom("id")[:nlocal]
        return xs[:, :2].copy(), (ids == self.puller_id)

    def close(self):
        self.lmp.close()

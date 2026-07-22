"""2D covalent carbon under oxygen bombardment -- the covalent-bonding member
of the set, alongside metallic Cu (EAM), van-der-Waals Ar (LJ), ionic NaCl,
and the soft molecular lipid/water systems.

What it teaches. Metallic, ionic and van-der-Waals bonds are all essentially
*non-directional*: an atom just wants as many neighbors as close as possible
(hence the close-packed hex crystals of Cu and Ar, or the Madelung
alternation of NaCl). Covalent bonding is the opposite -- a few *strong,
directional* shared-electron bonds at definite angles. The 2D covalent
crystal that expresses this is graphene: each carbon makes exactly three
sigma bonds at 120 degrees, so instead of close-packing it forms an *open
honeycomb*. (True diamond is the 3D sp3 version, 4 bonds tetrahedrally;
graphene is its honest 2D analog -- a one-atom-thick covalent sheet -- so
that is what a 2D "diamond" demo has to be.) The consequences of strong
directional bonds are all on screen:
  - the open honeycomb itself (three-fold, not close-packed, coordination);
  - extreme stiffness / a very high melting mark (covalent solids are the
    hardest, highest-melting materials -- contrast argon's ~84 K);
  - bonds that BREAK rather than flow: pull on the lattice and covalent bonds
    snap at a definite point, leaving reactive *dangling bonds* behind.

The interaction: chemical etching. The puller is an oxygen atom -- a reactive
radical you drive into the surface. Real oxygen plasmas etch carbon by exactly
this route: an O grabs a surface C, and if it can pull that C free (breaking
its covalent bonds to the sheet) the pair leaves as carbon monoxide, CO. So
here O binds a surface carbon (a strong C=O bond) and, if you drag it away hard
enough to snap the carbon's bonds to its neighbors, the carbon is etched off as
CO -- leaving a vacancy (an etch pit) ringed by glowing, under-coordinated
reactive carbons. The CO tally climbs, the surface roughens, and a fresh O is
sent in so you can keep etching. Re-selecting the demo regrows a pristine sheet.

Why this potential. The carbon sheet uses REBO (Brenner's reactive
bond-order potential, `pair_style rebo`, CH.rebo), the standard reactive
carbon potential: it reproduces graphene's cohesive energy (~ -7.4 eV/atom,
checked here) AND lets bonds form and break, which a fixed harmonic network
could never show. Genuine C=O reactive chemistry (making real CO/CO2) would
need a ReaxFF field, but ReaxFF forces LAMMPS into "real" units and an
every-step charge-equilibration solve -- both incompatible with this app's
all-metal-units, 60 fps interactive design -- so the oxygen here binds carbon
through a strong Morse C-O bond (deep enough to abstract a carbon), overlaid on
REBO via `pair_style hybrid/overlay`. That captures the pedagogy -- reactive O
snaps strong covalent bonds and carries carbon off -- in eV and in real time.
All constants are in LAMMPS "metal" units (eV, Angstrom, ps, amu).
"""
import math
import os

import numpy as np
from lammps import lammps

from .base import ForceFeedbackProfile, MDSystem, SliderSpec, SystemSpec


def _rebo_potential_path():
    """Locate CH.rebo in the installed LAMMPS' bundled potentials dir."""
    import lammps as _l
    return os.path.join(os.path.dirname(_l.__file__), "share", "lammps", "potentials", "CH.rebo")


A_GRAPHENE = 2.46          # Angstrom -- graphene lattice constant (C-C bond = a/sqrt(3) ~ 1.42)
CC_BOND = A_GRAPHENE / math.sqrt(3.0)   # ~1.42 Angstrom, the covalent bond length
ROW_HEIGHT = A_GRAPHENE * math.sqrt(3) / 2.0   # y-spacing of honeycomb rows
NX = 10                    # honeycomb cells across (box width = NX*a, periodic in x)
CRYSTAL_ROWS = 7           # rows of sheet built up from the bottom
FLOOR_ROWS = 1             # bottom rows clamped as fixed "bulk"
VACUUM = 34.0              # Angstrom of empty space above the sheet for the beam
PULLER_GAP = 6.0           # start height of the O above the surface
SETTLE_STEPS = 800
TIMESTEP = 0.0002          # ps -- small: covalent bonds are stiff and impacts violent

C_MASS = 12.011            # amu
O_MASS = 15.999            # amu
H_MASS = 1.008             # amu
CH_BOND = 1.09             # Angstrom -- REBO C-H bond (the passivating cap length)

# Reactive O: a strong Morse C-O bond, deep enough to abstract a surface carbon.
# The bond's peak pull (D0*alpha/2 ~ 12 eV/A here) must exceed the input force
# the user applies, or the C-O bond -- not the carbon's C-C bonds -- is what
# yields when they pull; then no carbon is ever extracted. Kept below the real
# ~11 eV C=O well only modestly (the point is grab-and-rip, not exact chemistry).
MORSE_CO_D0 = 10.0         # eV well depth
MORSE_CO_ALPHA = 2.4       # 1/Angstrom
MORSE_CO_R0 = 1.20         # Angstrom -- C=O-like bond length
MORSE_OO_D0 = 0.3          # eV -- weak O-O (rarely relevant; usually one O)
MORSE_OO_ALPHA = 2.0
MORSE_OO_R0 = 1.30
# O-H: REBO covers C and H but not O, so without an explicit O-H term the oxygen
# would fly straight through the hydrogen caps. A near-real O-H Morse (~4.5 eV,
# still below the C-O well) lets the O actually grab a cap and, pulled away, rip
# it off its carbon as OH -- de-passivating the surface into a reactive
# dangling-bond site the O can then attack. Deep enough to abstract H (its peak
# pull must beat the C-H bond it's stealing the H from), shallow enough that a
# carbon, once exposed, is bound far more strongly and is what the user etches.
MORSE_OH_D0 = 4.5          # eV
MORSE_OH_ALPHA = 2.4       # 1/Angstrom
MORSE_OH_R0 = 0.97         # Angstrom -- O-H-like bond length
MORSE_CUTOFF = 4.5         # Angstrom

BOND_CUTOFF = 1.85         # Angstrom -- draw/count a C-C or C-O bond within this
COORD_CUTOFF = 1.9         # Angstrom -- carbon coordination (covalent neighbor count)

PULLER_DAMPING_DEFAULT = 0.03   # eV*ps/Angstrom^2 -- caps the O's approach speed so a
                                # full-scale push seats rather than slams (see FORCE_FEEDBACK)
PULLER_DAMPING_MIN = 0.0
PULLER_DAMPING_MAX = 0.1

# Covalent solids are the highest-melting materials -- the dial reaches far
# higher than the metal/vdW systems, and the "melt" mark sits near graphene's
# ~4800 K sublimation, a deliberate contrast with argon's 84 K.
T_MIN = 1.0
T_MAX = 6000.0
T_MELT = 4800.0
THERMOSTAT_DAMP = 0.2
COLD_SEED_TEMP = 20.0

RDF_NBINS = 100
RDF_CUTOFF = 4.0 * CC_BOND
RDF_AVE_EVERY = 5
RDF_AVE_REPEAT = 40
RDF_AVE_FREQ = RDF_AVE_EVERY * RDF_AVE_REPEAT

# Species indices used by get_all_positions -> spec.species_colors/radii:
#   0 = fully-coordinated (inert, bulk) carbon
#   1 = under-coordinated reactive carbon (dangling bonds: surface + etch damage)
#   2 = oxygen (the reactive projectile / puller)
#   3 = clamped "bulk" carbon (the fixed bottom rows)
#   4 = hydrogen (the passivating cap on the top edge)
SP_C_BULK, SP_C_REACTIVE, SP_O, SP_C_FROZEN, SP_H = 0, 1, 2, 3, 4
C_BULK_COLOR = (120, 126, 138)      # graphite gray
C_REACTIVE_COLOR = (245, 150, 60)   # glowing dangling-bond orange
O_COLOR = (232, 72, 60)             # CPK oxygen red
C_FROZEN_COLOR = (72, 74, 84)       # dim clamped bulk
H_COLOR = (238, 240, 245)           # CPK hydrogen white (the passivating caps)

# Covalent contact/etch forces are very strong (snapping a C-C bond runs to
# ~10+ eV/A), so the profile gives plenty of input authority and a high knee.
# input_force_scale is deliberately large: the sheet is now a pristine, fully
# passivated crystal (stable at rest), so actually etching it -- ripping an
# H-terminated edge carbon free against its REBO C-C bonds -- takes a hard,
# committed pull (~14 eV/A at full deflection, empirically where abstraction
# becomes reliable within a couple of pushes). A gentler scale just heats the
# surface without ever breaking through. The puller's default viscous damping
# (below) is raised to match: it caps the O's approach *speed* so a full-scale
# push seats firmly instead of slamming, while leaving the steady pull force --
# the part that actually snaps bonds -- untouched.
FORCE_FEEDBACK = ForceFeedbackProfile(
    input_force_scale=14.0,
    ff_exaggeration=3.0,
    ff_knee=3.0,
    ff_max_mag=120.0,
    stiffness_threshold=0.1,
    stiffness_knee=1.0,
    damper_min_fraction=0.10,
    damper_max_fraction=0.50,
    vel_damp_max_fraction=0.5,
)

SPEC = SystemSpec(
    key="carbon_etch",
    name="Covalent carbon etch (O bombardment)",
    description="A 2D covalent carbon sheet (graphene, REBO) -- drive a reactive O in, snap covalent bonds and etch carbons off as CO.",
    element_label="C (covalent)",
    lattice_spacing=CC_BOND,   # bond-overlay draws the covalent network; it tears as bonds break
    timestep=TIMESTEP,
    temperature=SliderSpec("Temperature", T_MIN, T_MAX, T_MIN, fmt="{:.0f}", unit=" K"),
    damping=SliderSpec("Puller damping", PULLER_DAMPING_MIN, PULLER_DAMPING_MAX,
                        PULLER_DAMPING_DEFAULT, fmt="{:.4f}"),
    melt_temp=T_MELT,
    force_feedback=FORCE_FEEDBACK,
    puller_speed_cap=0.05 * A_GRAPHENE / TIMESTEP,
    species_colors=(C_BULK_COLOR, C_REACTIVE_COLOR, O_COLOR, C_FROZEN_COLOR, H_COLOR),
    species_radii_A=(0.70, 0.70, 0.66, 0.70, 0.40),
    bond_overlay=True,   # the faint near-1.42A network = the covalent bonds, drawn wrap-safe
)


class CarbonEtchSystem(MDSystem):
    spec = SPEC

    def __init__(self):
        self.lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
        self._input_fx = 0.0
        self._input_fy = 0.0
        self._co_count = 0
        self._h_stripped = 0
        self._build()
        self.set_input_force(0.0, 0.0)
        self._interactive_ps = 0.0

    # ---- construction -------------------------------------------------------

    def _carbon_coordination(self):
        """(coord, pos, ids, dx, dy, neigh) for the current carbon atoms, with
        minimum-image separations in the periodic x so neighbor counts are
        correct across the x-seam. coord[k] is atom k's carbon-neighbor count."""
        n = self.lmp.get_natoms()
        x = self.lmp.numpy.extract_atom("x")[:n]
        ids = self.lmp.numpy.extract_atom("id")[:n].copy()
        pos = np.array([[x[k][0], x[k][1]] for k in range(n)])
        Lx = self.xhi - self.xlo
        dx = pos[:, 0, None] - pos[None, :, 0]
        dx -= Lx * np.round(dx / Lx)                 # minimum image across the x-seam
        dy = pos[:, 1, None] - pos[None, :, 1]
        dist = np.hypot(dx, dy)
        np.fill_diagonal(dist, np.inf)
        neigh = dist < COORD_CUTOFF
        return neigh.sum(axis=1), pos, ids, dx, dy, neigh

    def _prune_and_cap_top(self, slab_top):
        """Turn the raw honeycomb cut into a clean sheet with a hydrogen-
        terminated top edge and a fully-clamped bottom edge. Returns
        (cap_positions, floor_cut) where floor_cut is the y below which carbons
        are clamped as inert bulk.

        The `region`-cut top leaves a ragged row of *monovalent* carbons -- apex
        atoms of half-cut hexagons, bonded to the sheet by a single bond each.
        Those radical-like sites are what made the sheet blow up at rest. Deleting
        them exposes the clean zigzag edge one row down (each of those carbons is
        left two-coordinate), which is then capped with one H apiece -- placed at
        CH_BOND along the missing-bond direction (the negative of the summed unit
        vectors to the two carbon neighbors, i.e. up and out of the sheet).

        The bottom edge is handled differently: it's the substrate, meant to sit
        frozen as bulk, so rather than cap it we clamp it. floor_cut is set just
        above the highest under-coordinated carbon in the bottom half, so the
        entire ragged bottom edge is frozen and every *mobile* carbon is
        fully-coordinated -- no dangling bonds left free to reconstruct and heat
        the sheet. The result is a textbook H-terminated ribbon that sits
        perfectly still."""
        y_mid = 0.5 * slab_top
        coord, pos, ids, _, _, _ = self._carbon_coordination()
        prune = [int(ids[k]) for k in range(len(ids)) if coord[k] <= 1 and pos[k, 1] > y_mid]
        if prune:
            self.lmp.command("group _prune id " + " ".join(str(i) for i in prune))
            self.lmp.command("delete_atoms group _prune compress yes")
            self.lmp.command("group _prune delete")

        coord, pos, ids, dx, dy, neigh = self._carbon_coordination()
        caps = []
        for k in range(len(ids)):
            if coord[k] != 2 or pos[k, 1] < y_mid:
                continue                             # only edge carbons on the TOP half
            js = np.where(neigh[k])[0]
            # unit vectors from carbon k toward each neighbor (k->j = -(dx, dy)[k, j])
            vecs = np.array([[-dx[k, j], -dy[k, j]] for j in js], dtype=float)
            vecs /= np.hypot(vecs[:, 0], vecs[:, 1])[:, None]
            miss = -vecs.sum(axis=0)
            m = np.hypot(miss[0], miss[1])
            if m < 1e-6:
                continue
            miss /= m
            caps.append((pos[k, 0] + CH_BOND * miss[0], pos[k, 1] + CH_BOND * miss[1]))

        # Clamp everything up to just above the highest under-coordinated carbon
        # in the bottom half (with a whole-row margin), so the bottom edge is
        # frozen bulk and no mobile carbon is left under-coordinated.
        bottom_bad = pos[(coord < 3) & (pos[:, 1] < y_mid), 1]
        floor_cut = self.ylo + FLOOR_ROWS * ROW_HEIGHT + 0.1
        if len(bottom_bad):
            floor_cut = max(floor_cut, float(bottom_bad.max()) + 0.5 * ROW_HEIGHT)
        return caps, floor_cut

    def _build(self):
        lmp = self.lmp
        lmp.command("dimension 2")
        lmp.command("units metal")
        lmp.command("atom_style atomic")
        lmp.command("boundary p f p")
        # Honeycomb (graphene) via the ORTHOGONAL 4-atom cell (a x a*sqrt(3),
        # armchair along x). Unlike the sheared 2-atom hex cell, this tiles a
        # rectangular periodic box cleanly: a1 is purely along x AND every row
        # ends flush with the x-seam, so there are no stranded under-coordinated
        # carbons at the seam (the sheared cell left a diagonal of them, which
        # both destabilized the sheet and forced the clamp region absurdly high).
        # y is non-periodic (free surface above, clamped rows below). The four
        # basis atoms sit at Cartesian (0,0), (0,1.42), (1.23,2.13), (1.23,3.55)
        # -- one full zigzag repeat -- giving C-C bonds of a/sqrt(3) ~ 1.42 A.
        lmp.command(
            f"lattice custom {A_GRAPHENE} a1 1.0 0.0 0.0 a2 0.0 1.7320508 0.0 "
            f"basis 0.0 0.0 0.0 basis 0.0 0.3333333 0.0 "
            f"basis 0.5 0.5 0.0 basis 0.5 0.8333333 0.0"
        )
        Lx = NX * A_GRAPHENE
        slab_top = CRYSTAL_ROWS * ROW_HEIGHT
        Ly = slab_top + VACUUM
        lmp.command(
            f"region simbox block 0 {Lx} 0 {Ly} {-0.25 * A_GRAPHENE} {0.25 * A_GRAPHENE} units box"
        )
        # Three atom types: 1 = carbon (the sheet), 2 = oxygen (the projectile),
        # 3 = hydrogen (the passivating caps on the top edge).
        lmp.command("create_box 3 simbox")

        boxlo, boxhi, *_ = lmp.extract_box()
        self.xlo, self.ylo = boxlo[0], boxlo[1]
        self.xhi, self.yhi = boxhi[0], boxhi[1]

        lmp.command(
            f"region slab block {self.xlo} {self.xhi} {self.ylo} {slab_top + 0.1} "
            f"-0.25 0.25 units box"
        )
        lmp.command("create_atoms 1 region slab")

        # Passivate the free top edge with hydrogen. Left bare, the freshly-cut
        # top edge is a row of radical-like under-coordinated carbons that
        # reconstruct violently and dump that energy as heat, melting the cold
        # sheet on the first frame (the whole reason the demo wasn't stable at
        # rest). _prune_and_cap_top cleans the cut into a proper zigzag edge and
        # terminates it with H -- the standard way real graphene edges are capped
        # -- so the sheet starts as a pristine, perfectly stable 2D crystal, like
        # the frozen argon lattice. The bottom edge stays clamped as bulk, so only
        # the top (the surface the O attacks) needs caps.
        h_positions, self._floor_cut = self._prune_and_cap_top(slab_top)
        self.n_carbon = lmp.get_natoms()
        for hx, hy in h_positions:
            lmp.command(f"create_atoms 3 single {hx:.4f} {hy:.4f} 0.0 units box")
        self.n_hydrogen = len(h_positions)

        lmp.command(f"mass 1 {C_MASS}")
        lmp.command(f"mass 2 {O_MASS}")
        lmp.command(f"mass 3 {H_MASS}")
        # REBO handles all carbon-carbon and carbon-hydrogen bonding (C and H are
        # mapped into it; O is NULL, i.e. invisible to REBO); the reactive oxygen
        # interacts with C and H only through the Morse overlay.
        lmp.command(f"pair_style hybrid/overlay rebo morse {MORSE_CUTOFF}")
        lmp.command(f"pair_coeff * * rebo {_rebo_potential_path()} C NULL H")
        lmp.command(f"pair_coeff 1 2 morse {MORSE_CO_D0} {MORSE_CO_ALPHA} {MORSE_CO_R0}")
        lmp.command(f"pair_coeff 2 2 morse {MORSE_OO_D0} {MORSE_OO_ALPHA} {MORSE_OO_R0}")
        lmp.command(f"pair_coeff 2 3 morse {MORSE_OH_D0} {MORSE_OH_ALPHA} {MORSE_OH_R0}")

        lmp.command("neighbor 2.0 bin")
        lmp.command("neigh_modify every 1 delay 0 check yes")
        lmp.command(f"comm_modify cutoff {max(RDF_CUTOFF, MORSE_CUTOFF) + 2.0}")

        # Oxygen projectile (the puller), started above the surface centre.
        # Created last, so its id is the highest -- past every carbon and every H.
        self._surface_top = slab_top
        px = (self.xlo + self.xhi) / 2
        py = slab_top + PULLER_GAP
        self.rest_pos = (px, py)
        self.puller_id = self.n_carbon + self.n_hydrogen + 1
        lmp.command(f"create_atoms 2 single {px} {py} 0.0 units box")

        # Groups: clamped bottom rows vs the mobile sheet (carbons + H caps) vs
        # the O beam. The thermostatted/integrated "sheet" is carbons and their
        # hydrogen caps together, minus the clamped floor. floor_top was chosen
        # (see _prune_and_cap_top) to bury the whole ragged bottom edge, so no
        # mobile carbon is left dangling.
        floor_top = self._floor_cut
        lmp.command(f"region floor_region block INF INF {self.ylo} {floor_top} INF INF units box")
        lmp.command("group carbon type 1")
        lmp.command("group hydro type 3")
        lmp.command("group beam type 2")
        lmp.command("group sheet type 1 3")
        lmp.command("group floor region floor_region")
        lmp.command("group sheet_mobile subtract sheet floor")
        # Remember the clamped bottom-row carbon ids (so they can be drawn as
        # dim "bulk"): the carbons whose initial y sits below the floor cut.
        nloc0 = lmp.get_natoms()
        ids0 = lmp.numpy.extract_atom("id")[:nloc0]
        x0 = lmp.numpy.extract_atom("x")[:nloc0]
        typ0 = lmp.numpy.extract_atom("type")[:nloc0]
        self._floor_ids = {int(ids0[k]) for k in range(nloc0)
                           if typ0[k] == 1 and x0[k][1] < floor_top}

        lmp.command("compute crystal_temp sheet_mobile temp/com")
        self._target_temp = T_MIN
        self._seed = 785412
        lmp.command("fix twod all enforce2d")
        lmp.command("fix freeze floor setforce 0.0 0.0 0.0")
        lmp.command("fix integ_c sheet_mobile nve")
        lmp.command(
            f"fix thermostat sheet_mobile temp/csvr {T_MIN} {T_MIN} "
            f"{THERMOSTAT_DAMP} {self._seed}"
        )
        lmp.command("fix_modify thermostat temp crystal_temp")
        # O beam: capped-velocity integrator (keeps a hard hit controllable) plus
        # the viscous damping dial.
        lmp.command(f"fix integ_o beam nve/limit {0.05 * A_GRAPHENE}")
        self._puller_damping = PULLER_DAMPING_DEFAULT
        lmp.command(f"fix damp_o beam viscous {PULLER_DAMPING_DEFAULT}")
        lmp.command(
            f"fix walls all wall/reflect ylo {self.ylo + 0.4 * A_GRAPHENE} "
            f"yhi {self.yhi - 0.4 * A_GRAPHENE} units box"
        )

        # Two coordination counts on the carbons:
        #  - crd_cc: carbon-carbon neighbors only, the covalent network, for the
        #    "bonds broken" tally (unaffected by the H caps);
        #  - crd_sat: carbon + hydrogen neighbors, how "satisfied" a carbon is,
        #    for recoloring -- an H-capped top carbon reads as fully bonded (bulk
        #    gray), and only turns reactive orange once its cap is stripped or a
        #    C-C bond snaps.
        lmp.command(f"compute crd_cc carbon coord/atom cutoff {COORD_CUTOFF} 1")
        # crd_sat counts neighbors of ALL types (no type filter -> a single
        # per-atom value): for a carbon that's C + H neighbors -- so an H-capped
        # edge carbon reads as satisfied. (Listing multiple types instead would
        # make coord/atom emit a per-type array, not one value.)
        lmp.command(f"compute crd_sat carbon coord/atom cutoff {COORD_CUTOFF}")
        # Net Morse force from the sheet (carbons + H caps) on the O, for the
        # reaction arrow / force feedback.
        lmp.command("compute iff beam group/group sheet")

        lmp.command(f"timestep {TIMESTEP}")
        lmp.command("thermo 100000")
        # Quenched settle: the passivated sheet is already near its relaxed
        # geometry, but the freshly-placed caps and cut edges still carry a small
        # perturbation. A strong viscous drag during the settle dissipates it,
        # leaving a pristine, relaxed, fully-capped slab that sits perfectly
        # still. The drag is then removed.
        lmp.command("fix settle_quench sheet_mobile viscous 1.0")
        lmp.command(f"run {SETTLE_STEPS}")
        lmp.command("unfix settle_quench")
        lmp.command("velocity sheet_mobile set 0.0 0.0 0.0")

        lmp.command("compute ke_atom all ke/atom")
        lmp.command("compute pe_atom all pe/atom")
        lmp.command(f"compute rdf_raw carbon rdf {RDF_NBINS} cutoff {RDF_CUTOFF}")
        lmp.command(
            f"fix rdf_avg carbon ave/time {RDF_AVE_EVERY} {RDF_AVE_REPEAT} {RDF_AVE_FREQ} "
            f"c_rdf_raw[*] mode vector"
        )
        self._rdf_bins_ready = False
        self._rdf_r = None
        self._rdf_ready_step = lmp.extract_global("ntimestep") + RDF_AVE_FREQ + RDF_AVE_EVERY

        # Baseline covalent bond count (fully-formed sheet) for the "bonds
        # broken" tally.
        self._initial_bonds = self._count_cc_bonds()

    # ---- controls -----------------------------------------------------------

    def set_input_force(self, fx, fy):
        self._input_fx = fx
        self._input_fy = fy
        self.lmp.command(f"fix input_force beam addforce {fx} {fy} 0.0")

    def set_target_temp(self, T):
        T = max(T_MIN, min(T_MAX, T))
        if T == self._target_temp:
            return
        self._target_temp = T
        current = self.lmp.extract_compute("crystal_temp", 0, 0)
        if T > current and current < COLD_SEED_TEMP:
            self._seed = (self._seed * 1103515245 + 12345) & 0x7FFFFFFF
            self.lmp.command(
                f"velocity sheet_mobile create {T} {self._seed} mom yes rot yes dist gaussian"
            )
        self.lmp.command(
            f"fix thermostat sheet_mobile temp/csvr {T} {T} {THERMOSTAT_DAMP} {self._seed}"
        )
        self.lmp.command("fix_modify thermostat temp crystal_temp")

    def set_puller_damping(self, gamma):
        gamma = max(PULLER_DAMPING_MIN, min(PULLER_DAMPING_MAX, gamma))
        if gamma == self._puller_damping:
            return
        self._puller_damping = gamma
        self.lmp.command(f"fix damp_o beam viscous {gamma}")

    # ---- stepping + reactive bookkeeping ------------------------------------

    def step(self, n=4):
        self.lmp.command(f"run {n}")
        self._interactive_ps += n * TIMESTEP
        self._handle_reactions()

    def _handle_reactions(self):
        """Detect what the O has grabbed and lifted clear of the surface, and
        desorb it: a carbon leaves as CO (an etch pit, the headline reaction) and
        a hydrogen cap leaves as OH (de-passivating the surface into a reactive
        dangling-bond site). In both cases a fresh O is then sent in. Also recycle
        an O that has simply flown off the top.

        An atom lifted several Angstrom above the surface while still gripped by
        the O has, in practice, had its bonds to the sheet snapped (a fully intact
        sheet resists that); so 'lifted clear and still gripped by O' is the
        robust, physical proxy for 'you pulled it off' -- more reliable than
        waiting for the pulled atom's coordination to hit exactly zero, which a
        stretched, still-tethered chain rarely does."""
        nlocal = self.lmp.get_natoms()
        ids = self.lmp.numpy.extract_atom("id")[:nlocal]
        typ = self.lmp.numpy.extract_atom("type")[:nlocal]
        x = self.lmp.numpy.extract_atom("x")
        oidx = np.where(ids == self.puller_id)[0]
        if len(oidx) == 0:
            return
        oi = int(oidx[0])
        opos = x[oi][:2].copy()

        # O escaped off the top with nothing attached -> recycle it.
        if opos[1] > self.yhi - 0.6 * A_GRAPHENE:
            self._respawn_oxygen()
            return

        escape_y = self._surface_top + 4.5
        if opos[1] <= escape_y:
            return
        # Consider both carbons (etch -> CO) and hydrogen caps (strip -> OH).
        targets = np.where((typ == 1) | (typ == 3))[0]
        if len(targets) == 0:
            return
        d = np.hypot(x[targets, 0] - opos[0], x[targets, 1] - opos[1])
        j = int(np.argmin(d))
        tj = targets[j]
        # The single atom the O is actually gripping, dragged clear of the
        # surface. One per event keeps it honest ("1 O + 1 C -> CO", "O + H -> OH").
        if d[j] < 1.8 and x[tj, 1] > self._surface_top + 3.5:
            if typ[tj] == 1:
                self._delete_carbon(int(ids[tj]))
                self._co_count += 1
            else:
                self._delete_atoms([int(ids[tj])])
                self._h_stripped += 1
            self._respawn_oxygen()

    def _delete_carbon(self, cid):
        """Desorb a carbon as CO, taking any hydrogen cap bonded to it along
        (it would otherwise be left as a stray, unbonded H flying loose)."""
        nlocal = self.lmp.get_natoms()
        ids = self.lmp.numpy.extract_atom("id")[:nlocal]
        typ = self.lmp.numpy.extract_atom("type")[:nlocal]
        x = self.lmp.numpy.extract_atom("x")
        ci = int(np.where(ids == cid)[0][0])
        gone = [cid]
        hydros = np.where(typ == 3)[0]
        for h in hydros:
            if math.hypot(x[h, 0] - x[ci, 0], x[h, 1] - x[ci, 1]) < 1.4:
                gone.append(int(ids[h]))
        self._delete_atoms(gone)

    def _delete_atoms(self, atom_ids):
        idlist = " ".join(str(i) for i in atom_ids)
        self.lmp.command(f"group _gone id {idlist}")
        self.lmp.command("delete_atoms group _gone compress no")
        self.lmp.command("group _gone delete")

    def _respawn_oxygen(self):
        """Teleport the (persistent) O projectile back to its launch point at
        rest. The O atom keeps its id, so its fixes stay bound to it."""
        nlocal = self.lmp.get_natoms()
        ids = self.lmp.numpy.extract_atom("id")[:nlocal]
        x = self.lmp.numpy.extract_atom("x")
        v = self.lmp.numpy.extract_atom("v")
        oidx = np.where(ids == self.puller_id)[0]
        if len(oidx) == 0:
            return
        oi = int(oidx[0])
        x[oi][0], x[oi][1] = self.rest_pos
        v[oi][0], v[oi][1] = 0.0, 0.0

    # ---- readouts -----------------------------------------------------------

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
        vec = self.lmp.extract_compute("iff", 0, 1)
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

    def _count_cc_bonds(self):
        nlocal = self.lmp.get_natoms()
        typ = self.lmp.numpy.extract_atom("type")[:nlocal]
        crd = self.lmp.numpy.extract_compute("crd_cc", 1, 1)[:nlocal]
        return int(round(crd[typ == 1].sum() / 2.0))

    def get_all_positions(self):
        nlocal = self.lmp.get_natoms()
        xs = self.lmp.numpy.extract_atom("x")[:nlocal]
        ids = self.lmp.numpy.extract_atom("id")[:nlocal].copy()
        typ = self.lmp.numpy.extract_atom("type")[:nlocal]
        # Coloring uses C+H coordination: an H-capped edge carbon reads as fully
        # bonded (bulk gray), turning reactive orange only once it loses a
        # neighbor (cap stripped or C-C bond snapped).
        crd = self.lmp.numpy.extract_compute("crd_sat", 1, 1)[:nlocal]
        species = np.empty(nlocal, dtype=int)
        for k in range(nlocal):
            if typ[k] == 2:
                species[k] = SP_O
            elif typ[k] == 3:
                species[k] = SP_H
            elif int(ids[k]) in self._floor_ids:
                species[k] = SP_C_FROZEN
            elif crd[k] >= 2.5:
                species[k] = SP_C_BULK
            else:
                species[k] = SP_C_REACTIVE
        return ids, xs[:, :2].copy(), (ids == self.puller_id), species

    def get_bond_pairs(self):
        """Only the O's live bonds (to a carbon it is abstracting, or a hydrogen
        cap it is stripping), drawn as bright sticks so the reactive C-O / O-H
        bond stands out. The C-C covalent network itself is drawn wrap-safely by
        the faint bond overlay (spec.bond_overlay)."""
        i, nlocal = self._puller_index()
        if i is None:
            return None
        xs = self.lmp.numpy.extract_atom("x")[:nlocal]
        typ = self.lmp.numpy.extract_atom("type")[:nlocal]
        opos = xs[i][:2]
        pairs = []
        for k in range(nlocal):
            if typ[k] not in (1, 3):
                continue
            if abs(xs[k][0] - opos[0]) > self.box_x() * 0.5:
                continue  # periodic-x wrap, not a real bond
            if math.hypot(xs[k][0] - opos[0], xs[k][1] - opos[1]) < BOND_CUTOFF:
                pairs.append((i, k))
        return np.array(pairs, dtype=int) if pairs else None

    def box_x(self):
        return self.xhi - self.xlo

    def get_hud_lines(self):
        broken = max(0, self._initial_bonds - self._count_cc_bonds())
        nlocal = self.lmp.get_natoms()
        typ = self.lmp.numpy.extract_atom("type")[:nlocal]
        crd = self.lmp.numpy.extract_compute("crd_sat", 1, 1)[:nlocal]
        cmask = typ == 1
        reactive = int(((crd < 2.5) & cmask).sum())
        return [
            "Drive the O in: strip an H cap (as OH), then bond a C and pull it free to etch CO.",
            f"CO etched: {self._co_count}     H stripped: {self._h_stripped}     covalent bonds broken: {broken}",
            f"reactive dangling-bond sites (orange): {reactive}",
        ]

    def get_box_size(self):
        return self.xhi - self.xlo, self.yhi - self.ylo

    def close(self):
        self.lmp.close()

"""Salt: an ionic crystal, pulled at.

The third bonding type in the picker, alongside metallic Cu (EAM, cu_deposition)
and van-der-Waals Ar (Lennard-Jones, lj_argon). Instead of neutral atoms bound by
an embedding energy or by dispersion, this is a lattice of alternating Na(+) and
Cl(-) held together by long-range Coulomb (Madelung) attraction, balanced against
a short-range Born-Mayer repulsion -- the classic rigid-ion model of an alkali
halide. The puller is a Na(+), so it is drawn down onto the Cl(-) sites
electrostatically.

The lattice is a checkerboard, not the close-packed triangle the neutral systems
crystallise on, and that difference is the physics: see IonicSlab2D in
playground/deposition.py for why ionic bonding needs a bipartite lattice.

The `charge` slider is the one to reach for first. Take it to zero and the
Madelung bonding that holds the whole thing together switches off, leaving a bare
repulsion; bring it back up and the checkerboard reassembles.

Units are LAMMPS metal (eV, Angstrom, ps, amu, charge in electrons). The
constants were measured the same way argon's and copper's were -- see the force
field (forcefields/stock.py) for the Born-Mayer parameters and the pressure sweep
that set the spacing.
"""
from ..playground import Control, ForceFeedbackProfile, Playground, ionic_slab_2d

# Zero-pressure nearest-neighbour (opposite-charge) distance of the 2D
# checkerboard with these Born parameters -- close to real NaCl's 2.82 A.
LATTICE_SPACING = 2.892      # Angstrom
# The same 5x-smaller-than-copper step argon needs: the stiff ionic contact
# forces (peaking ~6 eV/A, comparable to EAM copper) were checked to integrate a
# real deposition impact at this step without gaining energy or losing atoms.
TIMESTEP = 0.0002            # ps

# Ionic contact forces run at a scale comparable to EAM copper's, so the
# force-feedback tuning is copper-like rather than argon's much softer profile.
FORCE_FEEDBACK = ForceFeedbackProfile(
    ff_exaggeration=4.0,
    ff_knee=1.5,
    ff_max_mag=120.0,
    stiffness_threshold=0.05,
    stiffness_knee=0.5,
    damper_min_fraction=0.10,
    damper_max_fraction=0.50,
    vel_damp_max_fraction=0.5,
)

PLAYGROUND = Playground(
    name="Salt crystal (ionic, NaCl)",
    description="A 2D Na(+)/Cl(-) checkerboard held by Coulomb (Madelung) "
                "bonding -- pull an ion onto the lattice.",
    force_field="born_dsf",
    # 16 spacings across = 8 whole 2x2 checkerboard cells, so periodic-x wrapping
    # preserves the alternation (and thus exact neutrality) across the seam.
    scenario=ionic_slab_2d(a=LATTICE_SPACING, n_cols=16, crystal_rows=6,
                           floor_rows=0, gap_rows=2, puller_gap=3.0,
                           settle_steps=600, timestep=TIMESTEP,
                           sim_time_per_frame=0.002, thermostat_damp=0.1),
    mode="game",
    control=Control(
        atom="last",
        plane="xy",
        confine=False,
        # nve/limit, not plain nve: under a sustained user force the puller
        # would otherwise accelerate without bound and tunnel through the
        # lattice. The stiff ionic core wants argon's tighter cap, not copper's.
        displacement_cap=0.05 * LATTICE_SPACING,
        max_input_force=3.0,        # eV/A at full deflection
        damping_default=0.02,       # eV*ps/A^2
        damping_range=(0.0, 0.1),
    ),
    # No observable declared: `coordination` counts neighbours inside the force
    # field's interaction range, and here that is the 12 A Coulomb cutoff -- some
    # 39 ions, which says nothing about the lattice. The live g(r), whose first
    # peaks are the alternating +/- shells, is the readout that does.
    observables=(),
    presets={
        "rock_salt": {},
        # No Madelung bonding at all: what is left is a bare Born-Mayer
        # repulsion, and the lattice flies apart. The useful null comparison.
        "no_charge": {"charge": 0.0},
        # Softer ions: a longer hardness length lets the lattice compress.
        "soft_ions": {"born_rho": 0.5},
    },
    # csvr toward 1 K is a deterministic quench; T_MELT is real NaCl's melting
    # point as a dial marker -- the 2D model's own disordering, visible in the
    # live RDF, is the trustworthy signal.
    temperature=(1.0, 3000.0),
    temperature_default=1.0,
    melt_temp=1074.0,
    render_3d=False,
    reduced_units=False,
    element_label="NaCl (ionic)",
    lattice_spacing=LATTICE_SPACING,
    species_colors=((235, 205, 90), (90, 190, 235)),   # 0 = Na+, 1 = Cl-
    species_labels=("+", "-"),
    # Real ionic radii (Na+ ~1.02 A, Cl- ~1.81 A) scaled down uniformly to leave
    # a gap for the bond overlay while preserving the ~1.8x ratio: the big soft
    # anion and the small dense cation are a core lesson of ionic bonding (and
    # why the small cation slots between the large anions), so the drawn sizes
    # teach it directly rather than showing two equal dots.
    species_radii=(0.72, 1.28),
    particle_radius=1.0,
    bond_overlay=True,
    puller_speed_cap=0.05 * LATTICE_SPACING / TIMESTEP,
    force_feedback=FORCE_FEEDBACK,
)

"""2D copper deposition on the real EAM potential.

A Cu atom pulled onto a cold Cu(001)-like 2D crystal, sticking via genuine
metallic bonding from a tabulated embedded-atom potential. The contrast with
lj_argon.py is the point of having both: EAM's coordination-hungry many-body
bonding versus Lennard-Jones' soft pairwise van der Waals, at the same
interaction and side by side in the picker.

EAM's energy is NOT pairwise-additive -- it has a many-body embedding term -- so
there is no additive decomposition to show and no Python reference expression to
cross-check. The EAM force field declares no energy terms, and `--verify` skips
it, which is the honest answer for this potential rather than a gap.

Same deposition scenario as argon, differing only in the calibrated constants:
EAM's smoother embedding repulsion tolerates a 5x larger timestep than LJ's bare
r^-12 core, and copper's contact forces run an order of magnitude stronger.
"""
from ..playground import (
    Control, ForceFeedbackProfile, Playground, deposition_2d,
)

LATTICE_SPACING = 2.4605     # Angstrom -- empirical 2D-hex EAM Cu equilibrium
TIMESTEP = 0.0005            # ps

# Copper's contact forces run ~0.1-6 eV/A, an order of magnitude above argon's.
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
    name="Copper deposition (EAM)",
    description="A Cu atom pulled onto a cold Cu(001)-like 2D crystal -- sticks "
                "via real metallic bonding (EAM).",
    force_field="eam",
    scenario=deposition_2d(a=LATTICE_SPACING, n_cols=16, crystal_rows=7,
                           floor_rows=0, puller_gap=3.0, settle_steps=600,
                           timestep=TIMESTEP, sim_time_per_frame=0.003,
                           thermostat_damp=0.5),
    mode="game",
    control=Control(
        atom="last",
        plane="xy",
        confine=False,
        # nve/limit, not plain nve: under a sustained user force the puller would
        # otherwise accelerate without bound and tunnel through the lattice.
        # EAM's smoother repulsion tolerates twice argon's cap.
        displacement_cap=0.1 * LATTICE_SPACING,
        max_input_force=2.0,       # eV/A at full deflection
        damping_default=0.01,      # eV*ps/A^2
        damping_range=(0.0, 0.05),
    ),
    observables=["coordination"],
    presets={"copper": {}},
    # csvr toward 0 K is a pure deterministic quench. T_MELT is an approximate
    # dial marker here, unlike argon's, whose LJ parameters were fit to reproduce
    # its real melting point.
    temperature=(1.0, 10000.0),
    temperature_default=1.0,
    melt_temp=1000.0,
    render_3d=False,
    reduced_units=False,
    element_label="Cu (EAM)",
    lattice_spacing=LATTICE_SPACING,
    # Warm metallic copper, so it reads as a dense metal against argon's cold
    # noble-gas tint. Atoms at ~0.3x the spacing: big enough to show the
    # close-packed hex coordination, small enough to leave a clear gap for the
    # near-equilibrium bond overlay glowing between them.
    crystal_color=(210, 128, 66),
    particle_radius=0.30 * LATTICE_SPACING,
    bond_overlay=True,
    puller_speed_cap=0.1 * LATTICE_SPACING / TIMESTEP,
    force_feedback=FORCE_FEEDBACK,
)

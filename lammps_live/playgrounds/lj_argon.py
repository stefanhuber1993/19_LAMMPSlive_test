"""2D Lennard-Jones argon: pull an atom onto a cold crystal and watch it stick.

Soft van-der-Waals bonding instead of EAM's coordination-hungry metallic bonding
(see cu_deposition.py), so the same deposition interaction feels noticeably
different: weaker, softer contact forces (~0.01-0.5 eV/A vs copper's ~0.1-6) and a
much lower melting point.

epsilon/sigma/mass are literally argon's textbook LJ parameters, so the melting
marker is real argon's ~84 K rather than a dial guess. And because parameters are
now cheap to declare, epsilon and sigma are LIVE here -- they were module constants
in the hand-written version, so you could not feel what the well depth does.

The constants below were empirically checked in the original implementation, not
guessed:
  - lattice spacing: found by bisecting a periodic 2D-hex bulk on zero pressure.
    Came out at ~1.113*sigma, compressed from the bare pair minimum of
    2^(1/6)*sigma ~= 1.122*sigma by the same lattice-sum effect that compresses
    3D FCC LJ crystals.
  - timestep: LJ's bare r^-12 core is stiffer at hard contact than EAM's smoother
    embedding repulsion, so copper's 0.001 ps is NOT safe -- a deposition impact
    integrated there visibly GAINS energy and the puller never sticks. 0.0002 ps
    was the largest value that stayed stable through a real impact.
  - puller damping: swept at that timestep while depositing a 1 eV atom, picked
    from the range that reliably settles into a stuck, non-oscillating state.
"""
from ..playground import (
    Control, ForceFeedbackProfile, Playground, deposition_2d,
)

SIGMA = 3.40                 # Angstrom
LATTICE_SPACING = 3.784884   # Angstrom -- empirical 2D-hex zero-pressure spacing
TIMESTEP = 0.0001            # ps

# Forces here run roughly two orders of magnitude weaker than EAM copper's
# (~0.01-0.5 eV/A vs ~0.1-6), so every force-feedback knob is scaled down to
# match; reusing copper's -- or a reduced-unit profile -- reads as permanently numb.
FORCE_FEEDBACK = ForceFeedbackProfile(
    ff_exaggeration=4.0,
    ff_knee=0.08,
    ff_max_mag=120.0,
    stiffness_threshold=0.005,
    stiffness_knee=0.05,
    damper_min_fraction=0.10,
    damper_max_fraction=0.50,
    vel_damp_max_fraction=0.5,
)

PLAYGROUND = Playground(
    name="Argon melting (Lennard-Jones)",
    description="A softer, weaker-bonded 2D crystal -- same deposition "
                "interaction, real argon LJ parameters. Now with live epsilon.",
    force_field="lj",
    scenario=deposition_2d(a=LATTICE_SPACING, n_cols=16, crystal_rows=7,
                           floor_rows=0, puller_gap=3.0, settle_steps=600,
                           timestep=TIMESTEP, sim_time_per_frame=0.003,
                           thermostat_damp=0.5),
    mode="game",
    control=Control(
        atom="last",              # the slab is built first, the free atom last
        plane="xy",
        # A genuinely free atom: it flies in, sticks, and can be knocked loose.
        # There is no leash and no control plane, so no net is drawn.
        confine=False,
        # Tighter displacement cap than copper's 0.1*a: LJ's stiffer repulsive
        # core needs it, the same reasoning as the smaller timestep.
        displacement_cap=0.05 * LATTICE_SPACING,
        max_input_force=0.3,      # eV/A at full deflection
        damping_default=0.0015,   # eV*ps/A^2
        damping_range=(0.0, 0.005),
    ),
    observables=["coordination"],
    presets={
        "argon": {},
        # Deeper well: sticks harder and melts higher.
        "strongly_bound": {"epsilon": 0.03},
        # Barely bound -- the crystal sublimates almost immediately.
        "weakly_bound": {"epsilon": 0.002},
    },
    # csvr toward 0 K is a pure deterministic quench; 800 K is well past melting,
    # into a clearly gas-like RDF.
    temperature=(1.0, 800.0),
    temperature_default=1.0,
    melt_temp=84.0,              # real argon's melting point
    render_3d=False,
    reduced_units=False,
    element_label="Ar (LJ)",
    lattice_spacing=LATTICE_SPACING,
    # Cold periwinkle, deliberately unlike copper's warm metal, to signal a
    # weakly-bound noble-gas solid that melts near liquid-nitrogen cold. Argon's
    # atom is physically far larger than copper's, and at the same box-relative
    # scaling that difference is visible side by side.
    crystal_color=(168, 172, 230),
    particle_radius=0.45 * SIGMA,
    bond_overlay=True,
    puller_speed_cap=0.05 * LATTICE_SPACING / TIMESTEP,
    force_feedback=FORCE_FEEDBACK,
)

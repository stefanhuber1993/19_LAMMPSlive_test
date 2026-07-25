"""MesoMem membrane sheet -- the paper's planar-stability test, made interactive.

The same force field as the patch, at the scale the paper (Sec. IV) uses to check
that the interaction scheme supports a stable planar membrane: beads on a
hexagonal lattice at spacing a = 0.8 sigma, periodic in-plane, relaxed under
Langevin dynamics with a barostat that drives the lateral pressure to zero so the
sheet equilibrates tension-free. The paper's benchmark is 50x50 sites; 30x30 still
limits finite-size effects while staying real-time under interactive control.

Differences from the patch: the periodic cell means the sheet holds itself flat
with no artificial tether, and the puller is whichever bead starts nearest the box
centre.

Units are the paper's LJ-reduced units (sigma = eps = m = 1).
"""
from ..playground import Control, Playground, hex_sheet

PLAYGROUND = Playground(
    name="MesoMem membrane sheet (3D)",
    description="Paper-scale hexagonal MesoMem sheet (periodic, barostat-relaxed): "
                "pull one bead out and watch tilt/splay propagate.",
    force_field="mesomem",
    scenario=hex_sheet(n_cols=30, n_rows=30, a=0.8, z_half=4.0,
                       settle_steps=1000),
    mode="game",
    control=Control(
        atom="nearest_center",
        plane="xz",
        leash=(5.0, 3.5),
        # A large cohesive membrane barely tented under the patch's forces, so
        # pull authority is raised here.
        max_input_force=12.0,
        grid_step=0.8,
    ),
    observables=["nematic_S", "thickness", "area_per_particle"],
    # A large membrane buckles under high splay rather than merely stiffening, so
    # the dial is worth taking far past the patch's useful span of 3.
    param_ranges={"k_splay": (0.0, 40.0)},
    presets={
        "paper": {},
        # k_splay reaches much further on the sheet than on the 7-bead patch --
        # high splay buckles a large membrane rather than just stiffening it,
        # which is the interesting failure mode to be able to reach.
        "buckled": {"k_splay": 30.0},
        "floppy": {"k_tilt": 2.0, "k_splay": 0.1},
        "isotropic_only": {"k_tilt": 0.0, "k_splay": 0.0, "wc": 0.0},
    },
    temperature=(0.0, 0.5),
    temperature_default=0.001,
    melt_temp=0.3,
    particle_radius=0.5,
    reduced_units=True,
)

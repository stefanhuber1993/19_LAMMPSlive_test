"""MesoMem membrane patch -- the smallest interactive demo of the real force field.

Seven beads: one central bead ringed by six at hexagonal spacing, all in the
world xy-plane with directors initially along +z. The smallest patch that already
shows the tilt/splay physics -- pull the middle bead out of plane and its
neighbours' directors splay to follow, resisting through force feedback.

The central bead is the puller. The joystick's two axes slide it in the world
xz-plane -- perpendicular to the membrane and facing the camera, drawn in the
scene as a faint net whose extents are exactly the leash. Q/E (or the stick's
twist axis) applies a steering torque to its director, and the membrane's own tilt
stiffness springs it back; twist hard enough and the director flips to the
opposite normal, because the tilt term is bistable (both +n and -n are minima).

Units are the paper's LJ-reduced units (sigma = eps = m = 1).
"""
from ..playground import Control, Playground, hex_patch

PLAYGROUND = Playground(
    name="MesoMem membrane patch (3D)",
    description="Real MesoMem force field: pull the center bead out of a 7-bead "
                "patch, feel tilt/splay resist.",
    force_field="mesomem",
    # Sphere radius = sigma (diameter 2), which is what gives the paper's moment
    # of inertia I = (2/5) m sigma^2 -- it sets how fast the directors swing, and
    # the yaw/tilt feel is tuned around it. The larger sheets use diameter = sigma
    # instead, so overlapping beads read as a continuous membrane.
    force_field_options={"bead_diameter": 2.0},
    # A cubic container sized snugly around the patch and its pull reach, rather
    # than large and arbitrary: the camera frames the box outline, so a tight box
    # makes the beads fill the view instead of floating in a cavernous cell.
    scenario=hex_patch(n_rings=1, a=1.0, box=6.0, settle_steps=300),
    mode="game",
    control=Control(
        atom="first",           # hex_patch orders the centre site first
        plane="xz",
        # Kept inside the ring's interaction range: a bead at height z sits
        # sqrt(1 + z^2) from its neighbours, and past the rc = 2.5 cutoff it would
        # detach and float free. Inside the leash the membrane always exerts a
        # restoring pull, so the bead snaps back on release.
        leash=(2.8, 2.8),
        max_input_force=9.0,
        grid_step=0.5,
    ),
    observables=["mean_tilt_deg", "thickness", "coordination"],
    # The 7-bead patch is small, so the splay modulus is kept on a tighter range
    # here than on the big sheets -- past ~3 it simply locks the patch rigid.
    params={},
    presets={
        # The paper's standard conditions (also the declared defaults).
        "paper": {},
        # Below k_tilt ~ 10 the patch stops holding itself planar.
        "floppy": {"k_tilt": 2.0, "k_splay": 0.1},
        # Stiff and strongly aligned: the ring barely yields to the puller.
        "rigid": {"k_tilt": 40.0, "k_splay": 2.5},
        # Orientation switched off entirely -- what is left is a plain
        # isotropic 4-2 fluid, which is the useful null comparison.
        "isotropic_only": {"k_tilt": 0.0, "k_splay": 0.0, "wc": 0.0},
    },
    # Cold -> rigid flat patch; hot -> directors disorder and the patch frays. The
    # default starts near-frozen so the first thing you see is the geometry.
    temperature=(0.0, 0.5),
    temperature_default=0.001,
    melt_temp=0.3,
    particle_radius=0.5,
    reduced_units=True,
)

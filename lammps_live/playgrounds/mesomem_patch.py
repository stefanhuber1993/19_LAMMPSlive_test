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
from ..render_style import DEFAULT_STYLE

# Seven beads filling the frame is not the dense box the default look was tuned
# on, and two of its effects have to come down for it:
#
#   DEPTH OF FIELD  the scene is barely two bead-diameters deep, so the whole
#     patch should read as one sharp object. The focus plane sits mid-scene
#     (0.5) rather than near the front, the ramp is stretched over the entire
#     span, and the bokeh is cut to a couple of pixels -- just enough to soften
#     the outermost ring, not enough to look like a mistake.
#   DEPTH CUE  with seven beads there is no depth to cue; a strong fade would
#     just darken the far half of a single object. Kept weak and stretched.
#
# Everything else -- the wet material, the contact shadows, the outline that
# separates touching beads -- is what makes the patch read as solid, and is left
# exactly as the showreel had it.
STYLE = DEFAULT_STYLE.varied(
    dof_focus=0.5,          # sharp plane mid-scene, not near the front
    dof_range=1.0,          # the whole span to go out of focus
    dof_bokeh_px=2.5,       # (8.0 by default) -- 0 switches DoF off entirely
    cue_end=0.9,            # fade completes only at the very back
    cue_strength=0.35,      # and only a third of the way to the background
    # Left at the defaults, listed because they are the next dials to reach for:
    ao_strength=5.83,           # contact darkening between the touching beads
    outline_strength=12.0,      # how black the line around each bead is
    outline_edge_fraction=0.90,  # how far out it starts (0.94 = hairline)
)

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
    # than large and arbitrary. 4.2 sigma leaves 0.1 of clearance outside the
    # 2.0-sigma leash below, which is all the room the constraint needs -- the
    # walls are there to catch a particle that escapes, not to be seen.
    scenario=hex_patch(n_rings=1, a=1.0, box=4.2, settle_steps=300),
    mode="game",
    control=Control(
        atom="first",           # hex_patch orders the centre site first
        plane="xz",
        # Kept inside the ring's interaction range: a bead at height z sits
        # sqrt(1 + z^2) from its neighbours, and past the rc = 2.5 cutoff it would
        # detach and float free. At 2.0 that separation is 2.24 even at full
        # extension, so the membrane is still pulling and the bead snaps back on
        # release -- which is the thing worth feeling. (It was 2.8, far enough out
        # that the bond had already let go.)
        leash=(2.0, 2.0),
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
    render_style=STYLE,
)

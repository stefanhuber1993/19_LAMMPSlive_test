"""One MesoMem bead, and nothing else -- the first thing to put on the screen.

This is the tutorial's opening slide: a single directored bead, filling a quarter
of the frame, with the control net it can be dragged around drawn underneath it
and no box outline to distract from either. There is no physics to discover here
-- one bead has no neighbours, so every pair term in the force field is silent --
and that is the point. What is being taught is the CONTROLS: which axis moves the
bead, which one twists its director, what the arrows and the arcs around it mean,
where the net's edge is. Everything the later playgrounds assume you already know.

WHY A HEX PATCH WITH NO RINGS. `hex_patch(n_rings=0)` is exactly one site -- the
centre -- so this needs no new scenario: the same geometry, the same camera, the
same framing dials, one shell fewer. It also switches the patch's three soft
correction forces off for free, since they act on everything EXCEPT the controlled
particle and here there is nothing else; they are set to zero below anyway, so the
declaration says what it means rather than relying on that.

WHAT IS DELIBERATELY SMALL.

  THE FORCE  a bead in a membrane is held by its neighbours, and the patch's 4.0
    is picked to win against them. A lone bead is held by nothing, so the same
    number flings it to the leash before the hand has registered pushing. At 1.0
    against the default viscous damping the bead settles to a quarter sigma per
    tau: it crosses the net in about three seconds of real time, which is slow
    enough to steer deliberately and fast enough not to feel broken.
  THE TORQUE  same argument, more sharply. The membrane's tilt term is what
    absorbs a twist on the patch, and it is absent here; the patch's yaw_torque of
    1.0 would spin the director roughly 24 degrees per frame with nothing to stop
    it. Three tenths of that turns the director once in about three seconds of
    real time -- fast enough to point it anywhere, slow enough to stop it there.
  THE BOX AND THE NET  small so the camera comes right in. The framing rectangle
    below puts the bead at just under a quarter of the frame width, which is big
    enough to read the director spike and the two torque arcs against it, and the
    net at a bit over half the width, so its edge is on screen and the leash is
    something you can see yourself arrive at.

Units are the paper's LJ-reduced units (sigma = eps = m = 1).
"""
from ..playground import Control, Playground, hex_patch
from .mesomem_patch import STYLE as PATCH_STYLE

# The patch's look, with five things changed for a scene that is one sphere.
#
#   NO RING WHILE IT IS BEING DRIVEN. The cyan ring answers "which of these am I
#     holding?", and with one bead on screen there is nothing to answer: it is a
#     circle drawn around the only object in shot, competing with the director
#     spike and the two arrows for the eye. Kept for the RELEASED state, where it
#     says something the picture does not -- that you have let go.
#   NO BOX. There is nothing for a container outline to give scale to, and at this
#     zoom its near face crosses the frame behind the bead. `box_alpha=0` here and
#     not upstream because these values come AFTER `.on_light()`, which sets its
#     own box and net alphas -- so a varied() before it (as mesomem_patch does) is
#     overwritten by it, and only a varied() after it sticks.
#   A NET THAT READS. It is the only thing in the picture besides the bead, and it
#     is what the tutorial points at when it says where the bead can go, so it is
#     taken well up from the light theme's 120.
#   DEPTH OF FIELD  off entirely. The patch keeps a couple of pixels of it to
#     soften its outer ring; a single sphere has no outer ring and no depth, so any
#     blur at all is just a soft edge on the one object the scene is about.
#   DEPTH CUE  off, for the same reason. There is nothing behind the bead for it
#     to separate the bead from.
STYLE = PATCH_STYLE.varied(puller_ring="released", box_alpha=0, net_alpha=200,
                           dof_bokeh_px=0.0, cue_strength=0.0)

PLAYGROUND = Playground(
    name="One MesoMem bead (3D)",
    description="A single bead and the net it moves in -- the controls, before "
                "any physics: move it, twist it, read the arrows.",
    force_field="mesomem",
    # Sphere radius = sigma, as on the patch: it fixes the moment of inertia the
    # director's swing is felt through, and the twist here is meant to feel like
    # the twist there with the membrane taken away.
    force_field_options={"bead_diameter": 2.0},
    scenario=hex_patch(
        n_rings=0,                  # one site: the centre, on its own
        a=1.0,
        box=3.0,                    # invisible (box_alpha=0), and only a backstop
        # The patch's three soft corrections, off. With one particle they would
        # not fire anyway (they act on everything but the controlled one), but a
        # tutorial scene should not have a homing spring in it that the text does
        # not mention, and switching them off here is what guarantees that.
        k_center=0.0, k_align=0.0, k_home=0.0,
        # Nothing to relax: there are no neighbours to settle against.
        settle_steps=50,
        # Framing: a rectangle in the control plane, centred on the bead. 1.5 x 1.2
        # puts the bead at ~24% of the frame width (see Camera3D.fit_to_points --
        # the tighter of the two axes binds, and it is height at the default
        # window).
        view_center_z=0.0,
        view_half_width=1.5,
        view_half_height=1.2,
    ),
    mode="game",
    control=Control(
        atom="first",               # the only one there is
        plane="xz",
        # Well inside the frame, so the net's edge is visible on all four sides
        # and arriving at it is something you watch happen.
        leash=(1.2, 1.0),
        max_input_force=1.0,        # (4.0 on the patch -- see the docstring)
        yaw_torque=0.3,             # (1.0 on the patch -- likewise)
        grid_step=0.25,             # ~10 cells across the 2.4-sigma net
    ),
    # Nothing a single bead can report is interesting: coordination is zero,
    # thickness and tilt are statements about a membrane. The panels stay, and
    # they are what the tutorial points at; there is just nothing to plot.
    observables=[],
    presets={"paper": {}},
    temperature=(0.0, 0.5),
    # Cold, and the slider is inert -- which is worth knowing before someone reaches
    # for it in front of an audience. The thermostat acts on the `bath` group, which
    # is everything EXCEPT the controlled particle (see modes.GameMode), and here
    # there is nothing else: the one bead in shot is the one bead the bath does not
    # touch. Left on the panel because it is the same panel every other playground
    # has, and starting at zero so it says nothing rather than something wrong.
    temperature_default=0.0,
    melt_temp=0.3,
    particle_radius=0.5,
    reduced_units=True,
    # Off, and not offered: this is the one scene where an individual bead's own
    # motion IS the subject, and smoothing it would soften exactly the response
    # the tutorial is pointing at.
    trajectory_smoothing=False,
    render_style=STYLE,
)

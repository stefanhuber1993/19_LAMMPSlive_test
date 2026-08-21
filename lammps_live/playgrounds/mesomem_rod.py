"""MesoMem membrane wrapping a rod -- steer a bacterium into a membrane.

The membrane of `mesomem_sheet` with one rigid rod added: a long, non-spherical
particle -- think of a rod-shaped bacterium against a cell membrane -- interacting
with the beads through Pietro Sillano's `rod_lj` pair style. That style is a
Lennard-Jones between a bead and the rod's AXIS SEGMENT, applied at the closest
point on it, so the rod feels a torque as well as a force and the membrane can
genuinely wrap it rather than just be dented by it.

WHAT THERE IS TO DO. The rod starts a little above the membrane, out of reach.
Bring it down (the joystick's two axes slide it in the world xz-plane, the same
control plane as the patch, drawn as the net) until adhesion grabs it. From there:

  * push it in, and the membrane tents around the body -- the reaction on the
    stick is the membrane's own resistance, recovered from the pair style;
  * twist (Q/E, or the stick's twist axis) to rotate the rod within the plane.
    Lying flat is what adhesion wants and what gets wrapped; stand it on end and
    you are pushing a blunt tip through a bilayer, which needs far more force;
  * once it is in, let go of the drive (B, or the trigger) and watch what the
    membrane does with it on its own. At the reference conditions it holds the rod
    in a groove with about a hundred beads on it; turn `eps_rod` up and the beads
    climb further round, turn it down and the rod barely dents the surface.

THE MEMBRANE IS AT CONSTANT LATERAL PRESSURE, not in a fixed cell, and that is
what makes an invagination possible at all: covering a rod costs area, and in a
frozen periodic cell the only place that area can come from is stretching the
lattice, so the membrane dents instead of engulfing. A barostat runs throughout
here -- as it does through all three rod phases of the reference deck -- so the
projected area shrinks as the wrap grows. You can watch it: the cell is visibly
smaller by a couple of per cent once the rod is in. `baro_press` is the dial for
putting the membrane under tension instead, which suppresses wrapping.

Where the wrapping transition sits -- how much adhesion it takes to make the
membrane pay the bending cost of covering a rod of a given radius -- is a live
question rather than something this file can assert. One thing worth knowing
while looking for it: a rod already held by adhesion cannot be twisted, because
the restoring torque on the arc is bigger than the steering. That clamping IS the
wrap.

The rod is drawn as a mottled capsule rather than in the membrane's own colours,
because it is not made of membrane: see `body_material` in STYLE below.

The HUD's three numbers are the wrapping story: how deep the rod sits (negative
once the mean membrane surface has closed over its centre), how many beads are
touching it, and whether it is lying flat or standing up.

The view is a SECTION. The near half of the cell is not drawn, because a monolayer
is opaque and the rows of beads between the camera and the rod sit at exactly the
height the rod is being pushed to -- so they hide the invagination from every
angle. Cut them away and the remaining face is the picture: the membrane as a
line, and the rod sinking into it.

Units are the paper's LJ-reduced units (sigma = eps = m = 1). The collaborator's
original LAMMPS deck is kept beside the pair style, at
`forcefields/mesomem_ff/planar_wrapping_rod.lmp`.
"""
from ..playground import Control, Playground, rod_on_sheet
from ..render_style import DEFAULT_STYLE

# The sheet's look, with the two depth effects pulled back. The subject here is a
# single object at a known distance -- the rod, and the dimple under it -- not the
# receding surface the sheet playground is a picture of, so a strong tilt-shift
# blur would soften exactly the contact the demo is about.
#
#   DEPTH OF FIELD  focused mid-scene, where the rod is: seen edge-on the cell's
#     46 sigma of depth all lies along the view axis, so the far rows recede into
#     a soft horizon and the near ones out of the bottom of the frame, and the
#     section through the rod is the sharp part.
#   SECTION  the near half of the cell is not drawn at all, which is what makes
#     an edge-on view of an opaque monolayer readable. See `section_min` below.
#   PERIODIC IMAGES  OFF, unlike the sheet. The sheet tiles its cell because 900
#     beads is a small raft and the copies are what make it read as a piece of
#     something endless. This cell is 3600 beads and 54 sigma across, the camera
#     frames the middle third of it, and the membrane already runs off all four
#     edges of the frame -- so the copies would be paying for geometry nobody can
#     see, and every one of them would carry another rod. One cell, drawn once.
#   BOX OUTLINE  off with them, for the same reason: at this size the container's
#     near face is behind the camera and its far face is off in the haze, so the
#     outline is four lines nobody can connect into a box.
STYLE = DEFAULT_STYLE.varied(
    periodic_images=(0, 0, 0),
    box_alpha=0,
    # THE CUT. Draw only the far half of the cell, so what faces the camera is a
    # section through the membrane at the rod's own plane. Without it this view
    # cannot work at all: a monolayer is opaque, and the rows of beads between the
    # camera and the rod sit at exactly the height the rod is being pushed to, so
    # they hide the invagination however the camera is angled. The rod's body is
    # exempt and stays whole -- see RenderStyle.section_min.
    section_axis=(0.0, 1.0, 0.0),
    section_min=0.0,
    net_alpha=150,
    dof_focus=0.5,
    dof_range=1.5,
    dof_bokeh_px=4.0,
    cue_end=0.90,
    cue_strength=0.55,
    ao_strength=9.83,           # contact darkening where the beads meet the rod
    outline_strength=12.0,
    outline_edge_fraction=0.90,
    # THE ROD IS NOT A BEAD, so it is not painted like one. Its body is drawn in
    # its own mottled material -- see RenderStyle.body_material for the two
    # reasons the bead colourings cannot serve it, the second of which is a hard
    # one: on the energy ramp the rod's own potential runs to several hundred eps
    # against a bead's single digits, so it sits pinned at the bright end in every
    # configuration and the brightest object in the frame is the one whose colour
    # means nothing. A cell wall instead, whichever colouring the beads are in.
    body_material="bacterium",
).on_light()

# Rod geometry, in sigma. Repeated here because three declarations below need to
# agree with each other: the force field's dials, the height the scenario places
# the rod at, and the leash that has to reach from there to the membrane. The
# reference deck's own numbers (L = 5, D = 3).
ROD_LENGTH = 5.0
ROD_RADIUS = 1.5
# Comfortably outside the rod-membrane cutoff (~3.24 at this radius), and inside
# the leash below -- the leash is centred on the origin, so a rod placed above it
# would be yanked down to the limit on the first frame. tests/test_rod_wrapping.py
# pins both, via RodOnSheet.verify_reach.
ROD_HEIGHT = 3.5

PLAYGROUND = Playground(
    name="MesoMem membrane + rod (3D)",
    description="Steer a rod-shaped 'bacterium' into a MesoMem membrane and feel "
                "it wrap: adhesion vs bending, live.",
    force_field="mesomem_rod",
    # Beads at diameter = sigma, as on the other sheets, so the overlapping
    # monolayer reads as a continuous membrane rather than a raft of marbles. The
    # rod's mass is the force field's business (see MesoMemRod.__init__): heavy
    # enough to feel substantial against a bead, light enough for a stick to
    # steer, which the reference deck's density matching is not.
    force_field_options={"bead_diameter": 1.0, "rod_mass": 6.0},
    # 3600 beads over 54 x 47 sigma. Four times the sheet's count, and the size is
    # the reason: an invagination is a long-ranged deformation, and at 24 sigma
    # across the rod was wrapping into a cell barely three body-lengths wide, so
    # the dimple met its own periodic image before it had decayed. This gives it
    # room to die away, and gives the barostat enough membrane that the area it
    # gives up to a wrap is a small strain rather than a visible squeeze.
    scenario=rod_on_sheet(
        n_cols=60, n_rows=60,
        # The reference deck's lattice spacing. The barostat settle then relaxes
        # it to whatever this force field's tension-free spacing actually is, so
        # this is a starting point rather than a claim.
        a=0.9,
        # Deeper than the sheet's 4.0: the rod travels several sigma out of plane
        # and the container has to hold the whole of that, plus the dimple it
        # pushes the membrane into.
        z_half=6.0,
        settle_steps=1000,
        # 6 steps a frame, not the sheet's 20, and this costs the demo nothing.
        # The step size itself is fixed by stability -- measured, dt = 0.0075 and
        # 0.01 both blow the membrane up at the top of the adhesion dial with the
        # temperature raised, where 0.005 survives -- so the only dial left is how
        # many steps go into a frame. And LAMMPS costs the same per step whatever
        # the chunk size, so a smaller chunk buys a smoother PICTURE at the same
        # rate of physics: measured on 3600 beads, 10 steps a frame is 35 ms
        # (28 fps, 283 steps/s) and 6 is 23 ms (44 fps, 261 steps/s). The wrap
        # therefore takes the same wall time to form either way; it just arrives
        # in more frames.
        timestep=0.005,
        sim_time_per_frame=0.08,
        rod_height=ROD_HEIGHT,
        # Lying flat, along the control plane's horizontal axis -- the orientation
        # adhesion wants, so the demo starts from the interesting configuration
        # and twisting away from it is what costs. `(0, 0, 1)` starts it on end.
        rod_axis=(1.0, 0.0, 0.0),
        # No diffusion tracer: on this playground the rod is the thing to watch,
        # and a second highlight elsewhere on the membrane only competes with it.
        tracer_fraction=None,
        # Frame the middle third, not the whole cell. `view_span` scales the
        # camera's DISTANCE as well as the zoom, so this is the same picture from
        # closer in: the rod fills a useful part of the frame and the membrane
        # runs off all four edges, which is what a big membrane should look like.
        # Aimed straight at the middle, since there are no receding copies to
        # leave room for.
        view_span=0.38, view_aim_ahead=0.0,
        # Edge-on, so the membrane reads as a line and the rod is seen sinking
        # INTO it in section -- which is what a wrap looks like. A few degrees up
        # rather than exactly zero, to lift the sight line clear of the near rows.
        # See RodOnSheet.camera.
        view_elevation_deg=8.0,
    ),
    mode="game",
    control=Control(
        atom="last",            # rod_on_sheet appends the rod after the sheet
        plane="xz",
        # z reaches from well clear of the membrane down past it, so the rod can
        # be lifted out of contact AND pushed through to the far side; x is wide
        # enough to drag a wrapped rod sideways through the membrane.
        leash=(7.0, 5.0),
        # A rod pushing on ~50 beads at once meets far more resistance than a
        # single bead does, and it weighs a dozen beads. Both want more authority
        # than the sheet's 7.
        max_input_force=30.0,
        # It is heavy, so it wants more damping than a bead to stop it coasting
        # after the stick is released.
        damping_default=8.0,
        damping_range=(0.0, 20.0),
        # Steering is an angular-velocity kick, so the rod's inertia does not
        # blunt it -- and at this scenario's 20 steps a frame the patch's 1.0
        # would spin the rod most of a radian per frame. 0.3 turns a free rod
        # about half a revolution a second, which is a rod turning rather than
        # flailing, and leaves adhesion able to win: a rod pressed into the
        # membrane holds its orientation against a sustained twist (measured:
        # ~60 of restoring torque against this steering), which is the clamping
        # that a wrap IS. Raise it and the rod plows through instead.
        yaw_torque=0.3,
        rot_damp=0.85,
        # Display normalisation for the reaction-torque arc, not a limit: the
        # restoring torque here is adhesion pulling on a lever arm of half the
        # rod, which runs an order above the membrane's own tilt term. Measured
        # peak while twisting a rod pressed into the membrane: ~60.
        reaction_torque_max=60.0,
        grid_step=1.0,
    ),
    # The three numbers a wrap is told in. `coordination` would work here too --
    # the rod finds its own long-ranged pairs rather than widening the membrane's
    # list to reach them (see MesoMemRod.extended_pairs) -- but it says nothing
    # about the rod, which is the subject.
    observables=["rod_height", "rod_contacts", "rod_tilt_deg"],
    params={"rod_length": ROD_LENGTH, "rod_radius": ROD_RADIUS},
    param_ranges={
        # The wrapping transition is the thing worth finding, and it sits well
        # below the membrane's own default stiffness -- a floppy membrane wraps a
        # rod that a stiff one merely holds.
        "k_tilt": (0.0, 30.0),
    },
    presets={
        # The reference deck's conditions: eps = 3, L = 5, D = 3, paper moduli.
        # Brought into contact and released, the membrane invaginates until the
        # rod's centre sits at the mean surface, with the beads closed most of the
        # way round it (measured: height +0.07, ~120 touching, and the cell 2%
        # smaller than it started -- that shrinkage IS the area the wrap took).
        "reference": {},
        # Adhesion too weak to pay for any bending: the rod rests on the surface
        # and dents it.
        "weak_adhesion": {"eps_rod": 0.8},
        # More adhesion is NOT more wrapping, which is worth seeing: at the top of
        # the dial the rod is gripped so hard at first contact that the beads pile
        # onto its flanks instead of closing over it, and it ends up sitting
        # HIGHER than at the reference (measured: +1.4 against +0.07). The
        # transition is a balance, not a monotone.
        "strong_adhesion": {"eps_rod": 8.0},
        # Strong adhesion against a membrane with no orientational stiffness to
        # resist it: the beads go all the way round (measured: ~210 touching,
        # against ~120 at the reference). A plain isotropic fluid rather than a
        # bilayer, so this is the adhesion-wins limit rather than physics.
        "engulfed": {"eps_rod": 8.0, "k_tilt": 0.0},
        # Twice the radius at the same adhesion: twice the curvature to pay for,
        # over more area. The comparison that shows the transition is adhesion
        # AGAINST bending rather than adhesion alone.
        "fat_rod": {"rod_radius": 3.0},
        # A short, stubby rod, which is nearly the spherical-nanoparticle limit
        # the wrapping literature usually treats.
        "stubby": {"rod_length": 1.5},
    },
    temperature=(0.0, 0.5),
    temperature_default=0.001,
    melt_temp=0.3,
    particle_radius=0.5,
    reduced_units=True,
    # The rod's panel is dominated by adhesion: it is one particle in contact with
    # tens of beads at once, where a bead touches a dozen, so its share of the
    # energy runs two orders above the force field's per-particle scale and the
    # panel would otherwise sit pinned at full deflection. Measured: the adhesion
    # bar reaches ~-370 on a rod driven hard into the membrane.
    pulled_energy_scale=450.0,
    # The energy panels are a pass over every pair, and at 3600 beads that is the
    # one thing in the frame big enough to be felt as a hitch when it lands. The
    # aggregate barely changes frame to frame, so it is evaluated half as often as
    # the default. (The other half of that problem -- the rod's long reach
    # dragging the whole pair list out with it -- is solved in the force field,
    # by MesoMemRod.extended_pairs.)
    analysis_energy_every=8,
    # The membrane ripples thermally at every wavelength at once; smoothing the
    # drawn beads leaves the wrap itself -- a slow, collective change -- legible
    # without the shimmer on top. Advanced slider, 0 (off) by default.
    trajectory_smoothing=True,
    render_style=STYLE,
)

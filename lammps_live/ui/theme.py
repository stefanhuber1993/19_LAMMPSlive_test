"""Shared colors and layout constants for the pygame UI."""
import math

BG = (18, 18, 24)
CRYSTAL_COLOR = (230, 140, 40)   # fallback flat atom color; systems set their own (spec.crystal_color)
INPUT_VEC_COLOR = (80, 220, 120)   # joystick/mouse-commanded input force
REACTION_VEC_COLOR = (255, 90, 90)  # crystal's interaction-force reaction
BOX_OUTLINE = (90, 90, 100)
PANEL_BG = (24, 24, 30)
PANEL_DIVIDER = (60, 60, 68)
TEXT_COLOR = (200, 200, 200)
DIM_TEXT_COLOR = (130, 130, 140)
HEADER_TEXT_COLOR = (235, 235, 240)
GRID_COLOR = (45, 45, 52)
SLIDER_TRACK = (70, 70, 80)
SLIDER_HANDLE = (230, 230, 235)
SLIDER_HANDLE_ACTIVE = (255, 210, 90)
MELT_MARK_COLOR = (255, 110, 90)
# Marker for a slider's recommended "optimum" value (the paper's sweet spot),
# drawn as a distinct tick + "opt" label so it reads differently from the red
# melt marker on the temperature slider.
OPTIMUM_MARK_COLOR = (120, 230, 170)

# Faint "bond" lines between atom pairs, with opacity encoding how close their
# separation is to the system's equilibrium nearest-neighbor spacing
# (spec.lattice_spacing -- the optimal bonding distance). Alpha peaks at
# BOND_PEAK_ALPHA when a pair sits exactly at the optimum and falls off
# exponentially in either direction:
#     alpha = BOND_PEAK_ALPHA * exp(-|d - d_opt| / (BOND_FALLOFF * d_opt))
# so BOND_FALLOFF is the decay length (lambda) as a fraction of d_opt. Pairs
# whose alpha would drop below BOND_MIN_ALPHA are skipped -- both a visibility
# floor and the distance cutoff that bounds how many pairs get drawn. All
# tunable:
BOND_LINES_ENABLED = True
BOND_COLOR = (255, 255, 255)   # white
BOND_PEAK_ALPHA = 128          # out of 255 -> 50% opacity exactly at the optimum
BOND_FALLOFF = 0.10            # decay length (lambda) as a fraction of the optimal distance
BOND_MIN_ALPHA = 3             # skip lines fainter than this (also sets the effective cutoff)
BOND_WIDTH = 1                 # line thickness, px

# Glyph color for per-species atom labels (e.g. the +/- stamped on ions).
# Per-species fill colors themselves live on each system's SystemSpec
# (species_colors), since they're system-specific.
ION_LABEL_COLOR = (20, 20, 26)

# Explicit molecular backbone bonds (e.g. lipid head-tail-tail chains):
# subtle gray for ordinary molecules, bright cyan for the control lipid so
# "your lipid" and the way it points stand out.
BOND_STICK_COLOR = (95, 95, 110)
PULLER_BOND_COLOR = (110, 220, 255)

# Hydrogen-bond overlay (water demo): a lighter, cooler, thinner line than the
# solid intramolecular bonds, drawn dashed so a transient H-bond reads as
# "weak/breakable" rather than a fixed covalent stick.
HBOND_COLOR = (150, 200, 245)
HBOND_WIDTH = 1
HBOND_DASH = 6   # px on/off dash period; 0 -> solid

# Small per-system HUD lines drawn in the sim view (etch tally, water phase...).
HUD_TEXT_COLOR = (210, 214, 224)
HUD_BG = (0, 0, 0, 130)

CRYSTAL_RADIUS = 5    # fixed-pixel fallback when a system sets no physical radius
PULLER_RADIUS = 8

# The interactively-controlled ("puller") atom is drawn in its true species
# color -- a deposited Cu atom IS a Cu atom, the pulled Na+ IS a cation -- so it
# no longer masquerades as a differently-colored species. What marks it as
# "the one you control" is instead a bright ring around it (a color no atom
# fill uses, so it always contrasts) plus a small size boost. This keeps the
# picture chemically honest while still making the controlled atom pop.
PULLER_RING_COLOR = (95, 230, 255)
# ...and how it reads once released (B / joystick trigger): still marked, because
# it is still the particle the ring will grab again, but dim and cool so "you are
# not holding this" is legible at a glance rather than only in the HUD line.
PULLER_RING_FREE_COLOR = (95, 110, 130)
PULLER_RING_WIDTH = 3
PULLER_RADIUS_BOOST = 3   # extra px on the controlled atom over its species size
# Per-atom energy readout stamped under the controlled atom.
PULLER_LABEL_COLOR = (240, 240, 248)
PULLER_LABEL_BG = (0, 0, 0, 120)

# Physical-radius drawing (spec.atom_radius_A / species_radii_A) is converted to
# pixels at the live box scale, then clamped to this band so atoms stay visible
# on a big box and don't swallow the view (and the bond overlay) on a small one.
ATOM_MIN_RADIUS = 4
ATOM_MAX_RADIUS = 22
# Arrow length: soft-saturating (tanh) rather than linear, since the
# interaction force can spike well past what the input force ever reaches --
# a fixed linear scale either makes small vectors invisible or huge ones fly
# off-screen.
VECTOR_MAX_PX = 130.0
ARROWHEAD_LEN = 8
ARROWHEAD_ANGLE = math.radians(25)

# Circular "torque" arrows drawn around the puller bead: an arc starting at the
# top of a ring and sweeping up to a semicircle left/right in proportion to the
# applied (green) / reaction (red) torque about the control-plane normal. Radius
# is a fraction of the puller's on-screen radius; the two arcs use slightly
# different radii so they don't overdraw each other.
TORQUE_ARC_APPLIED_RADIUS = 0.62  # x puller radius (green, user's steering torque)
TORQUE_ARC_REACTION_RADIUS = 0.82  # x puller radius (red, membrane's restoring torque)
TORQUE_ARC_WIDTH = 3
TORQUE_ARC_HEAD_LEN = 9

# --- 3D scene (MesoMem membrane patch and future 3D systems) ---------------
# Depth cueing ("haze"): distant beads are blended toward this color so the
# scene recedes into the background instead of reading as a flat cluster. Kept
# equal to BG so far atoms melt into the void; STRENGTH caps how much of the
# original color is washed out at the far plane (1.0 = fully background).
# The fog is now anchored to the scene's own depth extent: nothing hazes until
# DEPTH_FADE_START of the farthest bead's distance, then it ramps LINEARLY to a
# full wash-out at the farthest bead -- a strong, distance-proportional cue.
HAZE_COLOR = BG
HAZE_STRENGTH = 1.0
DEPTH_FADE_START = 0.5   # fraction of the max bead distance at which haze begins
# Screen-space edge vignette applied to the beads of periodic scenes (the
# membrane sheet): beads fade toward the background over the outer frame margin,
# uniformly in screen space, softening the frame edge and the periodic clip seam
# without darkening the white box outline. 0 disables it; 1 fully fades the very
# edge to BG. See gl3d._COMPOSITE_FS.
EDGE_VIGNETTE_STRENGTH = 0.65
# Light direction for the shaded-sphere sprite (points FROM the light, in
# screen space: up-left and toward the viewer). Sphere shading is baked once
# from this and reused for every bead.
SPHERE_LIGHT_DIR = (-0.5, -0.6, 0.62)
SPHERE_AMBIENT = 0.28          # floor brightness on the dark side of a bead
# The director "spike" (orientation vector n_i) drawn standing out of each
# bead, like the cones in the MesoMem figures.
DIRECTOR_COLOR = (245, 214, 66)     # warm yellow
DIRECTOR_LEN_R = 1.55               # spike length as a multiple of bead radius
DIRECTOR_BASE_R = 0.42              # spike base half-width as a multiple of bead radius
# The control "net": the plane (perpendicular to the membrane, facing the
# screen) that the joystick slides the active bead along, drawn as a faint grid.
NET_COLOR = (120, 150, 210)
NET_ALPHA = 60
NET_LINE_ALPHA = 100
# The net is redrawn, a little brighter, clipped to the active bead's disc, so
# the control plane visibly cuts THROUGH that bead (a plane through a solid
# sphere's centre is otherwise hidden by its near hemisphere) -- makes clear the
# bead rides in the plane.
NET_GHOST_ALPHA = 110
# Thin in-bead director arrow (cylinder shaft + cone head) marking which way n_i
# points -- so a flip between the two equivalent normals (+n vs -n) is visible.
DIRECTOR_ARROW_COLOR = (28, 30, 40)
# Live additive-potential breakdown panel (get_potential_terms): one distinct
# colour per term, plus a neutral colour for their sum, drawn as signed bars.
POTENTIAL_COLORS = ((120, 205, 225), (250, 180, 95), (200, 140, 240))  # isotropic, tilt, splay
POTENTIAL_TOTAL_COLOR = (225, 228, 236)
POTENTIAL_PANEL_BG = (10, 12, 18, 185)
POTENTIAL_TRACK_COLOR = (58, 62, 76)
# Membrane bead fill and the bright stick backbone linking neighbors in 3D.
MEMBRANE_BEAD_COLOR = (232, 104, 98)
BOND_3D_COLOR = (150, 96, 92)
# The simulation-box outline drawn around the 3D scenes: white, faded by depth
# cueing like everything else so far edges recede into the background.
BOX_3D_COLOR = (240, 240, 245)
BOX_3D_ALPHA = 150   # out of 255; the box is a subtle frame, not a hard cage
# The corner of the box nearest the eye hangs in front of everything and streaks
# across the scene, so it is faded out. FADE_DEPTH is how much of the box's own
# depth span the fade covers, measured back from its nearest corner: 0 disables
# it, 0.3 dissolves roughly the near third of the three edges meeting that
# corner. SUBDIVISIONS is how many pieces each edge is cut into to carry the
# gradient -- enough that it reads as smooth, few enough to stay 12*n lines.
BOX_EDGE_FADE_DEPTH = 0.30
BOX_EDGE_SUBDIVISIONS = 12
# Paper (MesoMem) bead coloring: each bead is a little sphere whose POLES (along
# its director n_i) are hydrophilic -> blue, and whose EQUATOR (perpendicular to
# n_i) is hydrophobic -> yellow. The yellow band therefore tilts with the
# director, so tilt/splay show directly in the coloring. BAND_HALFWIDTH is the
# band's half-height as |cos(latitude)| (0 = equator, 1 = pole); BAND_SOFT is
# the soft transition width between yellow and blue.
BEAD_POLE_COLOR = (122, 165, 217)      # blue, hydrophilic poles
BEAD_EQUATOR_COLOR = (247, 225, 122)   # yellow, hydrophobic equator
BEAD_BAND_HALFWIDTH = 0.30
BEAD_BAND_SOFT = 0.07
# One pole (the +director pole) is over-painted WHITE, so which way a bead's
# director points is unambiguous (the two normals +n / -n no longer look
# identical). The cap is a small dot right at the pole: white where the SIGNED
# cos-latitude s = N.n exceeds BEAD_WHITE_POLE_MIN, with a BEAD_WHITE_POLE_SOFT
# transition. The projected on-screen diameter of the cap scales as sqrt(1 -
# MIN^2), so MIN = 0.954 (sin = 0.30) makes it half the diameter of the older
# MIN = 0.80 (sin = 0.60) dot. Only s > 0 (the +n hemisphere) is affected, so
# the opposite pole keeps the blue/yellow banding.
BEAD_WHITE_POLE_COLOR = (245, 246, 250)
BEAD_WHITE_POLE_MIN = 0.954
BEAD_WHITE_POLE_SOFT = 0.03

# --- the other bead colouring: potential energy -------------------------------
# Instead of the director banding, paint each bead by ITS OWN potential energy in
# the force field (LAMMPS' per-atom pe). The banding answers "which way is this
# bead pointing"; this answers "how bound is it" -- so a bead pulled out of the
# membrane brightens as its bonds stretch, a well-packed one stays dark, and on
# the assembly box the ordered patches read as dark against the loose gas. The
# +director pole keeps its white cap in both modes, so orientation is never lost.
#
# Matplotlib's `inferno`, sampled at 32 points and interpolated in the shader:
# perceptually uniform (equal steps in the number look like equal steps in the
# colour) and dark-to-bright, which is the right direction for "more energy".
# Baked in rather than imported so matplotlib is not a runtime dependency.
INFERNO = (
    (0.0015, 0.0005, 0.0139), (0.0140, 0.0112, 0.0719), (0.0423, 0.0281, 0.1411), (0.0820, 0.0433, 0.2153),
    (0.1358, 0.0469, 0.2998), (0.1904, 0.0393, 0.3614), (0.2450, 0.0371, 0.4000), (0.2972, 0.0475, 0.4205),
    (0.3540, 0.0669, 0.4309), (0.4039, 0.0856, 0.4332), (0.4537, 0.1038, 0.4305), (0.5035, 0.1216, 0.4234),
    (0.5596, 0.1413, 0.4101), (0.6093, 0.1595, 0.3936), (0.6585, 0.1790, 0.3727), (0.7065, 0.2007, 0.3478),
    (0.7584, 0.2291, 0.3153), (0.8019, 0.2587, 0.2831), (0.8420, 0.2929, 0.2486), (0.8780, 0.3321, 0.2123),
    (0.9130, 0.3816, 0.1698), (0.9387, 0.4301, 0.1304), (0.9591, 0.4820, 0.0895), (0.9742, 0.5368, 0.0484),
    (0.9846, 0.6011, 0.0236), (0.9879, 0.6603, 0.0517), (0.9856, 0.7208, 0.1122), (0.9775, 0.7823, 0.1859),
    (0.9625, 0.8515, 0.2855), (0.9487, 0.9105, 0.3953), (0.9517, 0.9606, 0.5242), (0.9884, 0.9984, 0.6449),
)

# Play / Pause / Reset playback buttons (self-assembly system), drawn along the
# bottom of the sim view. The button matching the current state (e.g. Play while
# running) is highlighted with the active colors.
BUTTON_BG = (40, 42, 52)
BUTTON_TEXT = (215, 218, 228)
BUTTON_BORDER = (90, 94, 108)
BUTTON_ACTIVE_BG = (70, 150, 220)
BUTTON_ACTIVE_TEXT = (12, 14, 20)

PANEL_WIDTH = 480
PANEL_PAD = 14
PLOT_COLORS = {
    "temp": (255, 150, 90),
    "press": (120, 180, 255),
    "ke": (120, 220, 140),
    "pe": (230, 130, 230),
    "etotal": (240, 220, 100),
    "rdf": (100, 210, 255),
}

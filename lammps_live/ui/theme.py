"""Shared colors and layout constants for the pygame UI."""
import math

BG = (18, 18, 24)
CRYSTAL_COLOR = (230, 140, 40)
PULLER_COLOR = (60, 200, 255)
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
# Tail beads of the multi-bead control lipid -- dimmer than the head's
# PULLER_COLOR so the head (and thus the lipid's orientation) is legible.
PULLER_TAIL_COLOR = (45, 120, 165)

CRYSTAL_RADIUS = 5
PULLER_RADIUS = 8
# Arrow length: soft-saturating (tanh) rather than linear, since the
# interaction force can spike well past what the input force ever reaches --
# a fixed linear scale either makes small vectors invisible or huge ones fly
# off-screen.
VECTOR_MAX_PX = 130.0
ARROWHEAD_LEN = 8
ARROWHEAD_ANGLE = math.radians(25)

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

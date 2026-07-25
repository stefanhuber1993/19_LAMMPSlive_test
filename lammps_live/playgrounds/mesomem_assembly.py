"""MesoMem self-assembly -- the paper's spontaneous-lamellae test, made live.

The paper's first validation experiment: N = 1500 directored particles dropped at
random positions and orientations into a cubic, fully periodic box of side
L = 20 sigma -- a reduced volume fraction phi = N*Vp/Vbox ~ 0.1 with
Vp = (pi/6) sigma^3. Under Langevin dynamics the disordered gas coarsens: small
patches by t ~ 500 tau, coalescing into large planar membranes by t ~ 2000 tau.

k_tilt is the critical control parameter -- below ~10 the aggregates stay compact
and isotropic, above it they flatten into membranes. Dial it live and watch the
morphology change; the `nematic_S` observable is the number that transition shows
up in, so you can see it rather than squint at it.

Runs in sim mode: Play / Pause / Reset, where Reset re-randomizes the box so a new
run starts from a fresh disordered state with whatever coefficients the sliders
currently hold. Because the mode is not baked in, `--mode game` also works here --
it will pick a particle near the box centre and let you probe whatever has
assembled.

Units are the paper's LJ-reduced units (sigma = eps = m = 1).
"""
from ..playground import Playground, random_fill

PLAYGROUND = Playground(
    name="MesoMem self-assembly (3D)",
    description="Paper's spontaneous-assembly run: 1500 random beads in a periodic "
                "20-sigma box coarsen into membranes. Play / Pause / Reset.",
    force_field="mesomem",
    scenario=random_fill(n=1500, box=20.0, overlap=0.9, maxtry=200),
    mode="sim",
    observables=["nematic_S", "coordination", "thickness"],
    param_ranges={"k_splay": (0.0, 40.0)},
    presets={
        "paper": {},
        # Below the k_tilt ~ 10 threshold: compact isotropic droplets instead of
        # membranes. The clearest single-slider demonstration in the model.
        "compact_droplets": {"k_tilt": 4.0},
        "strongly_planar": {"k_tilt": 30.0},
        "isotropic_only": {"k_tilt": 0.0, "k_splay": 0.0, "wc": 0.0},
    },
    # Assembly needs finite T so beads can diffuse and anneal into flat membranes,
    # but below the ~eps attraction well so they stay condensed rather than boiling
    # back into a gas. The default sits in that fluid-membrane window.
    temperature=(0.0, 0.5),
    temperature_default=0.2,
    melt_temp=0.3,
    particle_radius=0.5,
    reduced_units=True,
)

"""MesoMem self-assembly on a cluster GPU, drawn here -- the same demo, bigger.

Physically this is `mesomem_assembly`: the paper's spontaneous-lamellae run, N
directored beads dropped at random into a fully periodic cube at reduced volume
fraction phi = 0.1, coarsening under Langevin dynamics. Every coefficient, the
thermostat, the two watchability nudges and the whole look are that playground's;
what changes is where it runs and therefore how big it can be.

    N = 10,000 beads in a 37.4 sigma cell, against 1,500 in a 20 sigma cell.

WHY 10,000 AND NOT 100,000. This is the size the *client* can keep up with, not
the size the GPU can run -- docs/a100-plan.md section 3 measures the Python
analysis at 1.5 us/bead/chunk, which against a 16.7 ms frame caps N near 11,000
however fast the simulation is. So 10k is the honest first target: it fills the
frame budget on the drawing machine while leaving the A100 at a few percent of
its capacity, which makes it the right size to find out whether the *pipeline*
works before spending effort on making the analysis cheap enough for 100k. The
two things that then have to change are named in that document, and neither is in
this file.

THE DECK IS NOT WRITTEN OUT ANYWHERE. `docs/snellius/in.mesomem_100k` exists to
benchmark with, and had to be maintained by hand against this app's own setup.
Here the server builds this playground -- this file, this force field, this
scenario -- so there is one definition and no drift. Set `n` below and both ends
follow.

RUNNING IT. Select it in the app like any other playground; it comes up
disconnected, with a Connect panel. That panel asks the cluster for a GPU
(`salloc --no-shell`), ships this package to it, starts
`lammps_live.remote.server` inside the allocation, tunnels a port back, and
cancels the job when the window closes. The one thing it cannot do for you is the
login prompt, which is why the panel has a field for it.

For the pipeline without the cluster -- what the tests use, and the way to work on
the renderer while offline -- run the server on this machine:

    python -m lammps_live.remote.server --playground mesomem_remote \\
           --profile local --token dev --port 5723
    lammps-live --playground mesomem_remote --remote 127.0.0.1:5723 --token dev
"""
import math

from ..playground import Playground, random_fill
from ..remote import RemoteTarget
from .mesomem_assembly import STYLE
from ..render_style import CameraOrbit

# The size, and the cell that puts it at the paper's volume fraction. Written as
# the relation rather than as a number so changing N keeps the physics: with
# Vp = (pi/6) sigma^3, phi = N*Vp/L^3, so L = (N*(pi/6)/phi)^(1/3). 10k at
# phi = 0.1 is 37.41 sigma, which holds several independent membranes rather than
# the single one the 1500-bead cell manages.
N_BEADS = 50_000
PHI = 0.1
BOX = (N_BEADS * (math.pi / 6.0) / PHI) ** (1.0 / 3.0)

# The visual style is imported, not copied: this is the same scene as the local
# assembly box and should look identical. It survives the change of scale because
# every parameter in it is a fraction -- of the scene's depth span, of a bead
# radius -- rather than an absolute length (see render_style.py).

PLAYGROUND = Playground(
    name="MesoMem self-assembly, remote GPU (3D)",
    description=f"{N_BEADS:,} beads assembling on a cluster A100, drawn here. "
                f"Play / Pause / Reset, and every slider, over the wire.",
    force_field="mesomem",
    scenario=random_fill(
        n=N_BEADS, box=BOX, overlap=0.9, maxtry=200,
        # Both as in mesomem_assembly. The centring correction is if anything more
        # useful here: at this size several aggregates form at once, and without it
        # they wander off the near clipping plane one at a time.
        k_upright=0.6,
        center_accel=0.05,
    ),
    mode="sim",
    observables=["nematic_S", "coordination", "thickness"],
    param_ranges={"k_splay": (0.0, 40.0)},
    presets={
        "paper": {},
        "compact_droplets": {"k_tilt": 4.0},
        "strongly_planar": {"k_tilt": 30.0},
        "isotropic_only": {"k_tilt": 0.0, "k_splay": 0.0, "wc": 0.0},
    },
    temperature=(0.0, 0.5),
    temperature_default=0.2,
    melt_temp=0.3,
    particle_radius=0.5,
    reduced_units=True,
    trajectory_smoothing=True,
    render_style=STYLE,
    camera_orbit=CameraOrbit(autostart=True, speed=0.16),
    # The frame budget, and the only concession this file makes to its size. The
    # energy panels are a pass over every pair -- 2.5 us/bead measured -- and at
    # 10k beads the default every-4-frames cadence costs 6 ms of a 16.7 ms frame
    # on its own. Every 8 halves that for a panel whose aggregate barely changes
    # between frames anyway. The observables keep their own declared cadences.
    analysis_energy_every=8,
    # Where it runs. Everything here is overridable from the environment
    # (LAMMPS_LIVE_REMOTE_USER, _TIME, _PARTITION, ...) -- see remote/target.py.
    remote=RemoteTarget(
        host="snellius.surf.nl",
        user="stefanh",
        label="Snellius gpu_a100",
        partition="gpu_a100",
        gpus=1,
        ntasks=1,
        cpus_per_task=18,
        time="01:00:00",
        remote_dir="~/Projects/MesoMemLive/mesomem_gpu",
        env_script="_build/hpc/env.sh",
        profile="cluster-gpu",
    ),
)

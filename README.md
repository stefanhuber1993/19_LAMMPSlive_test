# LAMMPS live

A molecular-dynamics simulation you can put your hands on. Real LAMMPS runs
under a 60 fps game loop; you steer it with a force-feedback joystick and feel
what the model pushes back with.

![Seven MesoMem beads](docs/images/mesomem_patch.png)

Those are seven beads of the **MesoMem** coarse-grained membrane model -- the
Idema group's own LAMMPS pair style ([arXiv:2602.24123](https://arxiv.org/abs/2602.24123)),
compiled in and running, not approximated. Each bead carries a director (the
yellow/blue banding is which way it points), and the model's whole behaviour --
membranes that stay flat, that heal, that assemble themselves out of a random
soup -- comes from how those directors want to line up with their neighbours.

## What it is for

Membrane elasticity is taught as numbers in a paper: a tilt modulus, a splay
modulus, a transition somewhere around `k_tilt ~ 10`. This turns them into
something you do with your hand.

- **Feel the model.** Grab a bead, pull it out of the sheet, and the membrane's
  resistance arrives in the stick -- limp when nothing is holding you, firm on
  contact, buzzing with the thermostat's temperature. Twist hard enough and the
  bead's director flips to the opposite normal, because the tilt term is
  bistable. That is the physics, rendered as force.
- **Turn the knobs while it runs.** Every coefficient is a live slider. Take
  `k_tilt` down through the transition and watch the same run stop making
  membranes and start making droplets.
- **Watch it build itself.** 1,500 beads dropped in at random coarsen into flat
  lamellae, in front of you, in a minute.

It is a demo you give standing up, and an instrument for getting a feel for a
model whose parameters otherwise stay abstract.

## Scenes

| | |
|---|---|
| `mesomem_patch` | seven beads. Pull the middle one out and feel tilt and splay resist |
| `mesomem_sheet` | ~900 beads, periodic: a piece of an endless membrane. Watch a deformation propagate |
| `mesomem_assembly` | 1,500 beads from a random start, assembling. Play / Pause / Reset |
| `mesomem_remote` | the same at 10,000 beads, integrated on a cluster A100 |
| `cu_deposition`, `lj_argon`, `nacl` | atomistic classics -- copper (EAM), argon melting, a salt lattice whose ionic charge you can switch off |

Press `1`-`9` or `Tab` to switch; `lammps-live --list` prints them all.

## It runs on a supercomputer

`mesomem_remote` puts the simulation on an A100 at [Snellius](https://www.surf.nl)
and keeps the picture here at 60 fps. Press `N` and the app does the rest:
allocates the GPU, ships itself over, starts the server, tunnels a port home,
and hands the allocation back when you close the window. Both ends build the
same scene file, so there is one definition of the demo and no input deck to
keep in sync by hand.

Everything works exactly as it does locally -- every slider, Play/Pause/Reset --
because what crosses the wire is the same LAMMPS commands the local app issues
on itself. The GPU stays yours while you go and show another scene.

How it works: [docs/remote-gpu.md](docs/remote-gpu.md).
How to run it: [docs/snellius/README.md](docs/snellius/README.md).

## The joystick

A Microsoft Sidewinder Force Feedback 2, driven over raw HID
([driver](https://github.com/stefanhuber1993/sidewinder)). The whole demo is
reachable from it, because a hand on a stick should not have to go looking for a
keyboard mid-sentence.

The stick has two axes and the demo has more than two things worth steering, so
**one control is live at a time** and the hat switch moves between them: the
camera, then each slider, then the bead colouring. A bright cyan frame says
which. Push the stick to drive it -- most of the travel is a deliberately slow
band for placing a value, with the last of it accelerating to full speed.
The trigger plays and pauses (or grabs the bead); buttons 2, 3 and 4 reset the
run and change scene.

Force feedback runs on the device's own sensing rather than being streamed frame
by frame: a spring whose centre and stiffness follow the contact force, a damper
that stiffens on contact, and a vibration standing in for thermal jitter.
Everything works without the hardware too -- `--input mouse` or
`--input keyboard`.

## Run it

```bash
brew install mpich git          # Linux: apt install build-essential mpich libmpich-dev git
python3 -m venv venv && source venv/bin/activate
pip install -e .
lammps-live --input mouse
```

The `lammps` wheel links MPICH, so MPICH (not Open MPI) has to be what resolves
at runtime. The MesoMem force field compiles itself into a LAMMPS plugin the
first time you open a 3D scene -- a one-time ~10 s build, nothing to download.

```bash
lammps-live --input joystick               # needs hidapi: brew install hidapi
lammps-live --playground mesomem_assembly  # start on a particular scene
lammps-live --ui-scale 1.5                 # bigger UI on a 4K screen
lammps-live --list                         # everything runnable
```

On Linux the joystick also needs one udev rule, for non-root access to
`/dev/hidraw*`:

```bash
echo 'SUBSYSTEM=="hidraw", ATTRS{idVendor}=="045e", ATTRS{idProduct}=="001b", TAG+="uaccess"' \
  | sudo tee /etc/udev/rules.d/99-sidewinder-ff2.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

## Adding a scene

One ~50-line file naming a force field, a scenario and a mode -- no class to
subclass:

```python
PLAYGROUND = Playground(
    name="MesoMem membrane patch",
    force_field="mesomem",
    scenario=hex_patch(n_rings=1),
    mode="game",
)
```

Every live parameter the force field declares becomes a slider by itself. Drop
the file in `lammps_live/playgrounds/`, or run one from anywhere with
`lammps-live --playground ./my_idea.py`.

## Under the hood

Worth knowing, not worth explaining here:

- 3D scenes are GPU sphere impostors through a deferred shading chain (ambient
  occlusion, contact shadows, depth of field). [The impostor
  book](docs/impostor-book/) is the long version.
- The MesoMem scenes run in the paper's reduced LJ units, the atomistic ones in
  real metal units, and the readouts follow the model rather than a house style.
- A slider that destroys the simulation is a legitimate place to go: the run
  recovers by itself and says what happened -- on the cluster too, where the
  alternative was losing the GPU.
- `docs/a100-plan.md` is the plan for making the remote demo bigger, with the
  measurements behind it.

# LAMMPS live: interactive real-time MD demos

Pull an atom around with your mouse (or a force-feedback joystick) and
watch it interact with a live, thermostatted 2D crystal or fluid, all
driven by a real LAMMPS simulation running underneath the game loop. Modeled
on the classic MD demo image: a single atom deposited on a cold surface
redistributes its kinetic energy into the crystal and sticks, rather than
bouncing off, due to attractive interatomic forces -- except here the
"shot" is under continuous interactive control instead of a single
ballistic drop.

The puller atom's deflection is a *force* command (zero at center -- no
anchor point or spring-to-center). The system's reaction force is isolated
via a LAMMPS `compute group/group`, drawn as an on-screen vector, and (in
`--input joystick` mode) shapes the joystick's native Spring effect: its
center-point offset saturates fast (small forces already read as "pulled
toward the crystal", not just a hard push), and its stiffness ramps from
limp (nothing pulling) to firm (in contact) as that force builds. A
separate always-on native Damper effect gives viscous resistance, and a
Sine (vibration) effect fakes thermal jitter from the system's
instantaneous temperature. All of this runs on the device itself from its
own real-time position/velocity sensing, not streamed from Python frame by
frame.

Everything is in real, physical **metal units** (eV, Angstrom, ps, amu) --
not the reduced Lennard-Jones units common in MD tutorials -- so numbers on
screen mean what they say regardless of which system is active: forces in
eV/Angstrom, temperature in K, energy in eV.

## Systems

Multiple systems (material + potential + geometry) are registered and
switchable *while the demo is running* -- press `1`/`2`/... or `Tab`:

| key | name | potential | notes |
|---|---|---|---|
| `mesomem_patch` | MesoMem membrane patch (**3D**) | the **real** MesoMem pair-style | 7-bead hexagonal patch (hexagon + center). Pull the **center bead** out along the on-screen **net** and feel tilt/splay resist through force feedback |
| `mesomem_sheet` | MesoMem membrane sheet (**3D**) | same **real** MesoMem pair-style | the paper's planar-stability test at scale: a ~900-bead hexagonal sheet, periodic in-plane and barostat-relaxed to a tension-free spacing. Pull one bead out and watch tilt/splay propagate. A brightened cluster tags beads you can follow as they diffuse; beads crossfade across the periodic seam |
| `mesomem_assembly` | MesoMem self-assembly (**3D**) | same **real** MesoMem pair-style | the paper's spontaneous-assembly run: 1500 beads at random positions and orientations in a periodic 20-sigma box coarsen into planar membranes. Play / Pause / Reset. Dial `k_tilt` through the compact-vs-planar transition (~10) and watch the morphology change |
| `cu_deposition` | Copper deposition (EAM) | real Cu EAM (`Cu_u3.eam`) | strong metallic bonding, high melting range (dial goes to 10000 K) |
| `lj_argon` | Argon melting (Lennard-Jones) | `lj/cut`, real argon parameters | much softer/weaker, melts near 84 K (argon's real melting point). `epsilon` is a live slider, so you can feel what the well depth does |
| `nacl` | Salt crystal (ionic, NaCl) | Born-Mayer + damped shifted-force Coulomb (`born/coul/dsf`) | alternating Na(+)/Cl(-) ions (labeled `+`/`-`, drawn at their real size ratio) on a checkerboard square lattice -- the bipartite, 2D-stable ionic arrangement -- bound by Coulomb (Madelung) attraction |
| `lipid` | Lipid membrane (coarse-grained) | soft repulsion + cosine-squared tail attraction (`cosine/squared`), harmonic bonds/angles, Langevin implicit solvent | a solvent-free 2D lipid bilayer of 3-bead amphiphiles (head + 2 tails); the puller is a lipid you also **orient** (joystick yaw / Q-E) to insert into the membrane. Inspired by the MesoMem model (Sillano, Marrink & Idema 2026); see the module docstring |
| `mb_water` | Mercedes-Benz water (ice floats) | rigid 3-arm molecules (`fix rigid/small` + Langevin), directional hydrogen bond via `cosine/squared` arm-tip well | the 2D Mercedes-Benz water model (Ben-Naim 1971): each molecule is an O with three 120-degree hydrogen-bonding arms. Starts as the open hydrogen-bonded honeycomb **ice**; heat it and the network **collapses to denser liquid** -- water's freezing-expansion anomaly (why ice floats), with a live O-O spacing / density readout. The puller is a water molecule you also **orient** (joystick yaw / Q-E) to line its arms up and catch hydrogen bonds |

Run `lammps-live --list` to print this from the code (and
`--list-presets` for each playground's named parameter sets). The first five
are **playgrounds** -- declarative ~40-line files; the last three are older
hand-written systems. See "Writing a playground" below.

### The MesoMem force field (3D systems)

The three `mesomem_*` playgrounds run the authors' **actual** custom
LAMMPS pair-style (`pair_membrane_sillano_v2.{cpp,h}` from
[gitlab.tudelft.nl/idema-group/mesomem](https://gitlab.tudelft.nl/idema-group/mesomem),
[arXiv:2602.24123](https://arxiv.org/abs/2602.24123)), rather than approximating
it. Because the pip-installed LAMMPS ships the `PLUGIN` package, we don't rebuild
LAMMPS: the sources are **vendored** in `lammps_live/forcefields/mesomem_ff/` and
compiled once into a small shared library that is pulled in at runtime with
`plugin load`. The build runs automatically the first time you open one of these
systems (and rebuilds only when a source file changes); it uses the C++ compiler
and MPICH headers from the [prerequisites](#1-system-prerequisites) above, so
there is no separate download or manual build step -- set `LAMMPS_LIVE_MPI_INCLUDE`
only if the `mpi.h` auto-detection ever misses.

The three general problems this system solves -- **compiling a custom LAMMPS
force field into a stock build, rendering 3D fast (GPU sphere impostors with
per-pixel intersections, Blinn-Phong + ambient occlusion, depth-cued haze), and
driving a 3D scene with a 2-axis stick** -- are all reusable for the richer
MesoMem systems to come.

**3D rendering (OpenGL).** The 3D bead systems (`mesomem`, `membrane`) are drawn
on the GPU via `moderngl`: each bead is an instanced sphere impostor ray-cast in
the fragment shader (so overlapping beads intersect exactly, per-pixel, with no
sorting artifacts), lit with Blinn-Phong, darkened in the crevices by SSAO, and
the control-plane net is occluded by the real depth buffer. This needs an OpenGL
**3.3+ core** context (macOS 4.1 core, or Linux via EGL/GLX); the window is a GL
context and the pygame instrumentation panel is composited over the scene as a
texture. If no GL context can be created, the app automatically falls back to the
original CPU (numpy sphere-sprite) renderer for the whole session.

Each bead is colored the MesoMem way -- a yellow hydrophobic equator between blue
hydrophilic poles, tilting with the director -- with the `+`director pole capped
**white** so its sense reads at a glance. Depth is cued by fading the far half of
the scene toward the background.

**Controls.** The two joystick (or mouse) axes slide the center bead in the
world *xz*-plane -- a plane perpendicular to the membrane and facing the camera,
drawn as the faint net. Axis x -> along the sheet; axis y -> out of the sheet.
Pulling out tents the patch; the directors tilt and splay to follow, and the red
arrow / force feedback report the membrane's restoring force. Both MesoMem
systems add three live sliders -- **k_tilt**, **k_splay**, and **eta**
(interaction range) -- to the panel, so you can retune the force field and watch
the membrane stiffen, soften, or change cohesion range in real time. Units are
the paper's LJ-reduced units (so the temperature dial reads `T*`, not Kelvin).

## Setup

Runs on macOS and Linux (including WSL2 on Windows). Mouse control (`--input
mouse`) needs nothing beyond this section; the joystick needs one extra,
OS-specific step -- see "Joystick setup" below.

### 1. System prerequisites

Beyond Python 3.9+, a full install needs a few things from your OS package
manager:

- **MPICH** -- the `lammps` wheel dynamically links it, so it's required for
  *every* system (without it `import lammps` fails);
- a **C++ compiler** and **MPICH's development headers** (`mpi.h`) -- these
  additionally build the real MesoMem force field into a LAMMPS plugin the first
  time you open a 3D system (see
  ["The MesoMem force field"](#the-mesomem-force-field-3d-systems) below);
- **git** -- pip fetches the joystick driver straight from GitHub.

An OpenGL 3.3+ driver is used for the fast GPU renderer when present and the app
silently falls back to a CPU renderer when not, so it is *not* required.

- **macOS** (Homebrew):
  ```bash
  xcode-select --install     # C++ compiler (clang++), if not already installed
  brew uninstall open-mpi     # only if present -- ABI-incompatible with the lammps wheel
  brew install mpich git      # mpich provides libmpi + mpi.h; git for the pip dependency
  ```
- **Linux / WSL2 (Debian/Ubuntu):**
  ```bash
  sudo apt install build-essential mpich libmpich-dev git
  ```
  `libmpich-dev` supplies the `mpi.h` the plugin compiles against. If Open MPI is
  also installed (e.g. `libopenmpi-dev`), remove it or make sure MPICH takes
  precedence -- the two are ABI-incompatible.
- **Linux (Fedora):**
  ```bash
  sudo dnf install gcc-c++ mpich mpich-devel git
  module load mpi/mpich-x86_64   # Fedora keeps MPICH behind environment-modules
  ```

Why MPICH specifically: the `lammps` PyPI wheel dynamically links MPICH's
`libmpi`/`libpmpi` on every platform it ships (Linux, macOS, Windows) and is
ABI-incompatible with Open MPI, so MPICH -- not Open MPI -- must be what your
system resolves at runtime *and* what the plugin build compiles against. (The
plugin build auto-detects `mpi.h` via `mpicxx`/the usual install paths; override
with `LAMMPS_LIVE_MPI_INCLUDE` if it ever misses.)

### 2. Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

This installs a `lammps-live` command (see [pyproject.toml](pyproject.toml)).
If you'd rather not install the package, `pip install -r requirements.txt`
and run `python3 -m lammps_live` instead -- both work identically.

Nothing else has to be downloaded by hand. The copper EAM potential
(`Cu_u3.eam`) is bundled in this repo (`lammps_live/systems/data/`), and the
MesoMem C++ force field is vendored here
and **compiled automatically the first time you open a 3D system** (`mesomem` /
`membrane`) -- a one-time ~10 s build, cached next to its sources and rebuilt
only when they change. The Sidewinder joystick driver is the one thing fetched
over the network: pip pulls it from GitHub during install.

## Joystick setup (Sidewinder FF2 force feedback)

Optional -- only needed for `--input joystick`. Skip this section entirely
for `--input mouse`. The driver itself is the
[`sidewinder`](https://github.com/stefanhuber1993/sidewinder) package,
installed automatically from its own repo by `pip install -e .` -- no
separate clone or sibling checkout needed, and no copy of the driver kept
in sync here.

Only the native hidapi library below has to be installed by hand: it's a C
library, so pip can't bring it in.

1. Install the native hidapi library:
   - macOS: `brew install hidapi`
   - Linux / WSL2 (Debian/Ubuntu): `sudo apt install libhidapi-hidraw0`
   - Linux (Fedora): `sudo dnf install hidapi`
2. **Linux/WSL2 only** -- grant your user non-root access to the device.
   macOS's IOKit HID Manager is already user-accessible, so this step is
   mac's equivalent of a no-op; on Linux, `/dev/hidraw*` is root-only by
   default:
   ```bash
   echo 'SUBSYSTEM=="hidraw", ATTRS{idVendor}=="045e", ATTRS{idProduct}=="001b", TAG+="uaccess"' \
     | sudo tee /etc/udev/rules.d/99-sidewinder-ff2.rules
   sudo udevadm control --reload-rules && sudo udevadm trigger
   ```
   Unplug/replug the joystick afterwards. If `uaccess` doesn't take effect
   (e.g. a WSL2 distro without systemd enabled), fall back to a
   group-based rule instead: replace `TAG+="uaccess"` with
   `GROUP="plugdev", MODE="0660"`, then `sudo usermod -aG plugdev $USER`
   and log out/in.
3. **WSL2 only** -- the joystick is a USB device, and WSL2 doesn't see USB
   hardware by default. Attach it from a Windows (admin) prompt with
   [usbipd-win](https://github.com/dorssel/usbipd-win):
   ```powershell
   usbipd list                     # find the Sidewinder's BUSID
   usbipd bind --busid <BUSID>     # one-time
   usbipd attach --wsl --busid <BUSID>
   ```
   `usbipd attach` needs to be re-run each time the device is replugged or
   WSL is restarted.

Then, on any OS: `lammps-live --calibrate` prints live stick position for a
few seconds -- use it to confirm the device and permissions are working
before trying `--input joystick`.

## Run

```bash
lammps-live --input mouse                        # pointer: position moves, L/R buttons rotate
lammps-live --input keyboard                       # WASD moves, Q/E rotate (no pointer)
lammps-live --input joystick                      # Sidewinder FF2
lammps-live --playground mesomem_sheet             # start on a specific playground
lammps-live --playground mesomem_assembly --mode game   # override its mode
lammps-live --playground mesomem_sheet --preset buckled # start from a named parameter set
lammps-live --playground ./my_idea.py              # your own playground file, from anywhere
lammps-live --list                                 # print everything runnable and exit
lammps-live --list-presets                         # print each playground's presets
lammps-live --verify                               # check the force fields against LAMMPS
```

**Controls:**
- `1`-`9` -- jump directly to a system; `Tab` -- cycle to the next one
  (rebuilds the simulation; takes a moment)
- Move the puller: **mouse position** (`--input mouse`), **WASD** (`--input
  keyboard`), or the **joystick** (`--input joystick`). The three input modes are
  separate -- mouse mode does not read WASD, keyboard mode does not read the
  pointer
- **Orientation** (`lipid` and `mb_water` systems): rotate the control molecule's
  in-plane angle with the joystick's **yaw (twist) axis**, the **`Q`/`E`** keys
  (keyboard mode), or the **left/right mouse buttons** (mouse mode) -- so you can
  turn a lipid head-out to insert it into the bilayer, or line a water molecule's
  arms up to catch hydrogen bonds
- Mouse-drag the **Temperature** / **Puller damping** sliders in the
  right-hand panel (works regardless of `--input` mode). Below them sits one
  slider **per live force-field parameter**, generated automatically from the
  force field's declarations -- **k_tilt** / **k_splay** / **zeta** (and
  **rc** / **wc** / **splay symmetry** under "Advanced") for MesoMem,
  **epsilon** / **sigma** / **cutoff** for Lennard-Jones
- `Space` / `R` -- Play-Pause / Reset, on playgrounds running in **sim** mode
  (`mesomem_assembly`, or anything started with `--mode sim`)
- `Up`/`Down` arrow keys or the mouse scroll wheel -- nudge Temperature
- `Esc` or closing the window -- quit

Every atom -- puller and crystal alike -- leaves a thin, same-colored
motion trail behind it (last 1 second of wall-clock time, fading to
transparent with age): useful for seeing the swoop of a recent puller
approach/bounce, and for spotting which crystal atoms are actually moving
(thermal jitter, a shockwave from an impact) versus sitting still.

While actively dragging a slider in `--input mouse` mode, the puller's
input force is held at zero instead of also reading that same mouse
position as a deflection.

No `sudo` is required on macOS, or on Linux/WSL2 once the udev rule above
is installed: the `sidewinder` driver talks to the device via `hid`
(hidapi), which goes through the OS's own non-exclusive HID path -- IOKit
HID Manager on macOS, hidraw on Linux -- rather than `pyusb`/libusb
claiming exclusive access to the USB interface (which requires detaching
the kernel driver, hence root).

If the spring/damper feel ever seems off, `lammps-live --calibrate` prints
live stick position for a few seconds as a basic sanity check.

## Layout

```
lammps_live/
  cli.py            argument parsing, entry point
  app.py            the control loop: input -> sim -> force-feedback shaping -> renderer
  config.py         global constants that don't vary by system (window size, smoothing, ...)
  forcefeedback.py  force -> device-feedback signal shaping, parameterized by each
                     system's ForceFeedbackProfile
  units.py          metal-units -> SI display conversions (sim time, puller speed)
  catalog.py        one listing of everything runnable: playgrounds + legacy systems
  playground/       the declarative layer (see "Writing a playground")
    spec.py           Playground + Control -- what a playground file declares
    params.py         Param/ParamSet + the STRUCTURAL / HOT / HOT_RESTYLE tiers
                       that decide file-time vs slider-time
    forcefield.py     ForceField interface + registry
    scenario.py       Scenario interface, the membrane geometries, housekeeping forces
    deposition.py     the 2D crystal-slab + free-puller scenario
    thermostat.py     Langevin (implicit solvent) and CSVR (velocity rescaling)
    modes.py          game (controlled particle, leash, haptics) vs sim (Play/Pause/Reset)
    observables.py    declared observables + the throttled analysis scheduler
    state.py          Box, FrameState, the shared pair list -- pure numpy, no LAMMPS
    rdf.py            in-plane, 3D-radial and LAMMPS-native radial distribution functions
    verify.py         cross-checks a force field's Python energy against LAMMPS
    system.py         PlaygroundSystem: composes the above into an MDSystem
    registry.py       discovers playground files (bundled, or any path)
  playgrounds/        one file per explorable setup
    mesomem_patch.py     7-bead MesoMem patch
    mesomem_sheet.py     ~900-bead periodic MesoMem sheet
    mesomem_assembly.py  1500-bead MesoMem self-assembly (sim mode)
    lj_argon.py          2D Lennard-Jones argon deposition
    cu_deposition.py     2D copper EAM deposition
  forcefields/
    mesomem.py        the real MesoMem pair-style + its one vectorized energy expression
    stock.py          LennardJones and EAM, on stock LAMMPS styles (no C++)
    mesomem_ff/       vendored MesoMem C++ sources, compiled on demand
    data/             bundled potential files (Cu_u3.eam)
  systems/            the older hand-written systems
    base.py           MDSystem / MDSystem3D interfaces + SystemSpec/SliderSpec/
                       ForceFeedbackProfile
    nacl.py           ionic NaCl (Born-Mayer + DSF Coulomb)
    lipid_membrane.py coarse-grained lipid bilayer
    mb_water.py       2D Mercedes-Benz water (rigid 3-arm, hydrogen bonds)
  input/
    base.py     InputSource interface
    mouse.py    mouse control
    keyboard.py WASD/QE keyboard control
    joystick.py Sidewinder FF2 wrapper -- maps the device onto the sim's
                conventions and shapes the force feedback. The HID PID
                driver itself is the external `sidewinder` package
                (https://github.com/stefanhuber1993/sidewinder), a
                dependency rather than a vendored copy.
  ui/
    theme.py        colors/sizes
    widgets.py      Slider
    plotting.py     RollingHistory + generic line-plot drawer
    trail.py        rolling per-atom position snapshots behind every atom's fading motion trail
    camera.py       perspective camera for the 3D scenes
    gl3d.py         GPU bead pipeline (sphere impostors + SSAO + fog)
    glcompositor.py composites the 2D panel over the GL scene
    renderer.py     the sim box + instrumentation panel (2D and 3D paths)
```

## Writing a playground

A **playground** is one file naming a **force field**, a **scenario** and a
**mode**. That is the whole thing -- no class to subclass, no methods to
implement, and no registry to edit:

```python
# lammps_live/playgrounds/my_idea.py
from ..playground import Control, Playground, compose, hex_patch

PLAYGROUND = Playground(
    name="MesoMem: two patches colliding",
    force_field="mesomem",
    scenario=compose(hex_patch(n_rings=2, at=(-6, 0, 0)),
                     hex_patch(n_rings=2, at=(+6, 0, 0))),
    mode="game",
    control=Control(atom="nearest:0,0,0", plane="xz", leash=(5.0, 3.0)),
    observables=["nematic_S", "mean_tilt_deg"],
    presets={"floppy": {"k_tilt": 2.0, "k_splay": 0.1}},
)
```

Drop it in `lammps_live/playgrounds/` and it appears in `--list`, or keep it
anywhere and run `lammps-live --playground path/to/my_idea.py`.

### What goes in the file, and what goes on a slider

One rule decides it: **does changing this require rebuilding the simulation?**

| tier | examples | where |
|---|---|---|
| `STRUCTURAL` | particle counts, box size, boundary conditions, lattice spacing | **file only** -- a slider here would tear down and rebuild the simulation on every drag frame, throwing away the state you were watching |
| `HOT` | `pair_coeff` arguments, thermostat target | **slider, generated automatically** from the declared range/default/optimum |
| `HOT_RESTYLE` | a cutoff that moves the pair style's *global* cutoff (`rc`) | same, but the whole `pair_style` is re-declared first |
| `DERIVED` | `wc <= rc` | a declared clamp, applied everywhere the value is read |

A force field declares each parameter **once** (see
`lammps_live/playground/params.py`) and the slider follows. A playground can
widen or narrow a slider's span for its own purposes with `param_ranges`, since
what is worth exploring on a 7-bead patch differs from a 900-bead sheet.

### Writing a force field

A force field owns its parameters, its LAMMPS commands, and -- optionally -- one
vectorized Python expression of its energy, decomposed into the additive terms it
is built from. Subclass `ForceField`, `@register` it, and you are done;
`forcefields/mesomem.py` is a custom C++ pair style and `forcefields/stock.py`
holds two built on stock LAMMPS styles with no C++ at all.

**Custom C++ pair styles are compiled on demand.** Point a `PluginSpec` at a
directory of `pair_*.cpp` sources and they are built into a shared library and
pulled into the stock pip-installed LAMMPS with `plugin load`, rebuilt only when
a source changes. Edit your C++, restart, and the sliders still work.

**The energy expression is a test, not just a panel.** It drives the live
additive-energy bars, and `lammps-live --verify` asserts that its terms sum to
the potential energy LAMMPS computed from the compiled pair style, across several
parameter regimes:

```
$ lammps-live --verify
[verify] OK   mesomem_patch [defaults]
    LAMMPS  potential energy : -12.3908...
    Python  sum of terms     : -12.3908...
    relative error           : 2.868e-16 (tolerance 1e-06)
      isotropic  (repel + attract)           -12.4046  (+100.1%)
      tilt  (directors normal to bonds)      +0.01377  (-0.1%)
      splay  (neighbour directors align)     +1.218e-06 (-0.0%)
```

If your C++ and the expression you meant to implement disagree, this is where it
shows up -- as a number, at startup. Force fields whose energy is not
pairwise-additive (EAM) declare no terms and are correctly skipped.

### Scenarios, modes and observables

- **Scenarios** (`playground/scenario.py`, `playground/deposition.py`) own
  geometry, the cell, relaxation and per-frame housekeeping. The core contract is
  pure numpy -- `build(params, rng) -> positions, directors, box` -- so a scenario
  is unit-testable with no LAMMPS instance, and its particles reach LAMMPS in a
  single `create_atoms` call.
- **Modes** (`playground/modes.py`) are chosen at run time, not baked in, so
  `--mode game` and `--mode sim` both work on *any* playground. That is what lets
  you watch a structure self-assemble and then grab a particle and probe it.
- **Observables** (`playground/observables.py`) are named quantities -- nematic
  order `S`, membrane thickness, coordination -- declared by name and plotted
  live. They run on a declared cadence with a shared pair list, so adding them
  does not cost the frame budget.

### The older hand-written systems

`nacl`, `lipid` and `mb_water` are still monolithic `MDSystem` subclasses (see
`lammps_live/systems/base.py`). They work unchanged and are listed alongside the
playgrounds, but new work should be a playground. Nothing in `app.py`,
`forcefeedback.py` or `ui/` depends on which kind it is.

## Units and physics notes, for readers who know physics but not MD

- Everything is LAMMPS "metal" units: eV, Angstrom, ps, amu -- real
  physical scales the whole way through, not reduced/dimensionless LJ
  units. A force reading of `1.2 eV/A` is literally `-dE/dx` in those
  units, the same as in any other physics context.
- **Sim time** (top-left of the sim view) is elapsed *simulated* MD time --
  nsteps x timestep -- not wall-clock time, and not counted during each
  system's silent pre-roll settle. It's shown auto-scaled (fs / ps / ns,
  see `lammps_live/units.py`) since a play session spans from
  sub-picosecond up to a few hundred ps, rarely a clean fit for one fixed
  unit. **Puller speed** in the side panel is converted from the native
  Angstrom/ps to m/s (`1 A/ps = 100 m/s` exactly, since 1 A = 1e-10 m and
  1 ps = 1e-12 s) -- both units are shown side by side. Typical values land
  in the hundreds of m/s, the same range as real atomic thermal speeds at
  room temperature; that's not a coincidence, it's the same physics.
- The non-membrane systems are 2D (a one-atom-thick cross-section); the two
  MesoMem membrane systems (`mesomem`, `membrane`) are genuinely 3D and run in
  the paper's reduced LJ units rather than metal units. The 2D systems change
  some physics from the 3D case you might expect:
  - The equilibrium lattice is a 2D close-packed **hexagonal/triangular**
    lattice (6 in-plane neighbors), not a square cross-section of a 3D FCC
    lattice -- there's no out-of-plane bonding to brace a square
    arrangement, so the true 2D energy minimum is hexagonal. See
    `cu_deposition.py`'s module docstring for the empirical lattice-spacing
    sweep that confirmed this.
  - 2D crystals don't have a sharp melting transition the way 3D ones do
    (long-wavelength Mermin-Wagner fluctuations smear it out), so each
    system's `T_MELT` dial mark is an approximate reference point, not a
    rigorously-derived phase boundary -- the live radial distribution
    function g(r) panel (sharp peaks -> broad humps as you heat past it) is
    the trustworthy signal, not that one number.
- **Pressure is quasi-2D.** LAMMPS reports the `press` thermo quantity as a
  real 3D-style pressure, computed using the simulation box's actual
  (tiny, arbitrary) out-of-plane thickness as its "volume". That thickness
  was chosen for numerical convenience (just wide enough to hold one atomic
  layer), not to represent a physical z-extent, so the absolute bar value
  on screen is an artifact of that choice and isn't directly comparable to
  a real 3D pressure reading. It's kept (labeled "quasi-2D" in the panel)
  because *trends* -- pressure rising under compression or heating -- are
  still physically meaningful; just don't read the absolute number as "the
  pressure of a 2D gas" in any rigorous sense.
- **Temperature** is measured via `compute temp` over the mobile crystal
  atoms only (excludes the permanently-frozen floor and the
  user-driven puller, both of which would otherwise dilute or bias the
  reading), and LAMMPS handles the 2D degrees-of-freedom accounting
  automatically via `dimension 2`. The crystal is held at temperature by a
  canonical-sampling velocity-rescaling thermostat (Bussi et al. 2007,
  LAMMPS `temp/csvr`), **not** a Langevin bath: atoms move under the real
  interatomic forces alone and the thermostat only rescales the crystal's
  *total* kinetic energy toward the target by one global factor per step --
  never a per-atom random kick -- so on screen you see genuine phonon motion
  rather than white-noise buzz, the RDF/temperature plots stay canonically
  faithful, and a quench actually reaches 0 K. The thermostat is bound (via
  `fix_modify ... temp`) to the same `compute temp` shown on screen, so its
  setpoint equals the measured temperature with no calibration fudge factor.
  A `fix momentum` zeroes the crystal's slow net drift once per frame so that
  drift isn't miscounted as a temperature floor (and the substrate stays
  put); see each system module's thermostat note for the full rationale.
- **Energy plot** is relative to the value at the start of the current
  session (`t=0`), not an absolute zero -- LAMMPS's zero of energy is
  potential-dependent and not physically meaningful on its own; only
  energy *differences* are.
- The on-screen force vectors and the joystick's force-feedback are
  deliberately **not** a literal 1:1 rendering of the physical force --
  see `forcefeedback.py`'s module docstring for the soft-saturation
  (tanh) shaping and why, and note that the shaping constants
  (`ForceFeedbackProfile`) are tuned per-system to each potential's
  characteristic force scale.

## Tuning knobs

- `lammps_live/systems/*.py`: lattice spacing, timestep, box/region sizes,
  the puller's `nve/limit` displacement cap, damping slider range, thermostat
  target range and dial marks, RDF resolution/averaging window, and the
  `ForceFeedbackProfile` -- each module's docstring explains which
  constants were empirically measured (and how) vs. chosen as reasonable,
  clearly-labeled approximations.
- `lammps_live/config.py`: window size, sim-time-advanced-per-frame,
  temperature key/wheel nudge rates, plot history window, force-feedback
  smoothing time constant.
- `lammps_live/input/joystick.py`: SDK-level spring/damper/jitter ranges
  (hardware-fixed, not per-system).

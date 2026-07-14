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
| `cu_eam` | Copper deposition (EAM) | real Cu EAM (`Cu_u3.eam`) | strong metallic bonding, high melting range (dial goes to 6000 K) |
| `lj_argon` | Argon melting (Lennard-Jones) | generic `lj/cut`, real argon parameters | much softer/weaker, melts near 84 K (argon's real melting point) |
| `nacl` | Salt crystal (ionic, NaCl) | Born-Mayer + damped shifted-force Coulomb (`born/coul/dsf`) | alternating Na(+)/Cl(-) ions (labeled `+`/`-`) on a checkerboard square lattice -- the bipartite, 2D-stable ionic arrangement -- bound by Coulomb (Madelung) attraction |
| `lipid` | Lipid membrane (coarse-grained) | soft repulsion + cosine-squared tail attraction (`cosine/squared`), harmonic bonds/angles, Langevin implicit solvent | a solvent-free 2D lipid bilayer of 3-bead amphiphiles (head + 2 tails); the puller is a lipid you also **orient** (joystick yaw / Q-E) to insert into the membrane. Inspired by the MesoMem model (Sillano, Marrink & Idema 2026); see the module docstring |

Run `lammps-live --list-systems` to print this from the code. See
"Adding a new system" below to add your own.

## Setup

Runs on macOS and Linux (including WSL2 on Windows). Mouse control (`--input
mouse`) needs nothing beyond this section; the joystick needs one extra,
OS-specific step -- see "Joystick setup" below.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

This installs a `lammps-live` command (see [pyproject.toml](pyproject.toml)).
If you'd rather not install the package, `pip install -r requirements.txt`
and run `python3 -m lammps_live` instead -- both work identically.

`Cu_u3.eam` (the copper EAM potential file, used by the `cu_eam` system) is
bundled at `lammps_live/systems/data/Cu_u3.eam`, copied from the `lammps`
wheel's bundled benchmark data.

The `lammps` PyPI wheel dynamically links against **MPICH**'s `libmpi`/
`libpmpi` on every platform it ships (Linux, macOS, Windows), and is ABI-
incompatible with Open MPI -- make sure MPICH, not Open MPI, is what your
system resolves at runtime:

- **macOS**:
  ```bash
  brew uninstall open-mpi   # only if you have it - ABI-incompatible with the lammps wheel
  brew install mpich        # provides libmpi.12/libpmpi.12, which the wheel needs
  ```
- **Linux / WSL2 (Debian/Ubuntu)**:
  ```bash
  sudo apt install mpich
  ```
  If Open MPI is also installed system-wide (e.g. via `libopenmpi3`),
  remove it or make sure MPICH's lib directory takes precedence on the
  loader path.
- **Linux (Fedora)**:
  ```bash
  sudo dnf install mpich
  module load mpi/mpich-x86_64   # Fedora keeps MPICH behind environment-modules
  ```

## Joystick setup (Sidewinder FF2 force feedback)

Optional -- only needed for `--input joystick`. Skip this section entirely
for `--input mouse`. The driver itself
(`lammps_live/hardware/ff2.py`) is vendored in this repo -- no separate
clone or sibling checkout needed.

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
lammps-live --input mouse                        # no hardware needed - puller follows mouse
lammps-live --input joystick                      # Sidewinder FF2
lammps-live --input mouse --system lj_argon        # start on a specific system
lammps-live --list-systems                          # print available systems and exit
```

**Controls:**
- `1`-`9` -- jump directly to a system; `Tab` -- cycle to the next one
  (rebuilds the simulation; takes a moment)
- Move the puller with the mouse / joystick, as usual
- **Orientation** (`lipid` system): the joystick's **yaw (twist) axis** -- or
  the **`Q`/`E`** keys in mouse mode -- rotate the control lipid's in-plane
  angle, so you can turn it head-out and insert it into the bilayer
- Mouse-drag the **Temperature** / **Puller damping** sliders in the
  right-hand panel (works regardless of `--input` mode)
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
is installed: `lammps_live/hardware/ff2.py` talks to the device via `hid`
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
  systems/
    base.py           MDSystem interface + SystemSpec/SliderSpec/ForceFeedbackProfile
    cu_deposition.py  copper EAM system
    lj_argon.py       argon Lennard-Jones system
    data/             bundled potential files (Cu_u3.eam)
  input/
    base.py     InputSource interface
    mouse.py    mouse control
    joystick.py Sidewinder FF2 wrapper
  hardware/
    ff2.py      vendored HID PID driver, originally from
                https://github.com/stefanhuber1993/sidewinder
  ui/
    theme.py    colors/sizes
    widgets.py  Slider
    plotting.py RollingHistory + generic line-plot drawer
    trail.py    rolling per-atom position snapshots behind every atom's fading motion trail
    renderer.py the sim box + instrumentation panel
```

## Adding a new system

Each system is one self-contained module implementing `MDSystem` (see
`lammps_live/systems/base.py`'s docstrings for the full interface). At
minimum:

1. Create `lammps_live/systems/my_system.py`. Set up your LAMMPS box in
   `__init__`/a private `_build` method -- `cu_deposition.py` and
   `lj_argon.py` are two working examples with the same region/group/fix
   layout (crystal + frozen floor + interactively-controlled puller +
   csvr velocity-rescaling thermostat + RDF), just with different pair
   styles/constants.
2. Define a module-level `SystemSpec` (temperature/damping slider ranges,
   melt-temp dial mark, and a `ForceFeedbackProfile` scaled to your
   potential's characteristic force magnitude -- see the comments on
   `ForceFeedbackProfile` in `base.py` for why this needs tuning per
   system) and set it as the class's `spec` attribute.
3. Register it in `lammps_live/systems/__init__.py`'s `REGISTRY` dict.

Nothing in `app.py`, `forcefeedback.py`, or `ui/` needs to change -- they
only depend on the `MDSystem`/`SystemSpec` interface. If you're adding a
system with a genuinely different interaction (e.g. no puller/deposition
concept at all), the interface will need extending; open an issue/PR
description explaining the shape rather than special-casing it in the
existing systems.

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
- Both bundled systems are 2D (a one-atom-thick cross-section), which
  changes some physics from the 3D case you might expect:
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

# LAMMPS live: interactive 2D copper deposition

Modeled on the classic MD demo image: a single Cu atom deposited on a cold
Cu(001) surface redistributes its kinetic energy into the crystal and
sticks, rather than bouncing off, due to attractive interatomic forces.
Reduced to 2D (a cross-section, as in the reference figure), using the real
EAM potential for copper (`Cu_u3.eam`, bundled with LAMMPS) rather than a
toy pair potential.

The deposited ("puller") atom is under continuous interactive control
instead of a single ballistic shot: mouse/joystick deflection is a *force*
command (zero at center -- there's no anchor point or spring-to-center).
The crystal's EAM reaction force is isolated via a LAMMPS
`compute group/group`, drawn as an on-screen vector, and shapes the
joystick's native Spring effect. That same spring center is also nudged
opposite the puller's own velocity -- a weak (max 25% of range), linear
built-in deceleration cue, on top of the interaction-force bias. A
separate, always-on native Damper effect gives strong viscous resistance.
All of this is computed by the device itself from its own real-time
position/velocity sensing, not streamed from Python frame by frame.

All physical constants in `sim.py` (lattice spacing, damping) were found by
empirical testing in isolated scripts, not guessed -- see the module
docstring there for how (pressure-sweep for the true 2D equilibrium
spacing, kinetic-energy-calibrated deposition for the damping coefficient).

## Setup

```bash
brew uninstall open-mpi   # only if you have it - ABI-incompatible with the lammps wheel
brew install mpich        # provides libmpi.12/libpmpi.12, which the wheel needs
brew install hidapi       # joystick force feedback, via ../sidewinder/sidewinder/ff2.py
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`Cu_u3.eam` (the copper EAM potential file) is already copied into this
directory from the `lammps` wheel's bundled benchmark data.

## Run

```bash
python3 main.py --input mouse            # no hardware needed - puller follows mouse
python3 main.py --input joystick         # Sidewinder FF2 -- no sudo needed
```

Esc or closing the window quits.

No `sudo` is required for the joystick: `../sidewinder/sidewinder/ff2.py`
talks to the device via `hid` (hidapi), which goes through macOS's IOKit
HID Manager the same way the OS's own HID stack does, rather than `pyusb`/
libusb claiming exclusive access to the USB interface (which requires
detaching the kernel driver, hence root). An earlier version of this
project reimplemented the raw HID-PID protocol directly over pyusb and
needed sudo for exactly that reason; it's been replaced with a thin wrapper
around the proven `ff2.py` driver instead (see `input_source.py`).

If the spring/damper feel ever seems off, `python3 main.py --calibrate`
prints live stick position for a few seconds as a basic sanity check.

## Files
- `sim.py` - LAMMPS setup and stepping: 2D EAM Cu crystal + puller atom
  (same element, identified by atom ID, not a separate type),
  `fix addforce` for input force, `fix nve/limit` on the puller (bounds
  velocity so a sustained push can't tunnel through the wall/neighbor skin
  -- this crashed once during testing without it), `compute group/group`
  for the isolated interaction force.
- `input_source.py` - mouse and joystick control sources. Joystick is a thin
  wrapper around `../sidewinder/sidewinder/ff2.py`'s `FF2Device`, driving
  native Spring (interaction force) and Damper (constant viscous feel)
  condition effects instead of streaming a hand-computed constant force.
- `render.py` - pygame drawing, including the two force-vector arrows.
- `main.py` - the control loop tying it together, the EAM-force shaping
  (tanh soft-saturation) and the velocity-opposing damping bias, both
  folded together before being sent to the Spring effect.

## Tuning knobs
- `sim.py`: `LATTICE_SPACING`, `VISCOUS_GAMMA`, `PULLER_GAP`,
  `CRYSTAL_ROWS`, `FLOOR_ROWS`, `SETTLE_STEPS`, the `nve/limit` cap.
- `main.py`: `INPUT_FORCE_SCALE` (joystick sensitivity, eV/A),
  `FF_EXAGGERATION` / `FF_KNEE` / `FF_MAX_MAG` (force-feedback shaping),
  `MAX_PULLER_SPEED` / `VEL_DAMP_MAX_FRACTION` (velocity-damping spring
  bias), `STEPS_PER_FRAME`.
- `input_source.py`: `SPRING_STIFFNESS`, `DAMPER_COEFFICIENT` (signed byte,
  -128..127; both currently maxed at 127 for "strong spring and damping").

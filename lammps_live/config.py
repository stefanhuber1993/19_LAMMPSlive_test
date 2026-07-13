"""Global constants that apply regardless of which system is active. Values
that vary by system (lattice spacing, damping range, force-feedback tuning,
...) live on each system's SystemSpec instead -- see systems/base.py."""

WINDOW_SIZE = (1300, 900)

# Real sim-time advanced per rendered frame, in ps. Converted to a per-system
# step count (SIM_TIME_PER_FRAME / system.spec.timestep) rather than a fixed
# step count, so systems with a smaller timestep (see lj_argon.py) still
# advance at the same physical rate instead of appearing in slow motion.
SIM_TIME_PER_FRAME = 0.01  # ps

TEMP_KEY_RATE_FRACTION = 0.05     # fraction of a system's temperature range, per second, while Up/Down is held
TEMP_WHEEL_STEP_FRACTION = 0.02   # fraction of a system's temperature range, per mouse-wheel notch
HISTORY_WINDOW_SECONDS = 20.0

# Force-feedback signal smoothing (see forcefeedback.py): recomputed from
# scratch every frame from instantaneous LAMMPS state, which is jerky
# frame-to-frame (individual atom vibrations, contact transients). Low-pass
# it with a simple exponential filter before it reaches the device, in real
# time rather than frame count so it's independent of frame rate.
FF_SMOOTHING_TAU = 0.1   # seconds

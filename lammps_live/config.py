"""Global constants that apply regardless of which system is active. Values
that vary by system (lattice spacing, damping range, force-feedback tuning,
...) live on each system's SystemSpec instead -- see systems/base.py."""

WINDOW_SIZE = (1300, 900)

# Real sim-time advanced per rendered frame, in ps. Converted to a per-system
# step count (SIM_TIME_PER_FRAME / system.spec.timestep) rather than a fixed
# step count, so systems with a smaller timestep (see lj_argon.py) still
# advance at the same physical rate instead of appearing in slow motion.
SIM_TIME_PER_FRAME = 0.003  # ps

# Joystick mode drives the puller mainly by the stick's input force, feeding the
# MD interaction force back INDIRECTLY through force feedback (see app.py). This
# is the fraction of that MD force also applied DIRECTLY to the puller in the
# simulation: 0.0 = puller feels only the stick (a pure haptic loop -- felt too
# detached in practice), 1.0 = puller feels the full MD force directly (like
# mouse mode). A middle value keeps some direct contact coupling. Mouse mode
# ignores this and always feels the full force.
JOYSTICK_MD_FORCE_FELT_FRACTION = 0.5

TEMP_KEY_RATE_FRACTION = 0.05     # fraction of a system's temperature range, per second, while Up/Down is held
TEMP_WHEEL_STEP_FRACTION = 0.02   # fraction of a system's temperature range, per mouse-wheel notch
HISTORY_WINDOW_SECONDS = 20.0
TRAIL_WINDOW_SECONDS = 2.0   # how far back every atom's fading motion trail reaches, in wall-clock seconds
# Trail snapshots are recorded every Nth rendered frame rather than every
# frame: with ~250+ atoms in cu_eam, drawing a full-rate (60Hz) trail for
# every atom costs several ms/frame just in per-segment pygame.draw.line
# calls (measured). Sampling at a lower rate cuts that roughly linearly
# while staying visually smooth -- the trail is a short, fast-fading smear,
# not a precision plot.
TRAIL_SAMPLE_EVERY_N_FRAMES = 5

# Force-feedback signal smoothing (see forcefeedback.py): recomputed from
# scratch every frame from instantaneous LAMMPS state, which is jerky
# frame-to-frame (individual atom vibrations, contact transients). Low-pass
# it with a simple exponential filter before it reaches the device, in real
# time rather than frame count so it's independent of frame rate.
FF_SMOOTHING_TAU = 0.1   # seconds

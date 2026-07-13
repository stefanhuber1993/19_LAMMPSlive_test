"""Translates a system's raw interaction force (and the puller's own
velocity) into force-feedback / on-screen-arrow signals, using a per-system
ForceFeedbackProfile (see systems/base.py) rather than global constants --
different systems have wildly different characteristic force scales (EAM
copper contact forces run ~0.1-6 eV/A, a soft LJ gas ~0.01-0.5 eV/A), and a
single global knee/threshold would make one of them feel numb or pegged at
max.

The joystick's native Spring effect represents the interaction force as a
center-point offset (soft-saturating -- exaggerates small/medium contact
forces for a clear feel, smoothly caps large spikes instead of a hard clip,
which would feel like an on/off switch) and its stiffness (how firmly it
holds that point, ramping from limp when idle to firm in contact). A
separate always-on Damper scales with that same contact signal. Both are
also nudged/scaled by the puller's own velocity as a built-in deceleration
cue. All of this is computed on the device itself from its own real-time
position/velocity sensing once told the target offset/stiffness/coefficient
-- not streamed frame by frame.
"""
import math


def shape_interaction_force(fx, fy, profile):
    """Soft-saturate (tanh) the raw interaction force for force feedback."""
    mag = math.hypot(fx, fy)
    if mag < 1e-9:
        return 0.0, 0.0
    shaped_mag = profile.ff_max_mag * math.tanh(mag * profile.ff_exaggeration / profile.ff_knee)
    scale = shaped_mag / mag
    return fx * scale, fy * scale


def shape_velocity_damping(vx, vy, profile, max_puller_speed, cp_offset_max):
    """Spring-center bias opposing the puller's velocity, linear up to
    max_puller_speed then flat, capped at profile.vel_damp_max_fraction of
    cp_offset_max."""
    mag = math.hypot(vx, vy)
    if mag < 1e-9:
        return 0.0, 0.0
    max_mag = profile.vel_damp_max_fraction * cp_offset_max
    shaped_mag = min(mag / max_puller_speed, 1.0) * max_mag
    scale = shaped_mag / mag
    return -vx * scale, -vy * scale


def contact_fraction(fx, fy, profile):
    """0..1 "is something pushing on the atom" fraction from the raw
    (physical-units) interaction force -- 0 below stiffness_threshold,
    ramping via tanh over stiffness_knee above it. Shared basis for both the
    spring's stiffness and the damper's coefficient -- limp/weak when idle,
    strong when in contact, from the same signal."""
    mag = math.hypot(fx, fy)
    if mag < profile.stiffness_threshold:
        return 0.0
    return math.tanh((mag - profile.stiffness_threshold) / profile.stiffness_knee)


def shape_stiffness(fx, fy, profile, spring_stiffness_max):
    """Limp-when-idle, strong-when-in-contact spring stiffness."""
    return spring_stiffness_max * contact_fraction(fx, fy, profile)


def shape_damper_coefficient(fx, fy, profile, damper_coefficient_max):
    """Damper coefficient, damper_min_fraction..damper_max_fraction of max,
    from the same contact signal as shape_stiffness."""
    frac = contact_fraction(fx, fy, profile)
    fraction = profile.damper_min_fraction + (profile.damper_max_fraction - profile.damper_min_fraction) * frac
    return damper_coefficient_max * fraction


class ExponentialSmoother2D:
    """First-order low-pass filter for a 2D signal, time-constant based (not
    frame-count based) so the amount of smoothing doesn't change with fps."""

    def __init__(self, tau):
        self.tau = tau
        self.x = 0.0
        self.y = 0.0

    def reset(self):
        self.x = 0.0
        self.y = 0.0

    def update(self, x, y, dt):
        alpha = 1.0 - math.exp(-dt / self.tau)
        self.x += (x - self.x) * alpha
        self.y += (y - self.y) * alpha
        return self.x, self.y

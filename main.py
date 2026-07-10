#!/usr/bin/env python3
"""
Interactive 2D copper deposition POC (LAMMPS + EAM), modeled on the classic
MD demo: a Cu atom deposited on a cold Cu(001) surface sticks rather than
bounces, because its kinetic energy redistributes into the crystal via
attractive interatomic forces. Here the deposited atom is under continuous
interactive control instead of a single ballistic shot.

    python3 main.py --input mouse            # no hardware needed
    python3 main.py --input joystick         # Sidewinder FF2 (via hidapi -- no sudo needed)

Joystick/mouse deflection is a force command on the puller atom (zero at
center, no anchor/spring-to-center). The crystal's EAM reaction force is
drawn as a second vector and shapes the joystick's native Spring effect,
biasing its center toward ongoing contact; that same Spring center is also
nudged opposite the puller's own velocity (weak, linear, capped at 25% of
range -- see shape_velocity_damping) as a built-in deceleration cue. A
separate, always-on native Damper effect provides strong viscous
resistance -- all computed on the device itself from its own real-time
position/velocity, not streamed from here frame by frame.
"""
import argparse
import math
import sys

import pygame

from sim import CrystalSim, LATTICE_SPACING, TIMESTEP
from render import Renderer
from input_source import MouseInput, JoystickInput, CP_OFFSET_MAX

WINDOW_SIZE = (800, 800)
STEPS_PER_FRAME = 10
INPUT_FORCE_SCALE = 3.0   # eV/Angstrom at full joystick deflection (real interaction forces run ~0.1-6 eV/A)

# Force-feedback shaping for the EAM interaction force -> the joystick's
# Spring effect center offset, which is a signed byte (-127..127) -- cap
# comfortably inside that range. Damping is handled entirely device-side
# (see JoystickInput's always-on Damper effect), not here.
FF_EXAGGERATION = 3.0     # amplifies small/medium contact forces
FF_KNEE = 6.0             # raw*exaggeration magnitude at the soft-saturation "knee" (~typical hard-contact force)
FF_MAX_MAG = 110.0        # cap, comfortably inside the +-127 spring-offset range

# Extra spring-center bias opposing the puller's own velocity: on top of the
# EAM reaction-force bias above, this nudges the spring center against the
# direction of motion, so the device itself resists a fast-moving puller
# (built-in deceleration cue) instead of only ever pulling toward contact.
# Scales linearly from 0 up to VEL_DAMP_MAX_MAG at MAX_PULLER_SPEED, then
# holds flat -- kept deliberately weak (25% of the hardware's +-127 range)
# so it reads as a gentle assist, not a second spring fighting the user.
MAX_PULLER_SPEED = 0.1 * LATTICE_SPACING / TIMESTEP  # A/ps -- the puller's own nve/limit displacement cap, expressed as a velocity ceiling
VEL_DAMP_MAX_FRACTION = 0.25
VEL_DAMP_MAX_MAG = VEL_DAMP_MAX_FRACTION * CP_OFFSET_MAX

# The combined force-feedback signal (EAM reaction + velocity damping) is
# recomputed from scratch every frame from instantaneous LAMMPS state, which
# is jerky frame-to-frame (individual atom vibrations, contact transients).
# Low-pass it with a simple exponential filter before it reaches the device,
# in real time rather than frame count so it's independent of frame rate:
# alpha = 1 - exp(-dt/tau), i.e. a first-order RC filter with time constant
# FF_SMOOTHING_TAU. Larger tau = smoother but more laggy; smaller = snappier
# but jerkier.
FF_SMOOTHING_TAU = 0.2   # seconds


def shape_lj_force(fx, fy):
    """Soft-saturate (tanh) the EAM force for force feedback: exaggerates small
    contact forces for a clear feel, smoothly caps large spikes instead of a
    hard clip (which would feel like an on/off switch)."""
    mag = math.hypot(fx, fy)
    if mag < 1e-9:
        return 0.0, 0.0
    shaped_mag = FF_MAX_MAG * math.tanh(mag * FF_EXAGGERATION / FF_KNEE)
    scale = shaped_mag / mag
    return fx * scale, fy * scale


def shape_velocity_damping(vx, vy):
    """Spring-center bias opposing the puller's velocity, linear up to
    MAX_PULLER_SPEED then flat, capped at VEL_DAMP_MAX_MAG."""
    mag = math.hypot(vx, vy)
    if mag < 1e-9:
        return 0.0, 0.0
    shaped_mag = min(mag / MAX_PULLER_SPEED, 1.0) * VEL_DAMP_MAX_MAG
    scale = shaped_mag / mag
    return -vx * scale, -vy * scale


class ExponentialSmoother2D:
    """First-order low-pass filter for a 2D signal, time-constant based (not
    frame-count based) so the amount of smoothing doesn't change with fps."""

    def __init__(self, tau):
        self.tau = tau
        self.x = 0.0
        self.y = 0.0

    def update(self, x, y, dt):
        alpha = 1.0 - math.exp(-dt / self.tau)
        self.x += (x - self.x) * alpha
        self.y += (y - self.y) * alpha
        return self.x, self.y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", choices=["mouse", "joystick"], default="mouse")
    parser.add_argument("--calibrate", action="store_true", help="dump raw joystick bytes and exit")
    args = parser.parse_args()

    if args.calibrate:
        js = JoystickInput()
        try:
            js.calibrate()
        finally:
            js.close()
        return

    # Not pygame.init() -- that also brings up SDL's joystick subsystem,
    # which grabs the Sidewinder as a native SDL game controller. When
    # JoystickInput then claims the same device exclusively via libusb, the
    # device vanishes out from under SDL mid-session and corrupts pygame's
    # event-translation state (observed as `KeyError: 0` inside
    # pygame.event.get()). We drive the joystick ourselves via raw HID
    # reports, so SDL's joystick subsystem is never needed.
    pygame.display.init()
    pygame.font.init()
    sim = CrystalSim()
    renderer = Renderer(WINDOW_SIZE, (sim.xhi - sim.xlo, sim.yhi - sim.ylo))
    clock = pygame.time.Clock()
    ff_smoother = ExponentialSmoother2D(FF_SMOOTHING_TAU)
    dt = 1.0 / 60  # seconds; seed value, replaced by the real measured frame time below

    if args.input == "mouse":
        max_radius_px = min(WINDOW_SIZE) * 0.35
        source = MouseInput(WINDOW_SIZE, max_radius_px)
    else:
        source = JoystickInput()

    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            jx, jy = source.poll()
            input_fx, input_fy = jx * INPUT_FORCE_SCALE, jy * INPUT_FORCE_SCALE
            sim.set_input_force(input_fx, input_fy)

            sim.step(STEPS_PER_FRAME)

            pos, vel = sim.get_puller_state()
            lj_force = sim.get_interaction_force()

            lj_fx, lj_fy = shape_lj_force(*lj_force)
            vel_damp_fx, vel_damp_fy = shape_velocity_damping(*vel) if vel is not None else (0.0, 0.0)
            smooth_fx, smooth_fy = ff_smoother.update(lj_fx + vel_damp_fx, lj_fy + vel_damp_fy, dt)
            source.send_force(smooth_fx, smooth_fy)

            positions, is_puller = sim.get_all_positions()
            renderer.draw(
                positions, is_puller, pos,
                (input_fx, input_fy), lj_force, clock.get_fps(),
            )

            dt = clock.tick(60) / 1000.0
    finally:
        source.close()
        sim.close()
        pygame.quit()


if __name__ == "__main__":
    sys.exit(main())

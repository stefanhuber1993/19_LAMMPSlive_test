"""Keyboard control -- always available, no hardware and no pointer needed.

WASD deflects the puller (D/A = +/-x, W/S = +/-y in sim convention, W = up) and
Q/E steer its in-plane rotation. This is the pure-keyboard counterpart to
MouseInput: neither reads the other's device, so `--input mouse` and
`--input keyboard` are cleanly separate control modes."""
import math

from .base import InputSource


class KeyboardInput(InputSource):
    def poll(self):
        import pygame
        keys = pygame.key.get_pressed()
        # D/A = right/left (+/-x); W/S = up/down (+/-y, sim convention).
        dx = (1.0 if keys[pygame.K_d] else 0.0) - (1.0 if keys[pygame.K_a] else 0.0)
        dy = (1.0 if keys[pygame.K_w] else 0.0) - (1.0 if keys[pygame.K_s] else 0.0)
        mag = math.hypot(dx, dy)
        if mag > 1.0:                     # diagonals stay within the unit disk
            dx, dy = dx / mag, dy / mag
        return dx, dy

    def poll_yaw(self):
        # Q / E rotate the puller's orientation (same sign convention as the
        # joystick twist axis: Q = +1, E = -1; both/neither = 0).
        import pygame
        keys = pygame.key.get_pressed()
        return (1.0 if keys[pygame.K_q] else 0.0) - (1.0 if keys[pygame.K_e] else 0.0)

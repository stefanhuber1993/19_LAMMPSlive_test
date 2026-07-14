"""Mouse control -- always available, no hardware needed."""
import math

from .base import InputSource


class MouseInput(InputSource):
    """Anchor tracks mouse position relative to a fixed center point (the
    sim box's center on screen -- NOT the full window, which also contains
    the instrumentation panel)."""

    def __init__(self, center_xy, max_radius_px):
        self.cx, self.cy = center_xy
        self.max_radius_px = max_radius_px

    def poll(self):
        import pygame
        mx, my = pygame.mouse.get_pos()
        dx = (mx - self.cx) / self.max_radius_px
        dy = (my - self.cy) / self.max_radius_px
        mag = math.hypot(dx, dy)
        if mag > 1.0:
            dx, dy = dx / mag, dy / mag
        return dx, -dy  # screen y grows downward; flip so up = positive

    def poll_yaw(self):
        # Q / E rotate the puller's orientation (the mouse has no twist axis).
        # E = +1 (counter-clockwise), Q = -1 (clockwise); both/neither = 0.
        import pygame
        keys = pygame.key.get_pressed()
        return (1.0 if keys[pygame.K_e] else 0.0) - (1.0 if keys[pygame.K_q] else 0.0)

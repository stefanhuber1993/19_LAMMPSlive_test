"""Mouse control -- always available, no hardware needed."""
import math

from .base import InputSource


class MouseInput(InputSource):
    """Pure-pointer control: the deflection tracks the mouse position relative to
    a fixed center point (the sim box's center on screen -- NOT the full window,
    which also contains the instrumentation panel), and the left/right mouse
    buttons steer the in-plane rotation whenever the pointer is over the sim view.
    Keyboard (WASD/QE) is deliberately NOT read here -- that is KeyboardInput, a
    separate `--input keyboard` mode."""

    def __init__(self, center_xy, max_radius_px, sim_rect=None):
        self.cx, self.cy = center_xy
        self.max_radius_px = max_radius_px
        # (x, y, w, h) of the sim view in window pixels; the mouse-button
        # rotation is only live while the pointer is inside it (so a click on
        # the instrumentation panel / sliders never also spins the molecule).
        self.sim_rect = sim_rect

    def poll(self):
        import pygame
        mx, my = pygame.mouse.get_pos()
        dx = (mx - self.cx) / self.max_radius_px
        dy = (my - self.cy) / self.max_radius_px
        mag = math.hypot(dx, dy)
        if mag > 1.0:
            dx, dy = dx / mag, dy / mag
        return dx, -dy  # screen y grows downward; flip so up = positive

    def _over_sim(self, pos):
        if self.sim_rect is None:
            return True
        x, y, w, h = self.sim_rect
        return x <= pos[0] < x + w and y <= pos[1] < y + h

    def poll_yaw(self):
        # The left / right mouse buttons rotate the puller's orientation (the
        # mouse has no twist axis). Signed to match the joystick twist
        # convention: left button = +1, right button = -1; both/neither = 0.
        # Only live while the pointer is over the sim view.
        import pygame
        if self._over_sim(pygame.mouse.get_pos()):
            left, _, right = pygame.mouse.get_pressed(num_buttons=3)
            return (1.0 if left else 0.0) - (1.0 if right else 0.0)
        return 0.0

"""
Shared control-input interface for the puller, plus two implementations:
mouse (always available, no hardware) and the Sidewinder FF2 joystick.

The joystick wraps ../sidewinder/sidewinder/ff2.py directly -- a
hardware-validated HID PID driver (built on `hid`/hidapi via macOS's IOKit
HID Manager, not exclusive libusb access, so **no sudo is needed**) that
does the proper "Create New Effect" block-allocation handshake and correct
signed-byte coefficient encoding. An earlier hand-rolled version of this
file reimplemented the raw protocol directly over pyusb and got several
things wrong as a result (needed sudo to detach the kernel driver; sent
stiffness/coefficient as an unsigned 0xFF, which as a *signed* byte is -1,
not "max" -- likely why the spring/damper felt weak or absent).
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sidewinder"))


class InputSource:
    def poll(self):
        """Return (x, y) deflection in [-1, 1], sim convention: +y is up."""
        raise NotImplementedError

    def send_force(self, fx, fy):
        """Feed back a force (sim units) the device should resist with. No-op if unsupported."""
        pass

    def close(self):
        pass


class MouseInput(InputSource):
    """Anchor tracks mouse position relative to the window center."""

    def __init__(self, window_size, max_radius_px):
        self.cx = window_size[0] / 2
        self.cy = window_size[1] / 2
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


# ---------------------------------------------------------------------------
# Sidewinder FF2 joystick -- thin wrapper around sidewinder.ff2.FF2Device
# ---------------------------------------------------------------------------

# Spring/damper coefficients are signed bytes (-128..127) in the real driver
# (ff2.py's _s8 helper) -- 127 is genuinely max, not 255. "Strong spring and
# strong damping": both maxed at 127 coefficient / 255 saturation.
SPRING_STIFFNESS = 127
SPRING_SATURATION = 255
DAMPER_COEFFICIENT = 127
DAMPER_SATURATION = 255
CP_OFFSET_MAX = 127


class JoystickInput(InputSource):
    """
    Drives the device's own Spring and Damper *condition* effects rather than
    streaming a hand-computed constant-force value every frame. Both run
    continuously on the device's own high-rate position/velocity sensing:
    - Spring: center offset is updated each frame from the desired reaction
      force (see send_force) -- this is what encodes the interaction force.
    - Damper: fixed coefficient, set once at startup -- constant viscous
      resistance to hand motion, independent of the simulation.
    """

    def __init__(self):
        from sidewinder.ff2 import FF2Device

        self.ff = FF2Device()
        self.spring = self.ff.spring(stiffness=SPRING_STIFFNESS, saturation=SPRING_SATURATION)
        self.damper = self.ff.damper(coefficient=DAMPER_COEFFICIENT, saturation=DAMPER_SATURATION)
        self._last_xy = (0.0, 0.0)

    def poll(self):
        result = self.ff.read_position()
        if result is not None:
            x, y = result
            self._last_xy = (x, -y)  # device convention -> sim convention (+y up)
        return self._last_xy

    def calibrate(self, n=200):
        """Print live stick position for a few seconds, for basic sanity checking."""
        import time
        print("Move the stick now; printing position for a few seconds...")
        for _ in range(n):
            result = self.ff.read_position()
            if result is not None:
                print(f"x={result[0]:+.3f}  y={result[1]:+.3f}")
            time.sleep(0.05)

    def send_force(self, fx, fy):
        """Bias the spring's center away from true center (0) by (fx, fy).

        F_device = k*(center - actual_pos). With center = bias (not "current
        position minus fx", which was an earlier bug -- it made the spring
        cancel itself out every frame and gave zero force at rest), the
        spring's baseline behavior at bias=0 is a plain centering spring,
        and it's *offset* by the combined bias on top of that -- "centering
        by default, extra pull for a reaction force, extra push-back against
        the puller's own velocity" (see main.py's shape_lj_force and
        shape_velocity_damping, both folded into (fx, fy) before this is
        called). Y is negated to express the sim-convention force back in
        the device's own raw axis convention (see poll()'s flip).
        """
        cx = max(-CP_OFFSET_MAX, min(CP_OFFSET_MAX, int(round(fx))))
        cy = max(-CP_OFFSET_MAX, min(CP_OFFSET_MAX, int(round(-fy))))
        self.spring.set_condition(axis=0, cp_offset=cx,
                                   pos_coeff=SPRING_STIFFNESS, neg_coeff=SPRING_STIFFNESS,
                                   pos_sat=SPRING_SATURATION, neg_sat=SPRING_SATURATION)
        self.spring.set_condition(axis=1, cp_offset=cy,
                                   pos_coeff=SPRING_STIFFNESS, neg_coeff=SPRING_STIFFNESS,
                                   pos_sat=SPRING_SATURATION, neg_sat=SPRING_SATURATION)

    def close(self):
        try:
            self.spring.free()
            self.damper.free()
        except Exception:
            pass
        self.ff.close()

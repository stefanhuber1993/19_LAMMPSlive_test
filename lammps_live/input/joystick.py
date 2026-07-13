"""Sidewinder FF2 joystick input -- thin wrapper around
lammps_live.hardware.ff2.FF2Device.

That driver (vendored, originally from
https://github.com/stefanhuber1993/sidewinder) is a hardware-validated HID
PID driver (built on `hid`/hidapi, which talks to the device through the
OS's native non-exclusive HID path -- IOKit HID Manager on macOS, hidraw on
Linux, WinAPI on Windows -- rather than exclusive libusb access, so **no
sudo is needed** once the device node is readable/writable by the user; see
README.md's Joystick setup section for the one-time Linux/WSL2 udev step)
that does the proper "Create New Effect" block-allocation handshake and
correct signed-byte coefficient encoding.

Drives the device's own Spring and Damper *condition* effects rather than
streaming a hand-computed constant-force value every frame. Both run
continuously on the device's own high-rate position/velocity sensing:
- Spring: center offset and stiffness are updated each frame from the
  desired reaction force (see send_force).
- Damper: coefficient is likewise updated each frame (see
  set_damper_coefficient), scaling with that same contact signal --
  viscous resistance to hand motion, weak when idle and stronger in
  contact, rather than one constant value throughout.
"""
import math
import random

from .base import InputSource

# Spring/damper coefficients are signed bytes (-128..127) in the real driver
# (ff2.py's _s8 helper) -- 127 is genuinely max, not 255.
#
# Stiffness (pos/neg_coeff) ramps from SPRING_STIFFNESS_MIN (limp -- barely
# resists being moved off-center when nothing is pulling) up to
# SPRING_STIFFNESS_MAX (strong, firmly holds the center point) as a function
# of the interaction-force magnitude -- see forcefeedback.py, folded into
# the same send_force call as the center-point offset. Saturation (how much
# force the spring can ultimately deliver before clipping) stays maxed
# regardless -- that's a hard ceiling on delivered force, not part of the
# limp/strong feel.
SPRING_STIFFNESS_MIN = 1
SPRING_STIFFNESS_MAX = 127
SPRING_SATURATION = 255
# Damper coefficient is likewise not fixed: forcefeedback.py scales it with
# the same contact-force signal that drives spring stiffness, from
# DAMPER_COEFFICIENT_MIN (idle) up to DAMPER_COEFFICIENT_MAX (in contact).
DAMPER_COEFFICIENT_MIN = 13   # ~10% of max
DAMPER_COEFFICIENT_MAX = 127
DAMPER_SATURATION = 255
CP_OFFSET_MAX = 127

# Thermal jitter is faked with the device's own vibration (periodic Sine)
# effect layered on top of the Spring, rather than streamed per-atom motion
# -- magnitude scales with heat_fraction (see update_jitter), direction is
# re-randomized on every call since real thermal motion has no preferred
# direction and this is called at ~60Hz from the render loop anyway, which
# reads as an incoherent buzz rather than a directional shake.
JITTER_PERIOD_MS = 60
JITTER_MAX_MAGNITUDE = 30   # of 255 -- deliberately short of max so it never overwhelms the spring/damper feel


class JoystickInput(InputSource):
    def __init__(self):
        try:
            from lammps_live.hardware.ff2 import FF2Device
        except ImportError as e:
            raise RuntimeError(
                "Could not load the joystick driver (lammps_live.hardware.ff2). "
                "The native hidapi library likely isn't installed for this OS "
                "(macOS: `brew install hidapi`; Debian/Ubuntu/WSL2: `sudo "
                "apt install libhidapi-hidraw0`; Fedora: `sudo dnf install "
                "hidapi`). See README.md's Joystick setup section.\n"
                f"Original error: {e}"
            ) from e

        try:
            self.ff = FF2Device()
        except Exception as e:
            raise RuntimeError(
                "Could not open the Sidewinder FF2 device. On Linux/WSL2 "
                "this is usually a udev permissions issue on /dev/hidraw* "
                "-- see README.md's Joystick setup section for the udev "
                "rule. On WSL2, also confirm the device is attached with "
                "`usbipd attach --wsl ...` (USB devices aren't visible to "
                "WSL2 by default). Otherwise, check the joystick is "
                "plugged in.\n"
                f"Original error: {e}"
            ) from e
        self.spring = self.ff.spring(stiffness=SPRING_STIFFNESS_MIN, saturation=SPRING_SATURATION)
        self.damper = self.ff.damper(coefficient=DAMPER_COEFFICIENT_MIN, saturation=DAMPER_SATURATION)
        self.jitter = self.ff.sine(magnitude=0, period_ms=JITTER_PERIOD_MS)
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

    def send_force(self, fx, fy, stiffness=None):
        """Bias the spring's center away from true center (0) by (fx, fy),
        and set how stiffly it holds that point.

        F_device = k*(center - actual_pos). With center = bias (not "current
        position minus fx", which was an earlier bug -- it made the spring
        cancel itself out every frame and gave zero force at rest), the
        spring's baseline behavior at bias=0 is a plain centering spring,
        and it's *offset* by the combined bias on top of that -- "centering
        by default, extra pull for a reaction force, extra push-back against
        the puller's own velocity" (see forcefeedback.py's shape_lj_force
        and shape_velocity_damping, both folded into (fx, fy) before this is
        called). Y is negated to express the sim-convention force back in
        the device's own raw axis convention (see poll()'s flip).

        stiffness (None = SPRING_STIFFNESS_MAX) is the ramped limp->strong
        coefficient computed from that same combined bias's magnitude -- see
        SPRING_STIFFNESS_MIN/MAX above.
        """
        if stiffness is None:
            stiffness = SPRING_STIFFNESS_MAX
        k = max(0, min(127, int(round(stiffness))))
        cx = max(-CP_OFFSET_MAX, min(CP_OFFSET_MAX, int(round(fx))))
        cy = max(-CP_OFFSET_MAX, min(CP_OFFSET_MAX, int(round(-fy))))
        self.spring.set_condition(axis=0, cp_offset=cx,
                                   pos_coeff=k, neg_coeff=k,
                                   pos_sat=SPRING_SATURATION, neg_sat=SPRING_SATURATION)
        self.spring.set_condition(axis=1, cp_offset=cy,
                                   pos_coeff=k, neg_coeff=k,
                                   pos_sat=SPRING_SATURATION, neg_sat=SPRING_SATURATION)

    def set_damper_coefficient(self, coefficient):
        k = max(0, min(127, int(round(coefficient))))
        self.damper.set_condition(axis=0, pos_coeff=k, neg_coeff=k,
                                   pos_sat=DAMPER_SATURATION, neg_sat=DAMPER_SATURATION)
        self.damper.set_condition(axis=1, pos_coeff=k, neg_coeff=k,
                                   pos_sat=DAMPER_SATURATION, neg_sat=DAMPER_SATURATION)

    def update_jitter(self, heat_fraction):
        heat_fraction = max(0.0, min(1.0, heat_fraction))
        # sqrt: jitter amplitude ~ thermal speed ~ sqrt(kinetic energy) ~
        # sqrt(T), not linear in T -- matches how the crystal's own Langevin
        # jitter grows too.
        magnitude = int(round(JITTER_MAX_MAGNITUDE * math.sqrt(heat_fraction)))
        self.jitter.set_base(direction_deg=random.uniform(0, 360),
                              axis_x=True, axis_y=True, direction_enable=True)
        self.jitter.set_periodic(magnitude=magnitude, period_ms=JITTER_PERIOD_MS)

    def close(self):
        try:
            self.spring.free()
            self.damper.free()
            self.jitter.free()
        except Exception:
            pass
        self.ff.close()

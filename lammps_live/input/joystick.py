"""Sidewinder FF2 joystick input -- thin wrapper around sidewinder.ff2.FF2Device.

The driver is the `sidewinder` package, installed straight from
https://github.com/stefanhuber1993/sidewinder (see pyproject.toml) rather
than vendored here, so the device protocol has exactly one home. This module
is only the sim-facing adapter: it maps the device's decoded input onto the
sim's conventions and shapes the force feedback -- everything about *how* to
talk to the hardware lives upstream.

That driver is a hardware-validated HID PID driver (built on `hid`/hidapi,
which talks to the device through the OS's native non-exclusive HID path --
IOKit HID Manager on macOS, hidraw on Linux, WinAPI on Windows -- rather than
exclusive libusb access, so **no sudo is needed** once the device node is
readable/writable by the user; see README.md's Joystick setup section for the
one-time Linux/WSL2 udev step) that does the proper "Create New Effect"
block-allocation handshake and correct signed-byte coefficient encoding.

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
import threading
import time

from .base import InputSource

# Spring/damper coefficients are signed bytes (-128..127) in the real driver
# (sidewinder.ff2's _s8 helper) -- 127 is genuinely max, not 255.
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

# Twist below this fraction of full deflection reads as zero yaw, so a hand
# resting on the stick doesn't slowly spin the molecule.
TWIST_DEADZONE = 0.15


class JoystickInput(InputSource):
    def __init__(self, background=True):
        try:
            from sidewinder.ff2 import FF2Device
        except ImportError as e:
            raise RuntimeError(
                "Could not load the joystick driver (the `sidewinder` "
                "package). Either it isn't installed (`pip install -e .` "
                "pulls it from GitHub), or the native hidapi library it "
                "needs isn't installed for this OS (macOS: `brew install "
                "hidapi`; Debian/Ubuntu/WSL2: `sudo apt install "
                "libhidapi-hidraw0`; Fedora: `sudo dnf install hidapi`). "
                "See README.md's Joystick setup section.\n"
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
        self._last_yaw = 0.0
        self._last_buttons = frozenset()
        self._twist_center = None  # captured from the first reading (see _process_twist)
        # Last force-feedback conditions actually written to the device, so a
        # redundant re-write is skipped. Each set_condition is a blocking HID
        # transfer; the shaped values quantize to signed bytes, so frame-to-frame
        # (and especially at rest) they are very often unchanged -- skipping those
        # writes avoids needless HID traffic. (Only ever touched on the I/O side --
        # the worker thread in background mode, else the calling thread.)
        self._last_axis0 = None   # (cp_offset, coeff) last sent on spring axis 0
        self._last_axis1 = None   # ... axis 1
        self._last_damper = None  # last damper coefficient
        self._last_jitter_mag = None  # last sine (jitter) magnitude

        # --- Background HID I/O -------------------------------------------------
        # Every HID read and write on this device is a BLOCKING transfer: the
        # driver's read waits for a report (a still stick emits none, so it burns
        # its whole timeout), and each force-feedback write is a synchronous
        # SetReport that can take milliseconds -- and a frame issues up to six of
        # them. Doing all that inline on the 60 fps render thread is what ate
        # 20-30 ms/frame. So all device I/O runs on a daemon thread: the render
        # thread only swaps values in memory (instant), and because the `hid`
        # package is a ctypes wrapper (its blocking C calls release the GIL), the
        # worker's waits never stall the main thread.
        #
        # The worker samples the stick and pushes the latest desired FF state each
        # loop; the render thread reads the cached input and stores its desired FF
        # under this lock. background=False keeps everything synchronous (used by
        # the --calibrate CLI, which reads the device directly on one thread).
        self._lock = threading.Lock()
        self._want_force = None     # (fx, fy, stiffness) requested by the app, or None
        self._want_damper = None    # requested damper coefficient, or None
        self._want_heat = None      # requested jitter heat fraction, or None
        self._running = False
        self._worker = None
        if background:
            self._running = True
            self._worker = threading.Thread(target=self._io_loop, name="ff2-io",
                                            daemon=True)
            self._worker.start()

    # ---- render-thread API (non-blocking: only touches memory) --------------

    def poll(self):
        # Return the latest sampled stick position. In background mode the worker
        # keeps _last_xy fresh; with no worker (calibrate/synchronous) read here.
        if self._worker is None:
            self._read_once(timeout_ms=0)
        with self._lock:
            return self._last_xy

    def _read_once(self, timeout_ms=0):
        """Sample the stick once (blocking up to timeout_ms) and cache it. Called
        on the worker thread in background mode, else inline from poll(). The
        device only emits on change, so an empty read just keeps the last value --
        exactly right: no report means nothing moved."""
        state = self.ff.read_input(timeout_ms=timeout_ms)
        if state is None:
            return
        xy = (state.x, -state.y)             # device convention -> sim (+y up)
        yaw = self._process_twist(state.twist)
        buttons = frozenset(int(b) for b in state.pressed)
        with self._lock:
            self._last_xy = xy
            self._last_yaw = yaw
            self._last_buttons = buttons

    def _io_loop(self):
        """Daemon loop: sample the stick, then flush the latest requested force-
        feedback state to the device. The read's timeout paces the loop (~250 Hz
        when the stick is idle and emitting nothing) so it never busy-spins, and
        drains instantly when reports are flowing. FF requests coalesce to the
        newest -- if the render thread updates faster than the device accepts, only
        the latest is written, so no backlog builds."""
        while self._running:
            self._read_once(timeout_ms=4)
            with self._lock:
                force = self._want_force; self._want_force = None
                damper = self._want_damper; self._want_damper = None
                heat = self._want_heat; self._want_heat = None
            if force is not None:
                self._apply_force(*force)
            if damper is not None:
                self._apply_damper(damper)
            if heat is not None:
                self._apply_jitter(heat)

    def _process_twist(self, twist):
        # state.twist arrives already normalized to -1..1 (the driver decodes
        # byte 5's 6-bit signed field and scales it). Auto-center on the first
        # reading anyway to absorb any small per-unit rest offset (and so a
        # constant mis-read degrades to "no yaw" rather than a steady spin),
        # then apply a deadzone against jitter.
        if self._twist_center is None:
            self._twist_center = twist
        val = twist - self._twist_center
        if abs(val) < TWIST_DEADZONE:
            return 0.0
        # Negate: the raw twist sign is opposite the handedness the molecule
        # should turn (twisting one way must rotate it the same way on screen).
        return -max(-1.0, min(1.0, val))

    def poll_yaw(self):
        with self._lock:
            return self._last_yaw

    def poll_buttons(self):
        # Held state, not an event: the caller edge-detects. The device only
        # emits on change, so this is the last reported set and stays correct
        # for as long as nothing moves.
        if self._worker is None:
            self._read_once(timeout_ms=0)
        with self._lock:
            return self._last_buttons

    def calibrate(self, n=200):
        """Print live decoded stick state -- a sanity check that the device is
        found, permissions are right, and the axes move as expected. Push the
        stick around and twist it; x/y should reach +-1 at the stops and 'yaw'
        should swing symmetrically about 0."""
        import time
        print("Move and twist the stick. x/y: -1..+1, yaw: -1..+1 after deadzone.")
        print("(None means no report yet -- the device only reports on change; nudge the stick.)\n")
        for _ in range(n):
            state = self.ff.read_input()
            if state is None:
                print("waiting for first report...")
            else:
                yaw = self._process_twist(state.twist)
                buttons = ",".join(str(b) for b in state.pressed) or "-"
                print(f"x={state.x:+.3f}  y={state.y:+.3f}  twist={state.twist:+.3f}  "
                      f"yaw={yaw:+.2f}  hat={state.hat_name:<13s} buttons={buttons}")
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

        This only records the request; the actual (blocking) HID write happens on
        the I/O worker in background mode -- see _apply_force / _io_loop.
        """
        req = (fx, fy, SPRING_STIFFNESS_MAX if stiffness is None else stiffness)
        if self._worker is None:
            self._apply_force(*req)
        else:
            with self._lock:
                self._want_force = req

    def set_damper_coefficient(self, coefficient):
        if self._worker is None:
            self._apply_damper(coefficient)
        else:
            with self._lock:
                self._want_damper = coefficient

    def update_jitter(self, heat_fraction):
        if self._worker is None:
            self._apply_jitter(heat_fraction)
        else:
            with self._lock:
                self._want_heat = heat_fraction

    # ---- I/O-thread write helpers (blocking HID; caches skip no-op re-sends) --

    def _apply_force(self, fx, fy, stiffness):
        k = max(0, min(127, int(round(stiffness))))
        cx = max(-CP_OFFSET_MAX, min(CP_OFFSET_MAX, int(round(fx))))
        cy = max(-CP_OFFSET_MAX, min(CP_OFFSET_MAX, int(round(-fy))))
        # Only re-send an axis whose (offset, stiffness) actually changed -- an
        # unchanged spring condition already lives on the device.
        a0 = (cx, k)
        if a0 != self._last_axis0:
            self.spring.set_condition(axis=0, cp_offset=cx,
                                       pos_coeff=k, neg_coeff=k,
                                       pos_sat=SPRING_SATURATION, neg_sat=SPRING_SATURATION)
            self._last_axis0 = a0
        a1 = (cy, k)
        if a1 != self._last_axis1:
            self.spring.set_condition(axis=1, cp_offset=cy,
                                       pos_coeff=k, neg_coeff=k,
                                       pos_sat=SPRING_SATURATION, neg_sat=SPRING_SATURATION)
            self._last_axis1 = a1

    def _apply_damper(self, coefficient):
        k = max(0, min(127, int(round(coefficient))))
        if k == self._last_damper:   # unchanged -> already on the device
            return
        self._last_damper = k
        self.damper.set_condition(axis=0, pos_coeff=k, neg_coeff=k,
                                   pos_sat=DAMPER_SATURATION, neg_sat=DAMPER_SATURATION)
        self.damper.set_condition(axis=1, pos_coeff=k, neg_coeff=k,
                                   pos_sat=DAMPER_SATURATION, neg_sat=DAMPER_SATURATION)

    def _apply_jitter(self, heat_fraction):
        heat_fraction = max(0.0, min(1.0, heat_fraction))
        # sqrt: jitter amplitude ~ thermal speed ~ sqrt(kinetic energy) ~
        # sqrt(T), not linear in T -- matches how the crystal's own Langevin
        # jitter grows too.
        magnitude = int(round(JITTER_MAX_MAGNITUDE * math.sqrt(heat_fraction)))
        # When cold there is no jitter: once it has reached zero, stop re-writing
        # the sine effect every loop (the common case at the default near-frozen
        # temperature). While warm, keep re-writing so the direction re-randomizes
        # into an incoherent buzz.
        if magnitude == 0 and self._last_jitter_mag == 0:
            return
        self._last_jitter_mag = magnitude
        self.jitter.set_base(direction_deg=random.uniform(0, 360),
                              axis_x=True, axis_y=True, direction_enable=True)
        self.jitter.set_periodic(magnitude=magnitude, period_ms=JITTER_PERIOD_MS)

    def close(self):
        # Stop the I/O worker first, so freeing the effects (which are themselves
        # HID writes) can't race with the worker's own device access.
        self._running = False
        if self._worker is not None:
            self._worker.join(timeout=1.0)
            self._worker = None
        try:
            self.spring.free()
            self.damper.free()
            self.jitter.free()
        except Exception:
            pass
        self.ff.close()

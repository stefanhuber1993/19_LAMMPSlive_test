"""Shared control-input interface for the puller: any device that can report
a 2D deflection and optionally accept force-feedback commands."""


class InputSource:
    def poll(self):
        """Return (x, y) deflection in [-1, 1], sim convention: +y is up."""
        raise NotImplementedError

    def poll_yaw(self):
        """Return a rotation control signal in [-1, 1] for steering the puller's
        in-plane orientation (joystick twist/yaw axis, or Q/E keys in mouse
        mode). 0 = no rotation. Default: no yaw input."""
        return 0.0

    def poll_buttons(self):
        """Return the set of device buttons currently held, as ints. Used for
        edge-triggered actions (grab / release the puller). Default: none -- a
        device with no buttons simply never fires them, and the keyboard binding
        for the same action still works."""
        return frozenset()

    def send_force(self, fx, fy, stiffness=None):
        """Feed back a force (sim units) the device should resist with, and
        optionally how stiffly (device-specific range; None = default/max).
        No-op if unsupported."""

    def update_jitter(self, heat_fraction):
        """Fake thermal jitter on the puller, heat_fraction in [0, 1]. No-op
        if unsupported."""

    def set_damper_coefficient(self, coefficient):
        """Update the always-on viscous-damper strength. No-op if unsupported."""

    def close(self):
        pass

"""Shared control-input interface for the puller: any device that can report
a 2D deflection and optionally accept force-feedback commands."""


class InputSource:
    def poll(self):
        """Return (x, y) deflection in [-1, 1], sim convention: +y is up."""
        raise NotImplementedError

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

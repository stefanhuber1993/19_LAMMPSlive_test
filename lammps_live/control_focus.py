"""What the joystick is driving right now, and how a stick axis reads as a value.

The stick has two axes and the demo has more than two things worth steering: the
camera flying around the box, and a handful of live force-field parameters. So
rather than inventing an axis per control, there is ONE focus at a time -- the
viewport, or one slider -- and the hat switch (left / right) walks it along the
cycle. Whatever holds the focus is what the stick moves, and the screen says
which: a focused viewport gets a bright cyan frame around the scene, a focused
slider gets the same frame around its row in the panel.

The cycle is `[viewport, *sliders, *choices]`, in panel order, and it deliberately
holds only the EVERYDAY sliders (the advanced group -- puller damping, the cutoffs,
the view smoothing -- stays mouse-only). On the MesoMem playgrounds that gives
exactly viewport, Temperature, k_tilt, k_splay, zeta and the bead colouring: six
stops, all of them worth reaching for mid-demo, rather than a dozen-stop cycle you
have to count your way through.

A stop is a slider or a CHOICE (see `Choice`) -- a value you move, or a picture you
pick. They read the stick differently on purpose: a slider is a rate control (hold
it over and the value walks), a choice is a latched step (one push, one change,
however long it is held).

Moving the focus off the viewport is also what releases the puller on the
interactive playgrounds -- the same state the trigger toggles. It has to be: the
stick cannot both hold a bead against a membrane and set a number, and a bead
left attached to a stick that is now driving a slider would be dragged across
the box by every value change. The app re-grabs it when the focus comes back
(see App._cycle_focus).
"""

# --- stick-to-value response ------------------------------------------------
# A slider is driven by the stick's own DEFLECTION, as a rate: hold it over and
# the value walks, let go and it stops where it is. The shape is a deadzone, then
# a wide flat SLOW plateau, then a smooth acceleration to FAST over the last of
# the travel -- because the two things asked of this control are opposites (park a
# value exactly, and cross a whole range quickly) and a single sensitivity does
# one of them badly:
#
#   0 .. 35%     dead. Enough that a hand resting on the stick, or coming back
#                from a swing, cannot nudge the value it just set -- and no more,
#                because every bit of deadzone is travel the fine control loses.
#   35 .. 80%    slow: a flat 2% of the slider's range per second. Most of the
#                stick, and deliberately crawling: this is the band you park
#                k_tilt on the ~10 transition with.
#   80 .. 100%   the ramp: accelerates smoothly off that plateau to 45%/s at the
#                stop (a range end to end in ~2.2 s). Squared rather than linear,
#                so the first part of the ramp is still gentle and the top of the
#                travel is where the speed actually is.
#
# Flat inside the slow band, not a ramp, and continuous at both joins: "slow" has
# to be a place on the stick you can find by feel rather than a sensitivity you
# hold steady, and a step change in rate at the band edge reads as the control
# snatching.
AXIS_DEADZONE = 0.35
AXIS_SLOW_END = 0.80
AXIS_SLOW_RATE = 0.02   # fraction of the slider's range, per second
AXIS_FAST_RATE = 0.45
AXIS_CURVE = 2.0        # exponent of the slow -> fast ramp


def band_rate(x, deadzone, slow_end, slow, fast, curve=AXIS_CURVE):
    """Deflection in [-1, 1] -> a signed rate, through deadzone / slow plateau /
    smooth ramp. Shared with the camera (see render_style.CameraOrbit), which
    wants the same feel in rad/s rather than in fractions of a slider."""
    magnitude = abs(x)
    if magnitude <= deadzone:
        return 0.0
    sign = 1.0 if x > 0 else -1.0
    if magnitude <= slow_end or slow_end >= 1.0:
        return sign * slow
    u = min(1.0, (magnitude - slow_end) / (1.0 - slow_end))
    return sign * (slow + (fast - slow) * u ** curve)


def axis_rate(x, deadzone=AXIS_DEADZONE, slow_end=AXIS_SLOW_END,
              slow=AXIS_SLOW_RATE, fast=AXIS_FAST_RATE):
    """Stick deflection -> signed rate, in fractions of the focused slider's
    range per second. See the band table above."""
    return band_rate(x, deadzone, slow_end, slow, fast)


VIEWPORT = "viewport"

# A choice steps on a firm push and re-arms only once the stick is back near
# centre -- a Schmitt trigger, not a threshold. Without the gap, one push across a
# single threshold chatters through every option in a few frames as the stick
# wobbles about it, and with a threshold but no re-arm the options simply scroll
# for as long as the stick is held. Both were tried in the obvious order.
CHOICE_STEP = 0.50      # push past this to step one option
CHOICE_REARM = 0.25     # ... and back inside this before it will step again


class Choice:
    """A focus stop that picks one of a short list instead of moving a value.

    The bead colouring is the case this exists for: its options are pictures
    (director banding, or potential energy on an inferno ramp), not points on a
    scale, so "hold right to go faster" means nothing and one push has to mean one
    option. The value lives wherever it already lived -- this holds an index and
    calls `on_change`, so the mouse toggle and the joystick are two ways of moving
    the same state rather than two copies of it.
    """

    def __init__(self, label, options, index=0, on_change=None):
        self.label = label
        self.options = tuple(options)
        self.index = index % len(self.options)
        self.on_change = on_change
        self._armed = True

    @property
    def option(self):
        return self.options[self.index]

    @property
    def caption(self):
        """What the panel says the stick is driving: the control and its setting,
        since for a choice the setting IS the whole state."""
        return f"{self.label} ({self.option})"

    def step(self, direction):
        """Move `direction` options along, wrapping. Returns the new index."""
        self.index = (self.index + direction) % len(self.options)
        if self.on_change is not None:
            self.on_change(self.index)
        return self.index

    def push(self, x):
        """One frame of stick deflection. Steps at most once per push -- the stick
        has to come back near centre before it will step again. True if it moved."""
        if abs(x) < CHOICE_REARM:
            self._armed = True
            return False
        if not self._armed or abs(x) < CHOICE_STEP:
            return False
        self._armed = False
        self.step(1 if x > 0 else -1)
        return True


class ControlFocus:
    """The cycle `[viewport, *sliders]` and where in it the focus sits.

    Geometry-free and device-free: it holds Slider objects and an index, and the
    app decides what a focused viewport or slider means. That keeps the one
    question this class answers -- "what is the stick driving?" -- answerable
    from the app, the renderer and a test alike.
    """

    def __init__(self):
        self._stops = []          # everything after the viewport, in panel order
        self.index = 0

    def set_stops(self, sliders, choices=()):
        """(Re)declare what is in the cycle, and put the focus back on the
        viewport. Called on every system switch: the slider objects themselves are
        rebuilt there, and a stale index would leave the focus on a parameter the
        new playground does not have. Choices come last, after the sliders, because
        that is where the panel draws them."""
        self._stops = [*sliders, *choices]
        self.index = 0

    @property
    def on_viewport(self):
        return self.index == 0

    @property
    def stop(self):
        """Whatever holds the focus -- a Slider, a Choice, or None (viewport)."""
        return None if self.on_viewport else self._stops[self.index - 1]

    @property
    def slider(self):
        """The focused Slider, or None if the focus is the viewport or a choice."""
        stop = self.stop
        return None if isinstance(stop, Choice) else stop

    @property
    def choice(self):
        """The focused Choice, or None."""
        stop = self.stop
        return stop if isinstance(stop, Choice) else None

    @property
    def label(self):
        stop = self.stop
        if stop is None:
            return VIEWPORT
        return stop.caption if isinstance(stop, Choice) else stop.label

    def cycle(self, step):
        """Move `step` places along the cycle, wrapping. Returns the new index."""
        self.index = (self.index + step) % (len(self._stops) + 1)
        return self.index

    def drive(self, x, dt):
        """Apply one frame of stick deflection to whatever holds the focus.

        A slider walks at the banded rate; a choice steps once per push. No-op on
        the viewport. Returns True if anything moved.
        """
        stop = self.stop
        if stop is None:
            return False
        if isinstance(stop, Choice):
            return stop.push(x)
        rate = axis_rate(x)
        if rate == 0.0:
            return False
        stop.nudge(rate * (stop.vmax - stop.vmin) * dt)
        return True

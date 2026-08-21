"""What the joystick is driving right now, and how a stick axis reads as a value.

The stick has two axes and the demo has more than two things worth steering: the
camera flying around the box, and a handful of live force-field parameters. So
rather than inventing an axis per control, there is ONE focus at a time -- the
viewport, or one slider -- and the hat switch moves it. Whatever holds the focus
is what the stick moves, and the screen says which: a focused viewport gets a
bright cyan frame around the scene, a focused slider gets the same frame around
its row in the panel.

THE HAT IS LAID OUT LIKE THE SCREEN, which is the whole of the navigation model:

    left / right   between the two AREAS -- the scene on the left of the window,
                   the control panel on its right. Right goes into the panel,
                   left comes back out to the scene.
    up / down      between the stops WITHIN the panel, once it has the focus.
                   They are drawn in one column, so up is the row above.

...and once the panel has the focus, THE STICK'S OWN UP/DOWN AXIS walks the rows
too (see `RowStepper`), so the hand does not have to leave the stick to change
which slider it is setting: push forward or back to pick the row, left or right to
move its value. The hat still does it, for a hand that prefers to.

The first version of this walked a single flat cycle with left/right, which is
what the hardware suggests and what the screen contradicts: the sliders sit on
top of each other, so "right" to reach the next one down is a direction that
means nothing. Up/down for a vertical list and left/right for the two panes is
the mapping somebody watching the screen can predict without being told.

The panel's stops are `[*choices, *sliders]`, in panel order, and they
deliberately hold only the EVERYDAY sliders (the advanced group -- puller
damping, the cutoffs, the view smoothing -- stays mouse-only). On the MesoMem
playgrounds that gives exactly the bead colouring, Temperature, k_tilt, k_splay
and zeta: five rows, all of them worth reaching for mid-demo, rather than a dozen
you have to count your way through.

A stop is a slider or a CHOICE (see `Choice`) -- a value you move, or a picture you
pick. They read the stick differently on purpose: a slider is a rate control (hold
it over and the value walks), a choice is a latched step (one push, one change,
however long it is held).

Moving the focus off the viewport is also what releases the puller on the
interactive playgrounds -- the same state the B key toggles. It has to be: the
stick cannot both hold a bead against a membrane and set a number, and a bead
left attached to a stick that is now driving a slider would be dragged across
the box by every value change. The app re-grabs it when the focus comes back
(see App._move_focus).
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

# --- stepping a list from a stick axis ---------------------------------------
# The stick's up/down axis walks the panel's rows while the panel holds the focus.
# An axis is a continuous thing being asked for discrete steps, and doing that
# naively gives one of two failures: a bare threshold sweeps the whole list in a
# few frames, and a latch that only ever fires once per push means reaching the
# fifth row is five separate pushes. So this is a keyboard's auto-repeat:
#
#   * ONE STEP the moment the axis crosses `ROW_PUSH`, so a flick is one row;
#   * then NOTHING for `ROW_FIRST_REPEAT` even if it is held -- that pause is what
#     makes a single row reachable without care;
#   * then a step every `ROW_REPEAT_INTERVAL`: about two rows a second, so the
#     cyan frame is visibly stepping from row to row and you can let go on the one
#     you wanted. A five-row panel takes a couple of seconds to walk end to end,
#     which is the right price for not overshooting;
#   * and it will not fire again from a held stick until it has come back inside
#     `ROW_REARM`. A gap, not one threshold: a stick parked near the edge of a
#     bare threshold chatters across it.
#
# Slower than a keyboard's repeat on purpose. The list is five rows long and every
# row is a control somebody is about to touch, so overshooting costs more than
# waiting does.
ROW_PUSH = 0.55
ROW_REARM = 0.30
ROW_FIRST_REPEAT = 0.60         # seconds held before the repeat starts
ROW_REPEAT_INTERVAL = 0.45      # ... and between repeats after that
# How far up/down has to out-deflect left/right before it counts as a row change
# rather than a diagonal. Without this, running a slider's value up at the fast
# end of the travel -- a hard push right, which no hand delivers at exactly zero
# elevation -- also walks off the row being set. The value change is what the hand
# asked for there, so the row change is the one that has to yield.
ROW_DOMINANCE = 1.25


class RowStepper:
    """One stick axis -> discrete steps along a list, with a repeat. See above.

    Time-based rather than frame-based (it is handed the frame's dt), so the feel
    does not change with the frame rate -- which on these playgrounds runs from
    the high fifties down to the twenties depending on the bead count.
    """

    def __init__(self, push=ROW_PUSH, rearm=ROW_REARM,
                 first_repeat=ROW_FIRST_REPEAT, interval=ROW_REPEAT_INTERVAL,
                 dominance=ROW_DOMINANCE):
        self.push_threshold = push
        self.rearm = rearm
        self.first_repeat = first_repeat
        self.interval = interval
        self.dominance = dominance
        # Armed, unlike after `reset()`: a stepper that has never seen the axis has
        # nothing to be suspicious of, and requiring a trip through centre first
        # would mean the very first push of a session did nothing.
        self._armed = True
        self._direction = 0
        self._held = 0.0
        self._next = 0.0

    def reset(self):
        """SUSPEND it: nothing fires until the axis has come back near centre.

        That is the useful meaning of "reset" for this control, not "ready to
        fire". It is called when something else moves the focus -- the hat, or a
        playground switch -- and at that moment the stick may well be held right
        over, because on the viewport the same axis was flying the camera. Armed,
        it would step a row on the next frame for nothing.
        """
        self._armed = False
        self._direction = 0
        self._held = 0.0
        self._next = 0.0

    def step(self, y, dt, cross=0.0):
        """One frame. `y` is the deflection to step from, `cross` the other axis's
        (for the dominance rule). Returns -1, 0 or +1: how far to move this frame,
        in the sign of `y`."""
        magnitude = abs(y)
        if magnitude < self.rearm:
            self._armed = True                 # back near centre: ready again
            self._direction = 0
            self._held = 0.0
            return 0
        if magnitude < self.push_threshold or magnitude < self.dominance * abs(cross):
            return 0                           # in the gap, or a diagonal
        direction = 1 if y > 0 else -1
        if self._armed or (self._direction and direction != self._direction):
            # A fresh push, or a reversal without going through centre -- which is
            # what going one row too far and coming straight back feels like, so it
            # has to step at once rather than wait out the repeat delay.
            self._armed = False
            self._direction = direction
            self._held = 0.0
            self._next = self.first_repeat
            return direction
        if not self._direction:
            return 0                           # suspended: still waiting for centre
        self._held += dt
        if self._held >= self._next:
            self._next += self.interval
            return direction
        return 0


# --- picking one of a short list ---------------------------------------------
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
    """Where the focus sits: the viewport (index 0), or one of the panel's stops.

    Geometry-free and device-free: it holds Slider objects and an index, and the
    app decides what a focused viewport or slider means. That keeps the one
    question this class answers -- "what is the stick driving?" -- answerable
    from the app, the renderer and a test alike.
    """

    def __init__(self):
        self._stops = []          # choices, then the sliders in panel order
        self.index = 0
        # The stick's up/down axis as a second way to walk the rows, alongside the
        # hat's. Lives here rather than in the app because it is part of "what is
        # the stick driving" and, like the index, has to be forgotten when the
        # stops are rebuilt.
        self.row_stepper = RowStepper()

    def set_stops(self, sliders, choices=()):
        """(Re)declare the panel's stops, and put the focus back on the viewport.
        Called on every system switch: the slider objects themselves are rebuilt
        there, and a stale index would leave the focus on a parameter the new
        playground does not have. Choices come first, at the top: that is the
        order they are drawn in, and up/down has to agree with the screen."""
        self._stops = [*choices, *sliders]
        self.index = 0
        self.row_stepper.reset()

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

    def to_viewport(self):
        """Hat left: back out of the panel to the scene. Returns the new index."""
        self.index = 0
        return self.index

    def enter_stops(self):
        """Hat right: into the panel, at its top row. A no-op on a playground
        with no everyday stops at all -- there is nowhere to go, and leaving the
        viewport would release the puller for nothing. Returns the new index."""
        if self._stops and self.on_viewport:
            self.index = 1
        return self.index

    def step_stop(self, step):
        """Hat up / down: `step` rows along the panel, wrapping within it.

        Wrapping inside the panel rather than falling out of it at either end:
        the way back to the scene is hat left, one direction with one meaning,
        rather than two places in a list where up or down happens to also mean
        "leave". Returns the new index.
        """
        if self.on_viewport or not self._stops:
            return self.index
        self.index = 1 + (self.index - 1 + step) % len(self._stops)
        return self.index

    def row_step(self, y, dt, cross=0.0):
        """How far the stick's up/down axis wants to move the focused ROW this
        frame: -1, 0 or +1, already in "places along the stops" (so forward on the
        stick, which is +y, is the row ABOVE -- one place back, as it is for the
        hat). 0 while the viewport holds the focus: the stick is flying the camera
        there and its vertical axis is already spoken for."""
        if self.on_viewport:
            self.row_stepper.reset()
            return 0
        return -self.row_stepper.step(y, dt, cross)

    def drive(self, x, dt):
        """Apply one frame of stick deflection to whatever holds the focus.

        A slider walks at the banded rate; a choice steps once per push. No-op on
        the viewport. Returns True if anything moved.

        This is the LEFT/RIGHT axis only -- up/down belongs to `row_step`, which
        the caller applies first, because which control the stick is setting has to
        be settled before its value is moved.
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

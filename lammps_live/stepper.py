"""Running the simulation and drawing the frame at the same time.

The obvious loop -- step, read, draw, repeat -- leaves one of the two halves idle
throughout the other, and on the big systems both halves are expensive. Measured
at 1500 beads in a 1300x900 window:

    mesomem_sheet     step 20.8 ms    draw 6.3 ms
    mesomem_assembly  step 18.3 ms    draw 5.4 ms

so roughly a quarter of every frame is the simulation waiting for pygame to
finish drawing the previous state.

They do not have to take turns. LAMMPS' `run` is C, reached through ctypes, which
RELEASES the GIL for the duration -- so a Python thread can hold the interpreter
and draw while LAMMPS integrates on another core. Measured directly (20 steps of
the sheet against a pure-Python workload): 64.5 ms sequential, 44.1 ms
overlapped, i.e. the step vanishes completely under the drawing.

WHAT IT COSTS: one frame of latency. The loop becomes "read the state, launch the
next step, draw what you read", so the picture on screen is one step behind the
one being computed, and a force applied this frame shows its effect in the next.
At 60 Hz that is 16 ms, well under the ~50 ms where a haptic loop starts to feel
disconnected, and the force-feedback shaping already smooths over a longer window
than that (see config.FF_SMOOTHING_TAU).

THE RULE THIS IMPOSES: between `start()` and the next `wait()`, NOTHING may touch
the LAMMPS instance -- not a command, not a readout. The app reads everything a
frame needs before launching, and every readout hands back a copy (`np.array` of
the extracted arrays) rather than a view into LAMMPS' own memory, so the drawing
is working from a snapshot that the worker cannot move under it. Rebuilding or
closing a system must wait first, which is why `App` funnels those through
`_sim_idle()`.

Set `config.OVERLAP_SIM_AND_RENDER = False` to go back to taking turns; the
stepper then runs the step inline and everything else is unchanged.
"""
import threading
from time import perf_counter


class SimStepper:
    """Advances an MDSystem, on a worker thread when overlapping is enabled."""

    def __init__(self, enabled=True):
        self.enabled = bool(enabled)
        self._thread = None
        self._error = None
        self._pending = 0
        self._done = 0
        # How long the last wait() actually blocked -- with the overlap on, this
        # is the part of the step the drawing did NOT cover, which is the number
        # worth watching in the --debug breakdown.
        self.wait_seconds = 0.0

    def start(self, system, n):
        """Begin advancing `system` by `n` steps. The caller must not touch the
        simulation again until wait() returns."""
        if self._thread is not None:
            self.wait()
        if n <= 0:
            return
        if not self.enabled:
            t0 = perf_counter()
            self._run(system, n)
            self.wait_seconds = perf_counter() - t0
            self._done += n
            return
        self._pending = n
        self._thread = threading.Thread(target=self._run, args=(system, n),
                                        name="lammps-step", daemon=True)
        self._thread.start()

    def _run(self, system, n):
        # PlaygroundSystem.step catches a user-induced blow-up itself and latches
        # it for the HUD; anything that escapes that is a real bug, and is carried
        # back to the main thread rather than dying silently in a daemon.
        try:
            system.step(n)
        except BaseException as exc:       # noqa: BLE001 -- re-raised in wait()
            self._error = exc

    def wait(self):
        """Block until the in-flight step (if any) has finished. Returns the
        number of steps completed since the last call, and re-raises whatever the
        worker hit."""
        if self._thread is not None:
            t0 = perf_counter()
            self._thread.join()
            self.wait_seconds = perf_counter() - t0
            self._thread = None
            self._done += self._pending
            self._pending = 0
        done, self._done = self._done, 0
        if self._error is not None:
            error, self._error = self._error, None
            raise error
        return done

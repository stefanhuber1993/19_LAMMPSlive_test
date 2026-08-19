"""RemoteSystem: an MDSystem3D whose simulation is somewhere else.

The app talks to this exactly as it talks to `PlaygroundSystem` -- same interface,
same spec, same readouts -- and cannot tell the difference. `step(n)` waits for the
next frame off the socket instead of integrating; the sliders send their values
instead of issuing LAMMPS commands. Everything else is genuinely the same code:
the analysis, the observables, the energy panels, the RDF and the trajectory
smoothing all run here, on the received frames, out of `lammps_live.playground`.

WHERE THE WORK LANDS, AND WHY IT FITS. `step()` is called on the stepper's worker
thread (see stepper.py), so the network wait AND the analysis that follows it
overlap the drawing of the previous frame. That is not a detail -- the analysis is
the expensive half of this end (measured 1.5 us/bead/chunk, so ~10 ms at 10k
beads), and overlapped it costs max(analysis, render) per frame rather than the
sum. It is the same trick that lets the local demo run 1500 beads at 60 fps, used
for a different expensive thing.

FRAMES ARE DROPPED, NEVER QUEUED. The reader thread keeps only the newest frame.
If this machine falls behind -- a hitch, a slow analysis frame, a window resize --
the next `step()` picks up where the simulation actually IS, not where it was three
frames ago. A queue would trade latency for smoothness and, over a link that
cannot be flow-controlled by the renderer, the latency would only ever grow. The
dropped-frame count is on the HUD.

DISCONNECTED IS A NORMAL STATE. Selecting this playground builds a RemoteSystem
with no connection: it holds the empty scene, reports what to do in the HUD, and
every readout answers safely. That is what lets the connect panel exist inside the
running app, and what lets the whole client be tested without a cluster.
"""
import socket
import threading
import time
from collections import deque

import numpy as np

from ..mdsystem import MDSystem3D
from ..playground import forcefield as ff_registry
from ..playground.modes import SimMode
from ..playground.faults import Fault
from ..playground.observables import Analysis
from ..playground.rdf import InPlaneRDF, RadialRDF3D
from ..playground.smoothing import TrajectorySmoother
from ..playground.state import Box, FrameState
from ..playground.system import SMOOTHING_KEY, make_spec
from . import protocol


class LinkClosed(Exception):
    """The server went away, cleanly or otherwise."""


class FrameLink:
    """One connection to a frame server: a reader thread and a send lock.

    Holds the LATEST frame, not a queue of them (see the module docstring), plus
    the counters the HUD needs to say how the link is doing.
    """

    def __init__(self, sock, welcome):
        self.sock = sock
        self.welcome = welcome
        self.error = None
        self.closed = threading.Event()
        self._new = threading.Event()
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._latest = None            # (header, payload)
        self.received = 0
        self.dropped = 0
        self.bytes_in = 0
        self.rtt_ms = None
        self._pings = {}
        # Simulation-destroying events reported by the server, oldest first. Small
        # and bounded: if more than a handful arrive before anyone looks, the older
        # ones have been superseded by the newer anyway.
        self.faults = deque(maxlen=8)
        # Byte and frame counts over the last second, for the HUD's rate readout.
        self._recent = deque(maxlen=240)
        self._thread = threading.Thread(target=self._read_loop, name="frame-reader",
                                        daemon=True)
        self._thread.start()

    # ---- connecting ---------------------------------------------------------

    # How long to wait for a welcome once the server has said it is BUILDING.
    # Building is LAMMPS' own setup on 10k+ beads -- tens of seconds, and minutes
    # if the size is raised -- and it is bounded by nothing this end can see, so
    # this is a "something has gone wrong" limit rather than an estimate.
    BUILD_TIMEOUT = 600.0

    @classmethod
    def connect(cls, host, port, token, timeout=10.0, on_notice=None):
        """Open a link, or raise LinkClosed with something a user can act on.

        `timeout` covers reaching the server and its answer -- EXCEPT that a server
        which says it is building first is then given BUILD_TIMEOUT instead. That
        distinction is the whole point: a silent server is broken and should be
        given up on quickly, while one that has told us it is building LAMMPS is
        working, and giving up on it drops a socket the server is about to answer
        (and, worse, makes it throw away the build and start over for the retry).
        `on_notice` receives such messages, so the panel can show the wait.
        """
        try:
            sock = socket.create_connection((host, int(port)), timeout=timeout)
        except OSError as exc:
            raise LinkClosed(f"could not reach {host}:{port} -- {exc.strerror or exc}. "
                             f"Is the tunnel up and the server running?") from exc
        protocol.set_socket_options(sock)
        try:
            sock.sendall(protocol.pack({"t": "hello", "version": protocol.VERSION,
                                        "token": token}))
            while True:
                header, _payload = protocol.recv_message(sock)
                if header is None or header.get("t") != "building":
                    break
                if on_notice is not None:
                    on_notice(str(header.get("msg", "the server is building")))
                sock.settimeout(cls.BUILD_TIMEOUT)
        except (OSError, protocol.ProtocolError) as exc:
            sock.close()
            raise LinkClosed(f"handshake failed: {exc}") from exc
        if header is None:
            sock.close()
            raise LinkClosed("the server closed the connection during the handshake")
        if header.get("t") == "error":
            sock.close()
            raise LinkClosed(str(header.get("msg", "the server refused us")))
        if header.get("t") != "welcome":
            sock.close()
            raise LinkClosed(f"unexpected first message {header.get('t')!r}")
        # Back to blocking with no timeout: create_connection leaves the socket on
        # a 10 s timeout, which would turn every quiet moment on the control
        # channel into a spurious read error.
        sock.settimeout(None)
        return cls(sock, header)

    # ---- reading ------------------------------------------------------------

    def _read_loop(self):
        try:
            while True:
                header, payload = protocol.recv_message(self.sock)
                if header is None:
                    self.error = "the server closed the connection"
                    break
                kind = header.get("t")
                if kind == "frame":
                    with self._lock:
                        if self._latest is not None:
                            self.dropped += 1
                        self._latest = (header, payload)
                        self.received += 1
                        self.bytes_in += len(payload)
                        self._recent.append((time.monotonic(), len(payload)))
                    self._new.set()
                elif kind == "fault":
                    # Queued, not folded into the latest-frame slot: this is the one
                    # kind of message that must not be dropped, and the frame slot
                    # exists precisely to drop things (see take_frame).
                    self.faults.append(header)
                elif kind == "pong":
                    sent = self._pings.pop(header.get("id"), None)
                    if sent is not None:
                        self.rtt_ms = 1000.0 * (time.monotonic() - sent)
                elif kind == "error":
                    self.error = str(header.get("msg"))
                    break
        except (OSError, protocol.ProtocolError) as exc:
            self.error = self.error or f"{type(exc).__name__}: {exc}"
        finally:
            self.closed.set()
            self._new.set()

    def take_frame(self, timeout=0.25):
        """The newest frame, or None if none arrived inside `timeout`."""
        self._new.wait(timeout)
        with self._lock:
            frame, self._latest = self._latest, None
            if frame is None:
                self._new.clear()
        return frame

    def rates(self):
        """(frames/s, MB/s) over the last second."""
        cutoff = time.monotonic() - 1.0
        with self._lock:
            recent = [(t, n) for t, n in self._recent if t >= cutoff]
        if not recent:
            return 0.0, 0.0
        return float(len(recent)), sum(n for _t, n in recent) / 1e6

    # ---- writing ------------------------------------------------------------

    def send(self, message):
        """Send one control message. Swallows a dead socket: the reader thread is
        the one authority on whether the link is up, and a slider must not raise."""
        if self.closed.is_set():
            return False
        try:
            with self._send_lock:
                self.sock.sendall(protocol.pack(message))
            return True
        except OSError as exc:
            self.error = self.error or f"send failed: {exc}"
            self.closed.set()
            return False

    def ping(self):
        ping_id = self.received + 1
        self._pings[ping_id] = time.monotonic()
        # Never let unanswered pings accumulate on a link that stopped answering.
        if len(self._pings) > 8:
            self._pings.clear()
        self.send({"t": "ping", "id": ping_id})

    def close(self, say_goodbye=True):
        if say_goodbye:
            self.send({"t": "bye"})
        self.closed.set()
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


class RemoteSystem(MDSystem3D):
    """A playground running elsewhere, drawn here."""

    def __init__(self, playground, preset=None, target=None):
        self.playground = playground
        self.preset = preset
        self.target = (target or playground.remote).resolved()
        self.mode_name = "sim"
        # SimMode supplies every puller-shaped readout its neutral answer, which is
        # the same object the local sim-mode playgrounds use -- so "no puller" means
        # exactly the same thing here as there rather than a second set of stubs.
        self.mode = SimMode()
        self.mode.attach(self)
        self.force_field = ff_registry.get(playground.force_field)(
            **playground.force_field_options)
        self.params = self.force_field.new_params(playground.resolved_params(preset))
        self.scenario = playground.scenario
        self.scenario_params = self.scenario.new_params()
        self.spec = make_spec(playground, "sim", preset)
        self.has_directors = self.force_field.has_directors

        # The cell is known from the scenario without building anything -- which is
        # what lets the camera frame the scene, and the box outline be drawn, before
        # a single frame has arrived. The server's own box replaces it on connect
        # (authoritative: a scenario that lets a barostat settle would differ).
        self.box = self.scenario.build(self.scenario_params,
                                       np.random.default_rng(0)).box
        self.natoms = int(self.scenario_params["n"]
                          if self.scenario_params.has("n") else 0)
        self.all_ids = np.arange(1, self.natoms + 1)
        self.bonds = []
        self.brightness = None
        self.controlled_index = None
        self.controlled_id = None

        self.analysis = Analysis(self.force_field, playground.observables,
                                 energy_every=playground.analysis_energy_every)
        self._rdf = self._make_rdf()
        self._smoother = TrajectorySmoother()
        self._smoothing_tau = 0.0
        self._render_cache = None
        self._render_frame = -1
        self._energy_cache = None
        self._energy_render_frame = -1
        self._frame = 0
        self._state = None
        self._energies = None
        self._thermo = (0.0, 0.0, 0.0, 0.0, 0.0)
        self._sim_time = 0.0
        self._last_step_dt = 0.0
        self._unstable = None
        self._seq = 0
        self.analysis_seconds = 0.0
        self.wait_seconds = 0.0

        self.link = None
        self.status = "not connected"
        self._playing = False
        self._target_temp = self.spec.temperature.default
        self._sent_params = {}
        self._fault = None             # the far side's last simulation-killing event
        # Which received frame last had `get_bead_energies` called against it. The
        # request is derived from that with a couple of frames of hysteresis rather
        # than from an edge, because how many times `step()` is called per drawn
        # frame is not something this end gets to assume (a stepper that polls, a
        # paused run that only draws) and an edge would flap between the two states.
        self._energy_asked_frame = -1000
        self._energy_requested = False
        self._ping_countdown = 0
        # Waiting for the far side to finish rebuilding after a Reset -- see reset()
        # and _ingest, which clears it when a frame from the new run arrives.
        self._resetting = False

    # ---- the connection -----------------------------------------------------

    def connect(self, host=None, port=None, token="", timeout=10.0):
        """Open a link to a server (the CLI path). The connect panel uses `attach`
        instead, because it has already opened the socket through its tunnel."""
        link = FrameLink.connect(host or "127.0.0.1",
                                 port or self.target.local_port, token, timeout,
                                 on_notice=lambda msg: print(f"[lammps-live] {msg}"))
        self.attach(link)
        return link

    def attach(self, link):
        """Adopt an open link and align the far end with the local state."""
        self.detach()
        self.link = link
        welcome = link.welcome or {}
        box = welcome.get("box")
        if box:
            self.box = Box(tuple(box["lo"]), tuple(box["hi"]),
                           tuple(bool(p) for p in box["periodic"]))
            self._rdf = self._make_rdf()
        n = int(welcome.get("natoms") or self.natoms)
        if n != self.natoms:
            self.natoms = n
            self.all_ids = np.arange(1, n + 1)
        self._sim_time = float(welcome.get("sim_time") or 0.0)
        self._unstable = None
        self._fault = None
        self._energies = None
        self._energy_cache = None
        self._energy_render_frame = -1
        self._smoother.reset()
        self.status = (f"{welcome.get('host', '?')} "
                       f"[{welcome.get('profile', '?')}], "
                       f"{self.natoms:,} beads, job {welcome.get('slurm_job') or '-'}")
        # The far end starts from its own defaults, so push everything this end is
        # currently showing: the sliders, the temperature, the play state, and the
        # frame configuration. Anything not sent here would silently disagree with
        # the panel the user is looking at.
        link.send({"t": "config", "fps": self.target.fps,
                   "codec": self.target.codec, "free_run": self.target.free_run,
                   "energies": self._energy_requested})
        link.send({"t": "temp", "value": self._target_temp})
        for param in self.params.live_params():
            value = float(self.params[param.name])
            self._sent_params[param.name] = value
            link.send({"t": "set", "key": param.name, "value": value})
        link.send({"t": "play" if self._playing else "pause"})
        return link

    def detach(self, say_goodbye=True):
        if self.link is not None:
            self.link.close(say_goodbye=say_goodbye)
            self.link = None
        self.status = "not connected"
        # Nothing is going to answer a reset that was in flight when the link went.
        self._resetting = False

    @property
    def connected(self):
        return self.link is not None and not self.link.closed.is_set()

    def link_error(self):
        """Why the link went down, once it has -- for the connect panel."""
        if self.link is not None and self.link.closed.is_set():
            return self.link.error or "the connection closed"
        return None

    # ---- what the app drives ------------------------------------------------

    def set_playing(self, playing):
        """Play/Pause, forwarded. Without this the server would keep integrating
        into a socket nobody is reading -- an idle A100 is cheaper than a busy one
        computing frames that are thrown away."""
        playing = bool(playing)
        if playing == self._playing:
            return
        self._playing = playing
        if self.connected:
            self.link.send({"t": "play" if playing else "pause"})

    def set_target_temp(self, T):
        t_min, t_max = self.playground.temperature
        T = max(t_min, min(t_max, float(T)))
        if T == self._target_temp:
            return
        self._target_temp = T
        if self.connected:
            self.link.send({"t": "temp", "value": T})

    def set_extra_param(self, key, value):
        """A live parameter moved. Held locally too, because the energy panels are
        computed HERE and must use the coefficients the far end is running."""
        if key == SMOOTHING_KEY:
            self._smoothing_tau = max(0.0, float(value))
            return
        if not self.params.has(key):
            return
        if not self.params.set(key, value):
            return
        # The clamped value, not the raw slider value: `wc` is capped at `rc`, and
        # sending the uncapped number would make the far end run coefficients the
        # local energy decomposition is not using.
        clamped = float(self.params[key])
        if self._sent_params.get(key) == clamped:
            return
        self._sent_params[key] = clamped
        if self.connected:
            self.link.send({"t": "set", "key": key, "value": clamped})

    def _take_wire_fault(self, message):
        """Adopt a fault the server reported.

        THE REVERTED VALUES MATTER AS MUCH AS THE MESSAGE. When the far side's
        rebuild had to put a parameter back, this end is still holding the value
        that killed it -- in `params`, which the energy panels are computed from,
        and in `_sent_params`, which decides what is worth sending. Left alone, the
        panels would describe coefficients nobody is running, and dragging the
        slider back to the bad value would send nothing at all (because this end
        thinks it already sent it) so the picture and the sliders would disagree
        with no way to resync.
        """
        fault = Fault.from_message(message)
        if fault is None:
            return None
        for key, value in fault.reverted.items():
            if self.params.has(key):
                self.params.set(key, value)
                self._sent_params[key] = float(self.params[key])
        return fault

    def take_fault(self):
        """The next fault the far side reported, once. See PlaygroundSystem.

        Drained here rather than when the message arrives, because adopting one
        writes to `params` and `_sent_params` -- and the message arrives on the
        link's reader thread while the caller of this is the app's own.
        """
        if self._fault is None and self.link is not None and self.link.faults:
            self._fault = self._take_wire_fault(self.link.faults.popleft())
        fault, self._fault = self._fault, None
        return fault

    def live_param_values(self):
        """The effective value of every live parameter, for the sliders to follow."""
        return {p.name: float(self.params[p.name])
                for p in self.params.live_params()}

    def reset(self):
        """Re-randomize the box on the far side, keeping the current parameters.

        The rebuild happens THERE and takes as long as it takes -- at 10k beads,
        LAMMPS' setup plus a rejection-sampled random fill, which is seconds during
        which no frames come back at all. So this also latches `_resetting`, and the
        HUD says the far side is rebuilding until a frame from the new run arrives
        (the server restarts its sequence at 0, which is how that is recognised).
        Without it the picture simply sits on the last frame of the OLD run, which
        is indistinguishable from a Reset that did nothing.

        Called only with the simulation thread idle (App._reset_simulation waits
        first): it replaces the analysis and clears the smoother, both of which the
        stepper thread reads mid-frame.
        """
        self._unstable = None
        self._energies = None          # the old run's colours are not the new box's
        self._energy_cache = None
        self._energy_render_frame = -1
        self._smoother.reset()
        self._rdf.reset()
        self.analysis = Analysis(self.force_field, self.playground.observables,
                                 energy_every=self.playground.analysis_energy_every)
        self._playing = False
        if self.connected:
            self._resetting = True
            self.link.send({"t": "reset"})

    def set_input_force(self, fx, fy):
        pass

    def set_puller_damping(self, gamma):
        pass

    def steer_orientation(self, rate, dt):
        pass

    # ---- the frame ----------------------------------------------------------

    def step(self, n):
        """Wait for the next frame and take it in.

        Called on the stepper's worker thread, which is what puts both the network
        wait and the analysis alongside the drawing rather than in front of it.
        `n` is ignored: how far the simulation advances per frame is the server's
        decision, and it reports it (the `dt` on each frame).
        """
        if not self.connected:
            # Nothing to wait for. Sleep out a frame rather than spinning, so a
            # disconnected system does not burn a core in the stepper thread.
            time.sleep(1.0 / 60)
            return
        self._sync_energy_request()
        self._ping_countdown -= 1
        if self._ping_countdown <= 0:
            self._ping_countdown = 60
            self.link.ping()
        t0 = time.perf_counter()
        frame = self.link.take_frame(timeout=0.25)
        self.wait_seconds = time.perf_counter() - t0
        if frame is not None:
            self._ingest(frame)

    ENERGY_REQUEST_HOLD = 3        # frames of asking-nothing before it is dropped

    def _sync_energy_request(self):
        """Ask for per-bead energies only while the bead colouring is painting them.

        They are a tenth of the frame's bytes and the server has to gather them from
        a per-atom compute, so the request follows whether `get_bead_energies` is
        being called -- i.e. whether the renderer is actually using them. Costs a
        frame or two of lag on the toggle, and nothing at all while it is off.
        """
        wanted = (self._frame - self._energy_asked_frame) < self.ENERGY_REQUEST_HOLD
        if wanted != self._energy_requested:
            self._energy_requested = wanted
            self.link.send({"t": "config", "energies": wanted})

    def _ingest(self, frame):
        """Decode one frame and run this end's analysis on it."""
        header, payload = frame
        codec = header.get("codec", "q16")
        arrays = protocol.decode_frame(
            header.get("arrays") or [], payload, self.box,
            energy_range=self.spec.render_style.energy_range, codec=codec)
        positions = arrays.get("positions")
        if positions is None:
            return
        n = len(positions)
        if n != len(self.all_ids):
            self.all_ids = np.arange(1, n + 1)
            self.natoms = n
        self._state = FrameState(positions=positions,
                                 directors=arrays.get("directors"),
                                 types=None, ids=self.all_ids, box=self.box)
        self._energies = arrays.get("energies")
        seq = int(header.get("seq") or 0)
        # The server restarts its sequence at 0 on a rebuild, so a frame numbered
        # at or below the one we were on is the first of the NEW run.
        if self._resetting and seq <= self._seq:
            self._resetting = False
        self._seq = seq
        self._sim_time = float(header.get("sim_time") or 0.0)
        self._last_step_dt = float(header.get("dt") or 0.0)
        thermo = header.get("thermo")
        if thermo and all(np.isfinite(v) for v in thermo):
            self._thermo = tuple(float(v) for v in thermo)
        self._unstable = header.get("unstable")
        self._frame += 1
        self._render_cache = None
        t0 = time.perf_counter()
        self.analysis.update(self._state, self.params)
        self.analysis_seconds = time.perf_counter() - t0

    def _ensure_current(self):
        """While the run is paused, keep the picture up to date from the readouts.

        Paused, the app does not call `step()` at all (see App._tick), so nothing
        would take in the frame that shows the state the server is holding -- and
        the scene would stay empty after connecting until Play was pressed. While
        playing, `step()` has already taken the newest frame and this does nothing.
        """
        if self._playing or not self.connected:
            return
        frame = self.link.take_frame(timeout=0.0)
        if frame is not None:
            self._ingest(frame)

    def _render_state(self):
        """The frame as drawn: the received state, optionally smoothed. Mirrors
        PlaygroundSystem._render_state, including that NOTHING which measures comes
        through here."""
        self._ensure_current()
        state = self._state
        if state is None:
            return None
        if self._smoothing_tau <= 0.0:
            if self._smoother.active:
                self._smoother.reset()
            return state
        if self._render_frame != self._frame or self._render_cache is None:
            self._render_cache = self._smoother.apply(
                state, self._smoothing_tau, self._last_step_dt)
            self._render_frame = self._frame
        return self._render_cache

    # ---- readouts -----------------------------------------------------------

    def _empty_positions(self):
        return np.zeros((0, 3))

    def get_positions_3d(self):
        state = self._render_state()
        if state is None:
            return np.zeros(0, dtype=int), self._empty_positions(), np.zeros(0, bool)
        return state.ids, state.positions, np.zeros(len(state.positions), bool)

    def get_dipoles_3d(self):
        state = self._render_state()
        if state is None or state.directors is None:
            n = 0 if state is None else len(state.positions)
            return np.zeros((n, 3))
        return state.directors

    def get_all_positions(self):
        state = self._render_state()
        if state is None:
            return (np.zeros(0, dtype=int), np.zeros((0, 2)), np.zeros(0, bool), None)
        return (state.ids, state.positions[:, :2],
                np.zeros(len(state.positions), bool), None)

    def get_bead_energies(self):
        """The per-bead energies from the newest frame that carried them, smoothed
        the same way the drawn positions are (see PlaygroundSystem._smooth_energies).

        None until some have actually arrived -- NOT an array of zeros, which is
        what this returned at first and which is a lie the colour ramp believes:
        zero sits at the TOP of the energy range, so a scene that had simply not
        been sent its energies yet came up painted uniformly white, as if every bead
        were free. None means "no data", the renderer keeps the director banding,
        and the colours appear when the numbers do -- one or two frames later, since
        asking is itself how the request gets made (see _sync_energy_request).
        """
        self._energy_asked_frame = self._frame
        if self._energies is None:
            return None
        if self._smoothing_tau <= 0.0:
            return self._energies
        if self._energy_render_frame != self._frame:
            self._energy_cache = self._smoother.smooth_scalar(
                "bead_energy", self._energies, self._smoothing_tau,
                self._last_step_dt)
            self._energy_render_frame = self._frame
        return self._energy_cache

    def get_bead_brightness(self):
        return None

    def get_bonds_3d(self):
        return []

    def get_thermo_state(self):
        self._ensure_current()
        return self._thermo

    def get_sim_time(self):
        return self._sim_time

    def get_rdf(self):
        state = self._state
        if state is None:
            return self._rdf.get()
        if all(self.box.periodic):
            self._rdf.add(state.positions)
        else:
            self._rdf.add(state.positions[:, :2])
        return self._rdf.get()

    def get_potential_terms(self):
        return None                     # no controlled particle in sim mode

    def get_total_potential_terms(self):
        n = max(1, len(self.all_ids))
        scale = self.force_field.energy_scale_per_particle * n
        return self.analysis.energy_panel(
            "Whole-system energy -- additive (reduced units)", scale)

    def get_hud_lines(self):
        """What the scene overlay says. The link's own state belongs here: when
        frames stop arriving, the picture simply freezes, and without a line saying
        so that is indistinguishable from a simulation that has stopped moving."""
        if not self.connected:
            error = self.link_error()
            # Short lines: the HUD sits bottom-left, and the Play/Pause row starts
            # about 240 px in, so anything longer runs under the buttons.
            return ["REMOTE: not connected",
                    (error[:40] if error else "press N to connect")]
        lines = []
        if self._resetting:
            lines.append("REMOTE: rebuilding from a fresh state...")
        if self._unstable:
            lines += ["SIMULATION UNSTABLE on the remote node -- these parameters "
                      "destroyed it.", str(self._unstable),
                      "Dial the sliders back, then press R to rebuild it there."]
        lines += list(self.analysis.hud_lines() or [])
        fps, mbs = self.link.rates()
        rtt = f"{self.link.rtt_ms:.0f} ms" if self.link.rtt_ms else "-"
        # One short line: the HUD stack is already three observables tall here, and
        # it grows upward from the bottom-left corner where the debug breakdown also
        # lives.
        lines.append(f"link {fps:.0f} f/s, {mbs:.2f} MB/s, {rtt}, "
                     f"{self.link.dropped} drop")
        return lines

    def get_puller_state(self):
        return self.mode.puller_state()

    def get_puller_energy(self):
        return None, None

    def get_interaction_force(self):
        return self.mode.interaction_force()

    def get_torque_signals(self):
        return self.mode.torque_signals()

    def get_control_grid(self):
        return self.mode.control_grid()

    def puller_attached(self):
        return False

    def toggle_puller_attached(self):
        return False

    def get_camera_params(self):
        return self.scenario.camera(self.box)

    def get_box_bounds_3d(self):
        return self.box.bounds_3d()

    def get_box_periodic(self):
        return tuple(bool(p) for p in self.box.periodic)

    def get_scene_fit_points(self):
        return self.scenario.fit_points(self.scenario_params, self.box)

    def get_box_size(self):
        return self.box.lengths[0], self.box.lengths[1]

    def get_bond_pairs(self):
        return None

    def get_hbond_pairs(self):
        return None

    def close(self):
        self.detach()

    # ---- construction helpers -----------------------------------------------

    def _make_rdf(self):
        """The same choice PlaygroundSystem makes, minus the scenario override --
        which takes a live LAMMPS instance to build, and there isn't one here."""
        lengths = self.box.lengths
        if all(self.box.periodic):
            return RadialRDF3D(min(0.5 * min(lengths), 6.0), lengths[0])
        if self.box.periodic[0] and self.box.periodic[1]:
            return InPlaneRDF(min(0.5 * min(lengths[0], lengths[1]), 6.0),
                              box=(lengths[0], lengths[1]))
        return InPlaneRDF(3.0, nbins=48, box=None, sample_every=1)

"""PlaygroundSystem: the single MDSystem implementation the app talks to.

It composes a ForceField, a Scenario, a Mode and an Analysis into the interface
the existing control loop and renderer already speak, so nothing downstream had
to be rewritten -- the app was already push-based and spec-driven with no
per-system branching, which is the part of the old design that was right.

Everything system-specific now lives in the four composed objects. What is left
here is the LAMMPS deck assembly (in one place instead of three near-identical
copies), the id-to-local-index bookkeeping LAMMPS forces on us, and the plumbing
of readouts into the shapes the renderer wants.
"""
import random

import numpy as np
from lammps import lammps

from ..mdsystem import MDSystem3D, SliderSpec, SystemSpec
from . import forcefield as ff_registry
from .modes import GameMode, SimMode, select_controlled
from .observables import Analysis
from .rdf import InPlaneRDF, RadialRDF3D
from .scenario import Scenario
from .smoothing import TrajectorySmoother
from .state import Box, FrameState, normalize_rows


def make_mode(playground, mode_name=None):
    """Modes are chosen at run time, not baked into the system, so `--mode game`
    and `--mode sim` both work on any playground."""
    name = (mode_name or playground.mode or "game").lower()
    if name == "sim":
        return SimMode()
    if name == "game":
        return GameMode(playground.effective_control())
    raise ValueError(f"unknown mode {name!r} (expected 'game' or 'sim')")


def make_spec(playground, mode_name=None, preset=None):
    """The SystemSpec for a playground, WITHOUT building a LAMMPS instance.

    Callable from the registry so `--list-playgrounds` and the app's system picker
    can show every playground without constructing any of them.
    """
    force_field = ff_registry.get(playground.force_field)(**playground.force_field_options)
    params = force_field.new_params(playground.resolved_params(preset))
    scenario = playground.scenario
    mode_name = (mode_name or playground.mode or "game").lower()
    control = playground.effective_control()
    t_min, t_max = playground.temperature

    return SystemSpec(
        key=playground.key or playground.name,
        name=playground.name,
        description=playground.description,
        element_label=(playground.element_label
                       or ("membrane bead (director)" if force_field.has_directors
                           else "particle")),
        lattice_spacing=playground.lattice_spacing,
        timestep=scenario.timestep,
        temperature=SliderSpec("Temperature", t_min, t_max,
                               playground.temperature_default, fmt="{:.3f}",
                               unit=" T*" if playground.reduced_units else " K"),
        damping=SliderSpec(
            "Puller damping" if mode_name == "game" else "Puller damping (unused)",
            control.damping_range[0], control.damping_range[1],
            control.damping_default if mode_name == "game" else 0.0,
            fmt="{:.2f}", advanced=True),
        melt_temp=playground.melt_temp,
        force_feedback=playground.force_feedback,
        max_input_force=control.max_input_force if mode_name == "game" else 0.0,
        puller_speed_cap=(playground.puller_speed_cap
                          or 0.06 * playground.lattice_spacing / scenario.timestep),
        crystal_color=playground.crystal_color,
        species_colors=playground.species_colors,
        species_labels=playground.species_labels,
        species_radii_A=playground.species_radii,
        atom_radius_A=playground.particle_radius,
        sim_time_per_frame=scenario.sim_time_per_frame,
        bond_overlay=playground.bond_overlay,
        render_3d=playground.render_3d,
        render_style=playground.render_style,
        camera_orbit=playground.camera_orbit,
        reduced_units=playground.reduced_units,
        director_arrows=scenario.director_arrows,
        wrap_fade_fraction=scenario.wrap_fade_fraction,
        playback_controls=(mode_name == "sim"),
        # Every live force-field parameter becomes a slider, from ONE
        # declaration. The old code wrote each of these out twice: as a
        # SliderSpec here and again as a string key in a hand-maintained
        # set_extra_param dispatch dict, in each of three system modules.
        extra_sliders=(params.slider_specs(playground.param_ranges)
                       + smoothing_slider_specs(playground, scenario)),
    )


# The view key `set_extra_param` recognises, alongside the force field's own live
# parameters. Not a ParamSet entry, because it never reaches LAMMPS.
SMOOTHING_KEY = "view_smoothing"
# The slider's top end, as a number of frames' worth of memory. 20 frames is a
# third of a second at 60 Hz -- enough to flatten the rattle completely while a
# real rearrangement still reads as prompt.
SMOOTHING_MAX_FRAMES = 20


def smoothing_slider_specs(playground, scenario):
    """The trajectory-smoothing slider, or nothing if the playground declines it.

    The span is expressed in the scenario's own simulated time, scaled to its
    per-frame slice, so "full right" means the same ~20 frames of averaging on
    every playground rather than an arbitrary constant that would be gentle on one
    and glacial on the next.
    """
    if not playground.trajectory_smoothing:
        return ()
    per_frame = scenario.sim_time_per_frame
    return (SliderSpec("Smoothing", 0.0, SMOOTHING_MAX_FRAMES * per_frame, 0.0,
                       fmt="{:.2f}", unit=" tau", key=SMOOTHING_KEY,
                       advanced=True),)


class PlaygroundSystem(MDSystem3D):
    """A playground, running."""

    def __init__(self, playground, mode_name=None, preset=None):
        self.playground = playground
        self.preset = preset
        self.force_field = ff_registry.get(playground.force_field)(
            **playground.force_field_options)
        self.scenario = playground.scenario
        self.params = self.force_field.new_params(playground.resolved_params(preset))
        self.scenario_params = self.scenario.new_params()
        self.mode_name = (mode_name or playground.mode or "game").lower()
        self.mode = make_mode(playground, self.mode_name)
        self.mode.attach(self)
        self.spec = make_spec(playground, self.mode_name, preset)

        self.has_directors = self.force_field.has_directors
        self._target_temp = self.spec.temperature.default
        self._sim_time = 0.0
        self.analysis = Analysis(self.force_field, playground.observables)
        # Frame-state cache: several readouts want the same id-ordered gather over
        # every particle, and rebuilding it per readout is what made the old code
        # walk the whole system several times a frame.
        self._frame = 0
        self._state = None
        self._state_frame = -1
        self._order_cache = None
        self._order_frame = -1
        # Visual-only trajectory smoothing (see smoothing.py). Its own frame cache,
        # deliberately separate from the analysis one: the filtered coordinates must
        # reach the renderer and NOTHING else, so the two states cannot share a
        # slot that an observable might pick up by accident.
        self._smoothing_tau = 0.0
        self._smoother = TrajectorySmoother()
        self._render_cache = None
        self._render_frame = -1
        # Same once-per-frame caching for the smoothed bead energies (see
        # _smooth_energies), kept separate because the colouring is read only while
        # it is switched on.
        self._energy_cache = None
        self._energy_render_frame = -1
        self._last_step_dt = 0.0
        self.analysis_seconds = 0.0
        # Latched message once the simulation has been driven unstable, else None,
        # plus the last finite thermo reading to fall back on.
        self._unstable = None
        self._last_thermo = None
        # Whether this scenario overrides housekeeping at all -- checked once so
        # the per-frame path can skip the gather entirely.
        self._has_housekeeping = (
            type(self.scenario).housekeeping is not Scenario.housekeeping
        )
        self._has_director_housekeeping = (
            type(self.scenario).director_housekeeping
            is not Scenario.director_housekeeping
        )

        # Whether the next `run` has to do a full setup (neighbour rebuild + force
        # evaluation) because something structural changed since the last one.
        # See command() and step().
        self._setup_dirty = True
        self.lmp = None
        self._setup(self._new_seed())

    def _new_seed(self):
        if self.playground.seed is not None:
            return int(self.playground.seed)
        return random.randint(1, 900_000_000)

    # ---- construction -------------------------------------------------------

    def _setup(self, seed):
        """Build a fresh LAMMPS instance from the composed pieces.

        The command order reproduces what the three hand-written MesoMem systems
        did, including the two distinct relaxation patterns: the sheet's barostat
        settle runs BEFORE any thermostat or control fix exists (so it relaxes a
        genuinely free sheet), while the patch's short settle runs with everything
        already in place.
        """
        self._seed = seed
        rng = np.random.default_rng(seed)
        ff, scenario, params = self.force_field, self.scenario, self.params
        sparams = self.scenario_params

        build = scenario.build(sparams, rng)
        self.box = build.box
        self.bonds = list(build.bonds)
        self.brightness = build.brightness

        self.lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
        lmp = self.lmp
        c = lmp.command
        ff.ensure_available(lmp)

        c(f"units {ff.units}")
        c(f"dimension {ff.dimension}")
        c(f"atom_style {ff.atom_style}")
        c(self.box.boundary_command())
        c("atom_modify map array")
        c(self.box.region_command("box"))
        c(f"create_box {ff.n_types} box")

        # Particles either come from the scenario's positions array in ONE call,
        # or are placed by LAMMPS itself (random fill with overlap rejection).
        creation = scenario.atom_creation_commands(sparams, seed)
        if creation:
            for cmd in creation:
                c(cmd)
        else:
            n = len(build.positions)
            lmp.create_atoms(n, list(range(1, n + 1)),
                             [int(t) for t in build.types],
                             build.positions.ravel().tolist())

        for cmd in ff.setup_commands(params):
            c(cmd)
        for cmd in scenario.create_commands(sparams, build, seed):
            c(cmd)
        for cmd in ff.pair_commands(params):
            c(cmd)
        c(f"neighbor {scenario.neighbor_skin} bin")
        c("neigh_modify every 1 delay 0 check yes")

        self.natoms = lmp.get_natoms()
        self.all_ids = np.arange(1, self.natoms + 1)
        self._pick_controlled(build)

        # Groups: the mode's first (it defines `controlled`), then the scenario's,
        # which may partition the rest (a frozen floor, a mobile crystal).
        for cmd in self.mode.group_commands():
            c(cmd)
        for cmd in scenario.group_commands(sparams, self.controlled_id):
            c(cmd)

        # Integrators: a scenario that partitions its particles installs its own;
        # otherwise the force field's single global one covers everything.
        scenario_integrators = scenario.integrator_commands(sparams)
        if scenario_integrators:
            for cmd in scenario_integrators:
                c(cmd)
        else:
            global_integrator = ff.integrator_command()
            if global_integrator:
                c(global_integrator)

        for cmd in scenario.wall_commands(self.box):
            c(cmd)
        for cmd in scenario.extra_setup_commands(sparams):
            c(cmd)
        c(f"timestep {scenario.timestep}")
        c("thermo 100000")

        settle = scenario.pre_control_settle(sparams, seed)
        if settle:
            for cmd in settle:
                c(cmd)
            for cmd in scenario.settle_cleanup_commands():
                c(cmd)

        # The thermostat's own temperature compute (if it has one) must exist
        # before the fix that binds to it.
        self._bath_group = scenario.thermostat_group() or self.mode.thermostat_group()
        self._thermostat = scenario.get_thermostat()
        tc = self._thermostat.temperature_compute(self._bath_group)
        if tc is not None:
            self._temp_compute, tc_cmd = tc
            c(tc_cmd)
        else:
            self._temp_compute = "bath_temp"
            c(f"compute bath_temp {self._bath_group} temp")
        for cmd in self._thermostat.initial_commands(
                self._bath_group, self._target_temp,
                scenario.thermostat_damp(sparams), seed):
            c(cmd)

        for cmd in self.mode.control_commands(params):
            c(cmd)

        c("compute ke_atom all ke/atom")
        c("compute pe_atom all pe/atom")

        post = scenario.post_control_settle(sparams)
        for cmd in (post or ["run 0"]):
            c(cmd)

        # The cell may have been rescaled by a barostat during settling, so read
        # it back rather than trusting the requested geometry.
        self._refresh_box_from_lammps()
        if hasattr(self.mode, "on_built"):
            self.mode.on_built()
        self._rdf = self._make_rdf()
        self._setup_dirty = True   # the first chunk after a build sets up in full
        # A rebuild is a new set of particles; a filter still holding the old ones
        # would drag the first frames of the new scene toward the previous one.
        self._smoother.reset()
        self._render_cache = None
        self._render_frame = -1
        self._energy_cache = None
        self._energy_render_frame = -1
        self._sim_time = 0.0
        # Populate the panels for the paused first frame (sim mode shows its fresh
        # state before Play is pressed, and would otherwise show empty bars).
        self._refresh_analysis(force=True)

    def _pick_controlled(self, build):
        """Choose the controlled particle from the INITIAL configuration, and
        remember it by atom id -- LAMMPS reorders its local arrays between steps
        (spatial sorting), so an index would not survive."""
        self.controlled_index = None
        self.controlled_id = None
        if not self.mode.needs_control_particle:
            return
        positions = build.positions
        if not len(positions):
            # LAMMPS placed the particles, so read them back to choose.
            positions = self._read_positions_by_id()
        idx = select_controlled(positions, self.box,
                               self.playground.effective_control().atom)
        if idx is None:
            return
        self.controlled_index = int(idx)
        self.controlled_id = int(idx) + 1

    def _measured_temp(self):
        return self.lmp.extract_compute(self._temp_compute, 0, 0)

    def _refresh_box_from_lammps(self):
        lo = tuple(self.lmp.extract_global(f"box{a}lo") for a in "xyz")
        hi = tuple(self.lmp.extract_global(f"box{a}hi") for a in "xyz")
        self.box = Box(lo, hi, self.box.periodic)

    def _spacing(self):
        """The scenario's nearest-neighbour spacing if it declares one, else 1.0
        -- used only to scale RDF ranges."""
        return self.scenario_params["a"] if self.scenario_params.has("a") else 1.0

    def _make_rdf(self):
        """A bulk periodic cell gets the 3D radial g(r); anything else (a patch,
        or a monolayer periodic only in-plane) gets the in-plane one, which is
        what actually reads out lateral order for a sheet. A scenario that knows
        better -- a crystal slab with vacuum above it, where an in-plane
        bounding-box density is meaningless -- overrides the choice.

        r_max spans a handful of lattice shells but never more than half the box
        (the minimum-image limit), so the hexagonal peaks -- and their collapse on
        melting -- are all visible.
        """
        override = self.scenario.make_rdf(self.scenario_params, self.lmp, self.box)
        if override is not None:
            return override
        lengths = self.box.lengths
        if all(self.box.periodic):
            return RadialRDF3D(min(0.5 * min(lengths), 6.0), lengths[0])
        if self.box.periodic[0] and self.box.periodic[1]:
            r_max = min(0.5 * min(lengths[0], lengths[1]), 6.0 * self._spacing())
            return InPlaneRDF(r_max, box=(lengths[0], lengths[1]))
        # A small non-periodic patch resolves only the first couple of neighbour
        # shells rather than a bulk phase, but it populates the RDF panel instead
        # of leaving it stuck on "warming up".
        return InPlaneRDF(3.0 * self._spacing(), nbins=48, box=None, sample_every=1)

    def reset(self):
        """Rebuild from a fresh random state, keeping the current live parameters.
        Used by sim mode's Reset."""
        if self.lmp is not None:
            self.lmp.close()
        self.analysis = Analysis(self.force_field, self.playground.observables)
        self._unstable = None      # a rebuild is the way out of an unstable state
        self._setup(self._new_seed())

    # ---- mid-flight commands ------------------------------------------------

    def command(self, cmd):
        """Issue a LAMMPS command between chunks, and remember that the next `run`
        must therefore do a full setup.

        EVERY mid-flight command goes through here rather than straight to
        self.lmp, because `run ... pre no` (see step()) is only valid while the
        system has not changed structurally, and the things this app changes
        between chunks -- redefining the thermostat fix, re-issuing pair
        coefficients or a whole pair style, a scenario's per-frame command --
        are exactly the changes that invalidate it.

        The flag is set for ANY command, not only the ones that genuinely matter
        (a bare `velocity` change does not need a re-setup), because a
        conservative flag costs one full setup on the frames where a control
        actually moved, while getting the distinction wrong costs silently wrong
        forces. The systems where the saving matters -- the big MesoMem ones --
        issue no per-frame commands at all, so they take `pre no` every frame.

        Direct writes into the extracted x/v/mu/omega arrays (housekeeping, the
        controlled particle's constraints) deliberately do NOT set the flag: they
        change no fix, pair style, box or atom count. What they do cost is that
        the first step of the next chunk sees forces computed before the write --
        one step of staleness on a clamp that moves an atom by a fraction of a
        lattice spacing, which is far below the thermostat noise it swims in.
        """
        self._setup_dirty = True
        self.lmp.command(cmd)

    # ---- id <-> local-index bookkeeping -------------------------------------

    def _order(self):
        """Local indices in stable atom-id order.

        Every readout needs this, because LAMMPS is free to reorder its local
        arrays between steps (periodic spatial sorting) and an array index cannot
        survive that -- the renderer's per-particle brightness and the motion
        trails both key off identity.

        Ids run 1..N, so sorting by id IS the id-order gather, and argsort does it
        in numpy instead of building an N-entry Python dict per call (which the
        old code did, several times per frame). Cached per frame.
        """
        if self._order_frame == self._frame and self._order_cache is not None:
            return self._order_cache
        n = self.lmp.get_natoms()
        ids = self.lmp.numpy.extract_atom("id")[:n]
        self.natoms = n
        self._order_cache = np.argsort(ids, kind="stable")
        self._order_frame = self._frame
        return self._order_cache

    def controlled_local(self):
        """Local array index of the controlled particle, via its stable id."""
        if self.controlled_id is None:
            return None
        order = self._order()
        k = self.controlled_id - 1
        return int(order[k]) if k < len(order) else None

    def controlled_position(self):
        ic = self.controlled_local()
        if ic is None:
            return None
        return np.array(self.lmp.numpy.extract_atom("x")[ic][:3], dtype=float)

    def _read_positions_by_id(self):
        order = self._order()
        x = np.array(self.lmp.numpy.extract_atom("x")[:self.natoms], dtype=float)
        return x[order]

    def frame_state(self):
        """This frame's particle state as plain arrays, in stable id order."""
        order = self._order()
        n = self.natoms
        x = np.array(self.lmp.numpy.extract_atom("x")[:n], dtype=float)
        dirs = None
        if self.has_directors:
            mu = np.array(self.lmp.numpy.extract_atom("mu")[:n], dtype=float)[:, :3]
            dirs = normalize_rows(mu[order])
        # Types are only gathered for a multi-species force field, where they
        # decide how each particle is drawn; with one type they are a constant
        # array nobody reads.
        types = None
        if self.force_field.n_types > 1:
            types = np.array(self.lmp.numpy.extract_atom("type")[:n], dtype=int)[order]
        return FrameState(positions=x[order], directors=dirs, types=types,
                          ids=self.all_ids[:len(order)], box=self.box)

    # ---- controls -----------------------------------------------------------

    def set_input_force(self, fx, fy):
        self.mode.set_input_force(fx, fy)

    def set_puller_damping(self, gamma):
        self.mode.set_damping(gamma)

    def steer_orientation(self, rate, dt):
        self.mode.steer_orientation(rate, dt)

    def set_target_temp(self, T):
        t_min, t_max = self.playground.temperature
        T = max(t_min, min(t_max, T))
        if T == self._target_temp:
            return
        self._target_temp = T
        # The thermostat may need to act before the new setpoint -- seeding a
        # Maxwell-Boltzmann distribution when heating a lattice that is at rest,
        # or zeroing the net momentum on a sharp quench. See thermostat.py.
        seed = random.randint(1, 900_000_000)
        for cmd in self._thermostat.pre_change_commands(
                self._bath_group, self._measured_temp(), T, seed):
            self.command(cmd)
        for cmd in self._thermostat.set_commands(
                self._bath_group, T,
                self.scenario.thermostat_damp(self.scenario_params), seed):
            self.command(cmd)

    def set_extra_param(self, key, value):
        """Live force-field dials. Generic: the Param's declared tier decides
        whether the coefficients or the whole pair style get re-issued, and its
        declared clamp is applied when the value is read. No per-system dispatch
        table, and no `if key == "rc"` special case."""
        # The one slider that is not a force-field parameter: it changes how the
        # frame is DRAWN and issues no command, so it is handled before the
        # ParamSet lookup (which would reject it) and never reaches self.command
        # -- a view setting must not cost the next chunk its `pre no`.
        if key == SMOOTHING_KEY:
            self._smoothing_tau = max(0.0, float(value))
            return
        if not self.params.has(key):
            return
        if not self.params.set(key, value):
            return
        for cmd in self.force_field.live_commands(self.params, key):
            self.command(cmd)

    def step(self, n):
        """Advance the simulation, surviving a user-induced blow-up.

        Exploring a parameter space means being able to reach settings that
        destroy the simulation -- switch off the cohesion, or push a length scale
        past what the fixed lattice can accommodate, and LAMMPS raises "simulation
        unstable" on the next run. Letting that propagate kills the whole app and
        loses the session, which is a far worse outcome than a visibly broken
        simulation the user can Reset or dial back out of. So it is caught, latched,
        and reported in the HUD; stepping stops until a rebuild.

        A chunk is a fresh `run`, and by default LAMMPS treats each one as a new
        simulation: it rebuilds the neighbour lists and evaluates the forces once
        before taking the first step. Nothing in a 20-step chunk warrants that --
        it repeats work the previous chunk's last step already did -- so it is
        skipped with `pre no` whenever nothing structural has changed since,
        which command() tracks. `post no` drops the end-of-run timing summary
        that -screen none throws away anyway.

        Measured on the MesoMem systems (900-1500 beads, 20 steps/chunk), this
        takes 1-2 ms off an 18 ms chunk, i.e. 5-14%; the end-of-run thermo
        evaluation still happens, so get_thermo and the per-atom computes the app
        reads every frame stay exact (verified against a forced `run 0`).
        """
        from time import perf_counter
        if self._unstable:
            return
        dt = n * self.scenario.timestep
        setup = "" if self._setup_dirty else " pre no post no"
        try:
            self.lmp.command(f"run {n}{setup}")
        except Exception as exc:
            self._unstable = str(exc).strip().splitlines()[0]
            return
        # Only now: a run that threw leaves the instance in an unknown state, and
        # the next one (after a rebuild) should set up in full.
        self._setup_dirty = False
        # Bump the frame counter immediately: `run` may have reordered LAMMPS'
        # local arrays, so every id-order and frame-state cache is now stale.
        # Everything below this line sees consistent, freshly-gathered data.
        self._frame += 1
        # The sim time this frame covered, which is the interval the visual
        # smoothing filter averages over (see _render_state). Taken from the actual
        # chunk rather than the scenario's nominal per-frame slice, so a capped or
        # short chunk filters by what it really advanced.
        self._last_step_dt = dt
        self._apply_housekeeping(dt)
        # Constrain AFTER the housekeeping kick, so the readouts (and the
        # renderer) see the post-constraint positions -- the order the original
        # systems used.
        self.mode.after_step(dt)
        # Per-frame commands that depend on measured state and so cannot be a fix
        # (e.g. a drag proportional to this frame's centre-of-mass velocity).
        for cmd in self.scenario.frame_commands(self.scenario_params, self.lmp):
            self.command(cmd)
        self._sim_time += dt
        t0 = perf_counter()
        self.analysis.update(self._analysis_state(), self.params)
        # Reported to the app's --debug breakdown as its own section, so the
        # Python-side analysis cost is visible rather than hidden inside "sim".
        self.analysis_seconds = perf_counter() - t0

    def _refresh_analysis(self, force=False):
        """Rebuild the frame state and tick the analysis. Used once at build time
        so the panels are populated on the paused first frame."""
        self._frame += 1
        self.analysis.update(self._analysis_state(), self.params, force=force)

    def _apply_housekeeping(self, dt):
        """Apply the scenario's soft corrections as momentum kicks (delta v = F*dt
        with m = 1). Pure numpy in the scenario; only the array write happens here.

        Skipped entirely -- including the id-order gather -- for scenarios that
        declare no housekeeping, which is most of them.
        """
        if not (self._has_housekeeping or self._has_director_housekeeping):
            return
        order = self._order()
        if not len(order):
            return
        x = np.array(self.lmp.numpy.extract_atom("x")[:self.natoms], dtype=float)
        # housekeeping() works in id order, so the controlled particle's index has
        # to be expressed there too: id k lives at id-order position k - 1.
        controlled = None
        if self.controlled_id is not None and self.controlled_id - 1 < len(order):
            controlled = self.controlled_id - 1
        if self._has_housekeeping:
            f = self.scenario.housekeeping(x[order], self.scenario_params,
                                           controlled, self.box)
            if f is not None:
                v = self.lmp.numpy.extract_atom("v")
                v[order] += f * dt
        # The second channel: corrections that turn the particles' ORIENTATIONS
        # rather than move them. A periodic cell cannot be rigidly rotated -- a
        # membrane spanning it would have to tear -- so a scenario that wants to
        # steer which way its structures face does it through the directors, and
        # lets the force field's own tilt coupling carry the geometry round.
        if self._has_director_housekeeping and self.has_directors:
            mu = np.array(self.lmp.numpy.extract_atom("mu")[:self.natoms],
                          dtype=float)[:, :3]
            w = self.scenario.director_housekeeping(
                x[order], normalize_rows(mu[order]), self.scenario_params,
                controlled, self.box)
            if w is not None:
                omega = self.lmp.numpy.extract_atom("omega")
                omega[order] += w * dt

    # ---- readouts -----------------------------------------------------------

    def get_puller_state(self):
        return self.mode.puller_state()

    def get_puller_energy(self):
        ic = self.controlled_local()
        if ic is None:
            return None, None
        n = self.natoms
        ke = self.lmp.numpy.extract_compute("ke_atom", 1, 1)[:n]
        pe = self.lmp.numpy.extract_compute("pe_atom", 1, 1)[:n]
        return float(ke[ic]), float(pe[ic])

    def get_interaction_force(self):
        return self.mode.interaction_force()

    def toggle_puller_attached(self):
        """Grab / release the controlled particle. Returns the new state."""
        return self.mode.toggle_attached()

    def puller_attached(self):
        return self.mode.attached

    def get_torque_signals(self):
        return self.mode.torque_signals()

    def get_thermo_state(self):
        # Hold the last finite reading once unstable, so the plots and the
        # heat-fraction haptics do not ingest NaN.
        if self._unstable and self._last_thermo is not None:
            return self._last_thermo
        state = (self._measured_temp(), self.lmp.get_thermo("press"),
                 self.lmp.get_thermo("ke"), self.lmp.get_thermo("pe"),
                 self.lmp.get_thermo("etotal"))
        if all(np.isfinite(v) for v in state):
            self._last_thermo = state
        elif self._last_thermo is not None:
            return self._last_thermo
        return state

    def get_sim_time(self):
        return self._sim_time

    def get_rdf(self):
        state = self._analysis_state()
        if all(self.box.periodic):
            self._rdf.add(state.positions)
        else:
            self._rdf.add(state.positions[:, :2])
        return self._rdf.get()

    def _analysis_state(self):
        """The frame state, rebuilt at most once per frame.

        Once the simulation has gone unstable the particle coordinates are NaN, so
        the last good frame is held instead -- the scene freezes on something
        renderable rather than feeding NaN into the camera and the pair list.
        """
        if self._unstable and self._state is not None:
            return self._state
        if self._state_frame != self._frame:
            self._state = self.frame_state()
            self._state_frame = self._frame
        return self._state

    def _render_state(self):
        """The frame state AS DRAWN: the physics state, optionally temporally
        smoothed (see smoothing.py). Rebuilt at most once per frame, which is also
        what advances the filter exactly one step per frame however many readouts
        ask for it.

        Everything that goes on screen comes through here, so the beads, their
        directors, the bond sticks between them and their wrapped ghosts all agree
        on where a particle is. Everything that MEASURES -- the observables, the
        RDF, the energy panels, the puller's own position and the haptics -- goes
        through _analysis_state() instead and never sees the filtered coordinates.
        """
        state = self._analysis_state()
        if self._smoothing_tau <= 0.0:
            if self._smoother.active:
                self._smoother.reset()
            return state
        if self._render_frame != self._frame:
            # The controlled particle is written straight through: the user is
            # moving it deliberately, so its motion is signal, not the jitter this
            # is here to hide -- and it has to keep agreeing with the puller marker
            # and force arrows, which are drawn from the unsmoothed physics.
            self._render_cache = self._smoother.apply(
                state, self._smoothing_tau, self._last_step_dt,
                keep_exact=(None if self.controlled_id is None
                            else self.controlled_id - 1))
            self._render_frame = self._frame
        return self._render_cache

    def get_potential_terms(self):
        """The controlled particle's share of the additive energy.

        The analysis works in id order, where id k sits at position k - 1, so no
        index translation is needed.
        """
        if self.controlled_id is None:
            return None
        scale = self.force_field.energy_scale_per_particle * 2.0
        return self.analysis.energy_panel(
            "Pulled bead energy -- additive (reduced units)", scale,
            index=self.controlled_id - 1)

    def get_total_potential_terms(self):
        """The whole system's additive energy. Comes from the SAME evaluation as
        get_potential_terms -- the energy expression runs once per analysis frame,
        not once per panel."""
        n = max(1, len(self.all_ids))
        scale = self.force_field.energy_scale_per_particle * n
        return self.analysis.energy_panel(
            "Whole-system energy -- additive (reduced units)", scale)

    def get_hud_lines(self):
        if self._unstable:
            return [
                "SIMULATION UNSTABLE -- these parameters destroyed it.",
                self._unstable,
                "Dial the sliders back, then press R (or restart) to rebuild.",
            ]
        lines = list(self.analysis.hud_lines() or [])
        # Whether the input device is holding the particle, always shown for a
        # game-mode system: it is a state the user can be in without having meant
        # to be, and the line is also where they find out the key exists.
        if self.mode.needs_control_particle and self.controlled_id is not None:
            lines.append("puller: STEERED (B / trigger releases)" if self.mode.attached
                         else "puller: RELEASED -- free in the simulation (B / trigger grabs)")
        return lines or None

    def get_all_positions(self):
        """2D shadow required by the interface; the 3D renderer uses
        get_positions_3d.

        `species` is the LAMMPS atom type, zero-based, for a force field with
        more than one -- it indexes the playground's species_colors/labels/radii,
        so an ionic lattice draws its cations and anions at their own sizes and
        colours. A single-type force field reports None and every particle is
        drawn the same."""
        state = self._render_state()
        species = None
        if self.force_field.n_types > 1 and state.types is not None:
            species = (np.asarray(state.types) - 1).astype(int)
        return (state.ids, state.positions[:, :2], self._is_controlled_mask(),
                species)

    def _is_controlled_mask(self):
        mask = np.zeros(len(self.all_ids), dtype=bool)
        if self.controlled_id is not None:
            mask[self.controlled_id - 1] = True
        return mask

    def get_bead_brightness(self):
        return self.brightness

    def get_bead_energies(self):
        """Per-particle potential energy, in id order -- what the energy colouring
        paints.

        THE FACTOR OF 2 IS THE POINT. LAMMPS' `pe/atom` gives each particle HALF
        of every pair energy it takes part in, so that summing over particles
        returns the total potential energy without double counting. That is the
        right convention for a total, and the wrong number to colour by: it is
        half of what it costs to pull the particle out, and half of what the
        additive-energy panel reports for the controlled one. Colouring by it put
        the same bead at -2.8 on the scale while the panel above said -5.8, which
        is exactly as confusing as it sounds.

        So this returns the WHOLE energy of the bonds touching each particle,
        which for a pairwise-additive style is twice the per-atom share. A force
        field that offers no pairwise decomposition (EAM's embedding term is
        many-body) has no such identity, and reports the share as LAMMPS gives it.
        """
        order = self._order()
        pe = np.array(self.lmp.numpy.extract_compute("pe_atom", 1, 1)[:self.natoms],
                      dtype=float)[order]
        energies = 2.0 * pe if self.force_field.energy_terms_labels else pe
        return self._smooth_energies(energies)

    def _smooth_energies(self, energies):
        """Put the colouring through the same low-pass as the drawn positions.

        It is a DRAWN quantity, so it belongs on the drawn side of the line this
        class draws everywhere else (see _render_state): the energy panels, the
        observables and everything that measures keep reading the raw per-atom
        compute. Without it, turning the smoothing up holds the beads still and
        leaves them twinkling -- each one's energy is a sum over neighbours that are
        still rattling, so the colour swings frame to frame far more than the
        configuration it is painting.

        Filtered once per frame however many readouts ask for it, like
        _render_state, so the filter advances with the simulation rather than with
        the number of callers.
        """
        if self._smoothing_tau <= 0.0:
            return energies
        if self._energy_render_frame != self._frame:
            self._energy_cache = self._smoother.smooth_scalar(
                "bead_energy", energies, self._smoothing_tau, self._last_step_dt)
            self._energy_render_frame = self._frame
        return self._energy_cache

    # ---- 3D rendering data --------------------------------------------------

    def get_positions_3d(self):
        state = self._render_state()
        return state.ids, state.positions, self._is_controlled_mask()

    def get_dipoles_3d(self):
        state = self._render_state()
        if state.directors is None:
            return np.zeros((len(state.positions), 3))
        return state.directors

    def get_bonds_3d(self):
        return list(self.bonds)

    def get_camera_params(self):
        return self.scenario.camera(self.box)

    def get_control_grid(self):
        return self.mode.control_grid()

    def get_box_bounds_3d(self):
        return self.box.bounds_3d()

    def get_box_periodic(self):
        """(x, y, z) periodicity of the cell -- which axes the renderer may
        legitimately repeat when drawing periodic images."""
        return tuple(bool(p) for p in self.box.periodic)

    def get_scene_fit_points(self):
        """World points the camera zooms to fit, or None to keep its fixed FOV.

        A scenario that names its own points has the final say. The control net
        is only added for one that does not: the net spans the full leash, and on
        a small patch that is several times the size of what you are actually
        looking at, so forcing it into frame is what pushed the camera back until
        the beads were a fifth of the width. A scenario that has stated what
        matters should not have that overridden by the interaction's outer limit.
        """
        pts = self.scenario.fit_points(self.scenario_params, self.box)
        if pts is not None:
            return pts
        grid = self.mode.control_grid()
        if grid is None:
            return None
        origin = np.asarray(grid["origin"], dtype=float)
        u = np.asarray(grid["u_axis"], dtype=float)
        v = np.asarray(grid["v_axis"], dtype=float)
        (u0, u1), (v0, v1) = grid["u_range"], grid["v_range"]
        return np.array([origin + uu * u + vv * v
                         for uu in (u0, u1) for vv in (v0, v1)])

    def get_box_size(self):
        return self.box.lengths[0], self.box.lengths[1]

    def close(self):
        if self.lmp is not None:
            self.lmp.close()
            self.lmp = None

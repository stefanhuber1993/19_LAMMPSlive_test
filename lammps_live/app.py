"""Main control loop: owns the active system, the input source, the
renderer, and the force-feedback shaping that connects them. Switching
systems at runtime (number keys / Tab) tears down the old LAMMPS instance
and rebuilds the UI state (renderer scale, sliders, history, smoothers) for
the new one, in place.
"""
import math
import os
import sys
from time import perf_counter

import pygame

from . import config, units
from .forcefeedback import (
    ExponentialSmoother2D, shape_damper_coefficient, shape_interaction_force,
    shape_stiffness, shape_velocity_damping,
)
from .input import (
    CP_OFFSET_MAX, DAMPER_COEFFICIENT_MAX, JoystickInput, KeyboardInput,
    MouseInput, SPRING_STIFFNESS_MAX,
)
from .systems import get_system_class, list_systems
from .ui import AtomTrails, Renderer, RollingHistory, Slider
from .ui.camera import Camera3D

STEPS_PER_FRAME_CAP = 200  # sanity cap if a system's timestep is set absurdly small


class App:
    def __init__(self, input_mode, initial_system_key, fullscreen=False, debug=False):
        self.input_mode = input_mode
        self.debug = debug
        # Exponential moving averages (ms) of the per-frame timing breakdown, and
        # the header line built from them -- shown only under --debug. The line is
        # built from the PREVIOUS frame (a frame's own render time isn't known
        # until after it's drawn), which is fine for a running average.
        self._prof_ms = {"sim": 0.0, "read": 0.0, "ff": 0.0, "render": 0.0, "other": 0.0}
        self._debug_line = None
        self.systems = list_systems()  # [(key, SystemSpec), ...], stable order for the picker/number keys

        # Not pygame.init() -- that also brings up SDL's joystick subsystem,
        # which grabs the Sidewinder as a native SDL game controller. When
        # JoystickInput then claims the same device exclusively via libusb,
        # the device vanishes out from under SDL mid-session and corrupts
        # pygame's event-translation state (observed as `KeyError: 0` inside
        # pygame.event.get()). We drive the joystick ourselves via raw HID
        # reports, so SDL's joystick subsystem is never needed.

        # macOS: keep the green (zoom) button OUT of a native fullscreen "Space".
        # A Space is animated and, with our OpenGL context, not cleanly
        # reversible -- returning from it either leaves the GL drawable stale (a
        # black window) or, if we re-set_mode mid-transition, traps the window in
        # the Space. Disabling Spaces (before the first video init, when SDL reads
        # the hint) makes the green button a plain, reversible window zoom; F11
        # still gives a real fullscreen we control. Must precede display.init().
        if sys.platform == "darwin":
            os.environ.setdefault("SDL_VIDEO_MAC_FULLSCREEN_SPACES", "0")
        pygame.display.init()
        pygame.font.init()

        self.renderer = Renderer(config.WINDOW_SIZE, fullscreen=fullscreen)
        self.clock = pygame.time.Clock()

        self.source = self._make_source()

        self.ff_smoother = ExponentialSmoother2D(config.FF_SMOOTHING_TAU)
        self.interaction_smoother = ExponentialSmoother2D(config.FF_SMOOTHING_TAU)

        self.temp_slider = None
        self.damping_slider = None
        self.extra_sliders = []
        self.extra_slider_keys = []
        # Whether the collapsible "Advanced" slider group is expanded. Toggled by
        # clicking its header (see _handle_events); pushed to the renderer each
        # frame so draw_panel knows whether to draw the advanced sliders.
        self.show_advanced = False
        self.history = None
        self.atom_trails = None
        self._trail_frame_counter = 0
        self.energy_baseline = None
        self.sim_wall_time = 0.0
        self.steps_per_frame = 1
        self.total_steps = 0
        # Playback state for systems driven by Play/Pause/Reset buttons (the
        # self-assembly system): only stepped while playing. Ignored by the
        # interactive puller systems, which always step. Set per-system in
        # _build_system (playback systems start paused on their fresh state).
        self.sim_playing = False

        self.system_key = None
        self.system = None
        self._build_system(initial_system_key)

    def _make_source(self):
        """Build the input source for the current mode and window geometry. Only
        MouseInput depends on the window geometry, so it's the only one rebuilt on
        a fullscreen toggle; the keyboard is geometry-free and the joystick is
        screen-independent (its hardware handle is kept)."""
        if self.input_mode == "mouse":
            h = self.renderer.window_size[1]
            max_radius_px = min(self.renderer.sim_width, h) * 0.35
            sim_rect = (0, 0, self.renderer.sim_width, h)
            return MouseInput(self.renderer.sim_center_px(), max_radius_px, sim_rect)
        if self.input_mode == "keyboard":
            return KeyboardInput()
        return JoystickInput()

    def _setup_viewport(self):
        """(Re)establish everything tied to the sim viewport size: the box<->
        screen mapping and, for 3D systems, the perspective camera. Called when
        the system changes and after a fullscreen toggle."""
        spec = self.system.spec
        self.renderer.set_box_size(self.system.get_box_size())
        if spec.render_3d:
            cam = self.system.get_camera_params()
            self.camera3d = Camera3D(
                cam["eye"], cam["target"], cam["up"], cam["fov_deg"],
                self.renderer.sim_width, self.renderer.window_size[1],
            )
            # Zoom to fit the scene to the (possibly fullscreen, any-aspect) sim
            # viewport instead of the fixed vertical FOV, so the beads fill the
            # available width/height rather than leaving big side margins.
            fit_pts = self.system.get_scene_fit_points()
            if fit_pts is not None:
                self.camera3d.fit_to_points(fit_pts)
        else:
            self.camera3d = None

    def _toggle_fullscreen(self):
        self.renderer.toggle_fullscreen()
        self._after_resize()

    def _exit_fullscreen(self):
        self.renderer.set_windowed()
        self._after_resize()

    def _after_resize(self):
        """Re-establish everything tied to the window size after the display
        surface changed (fullscreen toggle or an OS window resize)."""
        self._setup_viewport()
        if self.input_mode == "mouse":
            self.source = self._make_source()

    def _build_system(self, key):
        """(Re)build the active system and everything downstream of its
        SystemSpec -- renderer scale, sliders, history, smoothers. Safe to
        call again later to switch systems live."""
        if self.system is not None:
            self.system.close()

        cls = get_system_class(key)
        self.system = cls()
        self.system_key = key
        spec = self.system.spec

        # Box<->screen mapping and (for 3D systems) the perspective camera.
        self._setup_viewport()

        if self.temp_slider is None:
            self.temp_slider = Slider.from_spec((0, 0, 100, 4), spec.temperature)
            self.damping_slider = Slider.from_spec((0, 0, 100, 4), spec.damping)
        else:
            self.temp_slider.reset(spec.temperature)
            self.damping_slider.reset(spec.damping)

        # Extra live-tunable parameters (per-system, variable count -- e.g. the
        # MesoMem k_tilt / k_splay / eta dials), rebuilt from scratch since the
        # count and identities differ between systems. Each remembers the
        # SliderSpec key so set_extra_param knows which parameter it drives.
        self.extra_sliders = [Slider.from_spec((0, 0, 100, 4), ss)
                              for ss in spec.extra_sliders]
        self.extra_slider_keys = [ss.key for ss in spec.extra_sliders]

        if self.history is None:
            self.history = RollingHistory(config.HISTORY_WINDOW_SECONDS, ["temp", "press", "ke", "pe", "etotal"])
        else:
            self.history.reset()
        if self.atom_trails is None:
            self.atom_trails = AtomTrails(config.TRAIL_WINDOW_SECONDS)
        else:
            self.atom_trails.reset()
        self._trail_frame_counter = 0
        self.energy_baseline = None
        self.sim_wall_time = 0.0
        self.total_steps = 0
        # Playback systems start paused, showing their fresh initial state until
        # the user presses Play; puller systems ignore this flag and always step.
        self.sim_playing = False

        self.ff_smoother.reset()
        self.interaction_smoother.reset()

        sim_time_per_frame = spec.sim_time_per_frame or config.SIM_TIME_PER_FRAME
        self.steps_per_frame = max(1, min(STEPS_PER_FRAME_CAP, round(sim_time_per_frame / spec.timestep)))

    def _cycle_system(self, step=1):
        keys = [key for key, _ in self.systems]
        idx = keys.index(self.system_key)
        self._build_system(keys[(idx + step) % len(keys)])

    def _reset_simulation(self):
        """Restart a playback system from a fresh initial state (e.g. re-randomize
        the self-assembly box), keeping the current slider values, and clear the
        derived per-run state (plots, trails, energy baseline, step count). Leaves
        the run paused so the fresh state is visible before Play is pressed."""
        self.system.reset()
        self.history.reset()
        self.atom_trails.reset()
        self._trail_frame_counter = 0
        self.energy_baseline = None
        self.total_steps = 0
        self.sim_playing = False

    def _playback_action(self, name):
        """Apply a Play/Pause/Reset button (or its keyboard shortcut)."""
        if name == "play":
            self.sim_playing = True
        elif name == "pause":
            self.sim_playing = False
        elif name == "reset":
            self._reset_simulation()

    def run(self):
        dt = 1.0 / 60  # seconds; seed value, replaced by the real measured frame time below
        running = True
        try:
            while running:
                running = self._handle_events(dt)
                dt = self._tick(dt)
        finally:
            self.source.close()
            self.system.close()
            pygame.quit()

    def _handle_events(self, dt):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    # Escape leaves fullscreen (ours or a macOS-native space)
                    # first; only quits when already windowed.
                    if self.renderer.is_fullscreen():
                        self._exit_fullscreen()
                    else:
                        return False
                elif event.key == pygame.K_F11:
                    self._toggle_fullscreen()
                elif event.key == pygame.K_TAB:
                    self._cycle_system(1)
                elif event.key == pygame.K_SPACE and self.system.spec.playback_controls:
                    self.sim_playing = not self.sim_playing
                elif event.key == pygame.K_r and self.system.spec.playback_controls:
                    self._reset_simulation()
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    idx = event.key - pygame.K_1
                    if idx < len(self.systems):
                        self._build_system(self.systems[idx][0])
            elif event.type == pygame.VIDEORESIZE:
                # Window dragged to a new size, or the macOS green button sending
                # it into / out of a native fullscreen space -- relayout to fit.
                self.renderer.handle_resize(event.size)
                self._after_resize()
            elif event.type == pygame.MOUSEWHEEL:
                step = self.temp_slider.vmax - self.temp_slider.vmin
                self.temp_slider.nudge(event.y * config.TEMP_WHEEL_STEP_FRACTION * step)
            else:
                # A click on a Play/Pause/Reset button (playback systems) is
                # routed to the playback action and consumes the event, so it
                # never falls through to slider/puller handling below.
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    name = self.renderer.playback_hit(event.pos)
                    if name is not None:
                        self._playback_action(name)
                        continue
                # A click on the "Advanced" header flips the group open/closed.
                # When collapsing, cancel any in-progress drag on a now-hidden
                # slider so it can't stay stuck "dragging" (which would keep the
                # puller input suppressed).
                toggle = self.renderer.advanced_toggle_rect
                if (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                        and toggle is not None and toggle.collidepoint(event.pos)):
                    self.show_advanced = not self.show_advanced
                    if not self.show_advanced:
                        for s in self.extra_sliders:
                            if s.advanced:
                                s.dragging = False
                        if self.damping_slider.advanced:
                            self.damping_slider.dragging = False
                    continue
                self.temp_slider.handle_event(event)
                # Hidden advanced sliders don't receive events (their rects are
                # parked off-screen while collapsed anyway).
                if not (self.damping_slider.advanced and not self.show_advanced):
                    self.damping_slider.handle_event(event)
                for s in self.extra_sliders:
                    if s.advanced and not self.show_advanced:
                        continue
                    s.handle_event(event)

        keys = pygame.key.get_pressed()
        temp_range = self.temp_slider.vmax - self.temp_slider.vmin
        rate = config.TEMP_KEY_RATE_FRACTION * temp_range
        if keys[pygame.K_UP]:
            self.temp_slider.nudge(rate * dt)
        if keys[pygame.K_DOWN]:
            self.temp_slider.nudge(-rate * dt)
        return True

    def _tick(self, dt):
        t_frame_start = perf_counter()
        spec = self.system.spec
        ff_profile = spec.force_feedback

        self.system.set_target_temp(self.temp_slider.value)
        self.system.set_puller_damping(self.damping_slider.value)
        for key, s in zip(self.extra_slider_keys, self.extra_sliders):
            self.system.set_extra_param(key, s.value)

        # While actively dragging a slider (which lives in the right-hand
        # panel, off to the side of the sim box), don't also feed that mouse
        # position to the puller as a deflection -- zero the input force
        # instead of letting a slider drag yank the atom.
        ui_capturing_mouse = (self.temp_slider.dragging or self.damping_slider.dragging
                              or any(s.dragging for s in self.extra_sliders))
        # Device I/O is split into two debug fields, since on the joystick both are
        # blocking HID traffic that belongs in neither sim nor "other": "read" is
        # the stick poll here, "ff" is the force-feedback writes further down.
        read_seconds = 0.0
        ff_seconds = 0.0
        t_in = perf_counter()
        if self.input_mode == "mouse" and ui_capturing_mouse:
            jx, jy = 0.0, 0.0
            yaw = 0.0
        else:
            jx, jy = self.source.poll()
            yaw = self.source.poll_yaw()
        read_seconds += perf_counter() - t_in
        input_fx, input_fy = jx * spec.max_input_force, jy * spec.max_input_force
        # Joystick is a force-feedback loop: the puller is driven mainly by the
        # stick's input force, with the MD interaction force reaching it partly
        # *indirectly* -- it's rendered on the stick (force feedback, below), the
        # user's hand yields, and the resulting deflection changes this input
        # force. Cancel out the fraction (1 - felt) of the measured MD force from
        # what's applied to the atom, leaving `felt` of it as direct contact
        # coupling (fully cancelling it felt too detached). Spread across the
        # puller's atoms via puller_bead_count, since set_input_force applies its
        # force to each. Mouse mode keeps the full direct force-on-atom feel.
        if self.input_mode == "joystick":
            n_beads = max(1, self.system.puller_bead_count())
            md_fx, md_fy = self.system.get_interaction_force()
            cancel = (1.0 - config.JOYSTICK_MD_FORCE_FELT_FRACTION) / n_beads
            self.system.set_input_force(input_fx - cancel * md_fx, input_fy - cancel * md_fy)
        else:
            self.system.set_input_force(input_fx, input_fy)
        # Yaw (joystick twist axis, or Q/E in mouse mode) steers the puller's
        # in-plane orientation -- a no-op for systems whose puller is a lone
        # atom, used by the lipid system to rotate the control lipid's director.
        self.system.steer_orientation(yaw, dt)

        t_sim_start = perf_counter()
        # Playback systems (Play/Pause/Reset) step only while playing; every
        # interactive puller system always steps.
        should_step = self.sim_playing if spec.playback_controls else True
        if should_step:
            self.system.step(self.steps_per_frame)
            self.total_steps += self.steps_per_frame
        sim_seconds = perf_counter() - t_sim_start

        pos, vel = self.system.get_puller_state()
        interaction_force = self.system.get_interaction_force()

        shaped_fx, shaped_fy = shape_interaction_force(*interaction_force, ff_profile)
        vel_damp_fx, vel_damp_fy = (
            shape_velocity_damping(*vel, ff_profile, spec.puller_speed_cap, CP_OFFSET_MAX)
            if vel is not None else (0.0, 0.0)
        )
        combined_fx, combined_fy = shaped_fx + vel_damp_fx, shaped_fy + vel_damp_fy
        smooth_fx, smooth_fy = self.ff_smoother.update(combined_fx, combined_fy, dt)
        # Stiffness uses its own smoothed copy of the RAW physical
        # interaction force -- separate from smooth_fx/fy above, which is
        # the device-unit-shaped position signal and would saturate
        # stiffness_threshold/knee almost instantly if reused.
        smooth_ifx, smooth_ify = self.interaction_smoother.update(
            interaction_force[0], interaction_force[1], dt
        )
        stiffness = shape_stiffness(smooth_ifx, smooth_ify, ff_profile, SPRING_STIFFNESS_MAX)
        t_in = perf_counter()
        self.source.send_force(smooth_fx, smooth_fy, stiffness)
        self.source.set_damper_coefficient(
            shape_damper_coefficient(smooth_ifx, smooth_ify, ff_profile, DAMPER_COEFFICIENT_MAX)
        )
        ff_seconds += perf_counter() - t_in

        temp, press, ke, pe, etotal = self.system.get_thermo_state()
        puller_ke, puller_pe = self.system.get_puller_energy()
        if self.energy_baseline is None:
            self.energy_baseline = (ke, pe, etotal)
        ke0, pe0, etotal0 = self.energy_baseline
        self.history.add(self.sim_wall_time, temp=temp, press=press,
                          ke=ke - ke0, pe=pe - pe0, etotal=etotal - etotal0)
        t_min = spec.temperature.vmin
        t_max = spec.temperature.vmax
        heat_fraction = max(0.0, min(1.0, (temp - t_min) / (t_max - t_min)))
        t_in = perf_counter()
        self.source.update_jitter(heat_fraction)
        ff_seconds += perf_counter() - t_in
        rdf = self.system.get_rdf()

        sim_time_ps = self.system.get_sim_time()
        puller_speed_m_s = units.speed_to_m_per_s(math.hypot(*vel)) if vel is not None else None

        ids, positions, is_puller, species = self.system.get_all_positions()
        bond_pairs = self.system.get_bond_pairs()
        hbond_pairs = self.system.get_hbond_pairs()
        hud_lines = self.system.get_hud_lines()
        self._trail_frame_counter += 1
        if self._trail_frame_counter % config.TRAIL_SAMPLE_EVERY_N_FRAMES == 0:
            self.atom_trails.add(self.sim_wall_time, ids, positions, is_puller)

        scene_3d = None
        if spec.render_3d:
            ids3d, pos3d, is_puller3d = self.system.get_positions_3d()
            scene_3d = {
                "positions3d": pos3d,
                "dipoles3d": self.system.get_dipoles_3d(),
                "is_puller": is_puller3d,
                "bonds": self.system.get_bonds_3d(),
                "camera": self.camera3d,
                "control_grid": self.system.get_control_grid(),
                "potential_terms": self.system.get_potential_terms(),
                "total_potential_terms": self.system.get_total_potential_terms(),
                "torque_signals": self.system.get_torque_signals(),
                "brightness": self.system.get_bead_brightness(),
                "box_bounds": self.system.get_box_bounds_3d(),
            }

        t_render_start = perf_counter()
        self.renderer.show_advanced = self.show_advanced
        self.renderer.draw(
            positions, is_puller, pos,
            (input_fx, input_fy), interaction_force, self.clock.get_fps(),
            spec, self.systems, self.system_key,
            (self.temp_slider, self.damping_slider, *self.extra_sliders),
            (temp, press, ke, pe, etotal), (puller_ke, puller_pe),
            self.history, rdf, heat_fraction=heat_fraction,
            sim_time_ps=sim_time_ps, puller_speed_m_s=puller_speed_m_s,
            atom_trails=self.atom_trails, species=species, bond_pairs=bond_pairs,
            hbond_pairs=hbond_pairs, hud_lines=hud_lines, scene_3d=scene_3d,
            total_steps=self.total_steps, steps_per_frame=self.steps_per_frame,
            debug_line=self._debug_line,
            playback_playing=(self.sim_playing if spec.playback_controls else None),
        )
        if self.debug:
            render_seconds = perf_counter() - t_render_start
            self._update_debug(perf_counter() - t_frame_start, sim_seconds,
                               render_seconds, read_seconds, ff_seconds)

        new_dt = self.clock.tick(60) / 1000.0
        self.sim_wall_time += new_dt
        return new_dt

    def _update_debug(self, work_seconds, sim_seconds, render_seconds,
                      read_seconds, ff_seconds):
        """Fold this frame's timings into the smoothed breakdown and rebuild the
        header line for the next frame. 'work' is everything the app does per
        frame except the fps-cap sleep. The device I/O is split into 'read' (the
        stick poll) and 'ff' (the force-feedback writes), both broken out because
        on the joystick they are blocking HID traffic; 'other' is the remainder
        after sim, render, read and ff -- force shaping and the per-frame readouts
        (positions, RDF, potential terms)."""
        other_seconds = max(0.0, work_seconds - sim_seconds - render_seconds
                            - read_seconds - ff_seconds)
        alpha = 0.1   # EMA weight -- steady enough to read, quick enough to track
        for name, secs in (("sim", sim_seconds), ("read", read_seconds),
                           ("ff", ff_seconds), ("render", render_seconds),
                           ("other", other_seconds)):
            self._prof_ms[name] += alpha * (secs * 1000.0 - self._prof_ms[name])
        total = sum(self._prof_ms.values()) or 1e-9
        def part(name):
            ms = self._prof_ms[name]
            return f"{name} {100.0 * ms / total:2.0f}% ({ms:4.1f}ms)"
        self._debug_line = (
            f"DEBUG  {part('sim')}   {part('read')}   {part('ff')}   "
            f"{part('render')}   {part('other')}   frame {total:4.1f}ms"
        )

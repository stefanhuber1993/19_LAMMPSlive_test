"""Main control loop: owns the active system, the input source, the
renderer, and the force-feedback shaping that connects them. Switching
systems at runtime (number keys / Tab) tears down the old LAMMPS instance
and rebuilds the UI state (renderer scale, sliders, history, smoothers) for
the new one, in place.
"""
import math

import pygame

from . import config, units
from .forcefeedback import (
    ExponentialSmoother2D, shape_damper_coefficient, shape_interaction_force,
    shape_stiffness, shape_velocity_damping,
)
from .input import CP_OFFSET_MAX, DAMPER_COEFFICIENT_MAX, JoystickInput, MouseInput, SPRING_STIFFNESS_MAX
from .systems import get_system_class, list_systems
from .ui import AtomTrails, Renderer, RollingHistory, Slider
from .ui.camera import Camera3D

STEPS_PER_FRAME_CAP = 200  # sanity cap if a system's timestep is set absurdly small


class App:
    def __init__(self, input_mode, initial_system_key):
        self.input_mode = input_mode
        self.systems = list_systems()  # [(key, SystemSpec), ...], stable order for the picker/number keys

        # Not pygame.init() -- that also brings up SDL's joystick subsystem,
        # which grabs the Sidewinder as a native SDL game controller. When
        # JoystickInput then claims the same device exclusively via libusb,
        # the device vanishes out from under SDL mid-session and corrupts
        # pygame's event-translation state (observed as `KeyError: 0` inside
        # pygame.event.get()). We drive the joystick ourselves via raw HID
        # reports, so SDL's joystick subsystem is never needed.
        pygame.display.init()
        pygame.font.init()

        self.renderer = Renderer(config.WINDOW_SIZE)
        self.clock = pygame.time.Clock()

        if input_mode == "mouse":
            max_radius_px = min(self.renderer.sim_width, config.WINDOW_SIZE[1]) * 0.35
            self.source = MouseInput(self.renderer.sim_center_px(), max_radius_px)
        else:
            self.source = JoystickInput()

        self.ff_smoother = ExponentialSmoother2D(config.FF_SMOOTHING_TAU)
        self.interaction_smoother = ExponentialSmoother2D(config.FF_SMOOTHING_TAU)

        self.temp_slider = None
        self.damping_slider = None
        self.history = None
        self.atom_trails = None
        self._trail_frame_counter = 0
        self.energy_baseline = None
        self.sim_wall_time = 0.0
        self.steps_per_frame = 1

        self.system_key = None
        self.system = None
        self._build_system(initial_system_key)

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

        self.renderer.set_box_size(self.system.get_box_size())

        # 3D systems (e.g. the MesoMem membrane patch) render through a
        # perspective camera instead of the top-down 2D box. Build it here from
        # the system's requested view and the sim viewport.
        if spec.render_3d:
            cam = self.system.get_camera_params()
            self.camera3d = Camera3D(
                cam["eye"], cam["target"], cam["up"], cam["fov_deg"],
                self.renderer.sim_width, config.WINDOW_SIZE[1],
            )
        else:
            self.camera3d = None

        if self.temp_slider is None:
            self.temp_slider = Slider.from_spec((0, 0, 100, 4), spec.temperature)
            self.damping_slider = Slider.from_spec((0, 0, 100, 4), spec.damping)
        else:
            self.temp_slider.reset(spec.temperature)
            self.damping_slider.reset(spec.damping)

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

        self.ff_smoother.reset()
        self.interaction_smoother.reset()

        sim_time_per_frame = spec.sim_time_per_frame or config.SIM_TIME_PER_FRAME
        self.steps_per_frame = max(1, min(STEPS_PER_FRAME_CAP, round(sim_time_per_frame / spec.timestep)))

    def _cycle_system(self, step=1):
        keys = [key for key, _ in self.systems]
        idx = keys.index(self.system_key)
        self._build_system(keys[(idx + step) % len(keys)])

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
                    return False
                elif event.key == pygame.K_TAB:
                    self._cycle_system(1)
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    idx = event.key - pygame.K_1
                    if idx < len(self.systems):
                        self._build_system(self.systems[idx][0])
            elif event.type == pygame.MOUSEWHEEL:
                step = self.temp_slider.vmax - self.temp_slider.vmin
                self.temp_slider.nudge(event.y * config.TEMP_WHEEL_STEP_FRACTION * step)
            else:
                self.temp_slider.handle_event(event)
                self.damping_slider.handle_event(event)

        keys = pygame.key.get_pressed()
        temp_range = self.temp_slider.vmax - self.temp_slider.vmin
        rate = config.TEMP_KEY_RATE_FRACTION * temp_range
        if keys[pygame.K_UP]:
            self.temp_slider.nudge(rate * dt)
        if keys[pygame.K_DOWN]:
            self.temp_slider.nudge(-rate * dt)
        return True

    def _tick(self, dt):
        spec = self.system.spec
        ff_profile = spec.force_feedback

        self.system.set_target_temp(self.temp_slider.value)
        self.system.set_puller_damping(self.damping_slider.value)

        # While actively dragging a slider (which lives in the right-hand
        # panel, off to the side of the sim box), don't also feed that mouse
        # position to the puller as a deflection -- zero the input force
        # instead of letting a slider drag yank the atom.
        ui_capturing_mouse = self.temp_slider.dragging or self.damping_slider.dragging
        if self.input_mode == "mouse" and ui_capturing_mouse:
            jx, jy = 0.0, 0.0
            yaw = 0.0
        else:
            jx, jy = self.source.poll()
            yaw = self.source.poll_yaw()
        input_fx, input_fy = jx * ff_profile.input_force_scale, jy * ff_profile.input_force_scale
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

        self.system.step(self.steps_per_frame)

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
        self.source.send_force(smooth_fx, smooth_fy, stiffness)
        self.source.set_damper_coefficient(
            shape_damper_coefficient(smooth_ifx, smooth_ify, ff_profile, DAMPER_COEFFICIENT_MAX)
        )

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
        self.source.update_jitter(heat_fraction)
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
            }

        self.renderer.draw(
            positions, is_puller, pos,
            (input_fx, input_fy), interaction_force, self.clock.get_fps(),
            spec, self.systems, self.system_key,
            (self.temp_slider, self.damping_slider),
            (temp, press, ke, pe, etotal), (puller_ke, puller_pe),
            self.history, rdf, heat_fraction=heat_fraction,
            sim_time_ps=sim_time_ps, puller_speed_m_s=puller_speed_m_s,
            atom_trails=self.atom_trails, species=species, bond_pairs=bond_pairs,
            hbond_pairs=hbond_pairs, hud_lines=hud_lines, scene_3d=scene_3d,
        )

        new_dt = self.clock.tick(60) / 1000.0
        self.sim_wall_time += new_dt
        return new_dt

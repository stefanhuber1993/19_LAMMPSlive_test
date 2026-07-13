"""pygame rendering: the LAMMPS box (crystal/puller/force vectors) on the
left, a live instrumentation panel (system picker, draggable sliders, and
the canonical MD live plots -- T/P, energy, RDF) on the right."""
import math
import random

import pygame

from .. import units
from .plotting import draw_plot
from .theme import (
    ARROWHEAD_ANGLE, ARROWHEAD_LEN, BG, BOX_OUTLINE, CRYSTAL_COLOR, CRYSTAL_RADIUS,
    DIM_TEXT_COLOR, HEADER_TEXT_COLOR, INPUT_VEC_COLOR, MELT_MARK_COLOR, PANEL_BG,
    PANEL_DIVIDER, PANEL_PAD, PANEL_WIDTH, PLOT_COLORS, PULLER_COLOR, PULLER_RADIUS,
    REACTION_VEC_COLOR, TEXT_COLOR, VECTOR_MAX_PX,
)

KEY_HINTS = (
    "1-9: pick system    Tab: next system    Up/Down or wheel: temperature    Esc: quit"
)


class Renderer:
    def __init__(self, window_size):
        self.window_size = window_size
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("LAMMPS live")
        self.font = pygame.font.SysFont(None, 18)
        self.small_font = pygame.font.SysFont(None, 15)
        self.header_font = pygame.font.SysFont(None, 22, bold=True)

        self.sim_width = window_size[0] - PANEL_WIDTH
        self.panel_rect = pygame.Rect(self.sim_width, 0, PANEL_WIDTH, window_size[1])
        self.box_x = self.box_y = None
        self.scale = self.ox = self.oy = None

        # Per-pixel-alpha scratch surface for the puller's fading motion
        # trail, reused every frame (cleared, not reallocated) -- the sim
        # view's main screen surface has no per-pixel alpha, so a true fade
        # needs a separate SRCALPHA surface blitted on top.
        self.trail_surface = pygame.Surface((self.sim_width, window_size[1]), pygame.SRCALPHA)

    def set_box_size(self, box_size):
        """(Re)compute the sim<->screen mapping for a box size in Angstrom
        -- called on startup and again whenever the active system changes
        (different systems can have different box dimensions)."""
        self.box_x, self.box_y = box_size
        margin = 40
        self.scale = min(
            (self.sim_width - 2 * margin) / self.box_x,
            (self.window_size[1] - 2 * margin) / self.box_y,
        )
        self.ox = (self.sim_width - self.box_x * self.scale) / 2
        self.oy = (self.window_size[1] - self.box_y * self.scale) / 2

    def sim_center_px(self):
        return (self.sim_width / 2, self.window_size[1] / 2)

    def sim_to_screen(self, x, y):
        sx = self.ox + x * self.scale
        sy = self.window_size[1] - (self.oy + y * self.scale)  # flip: sim y up
        return int(sx), int(sy)

    def _draw_arrow(self, start, vec_sim, color, knee, width=3):
        mag = math.hypot(vec_sim[0], vec_sim[1])
        if mag < 1e-6:
            return
        length_px = VECTOR_MAX_PX * math.tanh(mag / knee)
        if length_px < 2.0:
            return
        # screen y is flipped relative to sim y (sim +y is up)
        ux, uy = vec_sim[0] / mag, -vec_sim[1] / mag
        vx, vy = ux * length_px, uy * length_px
        end = (start[0] + vx, start[1] + vy)
        pygame.draw.line(self.screen, color, start, end, width)
        angle = math.atan2(vy, vx)
        for sign in (-1, 1):
            head_angle = angle + math.pi - sign * ARROWHEAD_ANGLE
            hx = end[0] + ARROWHEAD_LEN * math.cos(head_angle)
            hy = end[1] + ARROWHEAD_LEN * math.sin(head_angle)
            pygame.draw.line(self.screen, color, end, (hx, hy), width)

    def _draw_trails(self, trails):
        """Every atom's motion trail: a thin same-colored (puller vs.
        crystal) polyline per atom over the last trails.window_seconds,
        fading (alpha towards 0) the further back in time each segment is.
        Drawn on a dedicated per-pixel-alpha surface (cleared and reused
        each frame) since the main screen surface has no alpha channel of
        its own -- pygame.draw.line would just ignore it.

        Segments are found by walking consecutive frame snapshots and
        matching atom ids present in both -- see AtomTrails' docstring for
        why id (not array position) is the right join key.

        The box is periodic in x (see systems' "boundary p f p"): an atom
        drifting across that edge has its coordinate wrap from one side of
        the box to the other between two samples, which -- read naively --
        looks like a giant one-frame jump straight across the box. Segments
        whose endpoints are more than half the box apart are that wrap
        artifact, not real motion, and are skipped rather than drawn."""
        self.trail_surface.fill((0, 0, 0, 0))
        frames = list(trails.frames)
        if len(frames) < 2:
            return
        now = frames[-1][0]
        window = trails.window_seconds
        max_dx, max_dy = self.box_x * 0.5, self.box_y * 0.5
        for (_, snap0), (t1, snap1) in zip(frames, frames[1:]):
            age = now - t1
            alpha = 255.0 * (1.0 - age / window)
            if alpha <= 1.0:
                continue
            alpha_i = int(alpha)
            for atom_id, (x1, y1, is_puller) in snap1.items():
                prev = snap0.get(atom_id)
                if prev is None:
                    continue
                x0, y0, _ = prev
                if abs(x1 - x0) > max_dx or abs(y1 - y0) > max_dy:
                    continue  # periodic-boundary wrap, not real motion
                r, g, b = PULLER_COLOR if is_puller else CRYSTAL_COLOR
                p0 = self.sim_to_screen(x0, y0)
                p1 = self.sim_to_screen(x1, y1)
                pygame.draw.line(self.trail_surface, (r, g, b, alpha_i), p0, p1, 1)
        self.screen.blit(self.trail_surface, (0, 0))

    def draw_sim(self, positions, is_puller, puller_pos, input_force, reaction_force,
                 fps, spec, heat_fraction=0.0, sim_time_ps=0.0, atom_trails=None):
        self.screen.fill(BG)

        top_left = self.sim_to_screen(0, self.box_y)
        size = (self.box_x * self.scale, self.box_y * self.scale)
        pygame.draw.rect(self.screen, BOX_OUTLINE, (*top_left, *size), width=1)

        if atom_trails is not None:
            self._draw_trails(atom_trails)

        puller_xy = None
        for (x, y), p in zip(positions, is_puller):
            sx, sy = self.sim_to_screen(x, y)
            if p:
                puller_xy = (sx, sy)
                continue
            pygame.draw.circle(self.screen, CRYSTAL_COLOR, (sx, sy), CRYSTAL_RADIUS)

        if puller_xy is None:
            puller_xy = self.sim_to_screen(*puller_pos)

        knee = spec.force_feedback.ff_knee
        self._draw_arrow(puller_xy, input_force, INPUT_VEC_COLOR, knee)
        self._draw_arrow(puller_xy, reaction_force, REACTION_VEC_COLOR, knee)

        # Visual jitter: a small random screen-space wobble on the drawn
        # puller, scaling with heat_fraction -- readable "temperature" cue
        # even without force-feedback hardware (mouse mode, or just as a
        # second channel alongside the joystick's own vibration effect).
        if heat_fraction > 1e-3:
            jitter_px = 5.0 * heat_fraction
            jx = puller_xy[0] + random.uniform(-jitter_px, jitter_px)
            jy = puller_xy[1] + random.uniform(-jitter_px, jitter_px)
            puller_xy = (jx, jy)

        pygame.draw.circle(self.screen, PULLER_COLOR, puller_xy, PULLER_RADIUS)

        ix, iy = input_force
        rx, ry = reaction_force
        sim_time_str = units.format_sim_time(sim_time_ps)
        label = self.font.render(
            f"{spec.name}  |  sim time: {sim_time_str}   input force: ({ix:4.1f}, {iy:4.1f}) eV/A   "
            f"interaction force: ({rx:5.1f}, {ry:5.1f}) eV/A   fps: {fps:4.0f}",
            True, (200, 200, 200),
        )
        self.screen.blit(label, (10, 10))

        legend = self.font.render(
            f"green = your input force, red = {spec.element_label} interaction (reaction) force",
            True, (140, 140, 140),
        )
        self.screen.blit(legend, (10, 30))

    def draw_panel(self, systems, current_key, sliders, thermo_now, puller_energy,
                    history, rdf, spec, puller_speed_m_s=None):
        pygame.draw.rect(self.screen, PANEL_BG, self.panel_rect)
        pygame.draw.line(self.screen, PANEL_DIVIDER, (self.panel_rect.x, 0),
                          (self.panel_rect.x, self.window_size[1]), 1)

        x = self.panel_rect.x + PANEL_PAD
        w = PANEL_WIDTH - 2 * PANEL_PAD
        y = 10

        picker_bits = []
        for i, (key, sys_spec) in enumerate(systems, start=1):
            marker = ">" if key == current_key else " "
            picker_bits.append(f"{marker}[{i}] {sys_spec.name}")
        picker_surf = self.small_font.render("   ".join(picker_bits), True, DIM_TEXT_COLOR)
        self.screen.blit(picker_surf, (x, y))
        y += 18

        name_surf = self.header_font.render(spec.name, True, HEADER_TEXT_COLOR)
        self.screen.blit(name_surf, (x, y))
        y += name_surf.get_height() + 2

        desc_surf = self.small_font.render(spec.description, True, DIM_TEXT_COLOR)
        self.screen.blit(desc_surf, (x, y))
        y += 16

        hint_surf = self.small_font.render(KEY_HINTS, True, DIM_TEXT_COLOR)
        self.screen.blit(hint_surf, (x, y))
        y += 20

        pygame.draw.line(self.screen, PANEL_DIVIDER, (x, y), (x + w, y), 1)
        y += 12

        temp_slider, damping_slider = sliders
        temp_slider.rect = pygame.Rect(x, y, w, 4)
        temp_slider.draw(self.screen, self.font, mark_value=spec.melt_temp, mark_label="melt")
        y += 46
        damping_slider.rect = pygame.Rect(x, y, w, 4)
        damping_slider.draw(self.screen, self.font)
        y += 34

        temp, press, ke, pe, etotal = thermo_now
        readout = self.small_font.render(
            f"instantaneous: T={temp:6.1f} K   P={press:9.1f} bar", True, DIM_TEXT_COLOR
        )
        self.screen.blit(readout, (x, y))
        y += 18

        puller_ke, puller_pe = puller_energy
        if puller_ke is not None:
            speed_bit = ""
            if puller_speed_m_s is not None:
                speed_a_per_ps = puller_speed_m_s / units.ANGSTROM_PER_PS_TO_M_PER_S
                speed_bit = f"   speed={puller_speed_m_s:7.1f} m/s ({speed_a_per_ps:.3f} A/ps)"
            puller_readout = self.small_font.render(
                f"puller atom:   KE={puller_ke:7.4f} eV   PE={puller_pe:8.4f} eV{speed_bit}", True, DIM_TEXT_COLOR
            )
            self.screen.blit(puller_readout, (x, y))
        y += 20

        pygame.draw.line(self.screen, PANEL_DIVIDER, (x, y), (x + w, y), 1)
        y += 10

        plot_h = 140
        draw_plot(
            self.screen, self.small_font, pygame.Rect(x, y, w, plot_h),
            "Temperature", "K",
            list(history.t), [("T", PLOT_COLORS["temp"], list(history.series["temp"]))],
            y_range=(0.0, spec.temperature.vmax * 1.05),
            ref_lines=[(spec.melt_temp, MELT_MARK_COLOR, "melt")],
        )
        y += plot_h + 10

        # LAMMPS reports this as a real 3D-style pressure using the box's
        # tiny, arbitrary out-of-plane thickness (~0.5*lattice spacing) as
        # its "volume" -- the absolute number is an artifact of that
        # thickness choice, not directly comparable to a real 3D bar
        # reading. Trends (rising under compression/heating) are still
        # physically meaningful, which is why it's kept and labeled
        # "quasi-2D" rather than dropped.
        draw_plot(
            self.screen, self.small_font, pygame.Rect(x, y, w, plot_h),
            "Pressure -- quasi-2D box (bar)", "bar",
            list(history.t), [("P", PLOT_COLORS["press"], list(history.series["press"]))],
        )
        y += plot_h + 10

        draw_plot(
            self.screen, self.small_font, pygame.Rect(x, y, w, plot_h),
            "Energy, relative to t=0 this session (eV)", "eV",
            list(history.t),
            [
                ("KE", PLOT_COLORS["ke"], list(history.series["ke"])),
                ("PE", PLOT_COLORS["pe"], list(history.series["pe"])),
                ("E_tot", PLOT_COLORS["etotal"], list(history.series["etotal"])),
            ],
        )
        y += plot_h + 10

        rdf_rect = pygame.Rect(x, y, w, plot_h)
        if rdf is None:
            pygame.draw.rect(self.screen, (30, 30, 36), rdf_rect)
            pygame.draw.rect(self.screen, PANEL_DIVIDER, rdf_rect, width=1)
            title_surf = self.small_font.render("Radial distribution g(r)", True, TEXT_COLOR)
            self.screen.blit(title_surf, (rdf_rect.x + 6, rdf_rect.y + 4))
            warm = self.small_font.render("warming up...", True, DIM_TEXT_COLOR)
            self.screen.blit(warm, (rdf_rect.x + 6, rdf_rect.y + plot_h // 2))
        else:
            r, g = rdf
            draw_plot(
                self.screen, self.small_font, rdf_rect,
                "Radial distribution g(r), r in Angstrom", "g(r)",
                list(r), [("g(r)", PLOT_COLORS["rdf"], list(g))],
                y_range=(0.0, max(2.0, float(max(g)) * 1.1) if len(g) else 2.0),
                ref_lines=[(1.0, DIM_TEXT_COLOR, "gas")],
            )

    def draw(self, positions, is_puller, puller_pos, input_force, reaction_force, fps,
              spec, systems, current_key, sliders, thermo_now, puller_energy,
              history, rdf, heat_fraction=0.0, sim_time_ps=0.0, puller_speed_m_s=None,
              atom_trails=None):
        self.draw_sim(positions, is_puller, puller_pos, input_force, reaction_force,
                       fps, spec, heat_fraction, sim_time_ps, atom_trails)
        self.draw_panel(systems, current_key, sliders, thermo_now, puller_energy,
                         history, rdf, spec, puller_speed_m_s)
        pygame.display.flip()

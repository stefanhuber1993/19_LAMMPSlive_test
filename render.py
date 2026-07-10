"""pygame rendering of the LAMMPS box: crystal atoms, puller atom, force vectors."""
import math
import pygame

BG = (18, 18, 24)
CRYSTAL_COLOR = (230, 140, 40)
PULLER_COLOR = (60, 200, 255)
INPUT_VEC_COLOR = (80, 220, 120)   # joystick-commanded acceleration
LJ_VEC_COLOR = (255, 90, 90)       # crystal reaction (LJ) acceleration
BOX_OUTLINE = (90, 90, 100)

CRYSTAL_RADIUS = 5
PULLER_RADIUS = 8
# Arrow length: soft-saturating (tanh) rather than linear, since the EAM
# interaction force can spike to several eV/Angstrom on hard contact while
# input force tops out at INPUT_FORCE_SCALE (3.0) -- a fixed linear scale
# either makes small vectors invisible or huge ones fly off-screen.
VECTOR_MAX_PX = 130.0
VECTOR_KNEE = 2.0    # force magnitude (eV/A) at which the arrow is ~most of VECTOR_MAX_PX
ARROWHEAD_LEN = 8
ARROWHEAD_ANGLE = math.radians(25)


class Renderer:
    def __init__(self, window_size, box_size):
        self.window_size = window_size
        self.box_x, self.box_y = box_size
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("LAMMPS live: pull the crystal")
        self.font = pygame.font.SysFont(None, 20)
        margin = 40
        self.scale = min(
            (window_size[0] - 2 * margin) / self.box_x,
            (window_size[1] - 2 * margin) / self.box_y,
        )
        self.ox = (window_size[0] - self.box_x * self.scale) / 2
        self.oy = (window_size[1] - self.box_y * self.scale) / 2

    def sim_to_screen(self, x, y):
        sx = self.ox + x * self.scale
        sy = self.window_size[1] - (self.oy + y * self.scale)  # flip: sim y up
        return int(sx), int(sy)

    def _draw_arrow(self, start, vec_sim, color, width=3):
        mag = math.hypot(vec_sim[0], vec_sim[1])
        if mag < 1e-6:
            return
        length_px = VECTOR_MAX_PX * math.tanh(mag / VECTOR_KNEE)
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

    def draw(self, positions, is_puller, puller_pos, input_force, lj_force, fps):
        self.screen.fill(BG)

        top_left = self.sim_to_screen(0, self.box_y)
        size = (self.box_x * self.scale, self.box_y * self.scale)
        pygame.draw.rect(self.screen, BOX_OUTLINE, (*top_left, *size), width=1)

        puller_xy = None
        for (x, y), p in zip(positions, is_puller):
            sx, sy = self.sim_to_screen(x, y)
            if p:
                puller_xy = (sx, sy)
                continue
            pygame.draw.circle(self.screen, CRYSTAL_COLOR, (sx, sy), CRYSTAL_RADIUS)

        if puller_xy is None:
            puller_xy = self.sim_to_screen(*puller_pos)

        self._draw_arrow(puller_xy, input_force, INPUT_VEC_COLOR)
        self._draw_arrow(puller_xy, lj_force, LJ_VEC_COLOR)
        pygame.draw.circle(self.screen, PULLER_COLOR, puller_xy, PULLER_RADIUS)

        ix, iy = input_force
        lx, ly = lj_force
        label = self.font.render(
            f"input force: ({ix:4.1f}, {iy:4.1f}) eV/A   EAM force: ({lx:5.1f}, {ly:5.1f}) eV/A   fps: {fps:4.0f}",
            True, (200, 200, 200),
        )
        self.screen.blit(label, (10, 10))

        legend = self.font.render(
            "green = your input force, red = crystal (EAM) reaction force", True, (140, 140, 140)
        )
        self.screen.blit(legend, (10, 30))

        pygame.display.flip()

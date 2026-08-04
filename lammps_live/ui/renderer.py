"""pygame rendering: the LAMMPS box (crystal/puller/force vectors) on the
left, a live instrumentation panel (system picker, draggable sliders, and
the canonical MD live plots -- T/P, energy, RDF) on the right."""
import math

import moderngl
import numpy as np
import pygame

from .. import units
from .gl3d import GLScene, proj_matrix, view_matrix
from .glcompositor import GLCompositor
from .plotting import draw_plot
from .widgets import Button
from .theme import (
    ARROWHEAD_ANGLE, ARROWHEAD_LEN, ATOM_MAX_RADIUS, ATOM_MIN_RADIUS, BG,
    BEAD_BAND_HALFWIDTH, BEAD_BAND_SOFT, BEAD_EQUATOR_COLOR, BEAD_POLE_COLOR,
    BEAD_WHITE_POLE_COLOR, BEAD_WHITE_POLE_MIN, BEAD_WHITE_POLE_SOFT, DEPTH_FADE_START,
    BOND_3D_COLOR, BOND_COLOR, BOND_FALLOFF, BOND_LINES_ENABLED, BOND_MIN_ALPHA,
    BOND_PEAK_ALPHA, BOND_STICK_COLOR, BOND_WIDTH, BOX_3D_ALPHA, BOX_3D_COLOR,
    BOX_EDGE_FADE_DEPTH, BOX_EDGE_SUBDIVISIONS,
    BOX_OUTLINE, CRYSTAL_COLOR,
    CRYSTAL_RADIUS, DIM_TEXT_COLOR, DIRECTOR_ARROW_COLOR, EDGE_VIGNETTE_STRENGTH,
    HAZE_COLOR, HAZE_STRENGTH, HBOND_COLOR, HBOND_DASH,
    HBOND_WIDTH, HEADER_TEXT_COLOR, HUD_BG, HUD_TEXT_COLOR, INPUT_VEC_COLOR,
    ION_LABEL_COLOR, MELT_MARK_COLOR, MEMBRANE_BEAD_COLOR, NET_COLOR,
    NET_LINE_ALPHA, PANEL_BG, PANEL_DIVIDER, PANEL_PAD, PANEL_WIDTH, PLOT_COLORS,
    POTENTIAL_COLORS, POTENTIAL_PANEL_BG, POTENTIAL_TOTAL_COLOR, POTENTIAL_TRACK_COLOR,
    PULLER_BOND_COLOR, PULLER_LABEL_BG, PULLER_LABEL_COLOR, PULLER_RADIUS_BOOST,
    PULLER_RING_COLOR, PULLER_RING_FREE_COLOR, PULLER_RING_WIDTH,
    REACTION_VEC_COLOR, SPHERE_AMBIENT,
    SPHERE_LIGHT_DIR, TEXT_COLOR, TORQUE_ARC_APPLIED_RADIUS,
    TORQUE_ARC_HEAD_LEN, TORQUE_ARC_REACTION_RADIUS, TORQUE_ARC_WIDTH,
    VECTOR_MAX_PX,
)


# Banded-bead sprites are shaded per-pixel in numpy on a cache miss, and in a
# fluid membrane almost every bead's director differs and drifts every frame, so
# misses are near-constant. Cap the shaded array at this resolution and upscale
# (smoothly) to the bead's on-screen size, so a fullscreen bead (up to ~90px
# radius = a 180x180 shade) doesn't cost ~20x a windowed one to generate -- the
# generation cost is what made the 900-bead sheet crawl in fullscreen.
BANDED_SPRITE_GEN_MAX = 40


def _lerp_color(a, b, t):
    t = max(0.0, min(1.0, t))
    return (int(a[0] + (b[0] - a[0]) * t),
            int(a[1] + (b[1] - a[1]) * t),
            int(a[2] + (b[2] - a[2]) * t))

KEY_HINTS = (
    "1-9: system   Tab: next   WASD/mouse: move   Q/E or L/R click: rotate   "
    "Up/Down or wheel: temperature   F11/green button: fullscreen   Esc: exit fullscreen / quit"
)
# Appended for systems with a turntable camera (SystemSpec.camera_orbit).
ORBIT_KEY_HINTS = "   drag: orbit camera   wheel: zoom   C: auto-orbit"


class Renderer:
    def __init__(self, window_size, fullscreen=False):
        pygame.display.set_caption("LAMMPS live")
        # Desktop resolution, captured before the first set_mode so it's the true
        # screen size (not a prior window's) -- used as the fullscreen size.
        info = pygame.display.Info()
        self.desktop_size = (info.current_w, info.current_h)
        self.windowed_size = tuple(window_size)
        self.fullscreen = fullscreen
        self._init_display()   # sets screen + all size-dependent layout state

        self.font = pygame.font.SysFont(None, 18)
        self.small_font = pygame.font.SysFont(None, 15)
        self.header_font = pygame.font.SysFont(None, 22, bold=True)

        # Collapsible "Advanced" slider group: the app owns the open/closed state
        # (self.show_advanced, set before each draw) and reads back the clickable
        # header rect (self.advanced_toggle_rect) to toggle it. None -> no
        # advanced sliders this frame (nothing drawn / clickable).
        self.show_advanced = False
        self.advanced_toggle_rect = None

        # Play / Pause / Reset buttons for playback systems (self-assembly). Their
        # rects are (re)positioned every frame in draw_playback_controls; the app
        # reads them back via playback_hit to route clicks. Empty-rect until first
        # drawn, so a stray click can't hit a button for a non-playback system.
        self.playback_buttons = [Button("play", "Play"),
                                 Button("pause", "Pause"),
                                 Button("reset", "Reset")]
        self._playback_visible = False

        # Bead-colouring toggle for the 3D scenes: director bands (which way each
        # bead points) or potential energy (how bound it is). A basic control, not
        # an "Advanced" one -- it changes what you are looking at, not how the
        # model behaves -- so it sits in the panel above the sliders. The app owns
        # the state (self.bead_color_energy, pushed in before each draw) and reads
        # the rect back through bead_color_hit.
        self.bead_color_button = Button("bead_color", "")
        self.bead_color_energy = False
        self._bead_color_visible = False

        # Per-species glyphs (e.g. "+"/"-" on ions) are stamped on a few
        # hundred atoms every frame, so each label string is rendered once and
        # cached rather than re-rasterized per atom.
        self._glyph_font = pygame.font.SysFont(None, 17, bold=True)
        self._glyph_cache = {}

        # A single high-res shaded WHITE sphere sprite, baked once from the
        # light direction. Every 3D bead is this sprite scaled to its
        # (perspective) radius and tinted -- multiplying white by a color yields
        # that color's shaded sphere -- so depth cueing is just choosing the
        # tint. Scaled copies are cached by radius (see _sphere_for_radius).
        self._sphere_base = self._bake_sphere_sprite(128)
        self._sphere_scaled = {}

        # Normalized light + Blinn half-vector (viewer along +z), shared by the
        # banded-bead shading. Cache of banded sphere sprites keyed by
        # (radius, director-in-view, fog) -- see _banded_sphere_sprite.
        lx, ly, lz = SPHERE_LIGHT_DIR
        ln = math.sqrt(lx * lx + ly * ly + lz * lz)
        self._light = (lx / ln, ly / ln, lz / ln)
        hx, hy, hz = self._light[0], self._light[1], self._light[2] + 1.0
        hn = math.sqrt(hx * hx + hy * hy + hz * hz)
        self._half = (hx / hn, hy / hn, hz / hn)
        self._banded_cache = {}

        # Depth-cue strength of the 3D scene currently being drawn (see _fog).
        # Set per frame by draw_sim_3d; the theme default covers the CPU path.
        self._fog_strength = HAZE_STRENGTH

    def _apply_layout(self, size):
        """Recompute everything sized to the window: the sim viewport width, the
        instrumentation panel rect, and the per-pixel-alpha scratch surfaces. The
        fixed-width panel stays put; the sim view absorbs the extra pixels, so a
        bigger window just gives a bigger simulation area (no bitmap scaling). The
        box<->screen mapping is reset here and re-established by the next
        set_box_size call.

        In GL mode `self.screen` is an offscreen SRCALPHA surface that all 2D
        drawing targets, later composited over the GL scene (see GLCompositor); in
        the CPU fallback it is the display surface set by the caller's set_mode."""
        self.window_size = tuple(size)
        self.sim_width = max(200, size[0] - PANEL_WIDTH)
        self.panel_rect = pygame.Rect(self.sim_width, 0, PANEL_WIDTH, size[1])
        self.box_x = self.box_y = None
        self.scale = self.ox = self.oy = None
        # Scratch surfaces for the puller's fading motion trail and the faint
        # bond lines, cleared and reused each frame -- the main screen surface
        # has no per-pixel alpha of its own, so a fade / sub-255 alpha needs
        # these blitted on top. Reallocated here only, on a size change.
        self.trail_surface = pygame.Surface((self.sim_width, size[1]), pygame.SRCALPHA)
        self.bond_surface = pygame.Surface((self.sim_width, size[1]), pygame.SRCALPHA)

        if self.gl_enabled:
            # 2D UI draws to an offscreen surface; GL owns the real framebuffer.
            self.screen = pygame.Surface(size, pygame.SRCALPHA)
            if self.gl_scene is None:
                self.gl_scene = GLScene(self.gl, self.sim_width, size[1])
            else:
                self.gl_scene.resize(self.sim_width, size[1])
            if self.compositor is None:
                self.compositor = GLCompositor(self.gl, size[0], size[1])
            else:
                self.compositor.resize(size[0], size[1])

    def _gl_set_attributes(self):
        """Request a 3.3 core, forward-compatible, double-buffered context with a
        depth buffer -- must be set BEFORE set_mode. macOS only hands out a modern
        (3.2+) core context when forward-compatible core is requested explicitly;
        without this it gives a legacy 2.1 context and the 330 shaders fail."""
        gl = pygame.display
        gl.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        gl.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        gl.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        gl.gl_set_attribute(pygame.GL_CONTEXT_FLAGS, pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG)
        gl.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
        gl.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)

    def _create_gl_display(self, size, mode_flag):
        """(Re)create the OpenGL display surface + moderngl context for `size`.
        A set_mode on a GL window invalidates all GL objects, so the scene and
        compositor are dropped here (while the old context is still current) and
        rebuilt, sized, by the next _apply_layout."""
        if self.gl_scene is not None:
            self.gl_scene.release()
            self.gl_scene = None
        if self.compositor is not None:
            self.compositor.release()
            self.compositor = None
        self._gl_set_attributes()
        self._gl_surface = pygame.display.set_mode(
            size, pygame.OPENGL | pygame.DOUBLEBUF | mode_flag)
        self.gl = moderngl.create_context()

    def _init_display(self):
        """(Re)create the display for the current fullscreen state and lay out to
        it. Tries an OpenGL context first (fast GPU sphere rendering for the 3D
        systems); if that fails on the very first attempt -- no GPU / no GL 3.3 --
        it falls back to the CPU pygame renderer for the whole session. Windowed
        mode is RESIZABLE (drag to any size; macOS green button -> native
        fullscreen space); F11 uses a real SDL fullscreen at the desktop size."""
        size = self.desktop_size if self.fullscreen else self.windowed_size
        mode_flag = pygame.FULLSCREEN if self.fullscreen else pygame.RESIZABLE
        first = not hasattr(self, "gl_enabled")
        if first:
            self.gl_enabled = True   # optimistic; may flip to False just below
            self.gl = self.gl_scene = self.compositor = None

        if self.gl_enabled:
            try:
                self._create_gl_display(size, mode_flag)
            except Exception as exc:
                if not first:
                    raise
                print(f"[lammps-live] OpenGL unavailable ({exc}); "
                      f"using the CPU renderer.")
                self.gl_enabled = False
                self.gl = None
                pygame.display.quit()
                pygame.display.init()
                pygame.display.set_caption("LAMMPS live")

        if not self.gl_enabled:
            self.screen = pygame.display.set_mode(size, mode_flag)
        self._apply_layout(size)

    def is_fullscreen(self):
        """True if the view currently fills the screen -- either our own SDL
        fullscreen (F11) or a macOS-native fullscreen space entered via the green
        button (which fills the display, so the window matches the desktop
        size)."""
        return self.fullscreen or self.window_size == self.desktop_size

    def handle_resize(self, size):
        """React to an OS window resize (drag handle, or the green button
        entering/leaving a macOS fullscreen space). While in our own SDL
        fullscreen we ignore these. Entering a native fullscreen space we adopt
        the OS-provided surface WITHOUT calling set_mode -- doing so would drop
        the window straight back out of the space; any other resize re-creates a
        matching RESIZABLE surface and is remembered as the windowed size."""
        if self.fullscreen:
            return
        if self.gl_enabled:
            # An OS-driven resize (drag handle, or the green zoom button now that
            # native fullscreen Spaces are disabled -- see App.__init__) keeps the
            # GL context alive; the drawable just changes size. A cheap relayout
            # (resize the FBOs + offscreen surface) is all that's needed -- do NOT
            # set_mode, which recreates the context and is what trapped/blacked
            # the window across the old Space transition.
            if tuple(size) != self.desktop_size:
                self.windowed_size = tuple(size)
            self._apply_layout(size)
            return
        if tuple(size) == self.desktop_size:
            surf = pygame.display.get_surface()
            if surf is not None:
                self.screen = surf
        else:
            self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
            self.windowed_size = tuple(size)
        self._apply_layout(size)

    def toggle_fullscreen(self):
        """Flip between our SDL fullscreen and windowed. The caller must re-run
        set_box_size (and rebuild any camera) afterwards, since the sim viewport
        dimensions have changed."""
        self.fullscreen = not self.fullscreen
        self._init_display()

    def set_windowed(self):
        """Leave fullscreen back to the last windowed size -- used by Escape.

        Two different fullscreen mechanisms need two different exits:

          - Our own SDL fullscreen (F11): we created it with set_mode(FULLSCREEN),
            so we just recreate a windowed surface.

          - A macOS green-button "zoom" (Spaces are disabled, see App.__init__):
            the window is screen-filling but is an ordinary window in a sticky
            "zoomed" state, not an SDL fullscreen. A plain set_mode(windowed_size)
            on that same NSWindow does NOT clear the zoom -- the window stays
            screen-filling while only the GL drawable shrinks, leaving the content
            in a black-bordered box (the reported bug). Tearing the window down
            completely and building a fresh one sidesteps the sticky zoom state.

          - A macOS-native fullscreen space, if one is ever entered anyway: SDL
            tracks it as a fullscreen-flagged window, so toggling that flag
            animates back out to the prior windowed size."""
        if self.fullscreen:
            self.fullscreen = False
            self._init_display()
        elif self.window_size == self.desktop_size:
            surf = pygame.display.get_surface()
            if surf is not None and (surf.get_flags() & pygame.FULLSCREEN):
                # SDL tracks a Space as a fullscreen window: toggle its flag to
                # animate back out to the prior windowed size (handle_resize then
                # relayouts on the VIDEORESIZE).
                pygame.display.toggle_fullscreen()
            else:
                # Plain green-button zoom: rebuild a fresh, non-zoomed window.
                self._recreate_windowed_fresh()

    def _recreate_windowed_fresh(self):
        """Fully tear down the display and rebuild a windowed one. Used to leave a
        macOS green-button zoom, whose sticky "zoomed" NSWindow state a plain
        set_mode(windowed_size) can't clear; a brand-new window is never zoomed.
        GL objects and the context are released first (while still current) so the
        subsequent display teardown doesn't strand them."""
        self.fullscreen = False
        if self.gl_enabled:
            for obj in (self.gl_scene, self.compositor):
                if obj is not None:
                    obj.release()
            self.gl_scene = self.compositor = None
            try:
                if self.gl is not None:
                    self.gl.release()
            except Exception:
                pass
            self.gl = None
        pygame.display.quit()
        pygame.display.init()
        pygame.display.set_caption("LAMMPS live")
        self._init_display()

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

    def _bond_segments(self, xa, ya, xb, yb):
        """Screen-space segment(s) for a bond between sim points a and b, drawn
        minimum-image: a bond whose atoms sit on opposite sides of a periodic
        seam (more than half the box apart) is shown as two short stubs running
        off each atom toward its edge -- the way the bond actually wraps -- rather
        than one long line drawn straight across the whole box. Non-periodic
        directions never trip this (bonded atoms are never a half-box apart)."""
        dx, dy = xb - xa, yb - ya
        wrap_x = abs(dx) > 0.5 * self.box_x
        wrap_y = abs(dy) > 0.5 * self.box_y
        if not wrap_x and not wrap_y:
            return [(self.sim_to_screen(xa, ya), self.sim_to_screen(xb, yb))]
        if wrap_x:
            dx -= math.copysign(self.box_x, dx)
        if wrap_y:
            dy -= math.copysign(self.box_y, dy)
        return [(self.sim_to_screen(xa, ya), self.sim_to_screen(xa + dx, ya + dy)),
                (self.sim_to_screen(xb, yb), self.sim_to_screen(xb - dx, yb - dy))]

    def _species_color(self, spec, sp):
        """Fill color for an atom of species index sp (or None for a
        single-species system): its species color, else the system's flat
        crystal color, else the theme fallback."""
        if spec.species_colors is not None and sp is not None:
            return spec.species_colors[sp]
        return spec.crystal_color or CRYSTAL_COLOR

    def _atom_radius_px(self, spec, sp):
        """On-screen radius (px) for an atom of species index sp. Derived from
        the system's PHYSICAL radius (Angstrom) at the current box scale so real
        size ratios show, then clamped to the theme's visible band. Falls back to
        the fixed-pixel CRYSTAL_RADIUS when a system declares no physical size."""
        r_a = None
        if spec.species_radii_A is not None and sp is not None:
            r_a = spec.species_radii_A[sp]
        elif spec.atom_radius_A is not None:
            r_a = spec.atom_radius_A
        if r_a is None:
            return CRYSTAL_RADIUS
        return int(max(ATOM_MIN_RADIUS, min(ATOM_MAX_RADIUS, r_a * self.scale)))

    def _crystal_trail_color(self, spec):
        """A single representative color for the (species-agnostic) motion
        trails of the non-puller atoms: the flat crystal color, or the mean of
        the species colors for multi-species systems, or the theme fallback."""
        if spec.crystal_color is not None:
            return spec.crystal_color
        cs = spec.species_colors
        if cs:
            return tuple(sum(c[k] for c in cs) // len(cs) for k in range(3))
        return CRYSTAL_COLOR

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

    def _draw_trails(self, trails, crystal_color, puller_color):
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
                r, g, b = puller_color if is_puller else crystal_color
                p0 = self.sim_to_screen(x0, y0)
                p1 = self.sim_to_screen(x1, y1)
                pygame.draw.line(self.trail_surface, (r, g, b, alpha_i), p0, p1, 1)
        self.screen.blit(self.trail_surface, (0, 0))

    def _draw_bonds(self, positions, bond_length):
        """Faint bond lines between atom pairs, opacity encoding how close each
        pair is to its equilibrium separation. For every pair, alpha peaks at
        BOND_PEAK_ALPHA when the distance d equals bond_length (the system's
        nearest-neighbor / optimal bonding distance) and decays exponentially in
        either direction with length BOND_FALLOFF*bond_length:
            alpha = BOND_PEAK_ALPHA * exp(-|d - bond_length| / lambda)
        Pairs whose alpha falls below BOND_MIN_ALPHA are skipped, which also
        bounds the drawn distance band. Color, peak alpha, falloff, floor and
        width are all theme constants so they can be tuned in one place.

        Drawn on a dedicated per-pixel-alpha surface (cleared and reused each
        frame) since the main screen surface has no alpha channel of its own.
        Neighbor distances use the minimum image in the periodic x, so a pair
        bonded across the x-seam counts as the real short neighbor it is (rather
        than a box-spanning non-neighbor); such a bond is then drawn wrapped --
        two short stubs off each atom toward its edge -- via _bond_segments."""
        if not BOND_LINES_ENABLED or bond_length <= 0:
            return
        self.bond_surface.fill((0, 0, 0, 0))
        pts = np.asarray(positions, dtype=float)
        n = len(pts)
        if n < 2:
            return
        lam = BOND_FALLOFF * bond_length
        # Beyond this |d - d_opt| the exponential alpha has decayed below the
        # BOND_MIN_ALPHA floor, so those pairs are invisible -- clip them out
        # before the (more expensive) per-line work.
        max_offset = lam * math.log(BOND_PEAK_ALPHA / BOND_MIN_ALPHA)
        # Full pairwise separations. With ~a few hundred atoms this NxN pass is
        # trivial and lets numpy do the distance filtering in one shot. The x
        # component is taken minimum-image (x is the periodic crystal direction),
        # so seam-crossing neighbors register at their true short distance; y is
        # left raw (the non-periodic direction), so far-apart rows never get
        # wrapped into a spurious bond.
        diff = pts[:, None, :] - pts[None, :, :]
        dxw = diff[..., 0] - self.box_x * np.round(diff[..., 0] / self.box_x)
        dist = np.hypot(dxw, diff[..., 1])
        iu, ju = np.triu_indices(n, k=1)
        d = dist[iu, ju]
        offset = np.abs(d - bond_length)
        mask = offset <= max_offset
        iu, ju, offset = iu[mask], ju[mask], offset[mask]
        alphas = (BOND_PEAK_ALPHA * np.exp(-offset / lam)).astype(int)
        r, g, b = BOND_COLOR
        for i, j, a in zip(iu, ju, alphas):
            for p0, p1 in self._bond_segments(pts[i][0], pts[i][1], pts[j][0], pts[j][1]):
                pygame.draw.line(self.bond_surface, (r, g, b, int(a)), p0, p1, BOND_WIDTH)
        self.screen.blit(self.bond_surface, (0, 0))

    def _draw_dashed(self, p0, p1, color, width, dash):
        """A dashed straight line p0->p1. dash<=0 draws it solid."""
        x0, y0 = p0
        x1, y1 = p1
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-3:
            return
        if dash <= 0:
            pygame.draw.line(self.screen, color, p0, p1, width)
            return
        ux, uy = dx / length, dy / length
        n = int(length // dash)
        for k in range(0, n + 1, 2):
            s = k * dash
            e = min(s + dash, length)
            a = (x0 + ux * s, y0 + uy * s)
            b = (x0 + ux * e, y0 + uy * e)
            pygame.draw.line(self.screen, color, a, b, width)

    def _draw_hud(self, lines):
        """Small stacked live-status lines in the lower-left of the sim view,
        over a faint backdrop (see get_hud_lines)."""
        if not lines:
            return
        pad = 6
        surfs = [self.font.render(t, True, HUD_TEXT_COLOR) for t in lines]
        h = sum(s.get_height() for s in surfs) + 2 * pad
        w = max(s.get_width() for s in surfs) + 2 * pad
        y0 = self.window_size[1] - h - 10
        bg = pygame.Surface((w, h), pygame.SRCALPHA)
        bg.fill(HUD_BG)
        self.screen.blit(bg, (10, y0))
        y = y0 + pad
        for s in surfs:
            self.screen.blit(s, (10 + pad, y))
            y += s.get_height()

    def _draw_debug_line(self, text):
        """The --debug per-frame timing breakdown, in the lower-right of the sim
        view (clear of the lower-left HUD and the top-left header/potential
        panel), over a faint backdrop so it stays legible on any scene."""
        if not text:
            return
        surf = self.small_font.render(text, True, (120, 235, 150))
        pad = 5
        x = self.sim_width - surf.get_width() - pad - 12
        y = self.window_size[1] - surf.get_height() - pad - 10
        bg = pygame.Surface((surf.get_width() + 2 * pad, surf.get_height() + 2 * pad), pygame.SRCALPHA)
        bg.fill(HUD_BG)
        self.screen.blit(bg, (x - pad, y - pad))
        self.screen.blit(surf, (x, y))

    def _blit_glyph(self, sx, sy, label):
        """Stamp a centered species glyph (e.g. "+"/"-") at screen (sx, sy).
        Rendered surfaces are cached by label string."""
        if not label:
            return
        glyph = self._glyph_cache.get(label)
        if glyph is None:
            glyph = self._glyph_font.render(label, True, ION_LABEL_COLOR)
            self._glyph_cache[label] = glyph
        self.screen.blit(glyph, glyph.get_rect(center=(sx, sy)))

    # ---- 3D scene (perspective + depth-cued spheres) -----------------------

    def _bake_sphere_sprite(self, res):
        """A white, Phong-ish shaded sphere on transparent background, rendered
        once at `res` px. RGB is the shading (ambient + diffuse + a small
        specular) so tinting by BLEND_MULT gives any colored sphere; alpha is a
        soft circular mask."""
        lx, ly, lz = SPHERE_LIGHT_DIR
        ln = math.sqrt(lx * lx + ly * ly + lz * lz)
        lx, ly, lz = lx / ln, ly / ln, lz / ln
        coords = (np.arange(res) + 0.5) / res * 2.0 - 1.0
        gx, gy = np.meshgrid(coords, coords, indexing="ij")
        rr = gx * gx + gy * gy
        inside = rr <= 1.0
        rr_c = np.clip(rr, 0.0, 1.0)
        nz = np.sqrt(1.0 - rr_c)
        # Screen y is down; flip so the light's "up" reads as up on screen.
        nx, ny = gx, -gy
        diffuse = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)
        # Specular from a Blinn half-vector with the viewer along +z.
        hx, hy, hz = lx, ly, lz + 1.0
        hn = math.sqrt(hx * hx + hy * hy + hz * hz)
        ndoth = np.clip((nx * hx + ny * hy + nz * hz) / hn, 0.0, 1.0)
        specular = 0.5 * ndoth ** 32
        shade = np.clip(SPHERE_AMBIENT + (1.0 - SPHERE_AMBIENT) * diffuse + specular, 0.0, 1.0)
        val = (shade * 255).astype(np.uint8)

        surf = pygame.Surface((res, res), pygame.SRCALPHA)
        rgb = pygame.surfarray.pixels3d(surf)
        rgb[:, :, 0] = val
        rgb[:, :, 1] = val
        rgb[:, :, 2] = val
        del rgb
        alpha = pygame.surfarray.pixels_alpha(surf)
        # Soft 2px rim for anti-aliasing before downscaling.
        edge = np.clip((1.0 - rr) / 0.05, 0.0, 1.0)
        alpha[:, :] = np.where(inside, (edge * 255), 0).astype(np.uint8)
        del alpha
        return surf

    def _sphere_for_radius(self, r):
        r = max(2, int(r))
        sprite = self._sphere_scaled.get(r)
        if sprite is None:
            sprite = pygame.transform.smoothscale(self._sphere_base, (2 * r, 2 * r))
            self._sphere_scaled[r] = sprite
        return sprite

    def _banded_sphere_sprite(self, r, d_view, fog, bright=1.0):
        """A shaded sphere colored the MesoMem way: blue at the poles (along the
        director) and a yellow band at the equator (perpendicular to it), with the
        +director pole over-painted white so its sense reads. d_view is the bead's
        unit director expressed in the view frame (right, screen-down, toward-
        viewer); the band tilts with it, so tilt/splay read straight off the
        coloring. `bright` multiplies the albedo (spotlighting a tagged bead).
        Cached by (radius, quantized director, quantized fog, quantized bright) --
        the six ring beads share a cache entry (director +z), only the pulled
        bead's entry changes as it swings."""
        r = max(2, int(r))
        dvx, dvy, dvz = d_view
        # Coarse director quantization (0.1 per view-component ~ a few degrees of
        # band tilt) so nearly-aligned beads share a cache entry -- the band shift
        # this hides is imperceptible but it turns most of the sheet's per-frame
        # regenerations into cache hits.
        key = (r, round(dvx, 1), round(dvy, 1), round(dvz, 1), round(fog, 1),
               round(bright, 1))
        spr = self._banded_cache.get(key)
        if spr is not None:
            return spr
        # The pulled bead's director changes continuously, so its key changes
        # every frame -- bound the cache so a long session can't accumulate
        # thousands of one-off sprites. The static ring beads re-populate at once.
        if len(self._banded_cache) > 4000:
            self._banded_cache.clear()

        n = 2 * r
        # Shade at a capped resolution, then upscale to n -- a smooth sphere
        # upsamples cleanly and the generation cost stops growing with bead size.
        gen = min(n, BANDED_SPRITE_GEN_MAX)
        coords = (np.arange(gen) + 0.5) / gen * 2.0 - 1.0
        gx, gy = np.meshgrid(coords, coords, indexing="ij")
        rr = gx * gx + gy * gy
        inside = rr <= 1.0
        nz = np.sqrt(np.clip(1.0 - rr, 0.0, 1.0))

        # Lighting (screen-up is -gy), matching the plain sphere baker.
        lx, ly, lz = self._light
        hx, hy, hz = self._half
        diffuse = np.clip(gx * lx + (-gy) * ly + nz * lz, 0.0, 1.0)
        ndoth = np.clip(gx * hx + (-gy) * hy + nz * hz, 0.0, 1.0)
        shade = np.clip(SPHERE_AMBIENT + (1.0 - SPHERE_AMBIENT) * diffuse + 0.5 * ndoth ** 32, 0.0, 1.0)

        # Latitude relative to the director: signed s = N_view . d_view, with the
        # true (unlit) surface normal N_view = (gx, gy, nz) in the view frame.
        sdir = gx * dvx + gy * dvy + nz * dvz
        cos_lat = np.abs(sdir)
        t = np.clip((cos_lat - (BEAD_BAND_HALFWIDTH - BEAD_BAND_SOFT)) / (2.0 * BEAD_BAND_SOFT),
                    0.0, 1.0)   # 0 at the equator (yellow) -> 1 at the poles (blue)
        yl = np.array(BEAD_EQUATOR_COLOR, dtype=float)
        bl = np.array(BEAD_POLE_COLOR, dtype=float)
        base = yl[None, None, :] * (1.0 - t[..., None]) + bl[None, None, :] * t[..., None]
        # Over-paint the +director pole white (down to ~80% latitude).
        wl = np.array(BEAD_WHITE_POLE_COLOR, dtype=float)
        w = np.clip((sdir - (BEAD_WHITE_POLE_MIN - BEAD_WHITE_POLE_SOFT))
                    / (2.0 * BEAD_WHITE_POLE_SOFT), 0.0, 1.0)
        base = base * (1.0 - w[..., None]) + wl[None, None, :] * w[..., None]
        base *= bright
        if fog > 0.0:
            haze = np.array(HAZE_COLOR, dtype=float)
            base = base * (1.0 - fog) + haze[None, None, :] * fog
        rgb = np.clip(base * shade[..., None], 0, 255).astype(np.uint8)

        surf = pygame.Surface((gen, gen), pygame.SRCALPHA)
        px = pygame.surfarray.pixels3d(surf)
        px[:, :, :] = rgb
        del px
        al = pygame.surfarray.pixels_alpha(surf)
        al[:, :] = np.where(inside, np.clip((1.0 - rr) / 0.05, 0.0, 1.0) * 255, 0).astype(np.uint8)
        del al
        if gen != n:
            surf = pygame.transform.smoothscale(surf, (n, n))
        self._banded_cache[key] = surf
        return surf

    def _puller_ring_color(self, attached):
        """Colour of the ring marking the controlled particle. Dim once released
        (B / joystick trigger): still marked, because it is still the particle
        the ring will grab again, but plainly not being steered."""
        return PULLER_RING_COLOR if attached else PULLER_RING_FREE_COLOR

    def _fog(self, depth, near, far):
        """How far a thing at `depth` has receded into the background, 0..1.

        The strength is per-frame state (`self._fog_strength`) rather than the
        theme constant, because the GL path's depth cue is a per-system
        RenderStyle setting and the overlays drawn on top of it -- director
        spikes, bond spokes, the box outline -- have to recede on the SAME ramp
        as the beads or they float in front of a scene that has faded away."""
        if far <= near:
            return 0.0
        return self._fog_strength * max(0.0, min(1.0, (depth - near) / (far - near)))

    def _net_world_segments(self, grid):
        """World-space (start, end) point pairs for every line of the control-
        plane net -- projected and depth-tested per frame so the net can be
        occluded by the beads."""
        origin = np.asarray(grid["origin"], dtype=float)
        u = np.asarray(grid["u_axis"], dtype=float)
        v = np.asarray(grid["v_axis"], dtype=float)
        (u0, u1), (v0, v1) = grid["u_range"], grid["v_range"]
        step = grid["step"]
        segs = []
        # Snap the spacing so an integer number of cells lands exactly on both
        # ends: the net then spans precisely [u0,u1] x [v0,v1] -- the puller's
        # movement limits -- with its outer boundary lines right at those limits,
        # rather than stopping a fractional step short of the far/top edge.
        n_u = max(1, int(round((u1 - u0) / step)))
        n_v = max(1, int(round((v1 - v0) / step)))
        su = (u1 - u0) / n_u
        sv = (v1 - v0) / n_v
        for i in range(n_u + 1):
            uu = u0 + i * su
            segs.append((origin + uu * u + v0 * v, origin + uu * u + v1 * v))
        for j in range(n_v + 1):
            vv = v0 + j * sv
            segs.append((origin + u0 * u + vv * v, origin + u1 * u + vv * v))
        return segs

    def _box_edge_segments(self, box_bounds, camera):
        """World-space (start, end) pairs for the simulation box's edges, cut into
        short sub-segments so the near corner can be DISSOLVED rather than
        dropped. box_bounds is (xlo, xhi, ylo, yhi, zlo, zhi).

        The problem this solves: the corner of the box nearest the eye hangs in
        empty space in front of everything, and its three edges streak across the
        scene. Dropping the single nearest edge outright (what this used to do)
        fixes that but pops -- swing the camera and a whole edge blinks out as the
        argmin changes, which on the orbiting assembly box happens every few
        seconds. Fading by DEPTH instead is continuous in the camera angle, since
        each vertex's depth is; see _box_edge_alpha. Sub-dividing is what lets the
        fade run along an edge, so an edge that starts near the eye and ends deep
        in the scene comes in gradually instead of taking one alpha for its whole
        length."""
        xlo, xhi, ylo, yhi, zlo, zhi = box_bounds
        xs, ys, zs = (xlo, xhi), (ylo, yhi), (zlo, zhi)
        corner = {(i, j, k): np.array([xs[i], ys[j], zs[k]], dtype=float)
                  for i in (0, 1) for j in (0, 1) for k in (0, 1)}
        edges = []
        for i in (0, 1):
            for j in (0, 1):
                for k in (0, 1):
                    if i == 0:
                        edges.append(((0, j, k), (1, j, k)))
                    if j == 0:
                        edges.append(((i, 0, k), (i, 1, k)))
                    if k == 0:
                        edges.append(((i, j, 0), (i, j, 1)))
        segs = []
        n = BOX_EDGE_SUBDIVISIONS
        for a, b in edges:                       # 12 unique edges
            pa, pb = corner[a], corner[b]
            for m in range(n):
                segs.append((pa + (pb - pa) * (m / n),
                             pa + (pb - pa) * ((m + 1) / n)))
        return segs

    def _box_edge_alpha(self, box_bounds, camera):
        """A function mapping a point's view depth to the box outline's opacity
        there: 0 at the box's nearest corner, rising to 1 over the front
        BOX_EDGE_FADE_DEPTH of its depth span.

        Anchored to the BOX's own depth extent, not the beads': the box is what is
        being faded, its span is what the fade has to cover, and both are computed
        from the same eight corners whatever the camera is doing."""
        xlo, xhi, ylo, yhi, zlo, zhi = box_bounds
        corners = np.array([[x, y, z] for x in (xlo, xhi)
                            for y in (ylo, yhi) for z in (zlo, zhi)])
        _, depths, _ = camera.project(corners)
        finite = depths[np.isfinite(depths)]
        if not len(finite):
            return lambda d: 1.0
        d_near, d_far = float(finite.min()), float(finite.max())
        width = max(BOX_EDGE_FADE_DEPTH * (d_far - d_near), 1e-6)

        def alpha(d):
            if not np.isfinite(d):
                return 0.0
            t = min(max((d - d_near) / width, 0.0), 1.0)
            return t * t * (3.0 - 2.0 * t)       # smoothstep: no visible seam
        return alpha

    def _build_bead_zbuffer(self, screen, depth, radii, phys_r, region=None):
        """A per-pixel depth buffer of the beads' front surfaces over the sim
        view: zbuf[x, y] = view-depth of the nearest sphere surface at that
        pixel (+inf where no bead covers it). Used to occlude the net so it is
        hidden inside or behind a bead and drawn only up to its silhouette. The
        front-surface depth is d0 - r_world*sqrt(1 - (rho/r_px)^2) (nearest at
        the centre, tapering to d0 at the silhouette rim).

        region=(x0, y0, x1, y1), if given, is the net's screen bounding box: only
        beads overlapping it are painted, since the net can only be occluded
        where it's actually drawn. On the large sheet this is a small local net
        near the puller, so this skips ~all of the hundreds of beads and is the
        difference between a cheap and a frame-dominating z-buffer pass."""
        W, H = self.sim_width, self.window_size[1]
        zbuf = np.full((W, H), np.inf, dtype=np.float32)
        rx0, ry0, rx1, ry1 = region if region is not None else (0, 0, W, H)
        for i in range(len(depth)):
            d0 = depth[i]
            r = int(radii[i])
            if not np.isfinite(d0) or r < 1:
                continue
            cx, cy = screen[i]
            if cx + r < rx0 or cx - r > rx1 or cy + r < ry0 or cy - r > ry1:
                continue  # bead can't overlap the net's screen region
            x0, x1 = max(0, int(cx - r)), min(W, int(cx + r) + 1)
            y0, y1 = max(0, int(cy - r)), min(H, int(cy + r) + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            xs = np.arange(x0, x1) - cx
            ys = np.arange(y0, y1) - cy
            rho2 = (xs[:, None] ** 2 + ys[None, :] ** 2) / float(r * r)
            front = (d0 - phys_r * np.sqrt(np.clip(1.0 - rho2, 0.0, 1.0))).astype(np.float32)
            front = np.where(rho2 <= 1.0, front, np.inf).astype(np.float32)
            sub = zbuf[x0:x1, y0:y1]
            np.minimum(sub, front, out=sub)
        return zbuf

    def _draw_net_occluded(self, camera, segs, zbuf):
        """Draw the control-plane net with real depth occlusion against the bead
        z-buffer: each line is sampled in 3D, projected, and only the runs whose
        depth is in front of the nearest bead surface (or over empty background)
        are drawn -- so the net stops at each sphere's silhouette and is hidden
        inside and behind beads."""
        W, H = self.sim_width, self.window_size[1]
        surf = self.trail_surface  # cleared per-pixel-alpha scratch surface
        surf.fill((0, 0, 0, 0))
        col = (*NET_COLOR, NET_LINE_ALPHA)
        eps = np.float32(0.03)   # small bias so the net reaches exactly to the silhouette
        for pa, pb in segs:
            sa, _, _ = camera.project_point(pa)
            sb, _, _ = camera.project_point(pb)
            n = max(2, int(math.hypot(sb[0] - sa[0], sb[1] - sa[1]) / 3.0))
            ss = np.linspace(0.0, 1.0, n)
            pts = pa[None, :] * (1.0 - ss[:, None]) + pb[None, :] * ss[:, None]
            scr, dep, _ = camera.project(pts)
            xi = np.round(scr[:, 0]).astype(int)
            yi = np.round(scr[:, 1]).astype(int)
            inb = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H) & np.isfinite(dep)
            zb = np.full(n, np.inf, dtype=np.float32)
            zb[inb] = zbuf[xi[inb], yi[inb]]
            vis = inb & (dep < zb + eps)
            run = []
            for k in range(n):
                if vis[k]:
                    run.append((scr[k, 0], scr[k, 1]))
                elif len(run) >= 2:
                    pygame.draw.aalines(surf, col, False, run)
                    run = []
                else:
                    run = []
            if len(run) >= 2:
                pygame.draw.aalines(surf, col, False, run)
        self.screen.blit(surf, (0, 0))

    def _draw_box_occluded(self, camera, box_segs, zbuf, near, far, near_alpha):
        """Draw the simulation-box edges with the same bead-z-buffer occlusion as
        the net, but in depth-cued white: each edge is sampled in 3D, projected,
        and only the runs in front of the nearest bead surface (or over empty
        background) are drawn, so the box reads as a frame the beads sit inside.
        Fog and the near-corner dissolve are applied per sub-segment (from its
        midpoint depth) -- a per-pixel gradient isn't worth it on this rare CPU
        fallback, and _box_edge_segments hands back short enough pieces that the
        difference does not show."""
        W, H = self.sim_width, self.window_size[1]
        surf = self.bond_surface   # reused scratch (already blitted above)
        surf.fill((0, 0, 0, 0))
        eps = np.float32(0.03)
        for pa, pb in box_segs:
            _, dmid, _ = camera.project_point(0.5 * (pa + pb))
            fog = self._fog(dmid, near, far) if np.isfinite(dmid) else 0.0
            alpha = int(BOX_3D_ALPHA * near_alpha(dmid))
            if alpha <= 1:
                continue
            col = (*_lerp_color(BOX_3D_COLOR, HAZE_COLOR, fog), alpha)
            sa, _, _ = camera.project_point(pa)
            sb, _, _ = camera.project_point(pb)
            n = max(2, int(math.hypot(sb[0] - sa[0], sb[1] - sa[1]) / 3.0))
            ss = np.linspace(0.0, 1.0, n)
            pts = pa[None, :] * (1.0 - ss[:, None]) + pb[None, :] * ss[:, None]
            scr, dep, _ = camera.project(pts)
            xi = np.round(scr[:, 0]).astype(int)
            yi = np.round(scr[:, 1]).astype(int)
            inb = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H) & np.isfinite(dep)
            zb = np.full(n, np.inf, dtype=np.float32)
            zb[inb] = zbuf[xi[inb], yi[inb]]
            vis = inb & (dep < zb + eps)
            run = []
            for k in range(n):
                if vis[k]:
                    run.append((scr[k, 0], scr[k, 1]))
                else:
                    if len(run) >= 2:
                        pygame.draw.aalines(surf, col, False, run)
                    run = []
            if len(run) >= 2:
                pygame.draw.aalines(surf, col, False, run)
        self.screen.blit(surf, (0, 0))

    def _draw_director_arrow(self, camera, center_world, n, phys_r, r_px, fog):
        """A thin arrow (cylinder shaft + cone head) through the bead along its
        director n_i, poking out the near pole -- marks which of the two
        equivalent normals the bead currently points along, so a flip is
        visible. Drawn on top of the bead it belongs to (occluded by nearer
        beads via the painter ordering of the caller)."""
        n = np.asarray(n, dtype=float)
        c = np.asarray(center_world, dtype=float)
        tail = c - n * (0.6 * phys_r)
        neck = c + n * (0.7 * phys_r)
        tip = c + n * (1.25 * phys_r)
        ts, _, _ = camera.project_point(tail)
        ns, _, _ = camera.project_point(neck)
        tp, _, _ = camera.project_point(tip)
        col = _lerp_color(DIRECTOR_ARROW_COLOR, HAZE_COLOR, fog)
        pygame.draw.line(self.screen, col, ts, ns, max(2, int(round(0.08 * r_px))))
        self._draw_cone(ns, tp, max(2.5, 0.19 * r_px), col)

    def _draw_director_arrows_batch(self, camera, pts, dips, phys_r, radii,
                                    depth, near, far, order):
        """Director spikes for many beads at once. Same shaft+cone as
        _draw_director_arrow, but the shaft/neck/tip points for every bead are
        projected in three vectorized camera.project calls instead of three
        per-bead project_point calls (which, on the 900-bead sheet, would be
        thousands of tiny numpy calls and dominate the frame). Drawn far -> near
        (`order`) so nearer spikes overlay farther ones."""
        pts = np.asarray(pts, dtype=float)
        dips = np.asarray(dips, dtype=float)
        ts, _, _ = camera.project(pts - dips * (0.6 * phys_r))
        ns, _, _ = camera.project(pts + dips * (0.7 * phys_r))
        tp, _, _ = camera.project(pts + dips * (1.25 * phys_r))
        for i in order:
            if not np.isfinite(depth[i]):
                continue
            col = _lerp_color(DIRECTOR_ARROW_COLOR, HAZE_COLOR,
                              self._fog(depth[i], near, far))
            r_px = int(radii[i])
            a = (float(ts[i][0]), float(ts[i][1]))
            b = (float(ns[i][0]), float(ns[i][1]))
            t = (float(tp[i][0]), float(tp[i][1]))
            pygame.draw.line(self.screen, col, a, b, max(2, int(round(0.08 * r_px))))
            self._draw_cone(b, t, max(2.5, 0.19 * r_px), col)

    def _draw_potential_panel(self, decomposition, x=12):
        """A compact live breakdown of an interaction energy into the force
        field's additive terms (see MDSystem.get_potential_terms): each term as a
        signed horizontal bar from a shared zero line, plus their sum, so the
        additive structure is visible at a glance and the bars move as the user
        pulls/twists. Drawn in the otherwise-empty upper-left of the sim view; the
        `x` offset lets a second panel (the whole-system total) sit beside the
        first."""
        if not decomposition:
            return
        title, terms, scale = decomposition
        scale = max(1e-6, float(scale))
        rows = list(terms) + [("total", sum(v for _, v in terms))]

        w, pad = 312, 10
        row_h, title_h = 30, 20
        y0 = 48
        h = title_h + len(rows) * row_h + pad
        bg = pygame.Surface((w, h), pygame.SRCALPHA)
        bg.fill(POTENTIAL_PANEL_BG)
        self.screen.blit(bg, (x, y0))
        pygame.draw.rect(self.screen, PANEL_DIVIDER, (x, y0, w, h), 1)

        self.screen.blit(self.small_font.render(title, True, HEADER_TEXT_COLOR), (x + pad, y0 + 6))

        cx = x + pad + (w - 2 * pad) / 2.0   # shared zero line
        half = (w - 2 * pad) / 2.0 - 2
        y = y0 + title_h + 6
        for i, (label, val) in enumerate(rows):
            is_total = (i == len(rows) - 1)
            col = POTENTIAL_TOTAL_COLOR if is_total else POTENTIAL_COLORS[i % len(POTENTIAL_COLORS)]
            if is_total:
                pygame.draw.line(self.screen, PANEL_DIVIDER, (x + pad, y - 4), (x + w - pad, y - 4), 1)
            else:
                pygame.draw.rect(self.screen, col, (x + pad, y + 2, 9, 9))
            lx = x + pad + (0 if is_total else 15)
            self.screen.blit(self.small_font.render(label, True, col if is_total else TEXT_COLOR), (lx, y))
            vs = self.small_font.render(f"{val:+.2f}", True, col)
            self.screen.blit(vs, (x + w - pad - vs.get_width(), y))
            # signed bar from the zero line
            by = y + 16
            pygame.draw.line(self.screen, POTENTIAL_TRACK_COLOR, (x + pad, by), (x + w - pad, by), 1)
            pygame.draw.line(self.screen, (95, 99, 116), (cx, by - 3), (cx, by + 3), 1)
            blen = max(-half, min(half, val / scale * half))
            if abs(blen) >= 1.0:
                pygame.draw.line(self.screen, col, (cx, by), (cx + blen, by), 4 if not is_total else 5)
            y += row_h

    def _draw_cone(self, base_screen, tip_screen, half_w, color):
        """A filled director 'spike': a triangle from a base of width 2*half_w
        (perpendicular to the spike axis) up to the projected tip."""
        dx = tip_screen[0] - base_screen[0]
        dy = tip_screen[1] - base_screen[1]
        d = math.hypot(dx, dy)
        if d < 1e-3:
            return
        px, py = -dy / d * half_w, dx / d * half_w
        pts = [(base_screen[0] + px, base_screen[1] + py),
               (base_screen[0] - px, base_screen[1] - py),
               tip_screen]
        pygame.draw.polygon(self.screen, color, pts)
        pygame.draw.polygon(self.screen, _lerp_color(color, (0, 0, 0), 0.45), pts, 1)

    def _draw_arrow_3d(self, camera, anchor_world, vec_world, color, knee):
        """Force arrow: project the anchor and a short world-space step along
        vec_world, then draw a tanh-scaled screen arrow (like the 2D path)."""
        mag = float(np.linalg.norm(vec_world))
        if mag < 1e-6:
            return
        start, _, _ = camera.project_point(anchor_world)
        tip_world = np.asarray(anchor_world, dtype=float) + np.asarray(vec_world, dtype=float) / mag * 0.5
        tip, _, _ = camera.project_point(tip_world)
        dirx, diry = tip[0] - start[0], tip[1] - start[1]
        dl = math.hypot(dirx, diry)
        if dl < 1e-6:
            return
        length_px = VECTOR_MAX_PX * math.tanh(mag / knee)
        if length_px < 2.0:
            return
        ux, uy = dirx / dl, diry / dl
        end = (start[0] + ux * length_px, start[1] + uy * length_px)
        pygame.draw.line(self.screen, color, start, end, 3)
        angle = math.atan2(uy, ux)
        for sign in (-1, 1):
            ha = angle + math.pi - sign * ARROWHEAD_ANGLE
            hx = end[0] + ARROWHEAD_LEN * math.cos(ha)
            hy = end[1] + ARROWHEAD_LEN * math.sin(ha)
            pygame.draw.line(self.screen, color, end, (hx, hy), 3)

    def _draw_torque_arc(self, center, radius, frac, color):
        """A circular arrow around `center` depicting a torque about the control-
        plane normal. The arc starts at the top of the ring and sweeps to the
        right (frac > 0) or left (frac < 0), reaching a full semicircle (top ->
        bottom) at |frac| = 1; the sweep is linear in frac. A positive frac is
        the same screen handedness as a positive yaw command (top edge swings
        right). The arrowhead sits at the moving end so the sense of rotation
        reads directly."""
        if abs(frac) < 0.05:   # hide sub-5%-of-max torque to declutter
            return
        frac = max(-1.0, min(1.0, frac))
        sweep = frac * math.pi   # radians; +/-pi = semicircle
        cx, cy = center
        # Point on the ring at parameter phi: phi=0 is the top; increasing phi
        # moves clockwise on screen (toward +x), matching a positive-y torque.
        n = max(3, int(abs(sweep) / 0.15))
        pts = []
        for k in range(n + 1):
            phi = sweep * k / n
            pts.append((cx + radius * math.sin(phi), cy - radius * math.cos(phi)))
        if len(pts) >= 2:
            # Solid width-`TORQUE_ARC_WIDTH` polyline rather than stacked aalines:
            # pygame's aaline blends its anti-aliased edge toward the destination,
            # which on the transparent (0,0,0,0) overlay surface leaves dark fringe
            # pixels that composite as speckles over the bright GL beads. A plain
            # thick polyline writes solid, full-alpha color -- no fringe.
            pygame.draw.lines(self.screen, color, False, pts, TORQUE_ARC_WIDTH)
        # Arrowhead at the moving end, pointing along the direction of travel.
        end = pts[-1]
        s = 1.0 if sweep >= 0 else -1.0
        tx, ty = s * math.cos(sweep), s * math.sin(sweep)  # unit tangent at the end
        tangent = math.atan2(ty, tx)
        for sign in (-1, 1):
            ha = tangent + math.pi - sign * ARROWHEAD_ANGLE
            hx = end[0] + TORQUE_ARC_HEAD_LEN * math.cos(ha)
            hy = end[1] + TORQUE_ARC_HEAD_LEN * math.sin(ha)
            pygame.draw.line(self.screen, color, end, (hx, hy), TORQUE_ARC_WIDTH)

    def draw_sim_3d(self, positions3d, dipoles3d, is_puller, spec, camera, bonds,
                    input_force, reaction_force, control_grid, fps,
                    sim_time_ps=0.0, puller_energy=None, hud_lines=None,
                    potential_terms=None, torque_signals=None,
                    total_steps=0, steps_per_frame=1, debug_line=None,
                    brightness=None, total_potential_terms=None, box_bounds=None,
                    puller_attached=True, bead_energies=None,
                    box_periodic=None):
        pts = np.asarray(positions3d, dtype=float)
        screen, depth, scale = camera.project(pts)
        # Depth cueing anchored to the scene's own near->far extent, not to an
        # absolute camera distance, so the cue reads the same whether the whole
        # scene sits close (the 7-bead patch) or far (the big sheet). near/far
        # here are the ramp, shared by the beads, bond spokes and director spikes
        # so they all recede together.
        #
        # The GL path takes the ramp from the system's RenderStyle (where the
        # rest of the look lives, and where it can be tuned per system); the CPU
        # fallback, which has none of that machinery, keeps the theme constants.
        finite = np.isfinite(depth)
        dmin = float(np.min(depth[finite])) if np.any(finite) else 0.0
        dmax = float(np.max(depth[finite])) if np.any(finite) else 1.0
        style = spec.render_style
        if self.gl_enabled:
            near, far = style.cue_range(dmin, dmax)
            self._fog_strength = style.cue_strength if style.depth_cue else 0.0
        else:
            near, far = dmin + DEPTH_FADE_START * (dmax - dmin), dmax
            self._fog_strength = HAZE_STRENGTH

        # Physical bead radius (Angstrom/sigma) -> px at each bead's depth. The
        # bead-sized overlays (puller ring, torque arcs, director spikes) key off
        # `radii`, so it must match how the active path actually sizes the beads:
        # the GL impostors scale freely with perspective (no cap), while the CPU
        # sprites are capped at 90 px. Clamping `radii` for the GL path -- as the
        # earlier shared clip did -- left the ring/arcs stuck small while the
        # beads grew in fullscreen (the reported bug).
        phys_r = spec.atom_radius_A or 0.5
        order = np.argsort(-depth)   # far -> near (painter's ordering)

        if self.gl_enabled:
            # GPU path: beads + depth-occluded bond/net lines are rendered by the
            # GL scene into the sim viewport; the 2D overlays go on the
            # (transparent) offscreen surface and are composited over the top.
            radii = np.maximum(phys_r * scale, 2.0)   # unclamped: matches GL beads
            self.screen.fill((0, 0, 0, 0))
            self._render_gl_3d(camera, pts, dipoles3d, spec, bonds, control_grid,
                               depth, near, far, (dmin, dmax), brightness,
                               box_bounds, bead_energies, box_periodic)
            # Director spikes and the puller ring stay 2D overlays (they poke out
            # the near pole, so drawing on top without depth occlusion reads fine).
            # The spikes are batch-projected -- per-bead projection would be a few
            # thousand tiny numpy calls on the 900-bead sheet and dominate the frame.
            if spec.director_arrows:
                self._draw_director_arrows_batch(camera, pts, dipoles3d, phys_r,
                                                 radii, depth, near, far, order)
            for i in order:
                if np.isfinite(depth[i]) and is_puller[i]:
                    cx, cy = int(screen[i][0]), int(screen[i][1])
                    pygame.draw.circle(self.screen,
                                       self._puller_ring_color(puller_attached),
                                       (cx, cy), int(radii[i]) + 2, PULLER_RING_WIDTH)
        else:
            radii = np.clip(phys_r * scale, 3, 90)   # capped: matches CPU sprites
            self._draw_sim_3d_cpu(pts, dipoles3d, is_puller, spec, camera, bonds,
                                  control_grid, screen, depth, near, far, phys_r,
                                  radii, order, brightness, box_bounds,
                                  puller_attached)

        self._draw_3d_overlays(camera, pts, screen, depth, radii, is_puller, spec,
                               input_force, reaction_force, fps, sim_time_ps,
                               total_steps, steps_per_frame, potential_terms,
                               torque_signals, hud_lines, debug_line,
                               total_potential_terms)

    # ---- GPU scene: hand the beads + occluded lines to the GL pipeline ------

    def _render_gl_3d(self, camera, pts, dipoles3d, spec, bonds, control_grid,
                      depth, near, far, depth_range, brightness=None,
                      box_bounds=None, energies=None, box_periodic=None):
        """Render the beads (and depth-occluded bond/net lines) with the GL scene
        and blit the result into the on-screen sim viewport. `near`/`far` are the
        depth-cue ramp used for the LINES; `depth_range` is the scene's raw
        (nearest, farthest) bead distance, from which the GL scene places its own
        cue ramp and focus plane. The projection near/far are taken a touch wider
        than the actual bead extent so no front cap is clipped."""
        W, H = self.sim_width, self.window_size[1]
        view = view_matrix(camera.eye, camera.right, camera.true_up, camera.forward)
        fx = camera.focal / (camera.viewport_w / 2.0)
        fy = camera.focal / (camera.viewport_h / 2.0)
        phys_r = spec.atom_radius_A or 0.5

        # Beads to draw = the real beads plus opaque wrapped ghost copies near
        # periodic seams; brightness spotlights any tagged cluster. For a periodic
        # scene the shader clips beads to the box faces (all opaque), so a bead
        # crossing a seam slides across continuously with no transparency; the soft
        # edge is a screen-space vignette in the composite, not a per-bead fade.
        style = spec.render_style
        tiling = (box_bounds is not None and box_periodic is not None
                  and any(float(n) > 0.0 and per
                          for n, per in zip(style.periodic_images, box_periodic)))
        if tiling:
            # The images ARE the seam handling: no clipping to the cell faces and
            # no wrapped ghosts, because every neighbouring cell is drawn whole.
            bfade = None
            bpts, bdips, bbright, benergy, bfade = self._periodic_image_instances(
                np.asarray(pts, dtype=float), np.asarray(dipoles3d, dtype=float),
                np.ones(len(pts)) if brightness is None
                else np.asarray(brightness, dtype=float),
                np.zeros(len(pts)) if energies is None
                else np.asarray(energies, dtype=float),
                style, box_bounds, box_periodic)
            box_half, edge_fade_on = None, False
        else:
            bpts, bdips, bbright, benergy = self._wrap_ghost_instances(
                pts, dipoles3d, brightness, spec, energies)
            bfade = None
            periodic = bool(spec.wrap_fade_fraction) and self.box_x is not None
            box_half = (self.box_x / 2.0, self.box_y / 2.0) if periodic else None
            edge_fade_on = periodic
        radii = np.full(len(bpts), phys_r, dtype=np.float32)

        # The frustum and the depth span are measured over what is ACTUALLY
        # drawn, not over the real cell alone. With periodic images that is the
        # difference between an infinite-looking membrane and one whose front and
        # back rows are sliced off by the near and far planes -- the images
        # nearest the eye sit well in front of anything in the real cell.
        _, bdepth, _ = camera.project(bpts)
        bfinite = np.isfinite(bdepth)
        dmin = float(np.min(bdepth[bfinite])) if np.any(bfinite) else 1.0
        dmax = float(np.max(bdepth[bfinite])) if np.any(bfinite) else 10.0
        depth_range = (dmin, dmax)
        near_clip = dmin - phys_r - 1.0
        far_clip = dmax + phys_r + 1.0
        # The box outline can extend nearer/farther than any bead; widen the clip
        # planes to enclose its corners so its edges aren't sliced at the frustum.
        if box_bounds is not None:
            xlo, xhi, ylo, yhi, zlo, zhi = box_bounds
            corners = np.array([[x, y, z] for x in (xlo, xhi)
                                for y in (ylo, yhi) for z in (zlo, zhi)])
            _, cdep, _ = camera.project(corners)
            cfin = cdep[np.isfinite(cdep)]
            if len(cfin):
                near_clip = min(near_clip, float(cfin.min()) - 1.0)
                far_clip = max(far_clip, float(cfin.max()) + 1.0)
        proj = proj_matrix(fx, fy, max(0.05, near_clip), far_clip)
        verts, cols, ov_verts, ov_cols = self._build_gl_lines(
            camera, pts, spec, bonds, control_grid, depth, near, far, box_bounds,
            box_on_top=tiling)
        # Screen-space edge fade on periodic scenes (the sheet): softens the
        # frame edge / clip seam uniformly across the whole depth column, instead
        # of the old per-bead world-face fade. The white box outline is drawn
        # after the composite, so it stays bright.
        edge_fade = EDGE_VIGNETTE_STRENGTH if edge_fade_on else 0.0
        self.gl_scene.render(view, proj, bpts, radii, bdips, depth_range,
                             verts, cols, brights=bbright, box_half=box_half,
                             edge_fade=edge_fade, style=style,
                             bead_radius=phys_r, focal_px=camera.focal,
                             energies=benergy if energies is not None else None,
                             fades=bfade, overlay_verts=ov_verts,
                             overlay_cols=ov_cols)
        # Sim viewport is the left sim_width columns, full height (GL origin is
        # bottom-left, so its y origin is 0).
        self.gl_scene.blit_to_viewport(0, 0, W, H)

    def _periodic_image_instances(self, pts, dips, bright, energy, style,
                                  box_bounds, box_periodic):
        """Every bead, repeated over the cell's periodic images, with a per-bead
        strength that trails off with distance from the real cell.

        A periodic cell is a window onto an infinite system; drawing one copy of
        it says otherwise. This draws `style.periodic_images` copies along each
        periodic axis, so the sheet reads as a piece of an endless membrane
        rather than a small square raft, and fades them with distance so the
        tiling has no outer edge of its own.

        Counts are per axis, and may be given per SIDE and FRACTIONALLY: 0.5 is
        half of the neighbouring cell, and (0.5, 3) reaches half a cell one way
        and three cells the other. That is what puts the real cell -- the one
        carrying the controlled particle, its control net and the box outline --
        near the front of the block, with a little in front of it and a long tail
        of copies receding behind, rather than buried in the middle of its own
        copies or hanging off the front edge of them. Declared rather than
        derived from the camera on purpose: worked out per frame it would flip
        from one side to the other as the view swung past a right angle, and the
        whole block would jump.

        Returns (positions, directors, brightness, energies, fades). No clipping
        and no seam ghosts are needed alongside this: a bead poking out of the
        real cell is exactly the neighbouring copy's bead poking in, which is
        what the true infinite tiling looks like."""
        def sides(entry, periodic):
            """(before, after) copies along one axis. A single number means the
            same either side; a pair says how far the block reaches in each
            direction, which is what lets the real cell sit near the front of a
            long tail of copies without there being NOTHING in front of it."""
            if not periodic:
                return 0.0, 0.0
            if isinstance(entry, (tuple, list, np.ndarray)):
                return float(entry[0]), float(entry[1])
            return float(entry), float(entry)

        reach = np.ones((3, 2))                # per axis, per side, in half-widths
        whole = [[0, 0], [0, 0], [0, 0]]
        for axis, (entry, per) in enumerate(zip(style.periodic_images, box_periodic)):
            before, after = sides(entry, per)
            # In half-widths, with the real cell spanning [-1, 1]: n copies out
            # from a boundary reach 1 + 2n. Fractions are allowed and are what
            # make a partial copy mean something -- 0.5 is half a neighbouring
            # cell -- so whole copies are generated and then clipped to this.
            reach[axis] = (1.0 + 2.0 * before, 1.0 + 2.0 * after)
            whole[axis] = [int(math.ceil(before)), int(math.ceil(after))]

        lo = np.array(box_bounds[0::2], dtype=float)
        hi = np.array(box_bounds[1::2], dtype=float)
        lengths, centre = hi - lo, 0.5 * (lo + hi)
        half = 0.5 * lengths

        offsets = [np.array([i, j, k], dtype=float) * lengths
                   for i in range(-whole[0][0], whole[0][1] + 1)
                   for j in range(-whole[1][0], whole[1][1] + 1)
                   for k in range(-whole[2][0], whole[2][1] + 1)]
        P = np.concatenate([pts + o for o in offsets])
        n_img = len(offsets)
        D = np.tile(dips, (n_img, 1))
        B = np.tile(bright, n_img)
        E = np.tile(energy, n_img)

        # How far out each bead is, as a FRACTION of the copies drawn in that
        # direction: 0 anywhere in the real cell, 1 at the outer edge of the
        # block. Per axis and per side, because the two sides can reach
        # different distances, and the furthest-out axis wins.
        #
        # Expressed as a fraction rather than in half-widths so the fade cannot
        # be left promising range that is not drawn: at 1.0 it reaches the
        # background exactly where the copies are cut, whatever the counts, and
        # there is no straight edge to see. That is the one thing this has to get
        # right -- an abrupt boundary is worse than no images at all.
        signed = (P - centre) / half
        side = (signed > 0.0).astype(int)
        limit = np.take_along_axis(reach[None, :, :].repeat(len(P), 0),
                                   side[:, :, None], axis=2)[:, :, 0]
        out = np.clip((np.abs(signed) - 1.0) / np.maximum(limit - 1.0, 1e-9),
                      0.0, None)
        u = np.max(np.where(limit > 1.0, out, 0.0), axis=1)
        span = max(style.image_fade_end - style.image_fade_start, 1e-6)
        F = np.clip(1.0 - (u - style.image_fade_start) / span, 0.0, 1.0)

        # Clip to the extent asked for, one axis and side at a time, and drop
        # anything that has faded out entirely: a fully faded copy is
        # background-coloured but still writes depth, so it would punch a hole in
        # whatever is behind it.
        keep = (F > 0.01) & np.all(np.abs(signed) <= limit + 1e-9, axis=1)
        return P[keep], D[keep], B[keep], E[keep], F[keep]

    def _wrap_ghost_instances(self, pts, dips, brightness, spec, energies=None):
        """Real beads + OPAQUE wrapped ghost copies near the periodic x/y seams.

        Returns (positions, directors, brightness, energies), one entry per real bead
        followed by ghost entries for beads near a seam. A bead within about a
        bead-radius of a seam also gets an opaque copy at the wrapped position;
        the renderer clips every bead to the box faces (GL) / darkens the box edge
        (both paths), so a bead crossing a seam has its sliced area move
        continuously to its ghost on the opposite face -- no fade, no transparency
        (which is ill-posed for the dense, overlapping seam beads). Corners near
        both seams also get a diagonal ghost. Non-periodic scenes return the real
        beads unchanged."""
        pts = np.asarray(pts, dtype=float)
        dips = np.asarray(dips, dtype=float)
        n = len(pts)
        bright = np.ones(n) if brightness is None else np.asarray(brightness, dtype=float)
        # A ghost is the same bead seen through the periodic boundary, so it
        # carries the same per-bead quantities -- brightness and energy ride
        # along by construction.
        energy = np.zeros(n) if energies is None else np.asarray(energies, dtype=float)
        frac = spec.wrap_fade_fraction
        if not frac or frac <= 0 or self.box_x is None:
            return pts, dips, bright, energy

        # A bead within `band` of a seam is copied to the wrapped position. The
        # band is a bit over one bead radius so the ghost already exists by the
        # time its clipped cap becomes visible at the opposite face.
        r = spec.atom_radius_A or 0.5

        def axis_ghost(coord, half):
            band = min(1.5 * r, 0.49 * half)
            shift = np.zeros(n)
            hi = coord > half - band
            lo = coord < -(half - band)
            shift[hi] = -2.0 * half
            shift[lo] = 2.0 * half
            return shift, (hi | lo)

        sx, near_x = axis_ghost(pts[:, 0], self.box_x / 2.0)
        sy, near_y = axis_ghost(pts[:, 1], self.box_y / 2.0)

        groups = [(pts, dips, bright, energy)]   # the real beads

        def add_ghost(mask, dx, dy):
            if not np.any(mask):
                return
            p = pts[mask].copy()
            p[:, 0] += dx[mask]
            p[:, 1] += dy[mask]
            groups.append((p, dips[mask], bright[mask], energy[mask]))

        zero = np.zeros(n)
        add_ghost(near_x, sx, zero)
        add_ghost(near_y, zero, sy)
        add_ghost(near_x & near_y, sx, sy)

        return tuple(np.concatenate([g[k] for g in groups]) for k in range(4))

    def _build_gl_lines(self, camera, pts, spec, bonds, control_grid, depth, near,
                        far, box_bounds=None, box_on_top=False):
        """World-space line vertices (M,3) and per-vertex rgba (M,4, 0..1) for the
        bond spokes, the control-plane net, and the simulation-box outline, fogged
        like the CPU path. Returned as GL line-list pairs; the depth test against
        the beads does the occlusion (replacing the CPU z-buffer).

        Returns (verts, cols, overlay_verts, overlay_cols). `box_on_top` moves the
        box outline into the second, un-depth-tested set: once the cell is drawn
        surrounded by its own periodic images, the images bury the outline, and
        the one line that says WHERE THE REAL CELL IS is the one that must not be
        hidden by copies of it."""
        verts, cols = [], []
        box_verts, box_cols = [], []
        d_opt = spec.lattice_spacing
        lam = BOND_FALLOFF * d_opt
        max_offset = lam * math.log(max(BOND_PEAK_ALPHA / BOND_MIN_ALPHA, 1.0001))
        for a, b in (bonds or []):
            offset = abs(float(np.linalg.norm(pts[a] - pts[b])) - d_opt)
            if offset > max_offset:
                continue
            alpha = BOND_PEAK_ALPHA * math.exp(-offset / lam)
            if alpha < BOND_MIN_ALPHA:
                continue
            for idx in (a, b):
                fog = self._fog(depth[idx], near, far)
                r, g, bb = _lerp_color(BOND_3D_COLOR, HAZE_COLOR, fog)
                verts.append(pts[idx])
                cols.append((r / 255.0, g / 255.0, bb / 255.0, alpha / 255.0))
        if control_grid is not None:
            for pa, pb in self._net_world_segments(control_grid):
                for p in (pa, pb):
                    _, dp, _ = camera.project_point(p)
                    fog = self._fog(dp, near, far) if np.isfinite(dp) else 0.0
                    r, g, bb = _lerp_color(NET_COLOR, HAZE_COLOR, fog)
                    verts.append(np.asarray(p, dtype=float))
                    cols.append((r / 255.0, g / 255.0, bb / 255.0, NET_LINE_ALPHA / 255.0))
        if box_bounds is not None:
            near_alpha = self._box_edge_alpha(box_bounds, camera)
            bv, bc = (box_verts, box_cols) if box_on_top else (verts, cols)
            for pa, pb in self._box_edge_segments(box_bounds, camera):
                for p in (pa, pb):
                    _, dp, _ = camera.project_point(p)
                    # Un-fogged when it is on top: the depth cue exists to push
                    # SCENERY into the distance, and once the outline is the only
                    # thing saying where the real cell is among its images, it is
                    # annotation rather than scenery. Fogged against a tiled
                    # scene's depth span it all but vanished.
                    fog = (0.0 if box_on_top else
                           (self._fog(dp, near, far) if np.isfinite(dp) else 0.0))
                    r, g, bb = _lerp_color(BOX_3D_COLOR, HAZE_COLOR, fog)
                    bv.append(np.asarray(p, dtype=float))
                    # Per-VERTEX alpha, so the near corner dissolves along the
                    # edges rather than the whole edge switching off.
                    bc.append((r / 255.0, g / 255.0, bb / 255.0,
                               BOX_3D_ALPHA / 255.0 * near_alpha(dp)))

        def arrays(v, c):
            if not v:
                return None, None
            return np.array(v, dtype=np.float32), np.array(c, dtype=np.float32)
        return arrays(verts, cols) + arrays(box_verts, box_cols)

    # ---- CPU scene (fallback when no GL context is available) ---------------

    def _draw_sim_3d_cpu(self, pts, dipoles3d, is_puller, spec, camera, bonds,
                         control_grid, screen, depth, near, far, phys_r, radii, order,
                         brightness=None, box_bounds=None, puller_attached=True):
        """Numpy-shaded sphere sprites (painter-sorted), bond lines, and the
        z-buffer-occluded net + box outline -- the original CPU renderer, kept as
        the fallback for machines without an OpenGL 3.3 context. Draws onto
        self.screen (the display surface in fallback mode)."""
        self.screen.fill(BG)
        net_segs = self._net_world_segments(control_grid) if control_grid is not None else None

        # Connectivity lines, behind the beads: alpha peaks at d_opt and falls off
        # exponentially as the pair stretches/compresses (a spoke fades as its
        # bead is pulled off the sheet). Drawn on a per-pixel-alpha surface.
        d_opt = spec.lattice_spacing
        lam = BOND_FALLOFF * d_opt
        max_offset = lam * math.log(max(BOND_PEAK_ALPHA / BOND_MIN_ALPHA, 1.0001))
        self.bond_surface.fill((0, 0, 0, 0))
        for a, b in bonds:
            offset = abs(float(np.linalg.norm(pts[a] - pts[b])) - d_opt)
            if offset > max_offset:
                continue
            alpha = int(BOND_PEAK_ALPHA * math.exp(-offset / lam))
            if alpha < BOND_MIN_ALPHA:
                continue
            fog = self._fog(0.5 * (depth[a] + depth[b]), near, far)
            r, g, bb = _lerp_color(BOND_3D_COLOR, HAZE_COLOR, fog)
            w = max(1, int(round(0.28 * min(radii[a], radii[b]))))
            pygame.draw.line(self.bond_surface, (r, g, bb, alpha), screen[a], screen[b], w)
        self.screen.blit(self.bond_surface, (0, 0))

        # Beads to draw = the real beads plus opaque wrapped ghost copies near
        # periodic seams; real beads come first, so is_puller applies to the
        # leading n and ghosts carry no ring. brightness spotlights any tagged
        # cluster. (This fallback always draws the director banding -- the energy
        # colouring is a shader path, and this exists for machines with no GL.)
        bpts, bdips, bbright, _ = self._wrap_ghost_instances(pts, dipoles3d,
                                                            brightness, spec)
        m = len(bpts)
        aug_pull = np.zeros(m, dtype=bool)
        aug_pull[:len(pts)] = np.asarray(is_puller, dtype=bool)
        bscreen, bdepth, bscale = camera.project(bpts)
        bradii = np.clip(phys_r * bscale, 3, 90)
        # Director expressed in the view frame (right, screen-down, toward-viewer),
        # per bead, so the banded coloring tilts correctly on screen.
        rgt, tup, fwd = camera.right, camera.true_up, camera.forward
        dv = np.column_stack([bdips @ rgt, -(bdips @ tup), -(bdips @ fwd)])
        border = np.argsort(-bdepth)   # far -> near painter ordering

        # Opaque periodic-edge vignette (this fallback can't clip sphere sprites in
        # the shader, so it approximates the GL path by darkening beads toward the
        # background near the box faces). Computed on the wrapped-into-box position
        # so ghosts get the vignette of the face they poke through.
        edgefade = np.zeros(m)
        if spec.wrap_fade_fraction and self.box_x is not None:
            hx, hy = self.box_x / 2.0, self.box_y / 2.0
            eband = 2.0 * phys_r
            wx = (bpts[:, 0] + hx) % (2.0 * hx) - hx
            wy = (bpts[:, 1] + hy) % (2.0 * hy) - hy
            d_edge = np.minimum(hx - np.abs(wx), hy - np.abs(wy))
            edgefade = np.clip(1.0 - d_edge / eband, 0.0, 1.0)

        for i in border:
            if not np.isfinite(bdepth[i]):
                continue
            fog = self._fog(bdepth[i], near, far)
            # Fold the opaque edge vignette into the fog darken (both blend the
            # sprite toward the background), keeping the sprite fully opaque.
            total_fade = fog + (1.0 - fog) * float(edgefade[i])
            r = int(bradii[i])
            cx, cy = int(bscreen[i][0]), int(bscreen[i][1])
            sprite = self._banded_sphere_sprite(r, dv[i], total_fade, float(bbright[i]))
            self.screen.blit(sprite, (cx - r, cy - r))
            if aug_pull[i]:
                pygame.draw.circle(self.screen, self._puller_ring_color(puller_attached),
                                   (cx, cy), r + 2, PULLER_RING_WIDTH)

        # Director spikes on top (batch-projected -- see the GL path), so the
        # 900-bead sheet's arrows don't cost thousands of per-bead projections.
        if spec.director_arrows:
            self._draw_director_arrows_batch(camera, pts, dipoles3d, phys_r, radii,
                                             depth, near, far, order)

        # Control-plane net and the box outline, with true depth occlusion against
        # a per-pixel bead z-buffer: visible over empty background and up to each
        # sphere silhouette. The z-buffer is built once over the combined screen
        # region of both so a single pass serves both draws.
        box_segs = self._box_edge_segments(box_bounds, camera) if box_bounds is not None else None
        box_alpha = (self._box_edge_alpha(box_bounds, camera)
                     if box_bounds is not None else None)
        occ_segs = list(net_segs or []) + list(box_segs or [])
        if occ_segs:
            opts = np.array([p for seg in occ_segs for p in seg], dtype=float)
            oscr, _, _ = camera.project(opts)
            finite = np.isfinite(oscr[:, 0]) & np.isfinite(oscr[:, 1])
            if np.any(finite):
                xs, ys = oscr[finite, 0], oscr[finite, 1]
                region = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
            else:
                region = None
            zbuf = self._build_bead_zbuffer(screen, depth, radii, phys_r, region)
            if net_segs is not None:
                self._draw_net_occluded(camera, net_segs, zbuf)
            if box_segs is not None:
                self._draw_box_occluded(camera, box_segs, zbuf, near, far,
                                        box_alpha)

    # ---- shared 2D overlays over the 3D scene (both GL and CPU paths) --------

    def _draw_3d_overlays(self, camera, pts, screen, depth, radii, is_puller, spec,
                          input_force, reaction_force, fps, sim_time_ps,
                          total_steps, steps_per_frame, potential_terms,
                          torque_signals, hud_lines, debug_line,
                          total_potential_terms=None):
        # Force arrows at the puller (map control-plane (x,z) -> world x,z).
        puller_idx = int(np.argmax(is_puller)) if np.any(is_puller) else 0
        anchor = pts[puller_idx]
        knee = spec.force_feedback.ff_knee
        ivec = np.array([input_force[0], 0.0, input_force[1]])
        rvec = np.array([reaction_force[0], 0.0, reaction_force[1]])
        self._draw_arrow_3d(camera, anchor, rvec, REACTION_VEC_COLOR, knee)
        self._draw_arrow_3d(camera, anchor, ivec, INPUT_VEC_COLOR, knee)
        # Circular torque arrows around the puller, both about the control-plane
        # normal (in-plane director rotation): green = the user's steering torque,
        # red = the membrane's restoring torque. Both signals are the component
        # already projected onto the net's plane (see get_torque_signals).
        if torque_signals is not None:
            pcx, pcy = int(screen[puller_idx][0]), int(screen[puller_idx][1])
            r_px = float(radii[puller_idx])
            applied, reaction = torque_signals
            self._draw_torque_arc((pcx, pcy), r_px * TORQUE_ARC_REACTION_RADIUS,
                                  reaction, REACTION_VEC_COLOR)
            self._draw_torque_arc((pcx, pcy), r_px * TORQUE_ARC_APPLIED_RADIUS,
                                  applied, INPUT_VEC_COLOR)
        # (The pulled bead's energy is shown as the additive-potential panel in
        # the upper-left, and KE/PE in the right instrumentation panel -- no
        # floating label over the beads, which cluttered the scene and whose
        # per-atom PE didn't match the panel's full-bond term totals.)

        ix, iy = input_force
        rx, ry = reaction_force
        sim_time_str = units.format_sim_time(sim_time_ps, spec.reduced_units)
        # Input/reaction torque (director twist about the control-plane normal),
        # shown alongside the forces. torque_signals are the fractions [-1, 1] the
        # torque arcs use (green = your twist, red = membrane restoring twist).
        torque_str = ""
        if torque_signals is not None:
            applied, reaction = torque_signals
            torque_str = (f"input torque: {applied:+.2f}   "
                          f"membrane torque: {reaction:+.2f}   ")
        label = self.font.render(
            f"{spec.name}  |  sim time: {sim_time_str}   steps: {total_steps:,} ({steps_per_frame}/frame)   "
            f"input force: ({ix:4.1f}, {iy:4.1f}){units.force_unit(spec.reduced_units)}   "
            f"membrane force: ({rx:5.1f}, {ry:5.1f})   "
            f"{torque_str}fps: {fps:4.0f}",
            True, (200, 200, 200),
        )
        self.screen.blit(label, (10, 10))
        legend = self.font.render(
            "green = your pull/twist, red = membrane reaction   |   drag the center bead (WASD/mouse); twist / Q-E / L-R click rotates its director",
            True, (140, 140, 140),
        )
        self.screen.blit(legend, (10, 30))
        # Puller-bead breakdown on the left; the whole-system total (if the system
        # supplies one) as a second panel just to its right.
        self._draw_potential_panel(potential_terms, x=12)
        if total_potential_terms is not None:
            self._draw_potential_panel(total_potential_terms, x=336)
        self._draw_hud(hud_lines)
        self._draw_debug_line(debug_line)

    def draw_sim(self, positions, is_puller, puller_pos, input_force, reaction_force,
                 fps, spec, heat_fraction=0.0, sim_time_ps=0.0, atom_trails=None,
                 species=None, bond_pairs=None, puller_energy=None, hbond_pairs=None,
                 hud_lines=None, total_steps=0, steps_per_frame=1, debug_line=None,
                 puller_attached=True):
        self.screen.fill(BG)

        top_left = self.sim_to_screen(0, self.box_y)
        size = (self.box_x * self.scale, self.box_y * self.scale)
        pygame.draw.rect(self.screen, BOX_OUTLINE, (*top_left, *size), width=1)

        if atom_trails is not None:
            self._draw_trails(atom_trails, self._crystal_trail_color(spec), PULLER_RING_COLOR)

        # Precompute screen coords once -- bonds, atoms and the puller all index
        # into the same get_all_positions ordering.
        pts = [self.sim_to_screen(x, y) for (x, y) in positions]

        # Generic "near-equilibrium" faint bond overlay (crystals). Systems that
        # supply their own explicit bonds (lipids) turn this off in their spec.
        if spec.bond_overlay:
            self._draw_bonds(positions, spec.lattice_spacing)

        # Explicit molecular backbones (e.g. each lipid's head-tail-tail chain),
        # drawn as sticks beneath the beads. Bonds touching the control lipid
        # (the puller) are drawn brighter so "your lipid" stands out.
        # Transient hydrogen bonds (water), drawn beneath everything as light
        # dashed lines so they read as weak/breakable, not solid sticks. Both use
        # _bond_segments so a bond spanning a periodic seam wraps correctly toward
        # each edge instead of streaking across the box. Clip to the sim view so
        # a wrap stub can't spill onto the instrumentation panel.
        prev_clip = self.screen.get_clip()
        self.screen.set_clip(pygame.Rect(0, 0, self.sim_width, self.window_size[1]))
        if hbond_pairs is not None:
            for a, b in hbond_pairs:
                for p0, p1 in self._bond_segments(*positions[a], *positions[b]):
                    self._draw_dashed(p0, p1, HBOND_COLOR, HBOND_WIDTH, HBOND_DASH)

        if bond_pairs is not None:
            for a, b in bond_pairs:
                color = PULLER_BOND_COLOR if (is_puller[a] or is_puller[b]) else BOND_STICK_COLOR
                for p0, p1 in self._bond_segments(*positions[a], *positions[b]):
                    pygame.draw.line(self.screen, color, p0, p1, 2)
        self.screen.set_clip(prev_clip)

        labels = spec.species_labels

        # Crystal/membrane atoms, drawn in their species/material color and at
        # their real relative size (see _atom_radius_px). Single-species systems
        # (Cu, Ar) use the system's flat crystal color. Puller atoms are
        # collected and drawn afterwards, on top.
        puller_pts = []
        puller_species = []
        for i, p in enumerate(is_puller):
            sx, sy = pts[i]
            sp = int(species[i]) if species is not None else None
            if p:
                puller_pts.append((sx, sy))
                puller_species.append(sp)
                continue
            pygame.draw.circle(self.screen, self._species_color(spec, sp), (sx, sy),
                               self._atom_radius_px(spec, sp))
            if labels is not None and sp is not None:
                self._blit_glyph(sx, sy, labels[sp])

        # Puller anchor for the force-vector arrows: its single position, or the
        # centroid of the control lipid's beads. A 2D scene can legitimately have
        # no controlled atom at all (a playground running in sim mode), in which
        # case there is nothing to anchor arrows to and nothing to ring below.
        puller_xy = None
        if puller_pts:
            ax = sum(q[0] for q in puller_pts) / len(puller_pts)
            ay = sum(q[1] for q in puller_pts) / len(puller_pts)
            puller_xy = (ax, ay)
        elif puller_pos is not None:
            puller_xy = self.sim_to_screen(*puller_pos)

        if puller_xy is not None:
            knee = spec.force_feedback.ff_knee
            self._draw_arrow(puller_xy, input_force, INPUT_VEC_COLOR, knee)
            self._draw_arrow(puller_xy, reaction_force, REACTION_VEC_COLOR, knee)

        label_r = 0
        if puller_xy is None:
            pass          # nothing controlled: no ring, no glyph, no energy stamp
        elif len(puller_pts) > 1:
            # Multi-bead control lipid. Its beads are drawn in their true
            # head/tail species colors (like every other lipid) and linked by
            # the brighter backbone above; what marks it as YOURS is the bright
            # ring around the head (which also shows which way the lipid points
            # -- the orientation the user steers with yaw).
            for (sx, sy), sp in zip(puller_pts, puller_species):
                is_head = (sp == 0)
                r = self._atom_radius_px(spec, sp) + (PULLER_RADIUS_BOOST if is_head else 1)
                pygame.draw.circle(self.screen, self._species_color(spec, sp), (sx, sy), r)
                pygame.draw.circle(self.screen, self._puller_ring_color(puller_attached),
                                   (sx, sy), r, PULLER_RING_WIDTH if is_head else 1)
            label_r = self._atom_radius_px(spec, 0) + PULLER_RADIUS_BOOST
        else:
            # Single-atom puller (Cu, Ar, NaCl), drawn in its true species/
            # material color -- a deposited Cu atom is Cu, a pulled Na+ is a
            # cation -- with the bright control ring marking it as the one you
            # steer. No fake wobble is added: the atom's real thermal motion is
            # already in its live simulated position, so it jitters on its own
            # exactly as much as the physics says it should.
            sp0 = puller_species[0] if puller_species else None
            r = self._atom_radius_px(spec, sp0) + PULLER_RADIUS_BOOST
            pygame.draw.circle(self.screen, self._species_color(spec, sp0), puller_xy, r)
            pygame.draw.circle(self.screen, self._puller_ring_color(puller_attached),
                               puller_xy, r, PULLER_RING_WIDTH)
            # In a species system the puller is also (e.g.) an ion -- stamp its
            # glyph so its role is legible.
            if labels is not None and sp0 is not None:
                self._blit_glyph(puller_xy[0], puller_xy[1], labels[sp0])
            label_r = r

        # Kinetic and potential energy of the controlled atom, in eV, stamped
        # just beneath it on two lines -- KE (its motion) and PE (how bound it is,
        # the negative-when-attracted number) shown separately rather than summed,
        # over a faint backdrop so they stay legible on top of atoms and bonds.
        if (puller_xy is not None and puller_energy is not None
                and puller_energy[0] is not None):
            red = spec.reduced_units
            e_lines = [f"KE = {units.format_energy(puller_energy[0], red)}",
                       f"PE = {units.format_energy(puller_energy[1], red)}"]
            surfs = [self.small_font.render(t, True, PULLER_LABEL_COLOR) for t in e_lines]
            ey = int(puller_xy[1]) + label_r + 10
            for surf in surfs:
                rect = surf.get_rect(center=(int(puller_xy[0]), ey))
                bg = rect.inflate(6, 2)
                pad = pygame.Surface(bg.size, pygame.SRCALPHA)
                pad.fill(PULLER_LABEL_BG)
                self.screen.blit(pad, bg.topleft)
                self.screen.blit(surf, rect)
                ey += surf.get_height() + 1

        ix, iy = input_force
        rx, ry = reaction_force
        sim_time_str = units.format_sim_time(sim_time_ps, spec.reduced_units)
        fu = units.force_unit(spec.reduced_units)
        label = self.font.render(
            f"{spec.name}  |  sim time: {sim_time_str}   steps: {total_steps:,} ({steps_per_frame}/frame)   "
            f"input force: ({ix:4.1f}, {iy:4.1f}){fu}   "
            f"interaction force: ({rx:5.1f}, {ry:5.1f}){fu}   fps: {fps:4.0f}",
            True, (200, 200, 200),
        )
        self.screen.blit(label, (10, 10))

        legend = self.font.render(
            f"green = your input force, red = {spec.element_label} interaction (reaction) force",
            True, (140, 140, 140),
        )
        self.screen.blit(legend, (10, 30))

        self._draw_hud(hud_lines)
        self._draw_debug_line(debug_line)

    def draw_playback_controls(self, playing):
        """Play / Pause / Reset buttons centered along the bottom of the sim view
        (playback systems only). The button matching the current run state is
        highlighted: Play while running, Pause while stopped. Reset never latches.
        Positions the button rects so the app can hit-test clicks (playback_hit)."""
        bw, bh, gap = 96, 34, 12
        total = 3 * bw + 2 * gap
        x0 = (self.sim_width - total) // 2
        y0 = self.window_size[1] - bh - 16
        active = {"play": playing, "pause": not playing, "reset": False}
        for i, btn in enumerate(self.playback_buttons):
            btn.rect = pygame.Rect(x0 + i * (bw + gap), y0, bw, bh)
            btn.draw(self.screen, self.font, active=active[btn.name])
        self._playback_visible = True

    def playback_hit(self, pos):
        """Name of the playback button under `pos` ("play"/"pause"/"reset"), or
        None. Returns None while the controls aren't shown, so a click never hits
        a stale button rect from a system that has since been switched away."""
        if not self._playback_visible:
            return None
        for btn in self.playback_buttons:
            if btn.hit(pos):
                return btn.name
        return None

    def _draw_bead_color_toggle(self, x, y, w, spec):
        """The bead-colouring toggle plus a line saying what the colours mean.
        Returns the y to carry on from. Only for the 3D bead scenes -- the 2D
        crystals colour by species, which is not a choice."""
        self._bead_color_visible = False
        if not spec.render_3d:
            return y
        self._bead_color_visible = True
        energy = self.bead_color_energy
        self.bead_color_button.rect = pygame.Rect(x, y, 210, 26)
        self.bead_color_button.label = ("bead colour: ENERGY" if energy
                                        else "bead colour: DIRECTOR")
        self.bead_color_button.draw(self.screen, self.font, active=energy)
        # What the colours mean, on its own line under the button: a colour scale
        # nobody can read is decoration.
        lo, hi = spec.render_style.energy_range
        caption = ([f"each bead's potential energy, inferno {lo:g} to {hi:g} eps",
                    "dark = tightly bound, bright = strained or free"] if energy else
                   ["director bands: yellow hydrophobic equator, blue poles",
                    "the band tilts with the director, so tilt and splay show"])
        cy = y + 29
        for line in caption + ["white cap marks the +director pole, either way"]:
            self.screen.blit(self.small_font.render(line, True, DIM_TEXT_COLOR),
                             (x, cy))
            cy += 14
        return cy + 6

    def bead_color_hit(self, pos):
        """True if `pos` is on the bead-colouring toggle (and it is on screen)."""
        return self._bead_color_visible and self.bead_color_button.hit(pos)

    def draw_panel(self, systems, current_key, sliders, thermo_now, puller_energy,
                    history, rdf, spec, puller_speed_m_s=None):
        pygame.draw.rect(self.screen, PANEL_BG, self.panel_rect)
        pygame.draw.line(self.screen, PANEL_DIVIDER, (self.panel_rect.x, 0),
                          (self.panel_rect.x, self.window_size[1]), 1)

        x = self.panel_rect.x + PANEL_PAD
        w = PANEL_WIDTH - 2 * PANEL_PAD
        y = 10

        # Compact picker: number + short key (the full name of the active
        # system is shown in the header just below), so all systems fit on one
        # line. The current one is bracketed and drawn brighter.
        picker_bits = []
        for i, (key, sys_spec) in enumerate(systems, start=1):
            picker_bits.append(f"[{i}>{key}]" if key == current_key else f"{i}:{key}")
        picker_surf = self.small_font.render("  ".join(picker_bits), True, DIM_TEXT_COLOR)
        self.screen.blit(picker_surf, (x, y))
        y += 18

        name_surf = self.header_font.render(spec.name, True, HEADER_TEXT_COLOR)
        self.screen.blit(name_surf, (x, y))
        y += name_surf.get_height() + 2

        desc_surf = self.small_font.render(spec.description, True, DIM_TEXT_COLOR)
        self.screen.blit(desc_surf, (x, y))
        y += 16

        # The turntable keys are only listed for the systems that have one --
        # a hint for a key that does nothing is worse than no hint.
        hints = KEY_HINTS + (ORBIT_KEY_HINTS if spec.camera_orbit else "")
        hint_surf = self.small_font.render(hints, True, DIM_TEXT_COLOR)
        self.screen.blit(hint_surf, (x, y))
        y += 20

        y = self._draw_bead_color_toggle(x, y, w, spec)

        pygame.draw.line(self.screen, PANEL_DIVIDER, (x, y), (x + w, y), 1)
        y += 12

        # sliders = (temperature, damping, *extra_sliders). Temperature (always
        # sliders[0], never advanced) carries the melt marker and is drawn first.
        # The rest split into "basic" (drawn in order right after temperature) and
        # "advanced" (hidden behind a collapsible toggle -- see self.show_advanced).
        temp_slider = sliders[0]
        temp_slider.rect = pygame.Rect(x, y, w, 4)
        temp_slider.draw(self.screen, self.font, mark_value=spec.melt_temp, mark_label="melt")
        y += 46

        basic = [s for s in sliders[1:] if not s.advanced]
        advanced = [s for s in sliders[1:] if s.advanced]
        for extra in basic:
            extra.rect = pygame.Rect(x, y, w, 4)
            extra.draw(self.screen, self.font)
            y += 34

        if advanced:
            arrow = "v" if self.show_advanced else ">"
            toggle_surf = self.font.render(
                f"[ {arrow} ] Advanced ({len(advanced)})", True, HEADER_TEXT_COLOR)
            self.screen.blit(toggle_surf, (x, y))
            # Clickable hit-box, padded a little for easy targeting; read back by
            # the app to flip self.show_advanced.
            self.advanced_toggle_rect = pygame.Rect(
                x - 2, y - 2, toggle_surf.get_width() + 8, toggle_surf.get_height() + 6)
            y += toggle_surf.get_height() + 8
            if self.show_advanced:
                for extra in advanced:
                    extra.rect = pygame.Rect(x, y, w, 4)
                    extra.draw(self.screen, self.font)
                    y += 34
            else:
                # Collapsed: park the hidden sliders off-screen so a stale rect
                # from when they were last visible can't be clicked/dragged.
                for extra in advanced:
                    extra.rect = pygame.Rect(-1000, -1000, 0, 0)
        else:
            self.advanced_toggle_rect = None

        # MesoMem runs in reduced (LJ) units, so its readouts and plot axes drop
        # the Kelvin/bar/eV/m-s labels (meaningless here) for the dimensionless
        # reduced quantities; every other system stays in metal units.
        reduced = spec.reduced_units

        temp, press, ke, pe, etotal = thermo_now
        if reduced:
            readout_str = f"instantaneous: T*={temp:6.3f}   P*={press:8.3f}"
        else:
            readout_str = f"instantaneous: T={temp:6.1f} K   P={press:9.1f} bar"
        readout = self.small_font.render(readout_str, True, DIM_TEXT_COLOR)
        self.screen.blit(readout, (x, y))
        y += 18

        puller_ke, puller_pe = puller_energy
        if puller_ke is not None:
            speed_bit = ""
            if puller_speed_m_s is not None:
                # puller_speed_m_s carries A/ps * 100; in reduced units the raw
                # magnitude (A/ps analog = sigma/tau) is that / 100.
                speed_reduced = puller_speed_m_s / units.ANGSTROM_PER_PS_TO_M_PER_S
                if reduced:
                    speed_bit = f"   speed={speed_reduced:.3f} sigma/tau"
                else:
                    speed_bit = f"   speed={puller_speed_m_s:7.1f} m/s ({speed_reduced:.3f} A/ps)"
            if reduced:
                puller_str = f"puller bead:   KE={puller_ke:7.4f}   PE={puller_pe:8.4f}{speed_bit}"
            else:
                puller_str = f"puller atom:   KE={puller_ke:7.4f} eV   PE={puller_pe:8.4f} eV{speed_bit}"
            puller_readout = self.small_font.render(puller_str, True, DIM_TEXT_COLOR)
            self.screen.blit(puller_readout, (x, y))
        y += 20

        pygame.draw.line(self.screen, PANEL_DIVIDER, (x, y), (x + w, y), 1)
        y += 10

        # Four stacked plots share the space left below the readouts. Deriving the
        # per-plot height from what's actually left (rather than a fixed 140) keeps
        # all four on-screen whatever the slider count -- the MesoMem systems add
        # up to five extra sliders, which at a fixed height pushed the RDF plot off
        # the bottom of the default window.
        plot_h = int(max(96, min(140, (self.window_size[1] - y - 40) / 4)))
        draw_plot(
            self.screen, self.small_font, pygame.Rect(x, y, w, plot_h),
            "Temperature (reduced T*)" if reduced else "Temperature",
            "T*" if reduced else "K",
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
            "Pressure -- quasi-2D box (reduced P*)" if reduced else "Pressure -- quasi-2D box (bar)",
            "P*" if reduced else "bar",
            list(history.t), [("P", PLOT_COLORS["press"], list(history.series["press"]))],
        )
        y += plot_h + 10

        draw_plot(
            self.screen, self.small_font, pygame.Rect(x, y, w, plot_h),
            "Energy, relative to t=0 this session " + ("(reduced eps)" if reduced else "(eV)"),
            "eps" if reduced else "eV",
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
                "Radial distribution g(r), r in " + ("sigma" if reduced else "Angstrom"), "g(r)",
                list(r), [("g(r)", PLOT_COLORS["rdf"], list(g))],
                y_range=(0.0, max(2.0, float(max(g)) * 1.1) if len(g) else 2.0),
                ref_lines=[(1.0, DIM_TEXT_COLOR, "gas")],
            )

    def draw(self, positions, is_puller, puller_pos, input_force, reaction_force, fps,
              spec, systems, current_key, sliders, thermo_now, puller_energy,
              history, rdf, heat_fraction=0.0, sim_time_ps=0.0, puller_speed_m_s=None,
              atom_trails=None, species=None, bond_pairs=None, hbond_pairs=None,
              hud_lines=None, scene_3d=None, total_steps=0, steps_per_frame=1,
              debug_line=None, playback_playing=None, puller_attached=True):
        # In GL mode the default framebuffer is cleared to BG first; the 3D scene
        # (if any) is drawn straight into its sim viewport, and every 2D surface
        # is composited over it at the end. In CPU mode self.screen IS the display
        # and everything just draws to it.
        if self.gl_enabled:
            self.gl.screen.use()
            self.gl.clear(*(c / 255.0 for c in BG))

        if spec.render_3d and scene_3d is not None:
            self.draw_sim_3d(
                scene_3d["positions3d"], scene_3d["dipoles3d"], scene_3d["is_puller"],
                spec, scene_3d["camera"], scene_3d["bonds"], input_force, reaction_force,
                scene_3d["control_grid"], fps, sim_time_ps=sim_time_ps,
                puller_energy=puller_energy, hud_lines=hud_lines,
                potential_terms=scene_3d.get("potential_terms"),
                torque_signals=scene_3d.get("torque_signals"),
                total_steps=total_steps, steps_per_frame=steps_per_frame,
                debug_line=debug_line, brightness=scene_3d.get("brightness"),
                total_potential_terms=scene_3d.get("total_potential_terms"),
                box_bounds=scene_3d.get("box_bounds"),
                puller_attached=puller_attached,
                bead_energies=scene_3d.get("bead_energies"),
                box_periodic=scene_3d.get("box_periodic"),
            )
        else:
            self.draw_sim(positions, is_puller, puller_pos, input_force, reaction_force,
                           fps, spec, heat_fraction, sim_time_ps, atom_trails, species, bond_pairs,
                           puller_energy, hbond_pairs, hud_lines,
                           total_steps=total_steps, steps_per_frame=steps_per_frame,
                           debug_line=debug_line,
                           puller_attached=puller_attached)
        self.draw_panel(systems, current_key, sliders, thermo_now, puller_energy,
                         history, rdf, spec, puller_speed_m_s)
        # Play / Pause / Reset controls for playback systems, over the sim view.
        self._playback_visible = False
        if spec.playback_controls and playback_playing is not None:
            self.draw_playback_controls(playback_playing)
        if self.gl_enabled:
            self.compositor.present(self.screen)
        pygame.display.flip()

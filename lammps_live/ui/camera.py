"""A minimal pinhole camera for the 3D systems (e.g. the MesoMem membrane
patch). Just enough to project world points to the sim viewport with real
perspective and to hand back a per-point depth (for back-to-front painter
sorting) and a per-point scale (so nearer atoms are drawn bigger and the size
falloff reads as depth). Pure numpy, no GL -- the scenes here are a handful of
beads, so a per-frame vectorized projection is trivially fast and keeps the
renderer a plain pygame blitter.

Convention: the membrane lies in the world xy-plane with its normal along +z
(the beads' director). The camera looks at the patch from slightly below in y
and above in z, giving the tilted three-quarter view where the directors read
as spikes standing up out of the sheet.
"""
import numpy as np

from ..control_focus import band_rate


def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


class Camera3D:
    def __init__(self, eye, target, up, fov_deg, viewport_w, viewport_h):
        self.eye = np.asarray(eye, dtype=float)
        self.target = np.asarray(target, dtype=float)
        self.viewport_w = viewport_w
        self.viewport_h = viewport_h
        self.cx = viewport_w / 2.0
        self.cy = viewport_h / 2.0

        self.up_hint = np.asarray(up, dtype=float)
        self._rebuild_basis()

        # Focal length in pixels. Default from the vertical field of view (a
        # point one focal length ahead spans the half-viewport at the frame
        # edge); fit_to_points overrides it to frame a given content extent.
        self.focal = (viewport_h / 2.0) / np.tan(np.radians(fov_deg) / 2.0)

    def _rebuild_basis(self):
        """View basis: forward toward the target, right = forward x up_hint,
        true up = right x forward (right-handed, orthonormal)."""
        self.forward = _normalize(self.target - self.eye)
        self.right = _normalize(np.cross(self.forward, self.up_hint))
        self.true_up = np.cross(self.right, self.forward)

    def move_to(self, eye, target=None):
        """Put the camera somewhere else -- and, optionally, aim it somewhere else
        too, which is what a pan does (see OrbitController.pan). The focal length
        is left alone: an orbit dollies by moving the eye, and re-fitting the zoom
        as it went would make the scene breathe as the camera swung."""
        self.eye = np.asarray(eye, dtype=float)
        if target is not None:
            self.target = np.asarray(target, dtype=float)
        self._rebuild_basis()

    def fit_to_points(self, pts, fill_w=0.92, fill_h=0.92):
        """Set the focal length so the given world points just fit the viewport,
        filling `fill_*` of each half-axis and choosing the tighter (fully-
        visible) fit. Unlike the fixed vertical-FOV default -- which frames only
        by height and so wastes horizontal space on a wide (fullscreen) viewport
        -- this fills whichever dimension binds, so the scene uses the available
        area at any aspect ratio. The view direction is unchanged; only zoom
        adapts. Points are framed about the principal point (the target should be
        roughly centered, as it is for the symmetric control plane)."""
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        rel = pts - self.eye
        cz = rel @ self.forward
        valid = cz > 1e-6
        if not np.any(valid):
            return
        # tan of the horizontal / vertical angle of each point off the view axis.
        ax = np.abs((rel @ self.right)[valid] / cz[valid])
        ay = np.abs((rel @ self.true_up)[valid] / cz[valid])
        ax_max, ay_max = float(ax.max()), float(ay.max())
        focal_w = (self.viewport_w / 2.0 * fill_w) / ax_max if ax_max > 1e-9 else np.inf
        focal_h = (self.viewport_h / 2.0 * fill_h) / ay_max if ay_max > 1e-9 else np.inf
        focal = min(focal_w, focal_h)
        if np.isfinite(focal) and focal > 0:
            self.focal = focal

    def project(self, pts):
        """Project world points (N,3) -> (screen (N,2) float, depth (N,),
        scale (N,)). depth is distance along the view axis (bigger = farther);
        scale is focal/depth, the px-per-world-unit factor at that depth. Points
        at or behind the lens get depth=+inf and scale=0 so callers can drop
        them."""
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        rel = pts - self.eye
        cx = rel @ self.right
        cy = rel @ self.true_up
        cz = rel @ self.forward

        depth = cz.copy()
        valid = cz > 1e-6
        safe = np.where(valid, cz, 1.0)
        scale = np.where(valid, self.focal / safe, 0.0)

        sx = self.cx + cx * scale
        sy = self.cy - cy * scale   # screen y grows downward
        screen = np.column_stack([sx, sy])
        depth = np.where(valid, depth, np.inf)
        return screen, depth, scale

    def project_point(self, p):
        s, d, sc = self.project(np.asarray(p, dtype=float)[None, :])
        return s[0], float(d[0]), float(sc[0])


class OrbitController:
    """Turntable camera state: azimuth, elevation, distance about a target.

    There is ONE camera state and two things that write it -- the automatic
    orbit and the mouse drag -- which is what makes them compose instead of
    fight. Grabbing the mouse stops the animation (dragging against a moving
    target is unusable, and a drag applied as an offset ON TOP of the animation
    slides out from under you the moment you let go), and C resumes it from
    wherever the drag left the camera, because the animation increments the very
    same azimuth the drag was moving.

    The world here is z-up (the membrane normal), so azimuth 0 puts the eye on
    the -y axis -- the angle the fixed scene cameras look from -- and elevation
    lifts it toward +z.
    """

    def __init__(self, eye, target, spec):
        self.target = np.asarray(target, dtype=float)
        self.spec = spec
        rel = np.asarray(eye, dtype=float) - self.target
        self.dist0 = float(np.linalg.norm(rel)) or 1.0
        self.dist = self.dist0
        # Decomposed from the scenario's own camera, so switching the turntable
        # on does not also change where the scene is first seen from.
        self.elev = float(np.arcsin(np.clip(rel[2] / self.dist, -1.0, 1.0)))
        self.azimuth = float(np.arctan2(rel[0], -rel[1]))
        self.auto = bool(spec.autostart)

    def eye(self):
        ce = np.cos(self.elev)
        return self.target + self.dist * np.array([
            ce * np.sin(self.azimuth), -ce * np.cos(self.azimuth), np.sin(self.elev)])

    def update(self, dt):
        if self.auto:
            self.azimuth += self.spec.speed * dt

    def toggle_auto(self):
        self.auto = not self.auto

    def drag(self, dx, dy):
        """Mouse drag, in pixels (dy positive downward, as pygame reports it).

        The SCENE follows the pointer on both axes, like spinning a globe under
        your finger: drag right and the near face goes right, drag up and the
        near face goes up (which means the camera itself sinks). The other
        convention -- vertical drag moving the CAMERA, so dragging up lifts you
        into a top-down view -- is what many 3D viewers ship, and it reads as
        inverted here."""
        self.auto = False
        s = self.spec.drag_sensitivity
        limit = np.radians(self.spec.elev_limit_deg)
        self.azimuth -= dx * s
        self.elev = float(np.clip(self.elev + dy * s, -limit, limit))

    def _stick_rate(self, x, slow, fast):
        """One stick axis -> a rate, through the shared deadzone / slow plateau /
        smooth ramp response (control_focus.band_rate). The camera and the sliders
        deliberately use the same curve, in their own units."""
        s = self.spec
        return band_rate(x, s.stick_deadzone, s.stick_slow_end, slow, fast)

    def steer(self, x, y, dt):
        """Fly the camera from a held stick, in deflections (-1..1) per axis.

        Same three numbers the drag and the auto-orbit write, so this composes
        with both -- and, like a drag, taking the stick stops the automatic turn
        (a camera that is both flown and turning is unusable, and C hands it
        back).

        BOTH AXES MOVE THE SCENE, NOT THE CAMERA -- the same "globe under your
        finger" convention as drag(), which is why both rates go in negated. Push
        right and the near face of the box goes right (the eye travels left);
        push forward and the near face tips up (the eye sinks). The other
        mapping, the PILOT's -- push right and you travel right AROUND the box --
        was what this shipped with first, and in the hand it reads as inverted on
        both axes: what you are looking at is one object a metre in front of you,
        so the hand expects to be turning THAT, not flying around it. The mouse
        drag has always said so; this now agrees with it.
        """
        s = self.spec
        rate_x = self._stick_rate(x, s.stick_slow_speed, s.stick_speed)
        rate_y = self._stick_rate(y, s.stick_slow_speed, s.stick_speed)
        if rate_x == 0.0 and rate_y == 0.0:
            return
        self.auto = False
        limit = np.radians(s.elev_limit_deg)
        self.azimuth -= rate_x * dt
        self.elev = float(np.clip(self.elev - rate_y * dt, -limit, limit))

    def steer_zoom(self, twist, dt):
        """Dolly from a held twist axis, in wheel notches per second, through the
        same banded response as the other two axes.

        NEGATED against `zoom`'s notches: twisting the grip away from you has to
        push the scene away. The other sign was tried first and reads as inverted
        -- the hand expects the twist to move the SCENE, not to reel the camera in.
        """
        s = self.spec
        rate = self._stick_rate(twist, s.stick_zoom_slow_speed, s.stick_zoom_speed)
        if rate == 0.0:
            return
        self.zoom(-rate * dt)

    def pan(self, dx, dy):
        """Shift-drag: slide the scene sideways instead of turning it.

        Moves the TARGET across the view plane (the eye follows it, since `eye()`
        is derived from the target), which is what makes this compose with the
        orbit and the dolly rather than fighting them: after a pan the turntable
        still turns about whatever you have brought to the middle. That is the
        whole point -- one membrane in a 37-sigma cell of them is off-centre by
        definition, and until now the only way to look at it was to zoom out.

        Same "globe under your finger" convention as drag(): the scene follows the
        pointer, so dragging right slides the scene right (the camera goes left).
        """
        s = self.spec
        # World-space screen axes at the current angles: `right` is horizontal in
        # the world's xy-plane (z is up), and screen-up is what is left after the
        # view direction is taken out of world up.
        right = np.array([np.cos(self.azimuth), np.sin(self.azimuth), 0.0])
        forward = _normalize(self.target - self.eye())
        up = np.cross(right, forward)
        step = s.pan_sensitivity * self.dist
        self.target = self.target + (up * dy - right * dx) * step

    def zoom(self, notches):
        """Wheel dolly. MULTIPLICATIVE, so one notch is the same PROPORTIONAL
        step wherever you are -- a fixed number of world units per notch crawls
        when you are far out and slams through the scene when you are close."""
        self.dist = float(np.clip(self.dist * np.exp(-notches * self.spec.zoom_step),
                                  self.spec.dist_min * self.dist0,
                                  self.spec.dist_max * self.dist0))

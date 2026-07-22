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

        # View basis: forward toward the target, right = forward x up_hint,
        # true up = right x forward (right-handed, orthonormal).
        self.forward = _normalize(self.target - self.eye)
        self.right = _normalize(np.cross(self.forward, np.asarray(up, dtype=float)))
        self.true_up = np.cross(self.right, self.forward)

        # Focal length in pixels from the vertical field of view: a point one
        # focal length ahead spans the half-viewport at the frame edge.
        self.focal = (viewport_h / 2.0) / np.tan(np.radians(fov_deg) / 2.0)

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

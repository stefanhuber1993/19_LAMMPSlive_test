"""Render the 7-bead MesoMem patch offscreen, no window and no GUI -- the README's
opening picture. Uses the app's own GL pipeline, so what it shows is exactly what
the demo draws.

    ./venv/bin/python <this> docs/images/mesomem_patch.png
"""
import sys

import moderngl
import numpy as np
from PIL import Image

from lammps_live.playground import registry
from lammps_live.ui.camera import Camera3D
from lammps_live.ui.gl3d import GLScene, proj_matrix, view_matrix
from lammps_live.ui.theme import BG

W, H = 1200, 760
OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/images/mesomem_patch.png"

system = registry.build("mesomem_patch")
for _ in range(30):                       # settle into a real configuration
    system.step(20)

spec = system.spec
style = spec.render_style
pts = np.asarray(system.get_positions_3d()[1], dtype=float)
dips = np.asarray(system.get_dipoles_3d(), dtype=float)

cam = system.get_camera_params()
camera = Camera3D(cam["eye"], cam["target"], cam["up"], cam["fov_deg"], W, H)
fit = system.get_scene_fit_points()
if fit is not None:
    camera.fit_to_points(fit, fill_w=0.92, fill_h=0.86)

# A standalone context has no default framebuffer, which is fine: GLScene renders
# into its own FBOs and hands the picture back with read_rgb().
ctx = moderngl.create_standalone_context(require=330)
scene = GLScene(ctx, W, H)

phys_r = spec.atom_radius_A or 0.5
_, depth, _ = camera.project(pts)
dmin, dmax = float(depth.min()), float(depth.max())
view = view_matrix(camera.eye, camera.right, camera.true_up, camera.forward)
proj = proj_matrix(camera.focal / (W / 2.0), camera.focal / (H / 2.0),
                   max(0.05, dmin - phys_r - 1.0), dmax + phys_r + 1.0)

scene.render(view, proj, pts, np.full(len(pts), phys_r, dtype=np.float32), dips,
             (dmin, dmax), style=style, bead_radius=phys_r, focal_px=camera.focal)
Image.fromarray(scene.read_rgb()).save(OUT)
print("wrote", OUT, f"{W}x{H}")
system.close()

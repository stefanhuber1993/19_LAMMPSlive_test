"""GPU rendering of the 3D bead scenes (the MesoMem patch and sheet) with
moderngl.

The CPU path in renderer.py drew each bead as a numpy-shaded sphere sprite,
painter-sorted back to front -- which has no per-pixel depth, so beads close in z
"jump" in front of / behind each other instead of intersecting, and the ~900-bead
sheet regenerates sprites almost every frame. This module replaces that with the
standard molecular-viz technique:

  * Instanced sphere IMPOSTORS. Each bead is one camera-facing quad; the fragment
    shader ray-casts the sphere and writes gl_FragDepth, so the hardware depth
    buffer resolves intersections exactly (no sorting, no z-jumps).
  * A deferred G-buffer (albedo / view-normal / view-position) feeds an SSAO pass
    so crevices between beads darken -- a real depth cue.
  * A final pass does Blinn-Phong shading + AO + the same depth fog the CPU path
    used, then bond/net lines are drawn as real depth-tested GL lines (occluded by
    the beads for free -- no hand-rolled z-buffer).

The pipeline is context-agnostic: it takes a moderngl context (from the pygame GL
window, or a headless standalone context for benchmarking/tests) and renders into
its own framebuffers. `render()` fills `self.final_fbo`; `blit_to_viewport()`
copies it to the on-screen sim viewport, and `read_rgb()` reads it back (used by
the benchmark / snapshot tooling).

Requires an OpenGL 3.3+ core context (GLSL 330); works on macOS 4.1 core and Linux
via EGL/GLX.
"""
import numpy as np

from .theme import (
    BEAD_BAND_HALFWIDTH, BEAD_BAND_SOFT, BEAD_EQUATOR_COLOR, BEAD_POLE_COLOR,
    BEAD_WHITE_POLE_COLOR, BEAD_WHITE_POLE_MIN, BEAD_WHITE_POLE_SOFT,
    BG, HAZE_COLOR, HAZE_STRENGTH, SPHERE_AMBIENT, SPHERE_LIGHT_DIR,
)

SSAO_KERNEL_SIZE = 24


# ---- small matrix helpers (row-major math; transposed on upload to GL) --------

def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def view_matrix(eye, right, true_up, forward):
    """World->view (camera at origin looking down -Z, x right, y up), matching
    Camera3D's basis. Row-major; view = R*(world - eye)."""
    R = np.array([right, true_up, -forward], dtype=np.float64)
    t = -R @ np.asarray(eye, dtype=np.float64)
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def proj_matrix(fx, fy, near, far):
    """Perspective projection consistent with Camera3D's pixel focal length:
    fx = focal/(W/2), fy = focal/(H/2). Maps view space (front = -z) to clip."""
    M = np.zeros((4, 4), dtype=np.float64)
    M[0, 0] = fx
    M[1, 1] = fy
    M[2, 2] = -(far + near) / (far - near)
    M[2, 3] = -2.0 * far * near / (far - near)
    M[3, 2] = -1.0
    return M


def _gl(m):
    """Row-major numpy matrix -> column-major bytes for a GLSL mat4 uniform."""
    return np.ascontiguousarray(m.T, dtype="f4").tobytes()


# ---- shader sources -----------------------------------------------------------

_GEOM_VS = """
#version 330 core
uniform mat4 view;
uniform mat4 proj;
in vec2 in_corner;      // quad corner in [-1,1]
in vec3 in_center;      // per-instance world center
in float in_radius;     // per-instance world radius
in vec3 in_dir;         // per-instance unit director (world)
in float in_fade;       // per-instance fade-to-background (0 opaque .. 1 gone)
in float in_bright;     // per-instance albedo brightness multiplier (1 = normal)
out vec3 v_centerView;
out float v_radius;
out vec3 v_dirView;
out vec3 v_rayView;
out float v_fade;
out float v_bright;
void main() {
    vec4 cv = view * vec4(in_center, 1.0);
    v_centerView = cv.xyz;
    v_radius = in_radius;
    v_dirView = mat3(view) * in_dir;
    v_fade = in_fade;
    v_bright = in_bright;
    // Camera-facing billboard, enlarged a touch so the perspective silhouette
    // (slightly bigger than the radius) is fully covered.
    vec3 corner = cv.xyz + vec3(in_corner * in_radius * 1.15, 0.0);
    v_rayView = corner;
    gl_Position = proj * vec4(corner, 1.0);
}
"""

_GEOM_FS = """
#version 330 core
uniform mat4 proj;
uniform float band_half;
uniform float band_soft;
uniform vec3 equator_col;
uniform vec3 pole_col;
uniform vec3 white_col;
uniform float white_min;
uniform float white_soft;
in vec3 v_centerView;
in float v_radius;
in vec3 v_dirView;
in vec3 v_rayView;
in float v_fade;
in float v_bright;
layout(location=0) out vec4 o_albedo;
layout(location=1) out vec3 o_normal;
layout(location=2) out vec3 o_viewpos;
void main() {
    vec3 dir = normalize(v_rayView);          // ray from the eye (origin) out
    vec3 c = v_centerView;
    float b = dot(dir, c);
    float disc = b * b - (dot(c, c) - v_radius * v_radius);
    if (disc < 0.0) discard;                  // ray misses the sphere
    float t = b - sqrt(disc);                 // near intersection
    if (t < 0.0) discard;
    vec3 hit = dir * t;
    vec3 N = normalize(hit - c);
    // Correct per-pixel depth so overlapping beads intersect via the depth test.
    vec4 clip = proj * vec4(hit, 1.0);
    gl_FragDepth = 0.5 * (clip.z / clip.w) + 0.5;
    // Banded MesoMem albedo: yellow equator (perp to the director) -> blue poles
    // (along it), same blend as the old CPU _banded_sphere_sprite.
    float s = dot(N, normalize(v_dirView));   // signed cos-latitude (+ = +n pole)
    float cosl = abs(s);
    float tt = clamp((cosl - (band_half - band_soft)) / (2.0 * band_soft), 0.0, 1.0);
    vec3 albedo = mix(equator_col, pole_col, tt);
    // Over-paint the +n pole white (down to ~80% latitude) so director sense reads.
    float w = smoothstep(white_min - white_soft, white_min + white_soft, s);
    albedo = mix(albedo, white_col, w);
    o_albedo = vec4(albedo * v_bright, v_fade);   // alpha carries the fade-to-bg
    o_normal = N;
    o_viewpos = hit;
}
"""

_FULLSCREEN_VS = """
#version 330 core
out vec2 uv;
void main() {
    // Single covering triangle; uv in [0,1].
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

_SSAO_FS = """
#version 330 core
uniform sampler2D posTex;
uniform sampler2D normalTex;
uniform sampler2D noiseTex;
uniform sampler2D depthTex;
uniform vec3 samples[%d];
uniform mat4 proj;
uniform vec2 noiseScale;
uniform float radius;
uniform float bias;
in vec2 uv;
out float frag;
void main() {
    if (texture(depthTex, uv).r >= 1.0) { frag = 1.0; return; }  // background
    vec3 P = texture(posTex, uv).xyz;
    vec3 N = normalize(texture(normalTex, uv).xyz);
    vec3 rvec = normalize(texture(noiseTex, uv * noiseScale).xyz);
    vec3 T = normalize(rvec - N * dot(rvec, N));
    vec3 B = cross(N, T);
    mat3 TBN = mat3(T, B, N);
    float occ = 0.0;
    for (int i = 0; i < %d; ++i) {
        vec3 samplePos = P + TBN * samples[i] * radius;
        vec4 off = proj * vec4(samplePos, 1.0);
        off.xyz /= off.w;
        off.xyz = off.xyz * 0.5 + 0.5;
        if (off.x < 0.0 || off.x > 1.0 || off.y < 0.0 || off.y > 1.0) continue;
        float sampleDepth = texture(posTex, off.xy).z;   // view z (front = -)
        float rangeCheck = smoothstep(0.0, 1.0, radius / max(abs(P.z - sampleDepth), 1e-4));
        occ += (sampleDepth >= samplePos.z + bias ? 1.0 : 0.0) * rangeCheck;
    }
    frag = 1.0 - occ / float(%d);
}
""" % (SSAO_KERNEL_SIZE, SSAO_KERNEL_SIZE, SSAO_KERNEL_SIZE)

_BLUR_FS = """
#version 330 core
uniform sampler2D aoTex;
uniform sampler2D posTex;
uniform vec2 texel;
in vec2 uv;
out float frag;
void main() {
    // Foreground-only 4x4 box blur. The background's AO is 1.0 (fully open); a
    // naive blur bleeds that into a bead's silhouette pixels, un-darkening them
    // into a bright halo around the cluster. Weighting each tap by whether it is
    // a bead fragment (view-space z < 0; the background G-buffer z is cleared to
    // 0) keeps the blur inside the geometry, so no halo forms.
    float sum = 0.0;
    float wsum = 0.0;
    for (int x = -2; x < 2; ++x)
        for (int y = -2; y < 2; ++y) {
            vec2 o = uv + vec2(x, y) * texel;
            float fg = texture(posTex, o).z < -1e-4 ? 1.0 : 0.0;
            sum += texture(aoTex, o).r * fg;
            wsum += fg;
        }
    frag = wsum > 0.0 ? sum / wsum : texture(aoTex, uv).r;
}
"""

_COMPOSITE_FS = """
#version 330 core
uniform sampler2D albedoTex;
uniform sampler2D normalTex;
uniform sampler2D posTex;
uniform sampler2D aoTex;
uniform sampler2D depthTex;
uniform vec3 lightDir;
uniform float ambient;
uniform vec3 bgColor;
uniform vec3 hazeColor;
uniform float fogNear;
uniform float fogFar;
uniform float fogStrength;
uniform float aoStrength;
uniform float aoPower;
in vec2 uv;
out vec4 frag;
void main() {
    if (texture(depthTex, uv).r >= 1.0) { frag = vec4(bgColor, 1.0); return; }
    vec4 albedoSample = texture(albedoTex, uv);
    vec3 alb = albedoSample.rgb;
    float boundaryFade = albedoSample.a;       // per-bead fade toward the background
    vec3 N = normalize(texture(normalTex, uv).xyz);
    vec3 P = texture(posTex, uv).xyz;
    float ao = texture(aoTex, uv).r;
    vec3 L = normalize(lightDir);
    float diff = max(dot(N, L), 0.0);
    vec3 V = normalize(-P);
    vec3 H = normalize(L + V);
    float spec = 0.5 * pow(max(dot(N, H), 0.0), 32.0);
    float shade = ambient + (1.0 - ambient) * diff + spec;
    vec3 col = alb * shade;
    // Contact darkening: multiply the whole lit color by the ambient-occlusion
    // term (ao=1 open -> ao=0 fully occluded), so crevices where beads pack
    // together read as shadowed. Deepened (aoPower) and scaled (aoStrength) so
    // it is clearly visible, not just a nudge on the ambient floor.
    float occ = pow(clamp(ao, 0.0, 1.0), aoPower);
    col *= mix(1.0, occ, aoStrength);
    float depth = -P.z;                        // distance along the view axis
    float fog = fogStrength * clamp((depth - fogNear) / max(fogFar - fogNear, 1e-4), 0.0, 1.0);
    col = mix(col, hazeColor, fog);
    // Periodic-seam crossfade: a bead leaving one edge dissolves toward the
    // background while its wrapped ghost fades in at the opposite edge, so it
    // slides across the boundary instead of popping.
    col = mix(col, bgColor, clamp(boundaryFade, 0.0, 1.0));
    frag = vec4(col, 1.0);
}
"""

_LINE_VS = """
#version 330 core
uniform mat4 view;
uniform mat4 proj;
in vec3 in_pos;
in vec4 in_col;
out vec4 v_col;
void main() {
    v_col = in_col;
    gl_Position = proj * view * vec4(in_pos, 1.0);
}
"""

_LINE_FS = """
#version 330 core
in vec4 v_col;
out vec4 frag;
void main() { frag = v_col; }
"""

_BLIT_FS = """
#version 330 core
uniform sampler2D srcTex;
in vec2 uv;
out vec4 frag;
void main() { frag = texture(srcTex, uv); }
"""


def _ssao_kernel(n):
    """Hemisphere sample kernel (points in the +z hemisphere, weighted toward
    the origin so nearby occluders dominate)."""
    rng = np.random.default_rng(1234)
    k = np.empty((n, 3), dtype=np.float32)
    for i in range(n):
        v = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0, 1)])
        v = _normalize(v) * rng.uniform(0.0, 1.0)
        scale = i / n
        v *= 0.1 + 0.9 * scale * scale         # accelerate toward the origin
        k[i] = v
    return k


def _ssao_noise(size=4):
    """A small tiled texture of random in-plane rotations for the SSAO kernel."""
    rng = np.random.default_rng(5678)
    noise = np.zeros((size * size, 3), dtype=np.float32)
    noise[:, 0] = rng.uniform(-1, 1, size * size)
    noise[:, 1] = rng.uniform(-1, 1, size * size)
    return noise


class GLScene:
    """The GPU bead pipeline for one moderngl context, sized to a sim viewport."""

    def __init__(self, ctx, width, height):
        self.ctx = ctx
        self.width = 0
        self.height = 0
        self._inst_capacity = 0
        self._inst_vbo = None
        self._geom_vao = None
        self._line_vbo = None
        self._line_cap = 0
        self._line_vao = None

        self._build_programs()
        self._build_static_buffers()
        self.resize(width, height)

    # ---- setup --------------------------------------------------------------

    def _build_programs(self):
        c = self.ctx
        self.geom_prog = c.program(vertex_shader=_GEOM_VS, fragment_shader=_GEOM_FS)
        self.ssao_prog = c.program(vertex_shader=_FULLSCREEN_VS, fragment_shader=_SSAO_FS)
        self.blur_prog = c.program(vertex_shader=_FULLSCREEN_VS, fragment_shader=_BLUR_FS)
        self.comp_prog = c.program(vertex_shader=_FULLSCREEN_VS, fragment_shader=_COMPOSITE_FS)
        self.line_prog = c.program(vertex_shader=_LINE_VS, fragment_shader=_LINE_FS)
        self.blit_prog = c.program(vertex_shader=_FULLSCREEN_VS, fragment_shader=_BLIT_FS)
        self.blit_prog["srcTex"].value = 0

        # Constant shader parameters (bead band, lighting, colors, SSAO kernel).
        self.geom_prog["band_half"].value = BEAD_BAND_HALFWIDTH
        self.geom_prog["band_soft"].value = BEAD_BAND_SOFT
        self.geom_prog["equator_col"].value = tuple(c_ / 255.0 for c_ in BEAD_EQUATOR_COLOR)
        self.geom_prog["pole_col"].value = tuple(c_ / 255.0 for c_ in BEAD_POLE_COLOR)
        self.geom_prog["white_col"].value = tuple(c_ / 255.0 for c_ in BEAD_WHITE_POLE_COLOR)
        self.geom_prog["white_min"].value = BEAD_WHITE_POLE_MIN
        self.geom_prog["white_soft"].value = BEAD_WHITE_POLE_SOFT

        kernel = _ssao_kernel(SSAO_KERNEL_SIZE)
        self.ssao_prog["samples"].write(kernel.tobytes())
        self.ssao_prog["radius"].value = 0.9   # ~ bead diameter: reaches neighbours
        self.ssao_prog["bias"].value = 0.015

        self.comp_prog["lightDir"].value = tuple(_normalize(np.array(SPHERE_LIGHT_DIR)))
        self.comp_prog["ambient"].value = SPHERE_AMBIENT
        self.comp_prog["bgColor"].value = tuple(c_ / 255.0 for c_ in BG)
        self.comp_prog["hazeColor"].value = tuple(c_ / 255.0 for c_ in HAZE_COLOR)
        self.comp_prog["fogStrength"].value = HAZE_STRENGTH
        # Contact-shadow strength/contrast (applied as a whole-color multiplier so
        # packed beads visibly darken between each other).
        self.comp_prog["aoStrength"].value = 0.85
        self.comp_prog["aoPower"].value = 1.6

        # Texture unit assignments.
        for name, unit in (("albedoTex", 0), ("normalTex", 1), ("posTex", 2),
                           ("aoTex", 3), ("depthTex", 4)):
            if name in self.comp_prog:
                self.comp_prog[name].value = unit
        self.ssao_prog["posTex"].value = 0
        self.ssao_prog["normalTex"].value = 1
        self.ssao_prog["noiseTex"].value = 2
        self.ssao_prog["depthTex"].value = 3
        self.blur_prog["aoTex"].value = 0
        self.blur_prog["posTex"].value = 1

    def _build_static_buffers(self):
        c = self.ctx
        # Quad corners for the impostor billboard (triangle strip).
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
        self._quad_vbo = c.buffer(quad.tobytes())
        # SSAO rotation-noise texture (RGB16F, nearest, repeat).
        noise = _ssao_noise(4)
        self._noise_tex = c.texture((4, 4), 3, noise.astype("f2").tobytes(), dtype="f2")
        self._noise_tex.filter = (c.NEAREST, c.NEAREST)
        self._noise_tex.repeat_x = True
        self._noise_tex.repeat_y = True
        # Empty VAO for fullscreen passes (vertices generated from gl_VertexID).
        self._fs_vao = c.vertex_array(self.ssao_prog, [])

    def resize(self, width, height):
        width = max(1, int(width))
        height = max(1, int(height))
        if width == self.width and height == self.height:
            return
        self.width, self.height = width, height
        c = self.ctx
        self._release_fbos()

        # Albedo is f2 (not f1) so a per-bead brightness boost (>1) survives into
        # the composite before the final clamp, and the alpha channel can carry
        # the periodic-seam fade fraction.
        self.albedo_tex = c.texture((width, height), 4, dtype="f2")
        self.normal_tex = c.texture((width, height), 3, dtype="f2")
        self.pos_tex = c.texture((width, height), 3, dtype="f2")
        self.depth_tex = c.depth_texture((width, height))
        self.depth_tex.compare_func = ""          # sample raw depth, not a compare
        for t in (self.albedo_tex, self.normal_tex, self.pos_tex):
            t.filter = (c.NEAREST, c.NEAREST)
        self.gbuffer = c.framebuffer(
            color_attachments=[self.albedo_tex, self.normal_tex, self.pos_tex],
            depth_attachment=self.depth_tex,
        )

        self.ao_tex = c.texture((width, height), 1, dtype="f2")
        self.ao_fbo = c.framebuffer(color_attachments=[self.ao_tex])
        self.ao_blur_tex = c.texture((width, height), 1, dtype="f2")
        self.ao_blur_fbo = c.framebuffer(color_attachments=[self.ao_blur_tex])

        self.final_tex = c.texture((width, height), 4, dtype="f1")
        # Share the geometry depth buffer so the line pass is occluded by beads.
        self.final_fbo = c.framebuffer(
            color_attachments=[self.final_tex], depth_attachment=self.depth_tex
        )

        self.ssao_prog["noiseScale"].value = (width / 4.0, height / 4.0)
        self.blur_prog["texel"].value = (1.0 / width, 1.0 / height)

    def _release_fbos(self):
        for attr in ("gbuffer", "ao_fbo", "ao_blur_fbo", "final_fbo",
                     "albedo_tex", "normal_tex", "pos_tex", "depth_tex",
                     "ao_tex", "ao_blur_tex", "final_tex"):
            obj = getattr(self, attr, None)
            if obj is not None:
                obj.release()
                setattr(self, attr, None)

    # ---- per-frame instance / line uploads ----------------------------------

    def _upload_instances(self, centers, radii, directors, fades, brights):
        n = len(centers)
        data = np.empty((n, 9), dtype="f4")
        data[:, 0:3] = centers
        data[:, 3] = radii
        data[:, 4:7] = directors
        data[:, 7] = fades
        data[:, 8] = brights
        raw = data.tobytes()
        if n > self._inst_capacity:
            if self._inst_vbo is not None:
                self._inst_vbo.release()
            if self._geom_vao is not None:
                self._geom_vao.release()
            self._inst_vbo = self.ctx.buffer(reserve=max(1, n) * 9 * 4, dynamic=True)
            self._inst_capacity = n
            self._geom_vao = self.ctx.vertex_array(
                self.geom_prog,
                [(self._quad_vbo, "2f", "in_corner"),
                 (self._inst_vbo, "3f 1f 3f 1f 1f /i", "in_center", "in_radius",
                  "in_dir", "in_fade", "in_bright")],
            )
        self._inst_vbo.write(raw)
        return n

    def _upload_lines(self, verts, colors):
        """verts: (M,3) segment endpoints (pairs), colors: (M,4) rgba in [0,1]."""
        m = len(verts)
        data = np.empty((m, 7), dtype="f4")
        data[:, 0:3] = verts
        data[:, 3:7] = colors
        raw = data.tobytes()
        if m > self._line_cap:
            if self._line_vbo is not None:
                self._line_vbo.release()
            if self._line_vao is not None:
                self._line_vao.release()
            self._line_vbo = self.ctx.buffer(reserve=max(1, m) * 7 * 4, dynamic=True)
            self._line_cap = m
            self._line_vao = self.ctx.vertex_array(
                self.line_prog,
                [(self._line_vbo, "3f 4f", "in_pos", "in_col")],
            )
        self._line_vbo.write(raw)
        return m

    # ---- render -------------------------------------------------------------

    def render(self, view, proj, centers, radii, directors, near, far,
               line_verts=None, line_colors=None, fades=None, brights=None):
        """Render the beads (+ optional depth-occluded lines) into self.final_fbo.
        view/proj are row-major 4x4 numpy matrices (see view_matrix/proj_matrix).
        `fades` (0 opaque .. 1 gone) and `brights` (albedo multiplier) are optional
        per-bead arrays; default to fully-opaque, unit-brightness."""
        c = self.ctx
        vb, pb = _gl(view), _gl(proj)
        n = len(centers)
        fades = np.zeros(n, "f4") if fades is None else np.asarray(fades, "f4")
        brights = np.ones(n, "f4") if brights is None else np.asarray(brights, "f4")
        n = self._upload_instances(np.asarray(centers, "f4"),
                                   np.asarray(radii, "f4"),
                                   np.asarray(directors, "f4"), fades, brights)

        # --- geometry pass -> G-buffer ---
        self.geom_prog["view"].write(vb)
        self.geom_prog["proj"].write(pb)
        self.gbuffer.use()
        c.viewport = (0, 0, self.width, self.height)
        # Clear albedo/normal to 0 and pos.z to 0 (background), depth to 1.
        self.gbuffer.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        c.enable(c.DEPTH_TEST)
        c.depth_func = "<"
        c.disable(c.BLEND)
        self._geom_vao.render(mode=c.TRIANGLE_STRIP, instances=n)

        # --- SSAO -> ao_fbo ---
        self.ssao_prog["proj"].write(pb)
        self.pos_tex.use(0)
        self.normal_tex.use(1)
        self._noise_tex.use(2)
        self.depth_tex.use(3)
        self.ao_fbo.use()
        c.disable(c.DEPTH_TEST)
        self.ao_fbo.clear(1.0, 1.0, 1.0, 1.0)
        self._render_fullscreen(self.ssao_prog)

        # --- blur AO -> ao_blur_fbo (foreground-masked, see _BLUR_FS) ---
        self.ao_tex.use(0)
        self.pos_tex.use(1)
        self.ao_blur_fbo.use()
        self._render_fullscreen(self.blur_prog)

        # --- composite (lighting + AO + fog) -> final_fbo ---
        self.albedo_tex.use(0)
        self.normal_tex.use(1)
        self.pos_tex.use(2)
        self.ao_blur_tex.use(3)
        self.depth_tex.use(4)
        self.comp_prog["fogNear"].value = float(near)
        self.comp_prog["fogFar"].value = float(far)
        self.final_fbo.use()
        c.disable(c.DEPTH_TEST)
        self._render_fullscreen(self.comp_prog)

        # --- depth-occluded lines (bonds + net) into the same FBO ---
        if line_verts is not None and len(line_verts) >= 2:
            m = self._upload_lines(np.asarray(line_verts, "f4"),
                                   np.asarray(line_colors, "f4"))
            self.line_prog["view"].write(vb)
            self.line_prog["proj"].write(pb)
            self.final_fbo.use()
            c.enable(c.DEPTH_TEST)
            c.depth_func = "<="
            c.depth_mask = False               # test against beads, don't write
            c.enable(c.BLEND)
            self._line_vao.render(mode=c.LINES, vertices=m)
            c.depth_mask = True

    def _render_fullscreen(self, prog):
        vao = self.ctx.vertex_array(prog, [])
        vao.render(mode=self.ctx.TRIANGLES, vertices=3)
        vao.release()

    # ---- output -------------------------------------------------------------

    def blit_to_viewport(self, x, y, w, h):
        """Draw the rendered sim view into the default framebuffer's sim viewport
        rectangle (x, y, w, h) in GL (origin bottom-left) coordinates. Both this
        texture and the target are GL, so no vertical flip is needed."""
        c = self.ctx
        c.screen.use()
        c.viewport = (int(x), int(y), int(w), int(h))
        c.disable(c.DEPTH_TEST)
        c.disable(c.BLEND)
        self.final_tex.use(0)
        self._render_fullscreen(self.blit_prog)

    def read_rgb(self):
        """Read the final color buffer back as an (H, W, 3) uint8 array (row 0 =
        top), for snapshots and the benchmark."""
        data = self.final_fbo.read(components=3, dtype="f1")
        img = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 3)
        return img[::-1]                       # GL origin is bottom-left

    def release(self):
        self._release_fbos()
        for obj in (self._inst_vbo, self._geom_vao, self._line_vbo, self._line_vao,
                    self._quad_vbo, self._noise_tex, getattr(self, "_fs_vao", None)):
            if obj is not None:
                obj.release()

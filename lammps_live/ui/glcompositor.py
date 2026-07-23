"""Composite the pygame-drawn 2D UI (instrumentation panel + all the 2D sim
overlays) over the GL scene.

Since the window is now an OpenGL context, pygame can no longer blit straight to
the display. Instead every 2D drawing method keeps drawing to an offscreen
SRCALPHA Surface (Renderer.screen); each frame that surface is uploaded to a GL
texture and drawn as one full-window quad with alpha blending -- transparent over
the sim viewport for the 3D systems (so the GL beads show through), opaque over
the sim viewport for the 2D systems (which draw their whole scene onto it as
before)."""
import numpy as np
import pygame


_VS = """
#version 330 core
out vec2 uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

_FS = """
#version 330 core
uniform sampler2D uiTex;
in vec2 uv;
out vec4 frag;
void main() { frag = texture(uiTex, uv); }
"""


class GLCompositor:
    def __init__(self, ctx, width, height):
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=_VS, fragment_shader=_FS)
        self.prog["uiTex"].value = 0
        self.vao = ctx.vertex_array(self.prog, [])
        self.tex = None
        self.resize(width, height)

    def resize(self, width, height):
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        if self.tex is not None:
            self.tex.release()
        self.tex = self.ctx.texture((self.width, self.height), 4)
        self.tex.filter = (self.ctx.NEAREST, self.ctx.NEAREST)

    def present(self, surface):
        """Upload `surface` (an SRCALPHA Surface the size of the window) and draw
        it over the whole default framebuffer with premultiplied-style alpha
        blending."""
        c = self.ctx
        # tostring with flipvert=True so the surface's top row lands at the GL
        # texture's top (uv.y is flipped again in the quad, giving upright UI).
        raw = pygame.image.tostring(surface, "RGBA", True)
        self.tex.write(raw)
        c.screen.use()
        c.viewport = (0, 0, self.width, self.height)
        c.disable(c.DEPTH_TEST)
        c.enable(c.BLEND)
        c.blend_func = (c.SRC_ALPHA, c.ONE_MINUS_SRC_ALPHA)
        self.tex.use(0)
        self.vao.render(mode=c.TRIANGLES, vertices=3)

    def release(self):
        if self.tex is not None:
            self.tex.release()
        self.vao.release()

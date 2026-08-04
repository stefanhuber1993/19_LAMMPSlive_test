"""Generate every figure in the impostor-renderer book as a PNG.

PNG rather than SVG on purpose: the book is read on an e-ink reader, and every
reader in existence renders a PNG, while SVG support is patchy. Black line art
on white, no colour, fairly heavy strokes and large type -- that is what survives
a 6" grayscale panel.

    ./venv/bin/python docs/impostor-book/figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle, Wedge

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "images")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "text.color": "black",
    "axes.edgecolor": "black",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

INK = "black"
GREY = "0.55"
PALE = "0.88"


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", path)


def blank(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def arrow(ax, a, b, **kw):
    kw.setdefault("arrowstyle", "-|>")
    kw.setdefault("mutation_scale", 14)
    kw.setdefault("color", INK)
    kw.setdefault("lw", 1.4)
    ax.add_patch(FancyArrowPatch(a, b, shrinkA=0, shrinkB=0, **kw))


def box(ax, x, y, w, h, title, lines=(), fill="white", lw=1.6):
    ax.add_patch(Rectangle((x, y), w, h, fill=True, facecolor=fill,
                           edgecolor=INK, lw=lw))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top",
            fontsize=11, fontweight="bold")
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.62 - 0.34 * i, ln, ha="center", va="top",
                fontsize=9, color="0.25")


# ---------------------------------------------------------------------------
# 1. The pipeline
# ---------------------------------------------------------------------------
def fig_pipeline():
    fig, ax = blank(9.0, 8.4, (0, 12.2), (0.0, 11.2))
    passes = [
        ("PASS 1  G-BUFFER", ["one instanced draw call", "the only per-BEAD cost"]),
        ("PASS 2  OCCLUSION", ["ambient occlusion + sun", "half resolution"]),
        ("PASS 3  BLUR", ["masked 4x4 box", "half resolution"]),
        ("PASS 4  COMPOSITE", ["light, outline, cue, tonemap"]),
        ("PASS 5  DEPTH OF FIELD", ["24-tap spiral gather"]),
        ("PASS 6  LINES", ["real GL lines, depth-tested"]),
        ("PASS 7  FXAA", ["luma edge antialiasing"]),
    ]
    h, gap = 1.35, 0.2
    y = 11.0
    top = y
    for i, (name, sub) in enumerate(passes):
        fill = PALE if i == 0 else "white"
        ax.add_patch(Rectangle((0.5, y - h), 6.6, h, facecolor=fill,
                               edgecolor=INK, lw=1.6))
        ax.text(3.8, y - 0.44, name, ha="center", va="center", fontsize=11,
                fontweight="bold")
        for j, ln in enumerate(sub):
            ax.text(3.8, y - 0.82 - 0.30 * j, ln, ha="center", va="center",
                    fontsize=9, color="0.25")
        if i < len(passes) - 1:
            arrow(ax, (3.8, y - h), (3.8, y - h - gap))
        y -= h + gap
    bottom = y + gap
    ax.plot([7.5, 7.5], [top, top - h], color=INK, lw=2.6)
    ax.text(7.8, top - h / 2, "N beads", va="center", fontsize=10,
            fontweight="bold")
    ax.plot([7.5, 7.5], [top - h - gap, bottom], color=INK, lw=2.6)
    ax.text(7.8, (top - h - gap + bottom) / 2,
            "PIXELS ONLY\n(the bead count\ndoes not appear)",
            va="center", fontsize=10, fontweight="bold")
    save(fig, "fig-pipeline.png")


# ---------------------------------------------------------------------------
# 2. Coordinate spaces
# ---------------------------------------------------------------------------
def fig_spaces():
    fig, ax = blank(9.5, 4.0, (0, 13.15), (0.6, 6.1))
    stages = [
        ("WORLD", ["bead centre", "(2.0, 0, 0) sigma"]),
        ("VIEW", ["eye at origin,", "looking down -z", "(2.0, 0, -12.0)"]),
        ("CLIP", ["x4 homogeneous", "(2.168, 0, -7.09, 12)"]),
        ("NDC", ["divide by w", "(0.181, 0, -0.591)"]),
        ("SCREEN", ["viewport map", "(484 px, 450 px)", "depth 0.204"]),
    ]
    x, w, gap = 0.25, 2.3, 0.26
    for i, (name, sub) in enumerate(stages):
        box(ax, x, 2.4, w, 2.1, name, sub)
        if i < len(stages) - 1:
            arrow(ax, (x + w, 3.45), (x + w + gap, 3.45))
        x += w + gap
    labels = ["view matrix\n(rotate + translate)", "proj matrix\n(fx, fy, near, far)",
              "perspective\ndivide", "viewport\ntransform"]
    x = 0.25
    for lab in labels:
        ax.text(x + w + gap / 2, 2.15, lab, ha="center", va="top", fontsize=8.5,
                color="0.25")
        x += w + gap
    ax.text(6.6, 5.35, "the same bead, five ways of saying where it is",
            ha="center", fontsize=11, style="italic")
    save(fig, "fig-spaces.png")


# ---------------------------------------------------------------------------
# 3. Off-axis: why a screen-parallel quad is too small
# ---------------------------------------------------------------------------
def fig_offaxis():
    fig, ax = blank(9.0, 5.0, (-1.2, 12.5), (-1.6, 6.6))
    d, r, th = 7.6, 1.15, np.radians(30.0)
    C = np.array([d * np.cos(th), d * np.sin(th)])

    ax.plot(0, 0, "o", color=INK, ms=6)
    ax.text(-0.15, -0.55, "eye", ha="center", fontsize=10)
    ax.plot([0, 11.6], [0, 0], color=GREY, lw=1.1, ls=(0, (6, 4)))
    ax.text(11.7, 0, "view axis", va="center", fontsize=9, color="0.35")

    alpha = np.arcsin(r / d)
    for s in (+1, -1):
        a = th + s * alpha
        ax.plot([0, 11.0 * np.cos(a)], [0, 11.0 * np.sin(a)], color=INK, lw=1.2)
    ax.add_patch(Circle(C, r, facecolor="white", edgecolor=INK, lw=1.8, zorder=3))
    ax.plot(*C, "o", color=INK, ms=3.5, zorder=4)

    xs = C[0]
    ys_lo, ys_hi = xs * np.tan(th - alpha), xs * np.tan(th + alpha)
    ax.plot([xs, xs], [-1.3, 6.2], color=GREY, lw=1.1, ls=(0, (2, 3)))
    ax.text(xs + 0.15, -1.35, "a plane through the centre,\nparallel to the screen",
            fontsize=8.5, color="0.35", va="bottom")

    ax.plot([xs - 0.14, xs - 0.14], [ys_lo, ys_hi], color=INK, lw=3.0,
            solid_capstyle="butt", zorder=5)
    ax.plot([xs + 0.14, xs + 0.14], [C[1] - r, C[1] + r], color="0.55", lw=3.0,
            solid_capstyle="butt", zorder=5)
    ax.annotate("what it actually needs:\nabout r / cos(theta)", (xs - 0.14, ys_hi),
                (xs - 4.6, 5.7), fontsize=9.5, ha="left",
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1))
    ax.annotate("the old screen-parallel\nquad: half-size r", (xs + 0.14, C[1] - r),
                (xs + 1.0, 1.2), fontsize=9.5, ha="left", color="0.3",
                arrowprops=dict(arrowstyle="-|>", color="0.4", lw=1.0))

    ax.add_patch(Wedge((0, 0), 2.6, 0, np.degrees(th), width=0.06, color=INK))
    ax.text(2.85, 0.72, "theta", fontsize=10)
    save(fig, "fig-offaxis.png")


# ---------------------------------------------------------------------------
# 4. The exact silhouette billboard
# ---------------------------------------------------------------------------
def fig_billboard():
    fig, ax = blank(9.0, 5.6, (-1.2, 12.4), (-4.6, 4.6))
    d, r = 8.0, 3.0
    C = np.array([d, 0.0])
    t = np.sqrt(d * d - r * r)
    rc = r * t / d
    Cc = t * t / d

    ax.plot(0, 0, "o", color=INK, ms=6)
    ax.text(-0.2, 0.3, "eye", ha="center", fontsize=10)
    ax.plot([0, d], [0, 0], color=GREY, lw=1.1, ls=(0, (6, 4)))

    alpha = np.arcsin(r / d)
    for s in (+1, -1):
        a = s * alpha
        ax.plot([0, 12.0 * np.cos(a)], [0, 12.0 * np.sin(a)], color=INK, lw=1.3)
        tx, ty = t * np.cos(a), t * np.sin(a)
        ax.plot([tx, C[0]], [ty, C[1]], color="0.5", lw=1.0, ls=(0, (2, 2)),
                zorder=4)
    ax.add_patch(Circle(C, r, facecolor="white", edgecolor=INK, lw=1.8, zorder=3))
    ax.plot(*C, "o", color=INK, ms=3.5, zorder=5)
    for s in (+1, -1):
        a = s * alpha
        ax.plot(t * np.cos(a), t * np.sin(a), "o", color=INK, ms=5, zorder=6)

    # the tangent circle, seen edge-on: this is where the quad goes
    ax.plot([Cc, Cc], [-rc, rc], color=INK, lw=4.0, solid_capstyle="butt", zorder=6)
    ax.plot(Cc, 0, "o", color=INK, ms=4, zorder=7)

    ax.annotate("the QUAD sits HERE, not at the centre:\n"
                "half-size rc = r*t/d, at distance t^2/d",
                (Cc, rc), (0.4, 3.9), fontsize=10,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
    ax.annotate("tangent point, at distance\nt = sqrt(d^2 - r^2)",
                (t * np.cos(alpha), -t * np.sin(alpha)), (0.4, -4.3), fontsize=10,
                ha="left", arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
    ax.annotate("", (0, -1.9), (d, -1.9),
                arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1.1))
    ax.text(d / 2 - 1.6, -2.5, "d = |C|", ha="center", fontsize=10)
    ax.annotate("", (d, 0), (d, r), arrowprops=dict(arrowstyle="<|-|>",
                                                    color=INK, lw=1.1), zorder=8)
    ax.text(d + 0.22, r / 2, "r", fontsize=11)
    save(fig, "fig-billboard.png")


# ---------------------------------------------------------------------------
# 5. Ray-sphere intersection
# ---------------------------------------------------------------------------
def fig_raysphere():
    fig, ax = blank(9.0, 5.0, (-1.0, 12.0), (-2.6, 3.6))
    C = np.array([9.0, 0.9])
    r = 1.9
    ax.plot(0, 0, "o", color=INK, ms=6)
    ax.text(-0.1, -0.5, "eye", ha="center", fontsize=10)
    ax.add_patch(Circle(C, r, fill=False, edgecolor=INK, lw=1.8))
    ax.plot(*C, "o", color=INK, ms=4)
    ax.text(C[0] + 0.15, C[1] + 0.2, "c", fontsize=11)

    dirv = np.array([1.0, 0.02])
    dirv /= np.linalg.norm(dirv)
    b = dirv @ C
    disc = b * b - (C @ C - r * r)
    t0 = b - np.sqrt(disc)
    t1 = b + np.sqrt(disc)

    ax.plot([0, 12.0 * dirv[0]], [0, 12.0 * dirv[1]], color=INK, lw=1.4)
    ax.text(11.3, 0.45, "dir", fontsize=10)
    foot = dirv * b
    ax.plot(*foot, "o", color="0.45", ms=4)
    ax.plot([foot[0], C[0]], [foot[1], C[1]], color="0.5", lw=1.0, ls=(0, (2, 2)))
    ax.text(foot[0] - 0.1, foot[1] - 0.65, "b = dot(dir, c)\n(the closest approach)",
            ha="center", va="top", fontsize=9)

    for t, lab in ((t0, "t = b - sqrt(disc)\nTHIS is the hit"), (t1, "far root,\nthrown away")):
        p = dirv * t
        ax.plot(*p, "o", color=INK, ms=6)
        ax.annotate(lab, p, (p[0] - 1.2, p[1] + (2.3 if t == t0 else 2.9)),
                    fontsize=9, ha="center",
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0))
    hit = dirv * t0
    n = (hit - C) / r
    arrow(ax, hit, hit + n * 1.5, lw=1.8)
    ax.text(*(hit + n * 1.75 + np.array([-0.15, -0.3])),
            "N = (hit - c)/r", fontsize=9.5, ha="right")
    ax.text(0.2, -2.2, "disc = b*b - (dot(c,c) - r*r)      disc < 0  ->  discard, the ray misses",
            fontsize=9.5, style="italic")
    save(fig, "fig-raysphere.png")


# ---------------------------------------------------------------------------
# 6. Depth buffer nonlinearity
# ---------------------------------------------------------------------------
def fig_depth():
    near, far = 10.411, 29.577
    d = np.linspace(near, far, 400)
    M22 = -(far + near) / (far - near)
    M23 = -2 * far * near / (far - near)
    z = (M22 * (-d) + M23) / d
    depth = 0.5 * z + 0.5

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(d, depth, color=INK, lw=2.0)
    ax.plot(d, (d - near) / (far - near), color=GREY, lw=1.4, ls=(0, (5, 4)))
    ax.text(24.0, 0.50, "linear, for comparison", color="0.35", fontsize=9.5,
            rotation=17)
    for dd, lab in ((12.0, "nearest bead\nrow"), (20.0, "middle of\nthe sheet"),
                    (28.077, "farthest bead")):
        v = (M22 * (-dd) + M23) / dd * 0.5 + 0.5
        ax.plot([dd, dd], [0, v], color="0.6", lw=0.9, ls=":")
        ax.plot([near, dd], [v, v], color="0.6", lw=0.9, ls=":")
        ax.plot(dd, v, "o", color=INK, ms=5)
        ax.annotate("%s\n%.3f" % (lab, v), (dd, v), (dd + 0.6, v - 0.16),
                    fontsize=8.5)
    ax.set_xlabel("distance along the view axis (sigma)")
    ax.set_ylabel("gl_FragDepth")
    ax.set_xlim(near, far)
    ax.set_ylim(0, 1.02)
    ax.set_title("half the depth buffer is spent on the first 3 sigma",
                 fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, "fig-depth.png")


# ---------------------------------------------------------------------------
# 7. G-buffer
# ---------------------------------------------------------------------------
def fig_gbuffer():
    fig, ax = blank(9.0, 5.9, (0, 12.6), (-0.9, 7.6))
    rows = [
        ("albedo", "RGBA16F", "rgb = bead colour (linear light)\na = per-bead strength (image fade)"),
        ("normal", "RGB16F", "view-space surface normal N"),
        ("view position", "RGBA32F", "the hit point in view space.\nz < 0 means 'a bead is here'"),
        ("depth", "DEPTH24", "written by hand with gl_FragDepth"),
    ]
    y = 6.4
    for name, fmt, desc in rows:
        ax.add_patch(Rectangle((0.4, y - 1.25), 3.1, 1.25, facecolor="white",
                               edgecolor=INK, lw=1.6))
        ax.text(1.95, y - 0.42, name, ha="center", fontsize=11, fontweight="bold")
        ax.text(1.95, y - 0.85, fmt, ha="center", fontsize=9, color="0.3",
                family="monospace")
        ax.text(3.9, y - 0.62, desc, va="center", fontsize=9.5)
        y -= 1.5
    ax.text(0.4, 7.15, "ONE fragment shader invocation writes all four",
            fontsize=10.5, fontweight="bold")
    ax.text(0.4, 0.30, "every later pass reads these four textures and nothing else --\n"
                       "no bead, no vertex, no scene graph is ever consulted again",
            fontsize=9.5, style="italic", va="top")
    save(fig, "fig-gbuffer.png")


# ---------------------------------------------------------------------------
# 8. SSAO: hemisphere sampling and reprojection
# ---------------------------------------------------------------------------
def fig_ssao():
    fig, ax = blank(9.0, 4.3, (-0.5, 12.0), (-0.9, 5.1))
    # surface
    xs = np.linspace(1.2, 10.5, 300)
    ys = 1.6 + 0.9 * np.exp(-((xs - 4.2) ** 2) / 2.2) + 1.5 * np.exp(-((xs - 8.6) ** 2) / 1.2)
    ax.plot(xs, ys, color=INK, lw=2.0)
    ax.fill_between(xs, ys, -0.8, color=PALE, zorder=0)
    P = np.array([6.2, 1.66])
    ax.plot(*P, "o", color=INK, ms=7)
    ax.text(P[0] - 0.1, P[1] - 0.55, "P (this pixel)", ha="center", fontsize=9.5)
    N = np.array([0.16, 1.0])
    N /= np.linalg.norm(N)
    arrow(ax, P, P + N * 1.5, lw=1.8)
    ax.text(*(P + N * 1.65 + np.array([0.12, 0.05])), "N", fontsize=11)

    R = 2.1
    thetas = np.linspace(0, np.pi, 200)
    base = np.array([-N[1], N[0]])
    pts = np.array([P + R * (np.cos(t) * base + np.sin(t) * N) for t in thetas])
    ax.plot(pts[:, 0], pts[:, 1], color="0.55", lw=1.1, ls=(0, (4, 3)))
    ax.text(P[0] + R * 0.75, P[1] + R * 0.85, "radius = 2 x bead radius", fontsize=9,
            color="0.35")

    rng = np.random.default_rng(3)
    for i in range(9):
        xi = np.array([(i * 0.7548776662) % 1.0, (i * 0.5698402909) % 1.0])
        rr = np.sqrt(xi[0]) * R * 0.92
        phi = np.pi * xi[1] * 0.9 + 0.15
        s = P + rr * (np.cos(phi) * base + np.sin(phi) * N)
        buried = np.interp(s[0], xs, ys) > s[1]
        ax.plot(*s, "o", color=INK if buried else "0.55", ms=5,
                markerfacecolor=INK if buried else "white")
    ax.plot([], [], "o", color=INK, ms=5, label="buried -> counts as occluded")
    ax.plot([], [], "o", color="0.55", markerfacecolor="white", ms=5,
            label="in open air -> free")
    ax.legend(loc="upper left", fontsize=9, frameon=False,
              bbox_to_anchor=(0.0, 1.0))
    save(fig, "fig-ssao.png")


# ---------------------------------------------------------------------------
# 9. Contact-shadow march
# ---------------------------------------------------------------------------
def fig_march():
    fig, ax = blank(9.0, 3.6, (-0.5, 12.0), (0.0, 5.6))
    ax.add_patch(Circle((3.0, 1.4), 1.35, fill=False, edgecolor=INK, lw=1.8))
    ax.add_patch(Circle((6.8, 2.2), 1.35, facecolor=PALE, edgecolor=INK, lw=1.8))
    ax.text(6.8, 2.2, "occluder", ha="center", va="center", fontsize=9.5)
    P = np.array([3.55, 2.55])
    ax.plot(*P, "o", color=INK, ms=7)
    ax.text(3.1, 2.95, "P", fontsize=11)
    L = np.array([1.0, 0.42])
    L /= np.linalg.norm(L)
    N = (P - np.array([3.0, 1.4]))
    N /= np.linalg.norm(N)
    origin = P + N * 0.28
    total = 6.4
    for i in range(16):
        f = (i + 0.5) / 16.0
        s = origin + L * (total * f * f)
        hit = np.linalg.norm(s - np.array([6.8, 2.2])) < 1.35
        ax.plot(*s, "o", color=INK, ms=4.6 if hit else 3.0,
                markerfacecolor=INK if hit else "white")
    ax.plot([origin[0], origin[0] + L[0] * total], [origin[1], origin[1] + L[1] * total],
            color="0.6", lw=0.9, ls=(0, (3, 3)), zorder=0)
    arrow(ax, (9.4, 4.4), (10.9, 5.03), lw=1.6)
    ax.text(9.7, 4.75, "to the sun", fontsize=9.5)
    save(fig, "fig-march.png")


# ---------------------------------------------------------------------------
# 10. Why the blur needs a mask
# ---------------------------------------------------------------------------
def fig_blurmask():
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.4))
    gx, gy = np.meshgrid(np.linspace(-1.6, 1.6, 160), np.linspace(-1.6, 1.6, 160))
    rr = np.hypot(gx, gy)
    inside = rr < 1.0
    # A bead whose ambient occlusion is fairly uniformly dark, on open background.
    ao = np.where(inside, 0.30 + 0.12 * rr, 1.0)
    ao += np.where(inside, 0.09 * np.random.default_rng(1).standard_normal(ao.shape), 0)

    def boxblur(a, k=9):
        p = np.pad(a, k // 2, mode="edge")
        out = np.zeros_like(a)
        for i in range(k):
            for j in range(k):
                out += p[i:i + a.shape[0], j:j + a.shape[1]]
        return out / (k * k)

    naive = boxblur(ao)
    num = boxblur(np.where(inside, ao, 0.0))
    den = boxblur(inside.astype(float))
    masked = np.where(den > 0, num / np.maximum(den, 1e-6), 1.0)
    masked = np.where(inside, masked, 1.0)

    for ax, img, title in zip(axes, (ao, naive, masked),
                              ("raw AO (noisy)", "naive blur:\nbright HALO", "masked blur")):
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.15, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(INK)
    fig.tight_layout()
    save(fig, "fig-blurmask.png")


# ---------------------------------------------------------------------------
# 11. The outline threshold
# ---------------------------------------------------------------------------
def fig_outline():
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    u = np.linspace(0, 0.985, 400)
    slope = u / np.sqrt(1 - u * u)
    ax.plot(u, slope, color=INK, lw=2.0)
    ax.axhline(2.0647, color=GREY, lw=1.3, ls=(0, (5, 4)))
    ax.plot([0.9, 0.9], [0, 2.0647], color="0.6", lw=1.0, ls=":")
    ax.plot(0.9, 2.0647, "o", color=INK, ms=6)
    ax.annotate("outline_edge_fraction = 0.90\nslope 2.065", (0.9, 2.0647),
                (0.44, 3.4), fontsize=9.5,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1))
    ax.annotate("0.94 -> hairline", (0.94, 2.7566), (0.62, 5.2), fontsize=9,
                arrowprops=dict(arrowstyle="-|>", color="0.4", lw=1.0))
    ax.annotate("0.85 -> heavy ink", (0.85, 1.6136), (0.30, 1.05), fontsize=9,
                arrowprops=dict(arrowstyle="-|>", color="0.4", lw=1.0))
    ax.set_xlabel("u = distance from the bead's centre, as a fraction of its radius")
    ax.set_ylabel("depth slope  u / sqrt(1 - u^2)")
    ax.set_ylim(0, 6)
    ax.set_xlim(0, 1)
    ax.set_title("a sphere's own depth gradient, and where you cut it", fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, "fig-outline.png")


# ---------------------------------------------------------------------------
# 12. Depth of field: CoC and the gather
# ---------------------------------------------------------------------------
def fig_dof():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.8))
    front, back = 11.911, 28.077
    span = back - front
    focus = front + 0.25 * span
    rng = 1.5 * span
    d = np.linspace(front, back, 300)
    coc = np.clip(np.abs(d - focus) / rng, 0, 1) * 6.0
    ax1.plot(d, coc, color=INK, lw=2.0)
    ax1.axvline(focus, color=GREY, lw=1.2, ls=(0, (5, 4)))
    ax1.text(focus + 0.4, 5.2, "focus plane\n(dof_focus = 0.25)", fontsize=9)
    ax1.set_xlabel("distance (sigma)")
    ax1.set_ylabel("blur radius (px)")
    ax1.set_title("circle of confusion, the sheet's settings", fontsize=10)
    ax1.set_ylim(0, 6.5)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    n = 24
    ga = 2.39996323
    i = np.arange(1, n + 1)
    t = i / n
    a = i * ga
    ax2.plot(np.sqrt(t) * np.cos(a), np.sqrt(t) * np.sin(a), "o", color=INK, ms=5)
    ax2.plot(0, 0, "o", color=INK, ms=7)
    ax2.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor=GREY, lw=1.2,
                         ls=(0, (4, 3))))
    ax2.set_aspect("equal")
    ax2.set_xlim(-1.25, 1.25)
    ax2.set_ylim(-1.25, 1.25)
    ax2.axis("off")
    ax2.set_title("24 taps, golden-angle spiral\n(sqrt(t) spaces them by AREA)",
                  fontsize=10)
    fig.tight_layout()
    save(fig, "fig-dof.png")


# ---------------------------------------------------------------------------
# 13. Gamma
# ---------------------------------------------------------------------------
def fig_gamma():
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    x = np.linspace(0, 1, 300)
    ax.plot(x, x ** 2.2, color=INK, lw=2.0, label="display -> linear  (^2.2)")
    ax.plot(x, x ** (1 / 2.2), color=INK, lw=1.6, ls=(0, (5, 3)),
            label="linear -> display  (^1/2.2)")
    ax.plot(x, x, color=GREY, lw=1.0, ls=":")
    v = (247 / 255.0)
    ax.plot([v, v], [0, v ** 2.2], color="0.6", lw=1.0, ls=":")
    ax.plot(v, v ** 2.2, "o", color=INK, ms=5)
    ax.annotate("the bead's yellow, 247/255\n-> 0.932 linear", (v, v ** 2.2),
                (0.30, 0.90), fontsize=9,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0))
    v2 = (122 / 255.0)
    ax.plot(v2, v2 ** 2.2, "o", color=INK, ms=5)
    ax.annotate("the blue, 122/255\n-> 0.198 linear  (not 0.48!)", (v2, v2 ** 2.2),
                (0.05, 0.42), fontsize=9,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0))
    ax.set_xlabel("value in")
    ax.set_ylabel("value out")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, "fig-gamma.png")


# ---------------------------------------------------------------------------
# 14. Instancing
# ---------------------------------------------------------------------------
def fig_instancing():
    fig, ax = blank(9.0, 4.4, (0, 12.6), (0, 5.8))
    box(ax, 0.4, 3.4, 3.2, 1.5, "quad VBO", ["4 corners", "32 bytes, uploaded once"])
    box(ax, 0.4, 1.1, 3.2, 1.8, "instance VBO",
        ["10 floats per bead:", "centre, radius, director,", "bright, energy, fade"])
    box(ax, 5.2, 2.0, 3.0, 2.4, "vertex shader",
        ["runs 4 x N times", "(4 corners x N beads)"])
    box(ax, 9.2, 2.0, 3.0, 2.4, "fragment shader",
        ["runs once per", "covered PIXEL"])
    arrow(ax, (3.6, 4.15), (5.2, 3.5))
    arrow(ax, (3.6, 2.0), (5.2, 2.9))
    arrow(ax, (8.2, 3.2), (9.2, 3.2))
    ax.text(0.4, 0.45, "ONE draw call: render(mode=TRIANGLE_STRIP, instances=900).\n"
                       "The 900 in that argument is the entire per-bead cost on the CPU.",
            fontsize=9.5, style="italic", va="bottom")
    save(fig, "fig-instancing.png")


if __name__ == "__main__":
    fig_pipeline()
    fig_spaces()
    fig_offaxis()
    fig_billboard()
    fig_raysphere()
    fig_depth()
    fig_gbuffer()
    fig_ssao()
    fig_march()
    fig_blurmask()
    fig_outline()
    fig_dof()
    fig_gamma()
    fig_instancing()

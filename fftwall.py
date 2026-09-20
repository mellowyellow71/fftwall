#!/usr/bin/env python3
"""fftwall — procedurally generated fourier transform wallpapers.

iterate a chaotic map, render its point-cloud density and the 2d fft
log-magnitude spectrum through a colormap. the spatial-domain attractor
can be saved too (--domain space|both).
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

# ---------------------------------------------------------------- colormaps

# custom ramps: (position, hex) anchors, interpolated in rgb.
# dragon/ember are kanagawa-dragon; gold is navy->yellow glow; dust is
# dark film-grain warm cream.
CUSTOM = {
    "dragon": [(0.00, "0d0c0c"), (0.35, "2d4f67"), (0.72, "8ba4b0"), (1.00, "c5c9c5")],
    "ember":  [(0.00, "0d0c0c"), (0.38, "3a2426"), (0.74, "c4746e"), (1.00, "c4b28a")],
    "gold":   [(0.00, "0b1229"), (0.55, "27508f"), (0.85, "9db3cf"), (1.00, "ffe94a")],
    "dust":   [(0.00, "0d0c0c"), (0.45, "3f382e"), (0.78, "a08d72"), (0.93, "e7d8b1"), (1.00, "f0b070")],
}


def lut(name):
    """256x3 uint8 lookup table for a colormap name."""
    if name in CUSTOM:
        pos = np.array([p for p, _ in CUSTOM[name]])
        rgb = np.array([[int(h[i:i + 2], 16) for i in (0, 2, 4)] for _, h in CUSTOM[name]])
        t = np.linspace(0, 1, 256)
        return np.stack([np.interp(t, pos, rgb[:, c]) for c in range(3)], axis=1).astype(np.uint8)
    from matplotlib import colormaps
    return (colormaps[name](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)


def render(v01, cmap):
    """normalized 0..1 field -> rgb image via colormap."""
    return Image.fromarray(lut(cmap)[(np.clip(v01, 0, 1) * 255).astype(np.uint8)])


def norm(a, lo, hi):
    l, h = np.percentile(a, [lo, hi])
    return np.clip((a - l) / (h - l + 1e-12), 0, 1)


def density(x, y, w, h):
    """accumulate points into a log-density field of size h x w."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    x0, x1 = np.percentile(x, [0.2, 99.8])
    y0, y1 = np.percentile(y, [0.2, 99.8])
    mx, my = (x1 - x0) * 0.04, (y1 - y0) * 0.04
    x0, x1, y0, y1 = x0 - mx, x1 + mx, y0 - my, y1 + my
    ix = ((x - x0) / (x1 - x0) * (w - 1)).astype(np.int64)
    iy = ((y - y0) / (y1 - y0) * (h - 1)).astype(np.int64)
    m = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    d = np.bincount(iy[m] * w + ix[m], minlength=h * w).reshape(h, w)
    return np.log1p(d.astype(np.float32))


# ---------------------------------------------------------------- systems

def gen_ikeda(w, h, rng):
    """ikeda map point cloud density."""
    u = rng.choice([0.90, 0.918, 0.97])
    p = rng.normal(0, 0.5, (4096, 2))
    xs, ys = [], []
    for i in range(2500):
        x, y = p[:, 0], p[:, 1]
        t = 0.4 - 6.0 / (1 + x * x + y * y)
        p = np.stack([1 + u * (x * np.cos(t) - y * np.sin(t)),
                      u * (x * np.sin(t) + y * np.cos(t))], axis=1)
        np.clip(p, -50, 50, out=p)
        if i >= 60:
            xs.append(p[:, 0].copy())
            ys.append(p[:, 1].copy())
    return density(np.concatenate(xs), np.concatenate(ys), w, h)


CLIFFORD = [(-1.4, 1.6, 1.0, 0.7), (1.7, 1.7, 0.6, 1.2), (-1.7, 1.3, -0.1, -1.2),
            (-1.8, -2.0, -0.5, -0.9), (1.5, -1.8, 1.6, 0.9), (-1.7, 1.8, -1.9, -0.4)]


def gen_clifford(w, h, rng):
    """clifford strange attractor point-cloud density."""
    a, b, c, d = np.array(CLIFFORD[rng.integers(len(CLIFFORD))]) + rng.normal(0, 0.03, 4)
    p = rng.uniform(-1, 1, (4096, 2))
    xs, ys = [], []
    for i in range(2500):
        x, y = p[:, 0], p[:, 1]
        p = np.stack([np.sin(a * y) + c * np.cos(a * x),
                      np.sin(b * x) + d * np.cos(b * y)], axis=1)
        if i >= 50:
            xs.append(p[:, 0].copy())
            ys.append(p[:, 1].copy())
    return density(np.concatenate(xs), np.concatenate(ys), w, h)


DEJONG = [(1.4, -2.3, 2.4, -2.1), (2.01, -2.53, 1.61, -0.33), (-2.7, -0.09, -0.86, -2.2),
          (-2.24, 0.43, -0.65, -2.43), (1.641, 1.902, 0.316, 1.525), (-2.0, -2.0, -1.2, 2.0),
          (0.970, -1.899, 1.381, -1.506), (-0.827, -1.637, 1.659, -0.943)]


def gen_dejong(w, h, rng):
    """peter de jong map — smoky folded filament clouds."""
    a, b, c, d = np.array(DEJONG[rng.integers(len(DEJONG))]) + rng.normal(0, 0.02, 4)
    p = rng.uniform(-2, 2, (4096, 2))
    xs, ys = [], []
    for i in range(2500):
        x, y = p[:, 0], p[:, 1]
        p = np.stack([np.sin(a * y) - np.cos(b * x),
                      np.sin(c * x) - np.cos(d * y)], axis=1)
        if i >= 50:
            xs.append(p[:, 0].copy())
            ys.append(p[:, 1].copy())
    return density(np.concatenate(xs), np.concatenate(ys), w, h)


SVENSSON = [(1.40, 1.56, 1.40, -6.56), (-2.538, 1.362, 1.315, 0.513), (1.913, 2.796, 1.468, 1.01),
            (-2.337, -2.337, 0.533, 1.378), (1.245, -1.251, 1.816, 1.098)]


def gen_svensson(w, h, rng):
    """svensson map — looping ribbon sheets."""
    a, b, c, d = np.array(SVENSSON[rng.integers(len(SVENSSON))]) + rng.normal(0, 0.02, 4)
    p = rng.uniform(-1, 1, (4096, 2))
    xs, ys = [], []
    for i in range(2500):
        x, y = p[:, 0], p[:, 1]
        p = np.stack([d * np.sin(a * x) - np.sin(b * y),
                      c * np.cos(a * x) + np.cos(b * y)], axis=1)
        if i >= 50:
            xs.append(p[:, 0].copy())
            ys.append(p[:, 1].copy())
    return density(np.concatenate(xs), np.concatenate(ys), w, h)


CURL = [(5, 1.5), (7, 2.5), (9, 4.0), (12, 6.0)]  # (fourier modes, rms cycles across the frame)


def gen_curl(w, h, rng):
    """streamlines of a divergence-free flow — silk and lace.

    psi is a sum of random fourier modes and the flow is (dpsi/dy, -dpsi/dx),
    so it is divergence-free by construction and particles ride the level
    sets of psi. unit speed, one pixel per step, random lifetimes.
    """
    nmodes, cycles = CURL[rng.integers(len(CURL))]
    f = rng.normal(0, cycles, (nmodes, 2))
    k = 2 * np.pi * f
    amp = 1 / (1 + (f * f).sum(1) / cycles ** 2)
    ph = rng.uniform(0, 2 * np.pi, nmodes)
    s = max(w, h)
    n = max(512, w * h // 2500)
    box = np.array([w - 1, h - 1], np.float64)

    def spawn(m):
        return rng.uniform(0, box, (m, 2)), rng.integers(150, 900, m)

    p, life = spawn(n)
    weight = rng.exponential(1.0, n)  # some threads catch more light
    xs, ys, ws = [], [], []
    for _ in range(1500):
        c = amp * np.cos(p / s @ k.T + ph)
        v = np.stack([c @ k[:, 1], -(c @ k[:, 0])], axis=1)
        p += v / (np.hypot(v[:, 0], v[:, 1])[:, None] + 1e-9)
        life -= 1
        dead = (life <= 0) | (p < 0).any(1) | (p >= box).any(1)
        if dead.any():
            p[dead], life[dead] = spawn(int(dead.sum()))
        xs.append(p[:, 0].copy())
        ys.append(p[:, 1].copy())
        ws.append(weight)
    x, y, wt = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    # bilinear splat so sub-pixel positions draw smooth lines
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x - x0, y - y0
    d = np.zeros(h * w, np.float64)
    for dx, dy, wgt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                        (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        d += np.bincount((y0 + dy) * w + x0 + dx, wt * wgt, minlength=h * w)
    return np.log1p(d.reshape(h, w).astype(np.float32))


# markus-lyapunov windows: (forcing pattern, a range, b range)
LYAPUNOV = [("AB", (2.4, 4.0), (2.4, 4.0)), ("AABAB", (2.5, 4.0), (2.5, 4.0)),
            ("BBBBBBAAAAAA", (3.4, 4.0), (2.5, 3.4)), ("AAB", (3.0, 4.0), (2.6, 4.0)),
            ("ABB", (2.6, 4.0), (3.0, 4.0)), ("AABA", (3.2, 4.0), (3.2, 4.0))]


def gen_lyapunov(w, h, rng):
    """markus-lyapunov fractal: lyapunov exponent of the logistic map
    x -> r x (1 - x), r alternating between a (across) and b (down) in a
    fixed pattern. stable regions glow, chaos stays dark."""
    seq, (a0, a1), (b0, b1) = LYAPUNOV[rng.integers(len(LYAPUNOV))]
    a0, b0 = a0 + rng.uniform(-0.1, 0.1), b0 + rng.uniform(-0.1, 0.1)
    a1, b1 = min(4.0, a1 + rng.uniform(-0.1, 0.05)), min(4.0, b1 + rng.uniform(-0.1, 0.05))
    a = np.linspace(a0, a1, w, dtype=np.float32)[None, :]
    b = np.linspace(b0, b1, h, dtype=np.float32)[:, None]
    x = np.full((h, w), 0.5, np.float32)
    lyap = np.zeros((h, w), np.float32)
    transient, iters = 150, 450
    for i in range(transient + iters):
        r = a if seq[i % len(seq)] == "A" else b
        if i >= transient:
            lyap += np.log(np.abs(r * (1 - 2 * x)) + 1e-12)
        x = r * x * (1 - x)
    return -lyap / iters


SYSTEMS = {"clifford": gen_clifford, "ikeda": gen_ikeda,
           "dejong": gen_dejong, "svensson": gen_svensson,
           "curl": gen_curl, "lyapunov": gen_lyapunov}


# ---------------------------------------------------------------- main

def spectrum(field, zoom=1, corner=False, gamma=1.25):
    F = np.abs(np.fft.fft2(field))
    S = np.log1p(F if corner else np.fft.fftshift(F))
    h, w = S.shape
    if zoom > 1:
        if corner:
            S = S[:h // zoom, :w // zoom]
        else:
            ch, cw = h // (2 * zoom), w // (2 * zoom)
            S = S[h // 2 - ch:h // 2 + ch, w // 2 - cw:w // 2 + cw]
    return norm(S, 45, 99.95) ** gamma


def main():
    ap = argparse.ArgumentParser(prog="fftwall", description=__doc__)
    ap.add_argument("system", nargs="?", default="clifford", choices=list(SYSTEMS) + ["all"])
    ap.add_argument("-s", "--size", default="3840x2160", help="WxH (default 3840x2160)")
    ap.add_argument("-c", "--cmap", default="inferno",
                    help="inferno, magma, viridis, cubehelix, bone, twilight, dragon, ember, ...")
    ap.add_argument("-d", "--domain", default="freq", choices=["freq", "space", "both"])
    ap.add_argument("-z", "--zoom", type=int, default=1,
                    help="magnify the spectrum center (2-4 suits the attractors)")
    ap.add_argument("-g", "--grain", type=float, default=0.0,
                    help="noise mixed into the field before the fft")
    ap.add_argument("--corner", action="store_true",
                    help="skip the fftshift: the spectrum glow sits in the corner")
    ap.add_argument("--gamma", type=float, default=1.25,
                    help="spectrum tone curve; <1 spreads the glow and lifts "
                         "the background, >1 darkens (default 1.25)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("-o", "--outdir", default=".")
    args = ap.parse_args()

    w, h = (int(v) for v in args.size.lower().split("x"))
    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
    os.makedirs(args.outdir, exist_ok=True)

    for name in (list(SYSTEMS) if args.system == "all" else [args.system]):
        rng = np.random.default_rng(seed)
        field = SYSTEMS[name](w, h, rng)
        outs = []
        fs = field
        if args.grain > 0:
            fs = fs + (args.grain * field.std()
                       * rng.standard_normal(field.shape).astype(np.float32))
        if args.domain in ("freq", "both"):
            p = os.path.join(args.outdir, f"{name}-freq-{seed}.png")
            v = spectrum(fs, args.zoom, args.corner, args.gamma)
            if v.shape != (h, w):
                v = np.asarray(Image.fromarray(v.astype(np.float32)).resize((w, h), Image.BILINEAR))
            render(v, args.cmap).save(p)
            outs.append(p)
        if args.domain in ("space", "both"):
            p = os.path.join(args.outdir, f"{name}-space-{seed}.png")
            render(norm(field, 0.5, 99.7), args.cmap).save(p)
            outs.append(p)
        print(f"{name} · seed {seed} · {w}x{h} · {args.cmap} · {' '.join(outs)}")


if __name__ == "__main__":
    sys.exit(main())

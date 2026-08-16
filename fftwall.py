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

# custom kanagawa-dragon ramps: (position, hex) anchors, interpolated in rgb
KANAGAWA = {
    "dragon": [(0.00, "0d0c0c"), (0.35, "2d4f67"), (0.72, "8ba4b0"), (1.00, "c5c9c5")],
    "ember":  [(0.00, "0d0c0c"), (0.38, "3a2426"), (0.74, "c4746e"), (1.00, "c4b28a")],
}


def lut(name):
    """256x3 uint8 lookup table for a colormap name."""
    if name in KANAGAWA:
        pos = np.array([p for p, _ in KANAGAWA[name]])
        rgb = np.array([[int(h[i:i + 2], 16) for i in (0, 2, 4)] for _, h in KANAGAWA[name]])
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


SYSTEMS = {"clifford": gen_clifford, "ikeda": gen_ikeda,
           "dejong": gen_dejong, "svensson": gen_svensson}


# ---------------------------------------------------------------- main

def spectrum(field, zoom=1):
    S = np.log1p(np.fft.fftshift(np.abs(np.fft.fft2(field))))
    if zoom > 1:
        h, w = S.shape
        ch, cw = h // (2 * zoom), w // (2 * zoom)
        S = S[h // 2 - ch:h // 2 + ch, w // 2 - cw:w // 2 + cw]
    return norm(S, 45, 99.95) ** 1.25


def main():
    ap = argparse.ArgumentParser(prog="fftwall", description=__doc__)
    ap.add_argument("system", nargs="?", default="clifford", choices=list(SYSTEMS) + ["all"])
    ap.add_argument("-s", "--size", default="3840x2160", help="WxH (default 3840x2160)")
    ap.add_argument("-c", "--cmap", default="inferno",
                    help="inferno, magma, viridis, cubehelix, bone, twilight, dragon, ember, ...")
    ap.add_argument("-d", "--domain", default="freq", choices=["freq", "space", "both"])
    ap.add_argument("-z", "--zoom", type=int, default=1,
                    help="magnify the spectrum center (2-4 suits the attractors)")
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
        if args.domain in ("freq", "both"):
            p = os.path.join(args.outdir, f"{name}-freq-{seed}.png")
            img = render(spectrum(field, args.zoom), args.cmap)
            if img.size != (w, h):
                img = img.resize((w, h), Image.NEAREST)
            img.save(p)
            outs.append(p)
        if args.domain in ("space", "both"):
            p = os.path.join(args.outdir, f"{name}-space-{seed}.png")
            render(norm(field, 0.5, 99.7), args.cmap).save(p)
            outs.append(p)
        print(f"{name} · seed {seed} · {w}x{h} · {args.cmap} · {' '.join(outs)}")


if __name__ == "__main__":
    sys.exit(main())

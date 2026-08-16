#!/usr/bin/env python3
"""fftwall — procedurally generated fourier transform wallpapers.

simulate a mathematical system, take the 2d fft of the field,
render the log-magnitude spectrum with a colormap. the spatial-
domain source can be saved too (--domain space|both).
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


# ---------------------------------------------------------------- helpers

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


def fractal_noise(w, h, rng, slope=1.6):
    """1/f^slope noise via spectral synthesis."""
    white = rng.standard_normal((h, w))
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f = np.hypot(fx, fy)
    filt = 1.0 / (f + 1.0 / max(w, h)) ** slope
    field = np.fft.ifft2(np.fft.fft2(white) * filt).real
    return (field - field.mean()) / (field.std() + 1e-12)


# ---------------------------------------------------------------- systems

def gen_kuramoto(w, h, rng, cells_x=45):
    """lattice of coupled phase oscillators (kuramoto on a torus grid).

    each cell shows its oscillator: a waveform trace over cell-periodic
    stripes. the repeating cells put sharp harmonics in the spectrum;
    the traces and noise fill it with speckle.
    """
    cell_w = max(24, w // cells_x)
    cell_h = max(16, int(cell_w / 1.74))
    nx, ny = w // cell_w, h // cell_h
    ox, oy = (w - nx * cell_w) // 2, (h - ny * cell_h) // 2

    omega = rng.normal(1.0, 0.28, (ny, nx))
    theta = rng.uniform(0, 2 * np.pi, (ny, nx))
    K, dt = 0.9, 0.05

    def step(th):
        s = sum(np.sin(np.roll(th, sh, ax) - th) for sh, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1)))
        return th + (omega + K * s) * dt

    for _ in range(400):
        theta = step(theta)
    trace = np.empty((ny, nx, cell_w), np.float32)
    for i in range(cell_w):
        for _ in range(2):
            theta = step(theta)
        trace[:, :, i] = np.sin(theta)

    field = np.zeros((h, w), np.float32)

    # cell-periodic stripes, frequency tied to each oscillator's omega
    f_int = np.clip(np.round(omega * 3), 1, 8)
    xx = np.arange(cell_w)
    stripes = 0.5 + 0.5 * np.sin(2 * np.pi * f_int[:, :, None] * xx / cell_w + theta[:, :, None])
    env = (0.25 + 0.75 * np.hanning(cell_h)).astype(np.float32)
    block = (env[None, :, None, None] * stripes[:, None, :, :] * 0.3).astype(np.float32)
    field[oy:oy + ny * cell_h, ox:ox + nx * cell_w] = \
        block.transpose(0, 1, 2, 3).reshape(ny, cell_h, nx, cell_w).reshape(ny * cell_h, nx * cell_w)

    # waveform traces
    cy = np.arange(ny)[:, None, None] * cell_h + oy + cell_h // 2
    cx = np.arange(nx)[None, :, None] * cell_w + ox + xx[None, None, :]
    ty = np.clip(cy - np.round(trace * 0.38 * cell_h).astype(np.int64), 0, h - 1)
    tx = np.broadcast_to(cx, ty.shape)
    for dy, amp in ((0, 0.9), (1, 0.45), (-1, 0.45)):
        np.add.at(field, (np.clip(ty + dy, 0, h - 1).ravel(), tx.ravel()), amp)

    # faint grid lines + noise
    field[:, np.clip(ox + np.arange(nx + 1) * cell_w, 0, w - 1)] += 0.07
    field[np.clip(oy + np.arange(ny + 1) * cell_h, 0, h - 1), :] += 0.07
    field += rng.normal(0, 0.03, (h, w)).astype(np.float32)
    return field


def gen_waves(w, h, rng, n=28):
    """interfering plane waves over fractal noise."""
    y, x = np.mgrid[0:h, 0:w]
    X, Y = x / w, y / w
    field = np.zeros((h, w), np.float32)
    k = 10 ** rng.uniform(np.log10(3), np.log10(80), n)
    ang = rng.uniform(0, np.pi, n)
    amp = 1.0 / k ** 0.7
    phs = rng.uniform(0, 2 * np.pi, n)
    for i in range(n):
        field += (amp[i] * np.sin(2 * np.pi * k[i] * (np.cos(ang[i]) * X + np.sin(ang[i]) * Y) + phs[i])).astype(np.float32)
    field = norm(field, 1, 99).astype(np.float32) + 0.18 * fractal_noise(w, h, rng).astype(np.float32)
    r2 = ((X - 0.5 * w / w) ** 2 + (Y - 0.5 * h / w) ** 2)
    return field * (0.35 + 0.65 * np.exp(-r2 * 9)).astype(np.float32)


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


def gen_lorenz(w, h, rng):
    """lorenz attractor trajectories, x-z projection density."""
    P = 2048
    s = np.stack([rng.normal(0, 8, P), rng.normal(0, 8, P), rng.normal(25, 8, P)])

    def f(s):
        x, y, z = s
        return np.stack([10 * (y - x), x * (28 - z) - y, x * y - (8 / 3) * z])

    dt, xs, zs = 0.005, [], []
    for i in range(3600):
        k1 = f(s); k2 = f(s + dt / 2 * k1); k3 = f(s + dt / 2 * k2); k4 = f(s + dt * k3)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if i >= 200:
            xs.append(s[0].copy())
            zs.append(s[2].copy())
    return density(np.concatenate(xs), -np.concatenate(zs), w, h)


SYSTEMS = {"kuramoto": gen_kuramoto, "waves": gen_waves, "ikeda": gen_ikeda, "lorenz": gen_lorenz}


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
    ap.add_argument("system", nargs="?", default="kuramoto", choices=list(SYSTEMS) + ["all"])
    ap.add_argument("-s", "--size", default="3840x2160", help="WxH (default 3840x2160)")
    ap.add_argument("-c", "--cmap", default="inferno",
                    help="inferno, magma, viridis, cubehelix, bone, twilight, dragon, ember, ...")
    ap.add_argument("-d", "--domain", default="freq", choices=["freq", "space", "both"])
    ap.add_argument("-z", "--zoom", type=int, default=1,
                    help="magnify the spectrum center (2-4 suits lorenz/ikeda)")
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

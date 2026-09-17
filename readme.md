fftwall

procedurally generated fourier transform wallpapers.

iterate a chaotic map → point-cloud density → 2d fft → log-magnitude → colormap.
the attractor itself and its spectrum are both wallpapers.

## setup

```
python3 -m venv .venv
.venv/bin/pip install numpy pillow matplotlib
```

## use

```
.venv/bin/python fftwall.py                        # clifford, 4k, inferno
.venv/bin/python fftwall.py svensson -d both       # spectrum + the attractor itself
.venv/bin/python fftwall.py dejong -c dragon       # kanagawa-dragon palette
.venv/bin/python fftwall.py ikeda -z 2             # magnify spectrum center
.venv/bin/python fftwall.py all -s 4320x2400       # every system at 4k+
.venv/bin/python fftwall.py clifford --seed 7 -o out
```

every run without `--seed` is a new wallpaper; the seed is printed and
baked into the filename, so re-render any keeper at a bigger size.
render at your display's native resolution — the speckle is per-pixel,
and downscaling averages it away.

## flags

- `-s WxH` size, `-c` colormap, `-o` output dir, `--seed` reproduce a run
- `-d freq|space|both` — spectrum, the attractor itself, or both
- `-z` magnifies the spectrum center (2-4 suits the attractors)
- `-g/--grain` mixes noise into the field before the fft
- `--corner` skips the fftshift so the glow sits in a corner
- `--gamma` spectrum tone curve: <1 spreads the glow and lifts the
  background, >1 darkens (default 1.25)

## systems

- `clifford` — clifford strange attractor; radiating filament starburst
- `ikeda` — ikeda map (light in a ring cavity); swirling interference rings
- `dejong` — peter de jong map; smoky folded filament clouds
- `svensson` — svensson map; looping ribbon sheets and glowing tori
- `curl` — divergence-free vector-field streamlines; silk and lace
- `lyapunov` — lyapunov exponent map of two coupled logistic maps;
  sharp fractal boundaries between order and chaos

each seed also picks a curated parameter set, so silhouettes vary a lot
run to run. the space domain (`-d space` or `-d both`) is often the
better wallpaper — the spectra like `-z 2`. `curl` and `lyapunov` are
especially vivid in the spatial domain (`-d space`), `curl` benefits from
a little grain (`-g 0.05`) to soften the lattice in its spectrum.

## colormaps

`dragon` and `ember` are kanagawa-dragon ramps, `gold` is a navy→yellow
glow, `dust` is dark film-grain cream; any matplotlib name works
(`inferno`, `magma`, `viridis`, `cubehelix`, `bone`, `twilight`, ...).

## set as wallpaper (macos)

```
osascript -e 'tell application "System Events" to set picture of every desktop to "/path/to/out.png"'
```

## credit

inspired by [PureMathArt](https://wallhaven.cc/user/PureMathArt)'s
fourier-transform series on wallhaven — reverse-engineering
[one of their spectra](https://wallhaven.cc/w/jeyg2w) (a coupled-oscillator
lattice) is what led to this tool.

## license

[mit](LICENSE)

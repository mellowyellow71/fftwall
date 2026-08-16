# fftwall

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
.venv/bin/python fftwall.py all -s 7680x4320       # every system at 8k
.venv/bin/python fftwall.py clifford --seed 7 -o out
```

every run without `--seed` is a new wallpaper; the seed is printed and
baked into the filename, so re-render any keeper at a bigger size.
render at your display's native resolution — the speckle is per-pixel,
and downscaling averages it away.

## systems

- `clifford` — clifford strange attractor; radiating filament starburst
- `ikeda` — ikeda map (light in a ring cavity); swirling interference rings
- `dejong` — peter de jong map; smoky folded filament clouds
- `svensson` — svensson map; looping ribbon sheets and glowing tori

each seed also picks a curated parameter set, so silhouettes vary a lot
run to run. the space domain (`-d space` or `-d both`) is often the
better wallpaper — the spectra like `-z 2`.

## colormaps

`dragon` and `ember` are kanagawa-dragon ramps; any matplotlib name works
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

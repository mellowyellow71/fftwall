# fftwall

procedurally generated fourier transform wallpapers.

simulate a mathematical system → 2d fft → log-magnitude → colormap.

## setup

```
python3 -m venv .venv
.venv/bin/pip install numpy pillow matplotlib
```

## use

```
.venv/bin/python fftwall.py                        # kuramoto lattice, 4k, inferno
.venv/bin/python fftwall.py waves -c dragon        # kanagawa-dragon palette
.venv/bin/python fftwall.py ikeda -d both          # save spectrum + source
.venv/bin/python fftwall.py all -s 7680x4320       # every system at 8k
.venv/bin/python fftwall.py lorenz -z 3               # magnify spectrum center
.venv/bin/python fftwall.py waves --seed 7 -o out
```

## systems

- `kuramoto` — lattice of coupled phase oscillators; each cell draws its
  oscillator's waveform. the repeating grid puts sharp harmonics in the
  spectrum (this is what your original wallpaper was)
- `waves` — interfering plane waves over 1/f noise; starburst spectra
- `ikeda` — ikeda map point-cloud density
- `lorenz` — lorenz attractor trajectory density
- `quasicrystal` — plane waves at n-fold symmetric angles; the spectrum
  is a diffraction star like a real quasicrystal's
- `chladni` — vibrating-plate nodal lines; constellation spectrum
- `clifford` — clifford strange attractor; radiating filament starburst
- `grayscott` — gray-scott reaction-diffusion; the pattern's single
  characteristic wavelength puts a glowing ring in the spectrum
  (try `-z 2`), ~15 s
- `sandpile` — abelian sandpile mandala; dot-grid constellation
  spectrum, the slow one (~1-2 min)

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

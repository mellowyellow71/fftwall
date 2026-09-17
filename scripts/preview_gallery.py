#!/usr/bin/env python3
"""Render a deterministic sampler of 22 mathematical wallpaper styles.

Run: .venv/bin/python scripts/preview_gallery.py
Requires only the project's existing numpy, pillow and matplotlib dependencies.
These are curated previews, not additions to fftwall's generator registry.
"""
from pathlib import Path
import argparse
import html
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib.colors import hsv_to_rgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fftwall import render, norm

W, H = 960, 540


def grid(scale=3, center=0j):
    x = np.linspace(-scale * W / H, scale * W / H, W)
    y = np.linspace(-scale, scale, H)
    return x[None, :] + 1j * y[:, None] + center


def color(a, cmap='dust', gamma=0.8):
    a = np.nan_to_num(a, nan=0, posinf=0, neginf=0)
    return render(norm(a, 0, 99.85) ** gamma, cmap)


def cloud(x, y, cmap='dust', gamma=0.65, bounds=None):
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if bounds is None:
        x0, x1 = np.percentile(x, [0.1, 99.9])
        y0, y1 = np.percentile(y, [0.1, 99.9])
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        span = max((y1 - y0) * 1.10, (x1 - x0) * H / W * 1.10, 1e-6)
        bounds = (cx - span * W / H / 2, cx + span * W / H / 2,
                  cy - span / 2, cy + span / 2)
    x0, x1, y0, y1 = bounds
    d, _, _ = np.histogram2d(y, x, bins=(H, W), range=((y0, y1), (x0, x1)))
    return color(np.log1p(d), cmap, gamma)


def map_points(kind, rng):
    x, y = rng.uniform(-0.5, 0.5, (2, 900))
    xs, ys = [], []
    if kind == 'tinkerbell':
        x, y = -0.72 + x * 0.015, -0.64 + y * 0.015
    def mira_f(v):
        return -0.48 * v + 2 * 1.48 * v * v / (1 + v * v)
    for i in range(4200):
        if kind == 'hopalong':
            x, y = y - np.sign(x) * np.sqrt(np.abs(0.7 * x - 1.0)), 0.4 - x
        elif kind == 'mira':
            nx = y + 0.008 * (1 - 0.05 * y * y) * y + mira_f(x)
            x, y = nx, -x + mira_f(nx)
        elif kind == 'tinkerbell':
            x, y = x*x - y*y + 0.9*x - 0.6013*y, 2*x*y + 2*x + 0.5*y
        else:
            x, y = np.sin(x*y / -0.81)*y + np.cos(-0.67*x-y), x + np.sin(y)/-0.81
        if i >= 250:
            xs.append(x.copy())
            ys.append(y.copy())
    return cloud(np.concatenate(xs), np.concatenate(ys),
                 {'hopalong': 'gold', 'mira': 'dust', 'tinkerbell': 'ember', 'bedhead': 'dragon'}[kind])


def curl_flow(rng):
    # Velocity = (d psi/dy, -d psi/dx): exactly divergence-free analytically.
    x, y = rng.uniform(-5, 5, (2, 7000))
    k = rng.normal(size=(9, 2)) * 1.5
    phase = rng.uniform(0, 2*np.pi, 9)
    xs, ys = [], []
    for _ in range(600):
        vx, vy = np.zeros_like(x), np.zeros_like(y)
        for (kx, ky), p in zip(k, phase):
            c = np.cos(kx*x + ky*y + p) / (kx*kx + ky*ky + 0.2)
            vx += ky*c
            vy -= kx*c
        x += 0.008*vx
        y += 0.008*vy
        xs.append(x.copy()); ys.append(y.copy())
    return cloud(xs, ys, 'dragon', bounds=(-5, 5, -2.8125, 2.8125))


def symmetric(rng):
    # A sixfold equivariant polynomial map (symmetric icon family).
    z = rng.uniform(-0.2, 0.2, 1800) + 1j*rng.uniform(-0.2, 0.2, 1800)
    zs = []
    for i in range(2600):
        r2 = np.abs(z)**2
        z = (-2.7 + 5*r2 + 1.5*np.real(z**6))*z + np.conj(z)**5
        if not np.isfinite(z).all() or np.abs(z).max() > 20:
            raise ValueError('Symmetric icon left its bounded regime')
        if i > 200:
            zs.append(z.copy())
    z = np.concatenate(zs)
    return cloud(z.real, z.imag, 'gold')


def quasicrystal(rng):
    # Z^4 -> physical and internal R^2; regular octagonal acceptance window.
    lattice = np.indices((15,)*4).reshape(4, -1).T - 7
    a = np.arange(4)*np.pi/4
    physical = lattice @ np.stack((np.cos(a), np.sin(a)), axis=1)
    internal = lattice @ np.stack((np.cos(3*a), np.sin(3*a)), axis=1)
    normals = np.stack((np.cos(a+np.pi/8), np.sin(a+np.pi/8)), axis=1)
    keep = np.max(np.abs(internal @ normals.T), axis=1) < 0.92
    xy = physical[keep]
    # Oversampled point raster, Hann window, and centered diffraction intensity.
    n = 1536
    d, _, _ = np.histogram2d(xy[:, 1], xy[:, 0], bins=n, range=((-18, 18), (-18, 18)))
    d *= np.hanning(n)[:, None]*np.hanning(n)[None, :]
    intensity = np.abs(np.fft.fftshift(np.fft.fft2(d)))**2
    intensity = intensity[n//2-H//2:n//2+H//2, n//2-W//2:n//2+W//2]
    return color(np.log1p(intensity), 'gold', 2.8)


def chaos_game(rng):
    a = np.arange(5)*2*np.pi/5 - np.pi/2
    vertices = np.exp(1j*a)
    z = np.zeros(1200, complex)
    last = rng.integers(5, size=z.size)
    zs = []
    for i in range(2000):
        last = (last + rng.integers(1, 5, size=z.size)) % 5
        z = (z + vertices[last])/2
        if i > 30:
            zs.append(z.copy())
    z = np.concatenate(zs)
    return cloud(z.real, z.imag, 'dust')


def billiard(rng):
    # Specular reflection on an ellipse: integrable billiard with a caustic.
    a, b = 2.8, 1.5
    p = np.array([a, 0.0])
    v = np.array([-0.7, 0.71414284285]); v /= np.linalg.norm(v)
    image = Image.new('L', (W*2, H*2))
    draw = ImageDraw.Draw(image)
    def pixel(q):
        return (int(W + q[0]*W/3.3), int(H + q[1]*W/3.3))
    for _ in range(620):
        t = -2*(p[0]*v[0]/a**2 + p[1]*v[1]/b**2)/(v[0]**2/a**2 + v[1]**2/b**2)
        q = p + t*v
        draw.line((pixel(p), pixel(q)), fill=160, width=1)
        normal = q / np.array([a*a, b*b]); normal /= np.linalg.norm(normal)
        v -= 2*np.dot(v, normal)*normal
        p = q
    image = image.resize((W, H), Image.Resampling.LANCZOS)
    return color(np.asarray(image), 'dragon')


def ginibre(rng):
    # Complex Ginibre eigenvalues repel but form a disk, not a spiral.
    n = 320
    matrix = (rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n)))/np.sqrt(2*n)
    z = np.linalg.eigvals(matrix)
    image = Image.new('RGB', (W, H), '#0b1229')
    d = ImageDraw.Draw(image)
    for p in z:
        x, y = W/2+p.real*H*0.41, H/2+p.imag*H*0.41
        d.ellipse((x-2, y-2, x+2, y+2), fill='#e7d8b1')
    return image


def littlewood(rng):
    zs = []
    for _ in range(1800):
        zs.extend(np.roots(rng.choice([-1., 1.], 33)))
    z = np.asarray(zs)
    return cloud(z.real, z.imag, 'gold', 0.5, (-2.65, 2.65, -1.49, 1.49))


def aizawa(rng):
    p = rng.normal(0, 0.01, (120, 3)) + [0.1, 0, 0]
    points = []
    def f(p):
        x, y, z = p.T
        return np.stack(((z-0.7)*x-3.5*y, 3.5*x+(z-0.7)*y,
                         0.6+0.95*z-z**3/3-(x*x+y*y)*(1+0.25*z)+0.1*z*x**3), axis=1)
    for i in range(14000):
        k1 = f(p); k2 = f(p+0.005*k1)
        k3 = f(p+0.005*k2); k4 = f(p+0.01*k3)
        p += 0.01*(k1+2*k2+2*k3+k4)/6
        if i > 2500 and i % 2 == 0:
            points.append(p.copy())
    points = np.concatenate(points)
    x, y, z = points.T
    return cloud(0.87*x+0.5*y, -0.2*x+0.35*y-0.915*z, 'ember')


def harmonograph(rng):
    t = np.linspace(0, 210, 250000)
    x = np.exp(-0.009*t)*np.sin(2.01*t+0.5) + 0.65*np.exp(-0.013*t)*np.sin(3*t)
    y = np.exp(-0.010*t)*np.sin(3.006*t+1.1) + 0.65*np.exp(-0.014*t)*np.sin(2*t)
    image = Image.new('L', (W*2,H*2))
    d = ImageDraw.Draw(image)
    pts = np.stack((W+x*H*0.51, H+y*H*0.51), axis=1)
    d.line([tuple(p) for p in pts], fill=210, width=1)
    return color(np.asarray(image.resize((W,H), Image.Resampling.LANCZOS)), 'dust', 0.7)


def hydrogen(rng):
    # Real hydrogen state n=6,l=3,m=2, x-z slice (a0=1).
    z = grid(55)
    x, zz = z.real, z.imag
    r = np.abs(z) + 1e-12
    rho = 2*r/6
    laguerre = 0.5*(rho*rho - 20*rho + 90)  # L_2^7(rho)
    angular = (zz/r)*(x/r)**2  # real Y_3^2 on y=0, up to normalization
    psi = np.exp(-rho/2)*rho**3*laguerre*angular
    return color(np.abs(psi)**0.65, 'gold', 1.1)


def quantum_carpet(rng):
    x = np.linspace(0, 1, W)[None, :]
    t = np.linspace(0, 2*np.pi, H)[:, None]
    psi = np.zeros((H,W), complex)
    for n in range(1, 36):
        c = np.exp(-((n-14)/5)**2)*np.sin(n*np.pi*0.28)
        psi += c*np.sin(n*np.pi*x)*np.exp(-1j*n*n*t)
    return color(np.log1p(np.abs(psi)**2), 'magma')


def complex_gamma(rng):
    # Lanczos approximation + reflection, rendered using phase and log modulus.
    coeff = [676.5203681218851, -1259.1392167224028, 771.32342877765313,
             -176.61502916214059, 12.507343278686905, -0.13857109526572012,
             9.9843695780195716e-6, 1.5056327351493116e-7]
    def direct(z):
        z = z-1
        a = np.full(z.shape, 0.99999999999980993, complex)
        for i, c in enumerate(coeff):
            a += c/(z+i+1)
        t = z+7.5
        return np.sqrt(2*np.pi)*t**(z+0.5)*np.exp(-t)*a
    z = grid(3.7, -1.2+0.013j)
    left = z.real < 0.5
    f = direct(np.where(left, 1-z, z))
    f[left] = np.pi/(np.sin(np.pi*z[left])*f[left])
    phase = (np.angle(f)/(2*np.pi)+1) % 1
    logmag = np.log(np.abs(f)+1e-30)
    rings = 0.5+0.5*np.cos(2*np.pi*logmag/np.log(4))
    rays = 0.5+0.5*np.cos(12*np.angle(f))
    hsv = np.stack((phase, np.full_like(phase, 0.62), 0.3+0.48*rings**0.18*rays**0.12), axis=-1)
    return Image.fromarray((hsv_to_rgb(hsv)*255).astype('uint8'))


def newton(rng):
    z = grid(1.5, 0.04+0.03j)
    iterations = np.zeros(z.shape)
    active = np.ones(z.shape, bool)
    for i in range(65):
        f = z**3-1
        converged = np.abs(f) < 1e-7
        iterations[active & converged] = i
        active &= ~converged
        z[active] -= f[active]/(3*z[active]**2+1e-20)
    roots = np.exp(2j*np.pi*np.arange(3)/3)
    basin = np.argmin(np.abs(z[...,None]-roots), axis=-1)
    palette = np.array([[0.86,0.62,0.38], [0.32,0.59,0.70], [0.70,0.36,0.46]])
    shade = (0.18+0.82*np.exp(-iterations/12))[...,None]
    return Image.fromarray((255*palette[basin]*shade).astype('uint8'))


def lyapunov(rng):
    a = np.linspace(2.5,4,W)[None,:]
    b = np.linspace(2.5,4,H)[:,None]
    x = np.full((H,W),0.5)
    total = np.zeros_like(x)
    sequence = 'AABAB'
    for i in range(420):
        r = a if sequence[i % len(sequence)] == 'A' else b
        if i >= 100:
            total += np.log(np.abs(r*(1-2*x))+1e-14)
        x = r*x*(1-x)
    exponent = total/320
    # Stable regions glow; chaotic regions stay navy.
    return color(np.sqrt(np.maximum(-exponent,0)), 'gold', 0.8)


def julia(rng):
    z = grid(1.03)
    dz = np.ones(z.shape,complex)
    active = np.ones(z.shape,bool)
    dist = np.zeros(z.shape)
    for i in range(240):
        dz[active] = 2*z[active]*dz[active]
        z[active] = z[active]**2 + (-0.74543+0.11301j)
        escaped = active & (np.abs(z)>100)
        r = np.abs(z[escaped])
        dist[escaped] = r*np.log(r)/(np.abs(dz[escaped])+1e-30)
        active[escaped] = False
    v = np.exp(-3*np.maximum(dist,0)**0.32)
    v[active] = 0.02
    return render(v, 'dust')


def berry(rng):
    z = grid(5)
    wave = np.zeros((H,W),complex)
    for _ in range(100):
        a, p = rng.uniform(0,2*np.pi,2)
        wave += np.exp(1j*(10*(np.cos(a)*z.real+np.sin(a)*z.imag)+p))/10
    return color(np.abs(wave)**2, 'inferno', 0.7)


def buddhabrot(rng):
    d = np.zeros((H,W),float)
    bounds = (-2.25,1.25,-0.984375,0.984375)
    for _ in range(28):
        c = rng.uniform(-2,1,16000)+1j*rng.uniform(-1.5,1.5,16000)
        # Discard known interior points; retain moderately long escape orbits.
        q = (c.real-0.25)**2+c.imag**2
        c = c[~((q*(q+c.real-0.25)<0.25*c.imag**2) | ((c.real+1)**2+c.imag**2<0.0625))]
        z = np.zeros_like(c)
        escape = np.zeros(c.size,int)
        for i in range(260):
            active = escape == 0
            z[active] = z[active]**2+c[active]
            escape[active & (np.abs(z)>2)] = i+1
        c, escape = c[(escape>15)&(escape<260)], escape[(escape>15)&(escape<260)]
        z = np.zeros_like(c)
        for i in range(escape.max(initial=0)):
            active = escape>i
            z[active] = z[active]**2+c[active]
            p = z[active]
            # Conjugate symmetry reduces sampling noise.
            x = np.concatenate((p.real,p.real)); y = np.concatenate((p.imag,-p.imag))
            hist, _, _ = np.histogram2d(y,x,bins=(H,W),range=((bounds[2],bounds[3]),(bounds[0],bounds[1])))
            d += hist
    return color(np.log1p(d), 'dragon', 0.85)


def lenia(rng):
    # Actual continuous CA: radial ring convolution and Gaussian growth.
    h,w = 180,320
    yy,xx = np.mgrid[:h,:w]
    r = np.sqrt(((xx-w//2)/13)**2+((yy-h//2)/13)**2)
    kernel = np.exp(-((r-0.5)/0.15)**2)*(r<1)
    kernel /= kernel.sum()
    fk = np.fft.fft2(np.fft.ifftshift(kernel))
    a = rng.uniform(0,0.7,(h,w))
    # Patchy initial conditions rather than a single named organism.
    envelope = np.asarray(Image.fromarray(rng.uniform(0,1,(12,20)).astype('float32')).resize((w,h),Image.Resampling.BILINEAR))
    a *= envelope>0.4
    for _ in range(180):
        u = np.fft.ifft2(np.fft.fft2(a)*fk).real
        growth = 2*np.exp(-0.5*((u-0.15)/0.017)**2)-1
        a = np.clip(a+0.1*growth,0,1)
    return color(a, 'ember', 0.75).resize((W,H),Image.Resampling.LANCZOS)


ITEMS = [
    ('hopalong', 'Hopalong', 'Lacy orbit structure from the Barry Martin map.', lambda r: map_points('hopalong',r)),
    ('mira', 'Gumowski–Mira', 'Feathered folds from a nonlinear two-dimensional map.', lambda r: map_points('mira',r)),
    ('tinkerbell', 'Tinkerbell', 'Looping filaments from a quadratic two-dimensional map.', lambda r: map_points('tinkerbell',r)),
    ('bedhead', 'Bedhead', 'Braided, folded density from a trigonometric map.', lambda r: map_points('bedhead',r)),
    ('curl', 'Curl flow', 'Particles following a divergence-free Fourier flow.', curl_flow),
    ('symmetric', 'Symmetric chaos', 'Sixfold equivariant polynomial map; spatial density.', symmetric),
    ('quasicrystal', 'Quasicrystal diffraction', 'Octagonal cut-and-project Z⁴ point set; FFT intensity.', quasicrystal),
    ('chaos-game', 'Restricted chaos game', 'Pentagon midpoint iteration; no repeated vertex.', chaos_game),
    ('billiard', 'Elliptical billiard', 'Specular ray paths enclosing a caustic; integrable, not chaotic.', billiard),
    ('ginibre', 'Ginibre eigenvalues', 'A complex random matrix: repelling eigenvalues in a disk.', ginibre),
    ('littlewood', 'Littlewood roots', 'Zeros of 1,800 degree-32 polynomials with ±1 coefficients.', littlewood),
    ('aizawa', 'Aizawa attractor', 'An oblique density projection of a 3D chaotic flow.', aizawa),
    ('harmonograph', 'Harmonograph', 'Four damped sinusoidal oscillators tracing a single line.', harmonograph),
    ('hydrogen', 'Hydrogen orbital', 'Real n=6, l=3, m=2 state; x–z slice, tone-mapped density.', hydrogen),
    ('quantum-carpet', 'Quantum carpet', 'Particle-in-a-box probability: position × time.', quantum_carpet),
    ('gamma', 'Complex Gamma', 'Γ(z): hue encodes phase; bands encode log magnitude.', complex_gamma),
    ('newton', 'Newton basins', 'Newton iteration for z³−1; basin hue and convergence shading.', newton),
    ('lyapunov', 'Lyapunov fractal', 'Logistic maps with AABAB forcing; stable regions glow.', lyapunov),
    ('julia', 'Julia distance field', 'Quadratic Julia set with exterior distance-estimate shading.', julia),
    ('berry', 'Berry random waves', 'Intensity of 100 equal-wavenumber random plane waves.', berry),
    ('buddhabrot', 'Buddhabrot', 'Density of escaping Mandelbrot orbits; low-sample preview.', buddhabrot),
    ('lenia', 'Lenia', 'Continuous cellular automaton with ring kernel and Gaussian growth.', lenia),
]


def font(size):
    for path in ['/System/Library/Fonts/Supplemental/Arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if Path(path).exists():
            return ImageFont.truetype(path,size)
    return ImageFont.load_default(size=size)


def gallery(out):
    cards=[]
    for i,(slug,title,desc,_) in enumerate(ITEMS,1):
        name=f'{i:02d}-{slug}.png'
        cards.append(f'<article><a href="{name}"><img loading="lazy" src="{name}" alt="{html.escape(title)}"></a><div class="caption"><h2><span>{i:02d}</span> {html.escape(title)}</h2><p>{html.escape(desc)}</p><a href="{name}" download>Save PNG ↗</a></div></article>')
    (out/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>fftwall · mathematical sampler</title><style>
*{box-sizing:border-box}body{margin:0;background:#0c0e12;color:#e5e6eb;font:15px/1.6 system-ui,sans-serif}header,main,footer{max-width:1480px;margin:auto;padding:32px}h1{font-size:32px;font-weight:500;margin:0}header p{max-width:820px;color:#a4aab7}main{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:28px;padding-top:0}article{background:#151820;border:1px solid #262b35;border-radius:12px;overflow:hidden}img{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#090b10}.caption{padding:16px 20px 20px}h2{font-size:18px;margin:0;font-weight:500}h2 span{color:#7b8495;margin-right:8px}p{margin:6px 0 12px;color:#a4aab7}a{color:#b9cce3;text-decoration:none}footer{color:#8992a2}@media(max-width:760px){main{grid-template-columns:1fr}header,main,footer{padding:20px}}
</style><header><h1>fftwall / mathematical sampler</h1><p>22 procedurally generated studies · 960 × 540 · deterministic seed 42<br>Click any image for the clean, unlabelled PNG. These are curated style previews, not final 4K renders. Most show spatial patterns; #07 shows diffraction. One representative per suggested family, not every named variant.</p></header><main>'''+''.join(cards)+'''</main><footer>Generated locally with NumPy + Pillow. No external assets, JavaScript, or network requests.</footer></html>''',encoding='utf-8')
    for page,start in enumerate(range(0,len(ITEMS),6),1):
        batch=ITEMS[start:start+6]
        rows=(len(batch)+1)//2
        sheet=Image.new('RGB',(1020,90+rows*324),'#0c0e12')
        d=ImageDraw.Draw(sheet)
        d.text((22,16),f'FFTWall / math sampler — {page:02d}',font=font(25),fill='#e5e6eb')
        d.text((22,51),'Spatial studies unless labelled diffraction · open index.html for full previews',font=font(14),fill='#9ca7b8')
        for j,(slug,title,desc,_) in enumerate(batch):
            x,y=20+(j%2)*500,90+(j//2)*324
            with Image.open(out/f'{start+j+1:02d}-{slug}.png') as im:
                sheet.paste(im.resize((480,270),Image.Resampling.LANCZOS),(x,y))
            d.text((x,y+280),f'{start+j+1:02d}  {title}',font=font(19),fill='#e5e6eb')
        sheet.save(out/f'contact-{page:02d}.jpg',quality=94)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--outdir',type=Path,default=Path('out/math-gallery'))
    ap.add_argument('--only',choices=[item[0] for item in ITEMS])
    args=ap.parse_args()
    args.outdir.mkdir(parents=True,exist_ok=True)
    for i,(slug,title,_,generate) in enumerate(ITEMS,1):
        if args.only and args.only != slug:
            continue
        t=time.monotonic()
        print(f'{i:02d} {title} ...',flush=True)
        image=generate(np.random.default_rng(42))
        image.save(args.outdir/f'{i:02d}-{slug}.png')
        print(f'   {time.monotonic()-t:.1f}s',flush=True)
    if all((args.outdir/f'{i:02d}-{item[0]}.png').exists() for i,item in enumerate(ITEMS,1)):
        gallery(args.outdir)
        print(f'Gallery: {args.outdir / "index.html"}')


if __name__=='__main__':
    main()

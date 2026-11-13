"""VERSION 1 (superseded by degradations.py v2). Kept unchanged to reproduce the v1 grids.

All functions take an RGB float image in [0, 1] of shape (H, W, 3) at the 512 px scale
(Data/Resized 512), a severity s in [0, 1] (0 = unchanged), and a numpy Generator for the
random parts (position, direction, ...). They return a new float image of the same shape.

Severity -> parameter mappings (first version, to be tuned by eye):
  defocus_blur   disk kernel radius r = 15 s px                             (plan: radius 1-15 px)
  motion_blur    line kernel length L = 35 s px, random angle 0-180 deg     (plan: 5-35 px; HELD OUT of training)
  overexposure   out = clip(gain * x ** gamma), gamma = 1 - 0.6 s, gain = 1 + 0.5 s   (plan: gamma 1.0 -> 0.4)
  underexposure  out = gain * x ** gamma,       gamma = 1 + 1.5 s, gain = 1 - 0.4 s   (plan: gamma 1.0 -> 2.5)
  glare          1-3 elliptical near-white spots, radius 10 + 70 s px, peak 0.5 + 0.5 s  (plan: 10-80 px, 0.5-1.0)
  occlusion      eyelid-like band from the top (70%) or bottom edge, covering 5 + 45 s % of the
                 height at its centre, curved edge, skin-like texture, lashes along the margin
  framing        off-centre zoomed crop so that 5 + 35 s % of the central "eye" box (50% x 50% of
                 the frame) lies outside the frame; crop resized back to the input size
  noise / jpeg   nuisance only (no severity index)

Order when several are combined (scene -> optics -> sensor):
  framing -> occlusion -> glare -> defocus / motion blur -> exposure -> noise -> JPEG
"""
import io

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.signal import fftconvolve

ORDER = ["framing", "occlusion", "glare", "defocus_blur", "motion_blur", "overexposure", "underexposure"]


# ---------------------------------------------------------------- helpers
def to_float(im):
    return np.asarray(im.convert("RGB")).astype(np.float64) / 255.0


def to_pil(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255).round().astype(np.uint8))


def _convolve(a, k):
    """Convolve each channel with kernel k, reflect padding, same size."""
    p = k.shape[0] // 2
    padded = np.pad(a, ((p, p), (p, p), (0, 0)), mode="reflect")
    out = np.stack([fftconvolve(padded[..., c], k, mode="same") for c in range(a.shape[2])], axis=2)
    return out[p:-p, p:-p] if p else out


def _supersampled(fn, size, ss=8):
    """Kernel of shape (size, size) from an indicator fn(x, y) evaluated on an ss x ss subgrid."""
    c = (size - 1) / 2
    g = (np.arange(size * ss) + 0.5) / ss - 0.5 - c
    x, y = np.meshgrid(g, g)
    k = fn(x, y).astype(np.float64).reshape(size, ss, size, ss).mean(axis=(1, 3))
    return k / k.sum()


# ---------------------------------------------------------------- blur
def disk_kernel(r):
    size = 2 * int(np.ceil(r)) + 1
    return _supersampled(lambda x, y: x ** 2 + y ** 2 <= r ** 2, size)


def defocus_blur(a, s, rng=None):
    r = 15.0 * s
    if r < 0.5:
        return a.copy()
    return _convolve(a, disk_kernel(r))


def motion_kernel(length, angle_deg):
    size = int(np.ceil(length)) | 1
    t = np.deg2rad(angle_deg)
    dx, dy = np.cos(t), np.sin(t)
    # points within 0.5 px of the segment of the given length through the centre
    return _supersampled(lambda x, y: (np.abs(-dy * x + dx * y) <= 0.5) & (np.abs(dx * x + dy * y) <= length / 2), size + 2)


def motion_blur(a, s, rng):
    length = 35.0 * s
    if length < 1.5:
        return a.copy()
    return _convolve(a, motion_kernel(length, rng.uniform(0, 180)))


# ---------------------------------------------------------------- exposure
def overexposure(a, s, rng=None):
    gamma, gain = 1 - 0.6 * s, 1 + 0.5 * s
    return np.clip(gain * a ** gamma, 0, 1)


def underexposure(a, s, rng=None):
    gamma, gain = 1 + 1.5 * s, 1 - 0.4 * s
    return gain * a ** gamma


# ---------------------------------------------------------------- glare
def glare(a, s, rng):
    if s <= 0:
        return a.copy()
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    out = a.copy()
    tint = 0.85 + 0.15 * a.reshape(-1, 3).mean(0) / max(a.mean(), 1e-6)  # near-white, slightly toward the light colour
    tint = np.clip(tint / tint.max(), 0, 1)
    n = 1 + int(rng.integers(0, 1 + round(2 * s)))  # 1 spot at low severity, up to 3 at high
    for _ in range(n):
        r = (10 + 70 * s) * rng.uniform(0.7, 1.0) if n > 1 else 10 + 70 * s
        cy, cx = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w
        aspect, rot = rng.uniform(1.0, 2.0), rng.uniform(0, np.pi)
        u = (xx - cx) * np.cos(rot) + (yy - cy) * np.sin(rot)
        v = (-(xx - cx) * np.sin(rot) + (yy - cy) * np.cos(rot)) * aspect
        d2 = (u ** 2 + v ** 2) / (r ** 2)
        g = np.clip(1.6 * np.exp(-2.0 * d2), 0, 1) * (0.5 + 0.5 * s)  # saturated core, soft halo
        out = out + g[..., None] * (tint - out)  # blend toward near-white
    return np.clip(out, 0, 1)


# ---------------------------------------------------------------- occlusion
def occlusion(a, s, rng):
    if s <= 0:
        return a.copy()
    h, w = a.shape[:2]
    from_top = rng.uniform() < 0.7
    depth = (0.05 + 0.45 * s) * h                      # band height at its centre
    x0 = rng.uniform(0.3, 0.7) * w
    curv = rng.uniform(0.15, 0.35) * h                 # edge rises by this much at the image sides
    xs = np.arange(w)
    edge = depth - curv * ((xs - x0) / w) ** 2 * 4     # parabola: deepest at x0
    yy = np.arange(h)[:, None]
    dist = edge[None, :] - (yy if from_top else (h - 1 - yy))  # >0 inside the lid
    mask = np.clip(dist / 4.0 + 0.5, 0, 1)             # ~4 px soft margin
    # skin-like colour: darkened median of the outer 8% border, with low-frequency texture
    b = max(1, int(0.08 * min(h, w)))
    border = np.concatenate([a[:b].reshape(-1, 3), a[-b:].reshape(-1, 3), a[:, :b].reshape(-1, 3), a[:, -b:].reshape(-1, 3)])
    base = 0.55 * np.median(border, axis=0)
    tex = ndimage.gaussian_filter(rng.standard_normal((h, w)), 6)
    tex = 1 + 0.6 * tex / (np.abs(tex).max() + 1e-9) * 0.25
    lid = np.clip(base[None, None, :] * tex[..., None], 0, 1)
    out = a * (1 - mask[..., None]) + lid * mask[..., None]
    # eyelashes: short dark strokes leaving the lid margin
    lash = np.zeros((h, w))
    n_lash = int(rng.integers(25, 60))
    for _ in range(n_lash):
        x = rng.uniform(0, w)
        xi = int(np.clip(x, 0, w - 1))
        y_edge = edge[xi]
        if y_edge <= 2:
            continue
        length = rng.uniform(8, 25) * (0.6 + 0.4 * s)
        ang = rng.uniform(-0.5, 0.5)
        for t in np.linspace(0, length, int(length * 2)):
            px = int(round(x + t * np.sin(ang)))
            py = int(round(y_edge + t * np.cos(ang)))
            if not from_top:
                py = h - 1 - py
            if 0 <= px < w and 0 <= py < h:
                lash[py, px] = 1
    lash = np.clip(ndimage.gaussian_filter(lash, 0.7) * 3, 0, 1)
    out = out * (1 - 0.8 * lash[..., None])
    return out


# ---------------------------------------------------------------- framing
def _outside_fraction(win, box=(0.25, 0.25, 0.75, 0.75)):
    """Share of the central box (x0, y0, x1, y1 in frame fractions) outside the window."""
    ix = max(0.0, min(win[2], box[2]) - max(win[0], box[0]))
    iy = max(0.0, min(win[3], box[3]) - max(win[1], box[1]))
    return 1 - ix * iy / ((box[2] - box[0]) * (box[3] - box[1]))


def framing(a, s, rng, return_window=False):
    """Off-centre zoomed crop: target share t = 0.05 + 0.35 s of the central 50% x 50% box
    ("eye proxy") ends up outside the frame. Window size z (fraction of each side) and offset
    along a random direction are found by search; the crop is resized back to (H, W)."""
    h, w = a.shape[:2]
    if s <= 0:
        return (a.copy(), (0.0, 0.0, 1.0, 1.0)) if return_window else a.copy()
    t = 0.05 + 0.35 * s
    theta = rng.uniform(0, 2 * np.pi)
    dx, dy = np.cos(theta), np.sin(theta)
    best = None
    for z in np.linspace(0.85, 0.55, 31):              # prefer the mildest zoom that reaches t
        # move the window from the centre toward (dx, dy) until it touches the frame edge
        lo, hi = 0.0, 1.0
        cx_max = (1 - z) / 2
        def win_at(f):
            cx = 0.5 + f * cx_max * np.sign(dx) * min(1, abs(dx) / max(abs(dx), abs(dy)))
            cy = 0.5 + f * cx_max * np.sign(dy) * min(1, abs(dy) / max(abs(dx), abs(dy)))
            return (cx - z / 2, cy - z / 2, cx + z / 2, cy + z / 2)
        if _outside_fraction(win_at(1.0)) < t:
            continue
        for _ in range(40):                            # bisection on the offset
            mid = (lo + hi) / 2
            if _outside_fraction(win_at(mid)) < t:
                lo = mid
            else:
                hi = mid
        best = win_at(hi)
        break
    if best is None:                                   # not reachable: use the strongest setting
        best = (0.45 - 0.0, 0.45, 1.0, 1.0)
    x0, y0, x1, y1 = best
    box = (x0 * w, y0 * h, x1 * w, y1 * h)
    im = to_pil(a).resize((w, h), Image.LANCZOS, box=box)
    out = to_float(im)
    return (out, best) if return_window else out


# ---------------------------------------------------------------- nuisance
def gaussian_noise(a, sigma, rng):
    return np.clip(a + rng.normal(0, sigma, a.shape), 0, 1)


def jpeg(a, quality):
    buf = io.BytesIO()
    to_pil(a).save(buf, format="JPEG", quality=int(quality))
    return to_float(Image.open(buf))


FUNCS = {
    "defocus_blur": defocus_blur, "motion_blur": motion_blur, "overexposure": overexposure,
    "underexposure": underexposure, "glare": glare, "occlusion": occlusion, "framing": framing,
}


def apply(a, severities, rng):
    """Apply several degradations in the fixed physical ORDER. severities: {name: s}."""
    out = a
    for name in ORDER:
        s = severities.get(name, 0.0)
        if s > 0:
            out = FUNCS[name](out, s, rng)
    return out

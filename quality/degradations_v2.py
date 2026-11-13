"""VERSION 2 (superseded by degradations.py v3). Kept unchanged to reproduce the v2 grids. Originally:
Version 1 is kept unchanged as degradations_v1.py; the v1 -> v2 changes are listed.

All functions take an RGB float image in [0, 1] of shape (H, W, 3) at the 512 px scale
(Data/Resized 512), a severity s in [0, 1] (0 = unchanged), and a numpy Generator for the
random parts (position, direction, ...). They return a new float image of the same shape.

Severity -> parameter mappings (v2):
  defocus_blur   disk kernel radius r = 15 s px                                (plan: radius 1-15 px)
  motion_blur    line kernel length L = 35 s px, random angle 0-180 deg        (plan: 5-35 px; HELD OUT of training)
  overexposure   OUTCOME-defined (v2b): a gain in linear light (sRGB ~ gamma 2.2) is chosen so that a
                 share c = c0 + (0.50 - c0) s of the pixels is blown out (brightest channel >= 0.98);
                 c0 = the image's own share; blown pixels roll off toward white (no neon hue shifts).
                 Same severity -> same visible damage for dark and bright images.
  underexposure  OUTCOME-defined: a gain < 1 in linear light is chosen so that the mean luminance becomes
                 m0 ** (1 - s) * 0.02 ** s (geometric slide from the image's own mean m0 to near-black
                 0.02). s = 1 is near black for every image.
  glare          1-3 elliptical near-white spots, radius 10 + 100 s px, peak 0.5 + 0.5 s  (plan: 10-80 px)
  occlusion      upper (70%) or lower eyelid: band whose margin is an arch (shallowest at x0), covering
                 5 + 45 s % of the height at x0; soft 6 px margin, darker margin line, shading away from
                 the margin, skin texture, curved lashes
  framing        off-centre crop window of a GIVEN zoom z (fraction of each side kept), placed so that a
                 share t = 0.05 + 0.35 s of the central "eye" box (50% x 50%) lies outside the frame; if t
                 is not reachable at z, the largest reachable t is used and returned. Every training
                 image (clean ones too) gets a random zoom z ~ U(0.55, 1.0) via centred_zoom, so zoom
                 itself carries no framing information.
  noise / jpeg   nuisance only (no severity index), mild: noise sigma <= 0.02, JPEG quality >= 30

Order when several are combined (scene -> optics -> sensor):
  framing (incl. zoom) -> occlusion -> glare -> defocus / motion blur -> exposure -> noise -> JPEG
"""
import io

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.signal import fftconvolve

ORDER = ["framing", "occlusion", "glare", "defocus_blur", "motion_blur", "overexposure", "underexposure"]
ZOOM_RANGE = (0.55, 1.0)
NOISE_MAX, JPEG_MIN = 0.02, 30


# ---------------------------------------------------------------- helpers
def to_float(im):
    return np.asarray(im.convert("RGB")).astype(np.float64) / 255.0


def to_pil(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255).round().astype(np.uint8))


def _lum(a):
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


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
    return np.clip(_convolve(a, disk_kernel(r)), 0, 1)


def motion_kernel(length, angle_deg):
    size = (int(np.ceil(length)) | 1) + 2
    t = np.deg2rad(angle_deg)
    dx, dy = np.cos(t), np.sin(t)
    return _supersampled(lambda x, y: (np.abs(-dy * x + dx * y) <= 0.5) & (np.abs(dx * x + dy * y) <= length / 2), size)


def motion_blur(a, s, rng):
    length = 35.0 * s
    if length < 1.5:
        return a.copy()
    return np.clip(_convolve(a, motion_kernel(length, rng.uniform(0, 180))), 0, 1)


# ---------------------------------------------------------------- exposure (outcome-defined)
def _gain_linear(a, g):
    """Multiply linear light by g (sRGB approximated by gamma 2.2), clip, back to display values."""
    return np.clip(np.clip(a, 0, 1) ** 2.2 * g, 0, 1) ** (1 / 2.2)


def _solve_gain(a, f, target, lo, hi, iters=40):
    """Bisection in log-gain for f(_gain_linear(a, g)) = target; f must increase with g."""
    llo, lhi = np.log(lo), np.log(hi)
    for _ in range(iters):
        mid = (llo + lhi) / 2
        if f(_gain_linear(a, np.exp(mid))) < target:
            llo = mid
        else:
            lhi = mid
    return float(np.exp((llo + lhi) / 2))


def _gain_linear_rolloff(a, g):
    """Like _gain_linear, but a pixel whose brightest channel exceeds 1 is blended toward white by
    1 - 1/max (camera-like highlight roll-off) instead of clipping each channel separately, which
    produced neon hue shifts in v2a ."""
    y = np.clip(a, 0, 1) ** 2.2 * g
    m = y.max(axis=2, keepdims=True)
    w = np.clip(1 - 1 / np.maximum(m, 1e-9), 0, 1)
    return (np.clip(y, 0, 1) * (1 - w) + w) ** (1 / 2.2)


def overexposure(a, s, rng=None, return_gain=False):
    """Gain chosen so that a share c0 + (0.50 - c0) s of the pixels is blown out, i.e. their
    brightest channel is >= 0.98 (v2b; v2a used luminance >= 0.98 and target 0.60)."""
    if s <= 0:
        return (a.copy(), 1.0) if return_gain else a.copy()
    clipped = lambda x: float((x.max(axis=2) >= 0.98).mean())
    c0 = clipped(a)
    target = c0 + (0.50 - c0) * s
    llo, lhi = 0.0, np.log(4096.0)
    for _ in range(40):
        mid = (llo + lhi) / 2
        if clipped(_gain_linear_rolloff(a, np.exp(mid))) < target:
            llo = mid
        else:
            lhi = mid
    g = float(np.exp((llo + lhi) / 2))
    out = _gain_linear_rolloff(a, g)
    return (out, g) if return_gain else out


def underexposure(a, s, rng=None, return_gain=False):
    if s <= 0:
        return (a.copy(), 1.0) if return_gain else a.copy()
    m0 = max(float(_lum(a).mean()), 1e-4)
    target = m0 ** (1 - s) * 0.02 ** s
    g = _solve_gain(a, lambda x: float(_lum(x).mean()), target, 1e-6, 1.0)
    out = _gain_linear(a, g)
    return (out, g) if return_gain else out


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
        r = (10 + 100 * s) * rng.uniform(0.7, 1.0) if n > 1 else 10 + 100 * s
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
    depth = (0.05 + 0.45 * s) * h                      # band depth at x0 (the arch's shallowest point)
    x0 = rng.uniform(0.3, 0.7) * w
    curv = rng.uniform(0.10, 0.25) * h                 # the margin drops by this much toward the sides
    xs = np.arange(w)
    edge = depth + curv * (2 * (xs - x0) / w) ** 2     # arch
    yy = np.arange(h)[:, None].astype(float)
    ydist = yy if from_top else (h - 1 - yy)           # distance from the lid's own image edge
    inside = edge[None, :] - ydist                     # > 0 inside the lid: px from the margin
    mask = np.clip(inside / 6.0 + 0.5, 0, 1)           # soft 6 px margin
    # skin colour: darkened median of the outer 8% border; texture; shading
    b = max(1, int(0.08 * min(h, w)))
    border = np.concatenate([a[:b].reshape(-1, 3), a[-b:].reshape(-1, 3), a[:, :b].reshape(-1, 3), a[:, -b:].reshape(-1, 3)])
    base = 0.6 * np.median(border, axis=0)
    fine = ndimage.gaussian_filter(rng.standard_normal((h, w)), 1.5)
    coarse = ndimage.gaussian_filter(rng.standard_normal((h, w)), 12)
    tex = 1 + 0.35 * coarse / (np.abs(coarse).max() + 1e-9) + 0.12 * fine / (np.abs(fine).max() + 1e-9)
    shade = 1.0 + 0.35 * np.clip(inside / (0.25 * h + 1e-9), 0, 1)       # lid brighter away from the margin
    margin_line = 1 - 0.45 * np.exp(-((inside - 4) / 3.0) ** 2)           # dark line along the margin
    lid = np.clip(base[None, None, :] * (tex * shade * margin_line)[..., None], 0, 1)
    out = a * (1 - mask[..., None]) + lid * mask[..., None]
    # eyelashes: curved dark strokes leaving the margin
    lash = np.zeros((h, w))
    for _ in range(int(rng.integers(40, 90))):
        x = rng.uniform(0, w)
        y_edge = edge[int(np.clip(x, 0, w - 1))]
        if y_edge <= 2 or y_edge >= h - 2:
            continue
        length = rng.uniform(10, 30) * (0.6 + 0.4 * s)
        ang = rng.uniform(-0.6, 0.6) + 0.4 * np.sign(x - x0) * abs(x - x0) / w
        bend = rng.uniform(0.01, 0.04)
        for t in np.linspace(0, length, int(length * 3)):
            px = x + t * np.sin(ang + bend * t)
            py = y_edge + t * np.cos(ang + bend * t)
            pxi, pyi = int(round(px)), int(round(py if from_top else h - 1 - py))
            if 0 <= pxi < w and 0 <= pyi < h:
                lash[pyi, pxi] = 1
    lash = np.clip(ndimage.gaussian_filter(ndimage.binary_dilation(lash > 0).astype(float), 0.6) * 2, 0, 1)
    return out * (1 - 0.85 * lash[..., None])


# ---------------------------------------------------------------- framing and zoom
def _outside_fraction(win, box=(0.25, 0.25, 0.75, 0.75)):
    """Share of the central box (x0, y0, x1, y1 in frame fractions) outside the window."""
    ix = max(0.0, min(win[2], box[2]) - max(win[0], box[0]))
    iy = max(0.0, min(win[3], box[3]) - max(win[1], box[1]))
    return 1 - ix * iy / ((box[2] - box[0]) * (box[3] - box[1]))


def _crop_resize(a, win):
    h, w = a.shape[:2]
    x0, y0, x1, y1 = win
    im = to_pil(a).resize((w, h), Image.LANCZOS, box=(x0 * w, y0 * h, x1 * w, y1 * h))
    return to_float(im)


def centred_zoom(a, z):
    """Centred crop keeping a fraction z of each side, resized back (zoom 1/z). Applied to EVERY
    training image (z ~ U(ZOOM_RANGE)) so that zoom is not a framing cue. The central eye box stays
    fully inside for z >= 0.5, so the framing label (0) is unchanged."""
    if z >= 0.999:
        return a.copy()
    return _crop_resize(a, (0.5 - z / 2, 0.5 - z / 2, 0.5 + z / 2, 0.5 + z / 2))


def framing(a, s, rng, z=0.55, return_info=False):
    """Off-centre window of zoom z along a random direction, placed so that t = 0.05 + 0.35 s of the
    central 50% x 50% box lies outside it. If t is not reachable at z (window already at the frame
    edge), the largest reachable t is used. With return_info also returns (achieved t, window)."""
    if s <= 0:
        out = centred_zoom(a, z)
        win = (0.5 - z / 2, 0.5 - z / 2, 0.5 + z / 2, 0.5 + z / 2)
        return (out, (0.0, win)) if return_info else out
    t = 0.05 + 0.35 * s
    theta = rng.uniform(0, 2 * np.pi)
    d = np.array([np.cos(theta), np.sin(theta)])
    d = d / np.abs(d).max()                            # square norm: f = 1 puts the window at the frame edge
    half_free = (1 - z) / 2

    def win_at(f):
        cx, cy = 0.5 + f * half_free * d[0], 0.5 + f * half_free * d[1]
        return (cx - z / 2, cy - z / 2, cx + z / 2, cy + z / 2)

    if _outside_fraction(win_at(1.0)) <= t:
        f = 1.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if _outside_fraction(win_at(mid)) < t:
                lo = mid
            else:
                hi = mid
        f = hi
    win = win_at(f)
    out = _crop_resize(a, win)
    return (out, (_outside_fraction(win), win)) if return_info else out


# ---------------------------------------------------------------- nuisance (no severity index)
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

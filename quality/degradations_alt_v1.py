"""ALTERNATIVE degradations for the generalisation test. Never used for training.

Purpose: test whether the quality module learnt the DEFECTS or the fingerprints of OUR generator
(quality/degradations.py). Each defect is re-made with a different defect model and different code
(PIL / scikit-image, not our scipy kernels), but its LABEL is measured with the same formula as in
degradations.py, so labels mean the same thing.

| Defect              | Ours (degradations.py)                       | Alternative (this file)                                |
| defocus_blur        | disk / Gaussian / blend kernel, FFT conv     | box blur (PIL) or hexagonal-aperture kernel            |
| motion_blur         | straight line, random angle                  | curved camera-shake trajectory (random walk)           |
| overexposure        | linear-light gain + highlight roll-off       | display-space gain with hard clipping (no roll-off)    |
| underexposure       | linear-light multiplication                  | gamma curve (skimage adjust_gamma), darkens mid-tones  |
| uneven_illumination | shadow edge / off-centre spot, linear light  | cos^4 vignette or full-width gradient, display space   |
| haze                | uniform blend with a milky veil              | scattering model, non-uniform transmission map         |
| glare               | soft spots replacing the image (weight map)  | additive bloom (+ optional streak) in linear light     |
| occlusion           | textured eyelid with lashes and wet margin   | flat skin-coloured ellipse, blurred edge, no lashes    |
| noise               | additive Gaussian                            | Poisson (shot) or speckle noise (skimage random_noise) |

Label formulas (same as degradations.py; x before, y after):
  blur: 1 - T(y)/T(x); over: (B(y)-B(x))/(1-B(x)); under: 1 - mean lin L(y) / mean lin L(x);
  uneven: 1 - min G / max G over the central box (G = light field in linear light);
  haze: 1 - std L(y) / std L(x); occlusion: covered share; noise: std(y - x);
  glare: mean over the central box of the measured pull toward white, clip((L(y)-L(x)) / (1-L(x))),
         the image-measured equivalent of our weight-map mean (no weight map exists for additive bloom).
Strength ranges calibrated on VALIDATION clean photos so the label ranges match the training labels.
"""
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage
from skimage import draw, exposure, util

import degradations as D

VALUES = D.VALUES
RANGES_ALT = {   # calibrated on validation photos; units in comments
    "defocus_blur": (2.0, 12.0),         # box radius px (box) / hexagon radius = 1.6 x this
    "motion_blur": (8.0, 48.0),          # trajectory length px (curved, so a longer path than a line)
    "overexposure": (0.05, 1.0),         # u: target blown share = c0 + (0.5 - c0) u (hard clip)
    "underexposure": (1.15, 3.5),        # gamma
    "uneven_illumination": (0.2, 0.9),   # darkest / brightest display gain
    "haze": (0.15, 0.75),                # 1 - mean transmission
    "glare": (12.0, 95.0),               # bloom radius px
    "occlusion": (0.5, 0.92),            # occluder depth, share of image height
    "noise": (0.01, 0.08),               # target noise std (0-1 scale)
}


def _pil(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255).round().astype(np.uint8))


def _arr(im):
    return np.asarray(im).astype(np.float64) / 255.0


# ---------------------------------------------------------------- blur
def _hexagon_kernel(r):
    n = 2 * int(np.ceil(r)) + 3
    c = (n - 1) / 2
    ang = np.deg2rad(np.arange(6) * 60 + 30)
    rr, cc = draw.polygon(c + r * np.sin(ang), c + r * np.cos(ang), (n, n))
    k = np.zeros((n, n))
    k[rr, cc] = 1
    return k / k.sum()


def defocus_blur(x, rng, strength=None):
    s = rng.uniform(*RANGES_ALT["defocus_blur"]) if strength is None else strength
    if rng.uniform() < 0.5:
        y = _arr(_pil(x).filter(ImageFilter.BoxBlur(s)))
        kind = "box"
    else:
        k = _hexagon_kernel(1.6 * s)
        y = np.stack([ndimage.convolve(x[..., c], k, mode="nearest") for c in range(3)], axis=2)
        kind = "hexagon"
    return np.clip(y, 0, 1), D._edge_loss(x, y), {"size": s, "kernel": kind}


def _shake_kernel(length, rng):
    """Camera-shake path: a random walk with momentum (curved), rasterised with anti-aliased lines."""
    n_steps = 64
    v = rng.normal(size=2)
    v /= np.linalg.norm(v)
    pts, p = [np.zeros(2)], np.zeros(2)
    for _ in range(n_steps):
        v = 0.85 * v + 0.35 * rng.normal(size=2)
        v /= np.linalg.norm(v)
        p = p + v
        pts.append(p.copy())
    pts = np.array(pts)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
    pts *= length / seg                                     # path length = length px
    pts -= pts.min(0)
    n = int(np.ceil(pts.max())) + 3
    k = np.zeros((n, n))
    for (r0, c0), (r1, c1) in zip(pts[:-1] + 1, pts[1:] + 1):
        rr, cc, val = draw.line_aa(int(round(r0)), int(round(c0)), int(round(r1)), int(round(c1)))
        ok = (rr < n) & (cc < n)
        k[rr[ok], cc[ok]] = np.maximum(k[rr[ok], cc[ok]], val[ok])
    return k / k.sum()


def motion_blur(x, rng, strength=None):
    length = rng.uniform(*RANGES_ALT["motion_blur"]) if strength is None else strength
    k = _shake_kernel(length, rng)
    y = np.stack([ndimage.convolve(x[..., c], k, mode="nearest") for c in range(3)], axis=2)
    return np.clip(y, 0, 1), D._edge_loss(x, y), {"length": length, "path": "shake"}


# ---------------------------------------------------------------- exposure
def overexposure(x, rng, strength=None):
    u = rng.uniform(*RANGES_ALT["overexposure"]) if strength is None else strength
    c0 = D._blown_share(x)
    target = c0 + (0.5 - c0) * u
    xs = x[::4, ::4]
    lo, hi = 0.0, np.log(64.0)
    for _ in range(30):                                      # display-space gain, hard clip
        mid = (lo + hi) / 2
        if D._blown_share(np.clip(xs * np.exp(mid), 0, 1)) < target:
            lo = mid
        else:
            hi = mid
    g = float(np.exp((lo + hi) / 2))
    y = np.clip(x * g, 0, 1)
    label = (D._blown_share(y) - c0) / (1 - c0) if c0 < 1 else 0.0
    return y, float(np.clip(label, 0, 1)), {"u": u, "gain": g}


def underexposure(x, rng, strength=None):
    gamma = rng.uniform(*RANGES_ALT["underexposure"]) if strength is None else strength
    y = exposure.adjust_gamma(x, gamma)
    m0 = D.lin(D.lum(x)).mean()
    label = 1 - D.lin(D.lum(y)).mean() / m0 if m0 > 0 else 0.0
    return y, float(np.clip(label, 0, 1)), {"gamma": gamma}


def uneven_illumination(x, rng, strength=None):
    r = rng.uniform(*RANGES_ALT["uneven_illumination"]) if strength is None else strength
    h, w = x.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    if rng.uniform() < 0.5:                                  # cos^4 vignette, centre off the middle
        cx, cy = rng.uniform(0.1, 0.9) * w, rng.uniform(0.1, 0.9) * h
        d = np.hypot(xx - cx, yy - cy) / np.hypot(w, h)
        p = np.cos(np.clip(d * np.pi / 1.4, 0, np.pi / 2)) ** 4
        kind = "cos4_vignette"
    else:                                                    # full-width linear gradient
        t = rng.uniform(0, 2 * np.pi)
        p = (xx / w - 0.5) * np.cos(t) + (yy / h - 0.5) * np.sin(t)
        kind = "gradient"
    p = (p - p.min()) / (p.max() - p.min() + 1e-12)
    M = r + (1 - r) * p                                      # display-space multiplier
    y = np.clip(x * M[..., None], 0, 1)
    G = M ** 2.2                                             # the same field in linear light
    ys, xs = D._box_slices(x.shape)
    Gr = G[ys, xs]
    return y, float(np.clip(1 - Gr.min() / Gr.max(), 0, 1)), {"gain_ratio": r, "field": kind}


def haze(x, rng, strength=None):
    a = rng.uniform(*RANGES_ALT["haze"]) if strength is None else strength
    h, w = x.shape[:2]
    depth = ndimage.gaussian_filter(rng.standard_normal((h // 8 + 1, w // 8 + 1)), 4)
    depth = np.asarray(Image.fromarray(depth.astype(np.float32)).resize((w, h), Image.BILINEAR))
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-12)
    t = np.clip((1 - a) * (0.75 + 0.5 * depth), 0.02, 1)     # transmission, mean about 1 - a
    A = rng.uniform(0.8, 0.95)                               # neutral airlight
    y = x * t[..., None] + A * (1 - t[..., None])
    s0 = D.lum(x).std()
    label = 1 - D.lum(y).std() / s0 if s0 > 0 else 0.0
    return np.clip(y, 0, 1), float(np.clip(label, 0, 1)), {"haze": a, "airlight": A}


# ---------------------------------------------------------------- glare
def glare_label(x, y):
    """Mean pull toward white over the central box (image-measured)."""
    Lx, Ly = D.lum(x), D.lum(y)
    pull = np.clip((Ly - Lx) / np.maximum(1 - Lx, 1e-3), 0, 1)
    ys, xs = D._box_slices(x.shape)
    return float(pull[ys, xs].mean())


def glare(x, rng, strength=None):
    r = rng.uniform(*RANGES_ALT["glare"]) if strength is None else strength
    h, w = x.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    x0, y0, x1, y1 = D.CENTRAL_BOX
    cy, cx = rng.uniform(y0, y1) * h, rng.uniform(x0, x1) * w
    bloom = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (r / 2) ** 2))
    if rng.uniform() < 0.4:                                  # horizontal-ish flare streak
        rot = rng.uniform(-0.3, 0.3)
        u = (xx - cx) * np.cos(rot) + (yy - cy) * np.sin(rot)
        v = -(xx - cx) * np.sin(rot) + (yy - cy) * np.cos(rot)
        bloom = np.maximum(bloom, 0.6 * np.exp(-(u / (2.5 * r)) ** 2 - (v / (0.08 * r + 1)) ** 2))
        kind = "bloom+streak"
    else:
        kind = "bloom"
    intensity = rng.uniform(0.5, 1.5)                        # soft bloom; only strong ones saturate
    yl = D.lin(x) + intensity * bloom[..., None]             # additive light, then clipped
    y = D.delin(np.clip(yl, 0, 1))
    return y, glare_label(x, y), {"radius": r, "kind": kind, "intensity": intensity}


# ---------------------------------------------------------------- occlusion
def occlusion(x, rng, strength=None):
    depth = rng.uniform(*RANGES_ALT["occlusion"]) if strength is None else strength
    h, w = x.shape[:2]
    from_top = rng.uniform() < 0.7
    ry, rx = depth * h, rng.uniform(0.9, 1.4) * w            # a large ellipse whose lower edge is the lid margin
    cy = 0.0 if from_top else h - 1.0
    rr, cc = draw.ellipse(cy, w / 2 + rng.uniform(-0.15, 0.15) * w, ry, rx, shape=(h, w))
    m = np.zeros((h, w))
    m[rr, cc] = 1.0
    m = _arr(_pil(np.repeat(m[..., None], 3, 2)).filter(ImageFilter.GaussianBlur(3)))[..., 0]
    b = max(4, int(0.08 * min(h, w)))                         # whole outer frame (all four borders)
    frame = np.concatenate([x[:b].reshape(-1, 3), x[-b:].reshape(-1, 3), x[:, :b].reshape(-1, 3), x[:, -b:].reshape(-1, 3)])
    skin = np.clip(np.median(frame, axis=0) * 0.9 + 0.08, 0, 1)
    tex = 1 + 0.05 * rng.standard_normal((h, w, 1))
    occl = np.clip(skin[None, None, :] * tex, 0, 1)
    y = x * (1 - m[..., None]) + occl * m[..., None]
    return y, float(m.mean()), {"depth": depth, "from_top": bool(from_top)}


# ---------------------------------------------------------------- noise
def noise(x, rng, strength=None):
    target = rng.uniform(*RANGES_ALT["noise"]) if strength is None else strength
    seed = int(rng.integers(0, 2**31 - 1))
    if rng.uniform() < 0.5:                                  # speckle: y = x + x * n, var chosen for the target std
        var = (target / max(np.sqrt((x ** 2).mean()), 1e-3)) ** 2
        y = util.random_noise(x, mode="speckle", var=var, rng=seed)
        kind = "speckle"
    else:                                                    # Poisson shot noise at a photon scale giving the target std
        scale = max(x.mean(), 1e-3) / target ** 2
        y = np.clip(np.random.default_rng(seed).poisson(x * scale) / scale, 0, 1)
        kind = "poisson"
    return np.clip(y, 0, 1), float(np.clip((y - x).std(), 0, 1)), {"target": target, "kind": kind}


FUNCS = {"defocus_blur": defocus_blur, "motion_blur": motion_blur, "overexposure": overexposure,
         "underexposure": underexposure, "uneven_illumination": uneven_illumination, "haze": haze,
         "glare": glare, "occlusion": occlusion, "noise": noise}

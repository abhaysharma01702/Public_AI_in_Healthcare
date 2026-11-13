"""Synthetic degradations for the quality module, VERSION 4 (v3).
v1, v2, v3 are kept unchanged as degradations_v1.py / _v2.py / _v3.py (v3 built Quality Data).
v4 changes: overexposure strength = exposure increase in stops and label = newly NEAR-WHITE share
(luminance >= 0.97) instead of a forced blown share / any channel at 255; uneven illumination label
measured on DISPLAYED brightness (light field ^ 1/2.2) instead of linear light.

The quality module predicts a vector of 9 values:
    defocus_blur, motion_blur, overexposure, underexposure, uneven_illumination, haze,
    glare, occlusion, noise
Each label is the MEASURED effect of the defect we applied, computed at its own step from
x (image entering the step) and y (image right after it), so later defects cannot leak into it.
Clean images and defects not applied: 0. No nuisance noise / JPEG (point 7a).

All functions take an RGB float image in [0, 1] (H, W, 3) at the 512 px scale and a numpy
Generator, draw their strength uniformly within RANGES (point 12b), and return
(y, label, params).

Order when combined (scene -> optics -> light -> sensor):
    occlusion -> glare -> uneven_illumination -> defocus_blur / motion_blur -> haze ->
    overexposure / underexposure -> noise
"""
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.signal import fftconvolve

VALUES = ["defocus_blur", "motion_blur", "overexposure", "underexposure", "uneven_illumination",
          "haze", "glare", "occlusion", "noise"]
ORDER = ["occlusion", "glare", "uneven_illumination", "defocus_blur", "motion_blur", "haze",
         "overexposure", "underexposure", "noise"]

# Strength ranges (drawn uniformly; reviewed visually in grids). Units in comments.
RANGES = {
    "defocus_blur": (1.0, 8.0),          # blur spread sigma, px at 512 (disk radius = 2 sigma)
    "motion_blur": (4.0, 30.0),          # smear length, px at 512
    "overexposure": (0.0, 1.0),          # v4: share of the way through the chosen kind's range (EXPOSURE_KINDS)
    "underexposure": (0.0, 1.0),         # v4: share of the way through the chosen kind's range (EXPOSURE_KINDS)
    "uneven_illumination": (0.02, 0.8),  # darkest / brightest gain of the light field, linear light (v4: 0.02)
                                         # (lower = stronger; display x0.90 .. x0.26); v3a was (0.15, 0.8),
                                         # v3c first tried (0.05, 0.5): labels started at 0.5 (gap 0-0.5)
    "haze": (0.1, 0.7),                  # veil weight h
    "glare": (15.0, 110.0),              # spot radius, px at 512
    "occlusion": (0.45, 0.85),           # lid depth at its shallowest point, share of image height
    "noise": (0.01, 0.08),               # noise sigma (0-1 scale)
}
CENTRAL_BOX = (0.25, 0.25, 0.75, 0.75)   # R: x0, y0, x1, y1 as fractions (glare drawn only here)
NET_SIZE = 224                           # network input (shorter side)
BLOWN = 254.5 / 255                      # v3 definition (a channel at 255); kept for _blown_share only
WHITE_LUM = 0.97                         # v4: a pixel is blown out when its luminance is >= 0.97 (practically white)

# Mixing policy (points 11 / 12; plan 5.2): number of defects per training image
N_DEFECTS_P = {0: 0.30, 1: 0.35, 2: 0.20, 3: 0.15}
NOT_TOGETHER = [{"defocus_blur", "motion_blur"}, {"overexposure", "underexposure"}]
# "extreme" draws are applied alone (fix 4): top part of the range for these three
EXTREME = {"underexposure": 0.80, "haze": 0.55, "overexposure": 0.80}   # share of the way through the range


# ---------------------------------------------------------------- helpers
def to_float(im):
    return np.asarray(im.convert("RGB")).astype(np.float64) / 255.0


def to_pil(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255).round().astype(np.uint8))


def lum(a):
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def lin(a):
    return np.clip(a, 0, 1) ** 2.2


def delin(a):
    return np.clip(a, 0, 1) ** (1 / 2.2)


_EYE_BOX = [CENTRAL_BOX]                 # eye region of the image being degraded (set per image)


def set_eye_box(box=None):
    """Set the eye region (x0, y0, x1, y1 fractions) used to place glare / defocus focus and to measure the glare and
    uneven-illumination labels for the NEXT images; None = CENTRAL_BOX. From labels/eye_regions.csv ."""
    _EYE_BOX[0] = tuple(box) if box is not None else CENTRAL_BOX


def eye_box():
    return _EYE_BOX[0]


def _box_slices(shape, box=None):
    box = eye_box() if box is None else box
    h, w = shape[:2]
    return slice(int(box[1] * h), int(box[3] * h)), slice(int(box[0] * w), int(box[2] * w))


def _convolve(a, k):
    p = k.shape[0] // 2
    padded = np.pad(a, ((p, p), (p, p), (0, 0)), mode="reflect")
    out = np.stack([fftconvolve(padded[..., c], k, mode="same") for c in range(a.shape[2])], axis=2)
    return np.clip(out[p:-p, p:-p] if p else out, 0, 1)


def _supersampled(fn, size, ss=8):
    c = (size - 1) / 2
    g = (np.arange(size * ss) + 0.5) / ss - 0.5 - c
    x, y = np.meshgrid(g, g)
    k = fn(x, y).astype(np.float64).reshape(size, ss, size, ss).mean(axis=(1, 3))
    return k / k.sum()


def tenengrad_net(a):
    """Tenengrad (mean squared Sobel gradient of luminance) at network size (shorter side 224)."""
    L = lum(a).astype(np.float32)
    h, w = L.shape
    f = NET_SIZE / min(h, w)
    Ls = np.asarray(Image.fromarray(L).resize((round(w * f), round(h * f)), Image.LANCZOS), dtype=np.float64)  # float32 -> mode F
    gx, gy = ndimage.sobel(Ls, axis=1), ndimage.sobel(Ls, axis=0)
    return float((gx ** 2 + gy ** 2)[1:-1, 1:-1].mean())


def _edge_loss(x, y):
    tx = tenengrad_net(x)
    return float(np.clip(1 - tenengrad_net(y) / tx, 0, 1)) if tx > 0 else 0.0


# ---------------------------------------------------------------- blur
def _blur_kernel(sigma, shape):
    """Blur kernel with spread sigma; shape: disk (radius 2 sigma), gaussian, or blend
    (disk radius 1.6 sigma softened by a Gaussian 0.6 sigma; total spread sigma)."""
    if shape == "disk":
        r = 2 * sigma
        return _supersampled(lambda x, y: x ** 2 + y ** 2 <= r ** 2, 2 * int(np.ceil(r)) + 1)
    if shape == "gaussian":
        size = 2 * int(np.ceil(3 * sigma)) + 1
        return _supersampled(lambda x, y: np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)), size)
    r, s2 = 1.6 * sigma, 0.6 * sigma
    disk = _supersampled(lambda x, y: x ** 2 + y ** 2 <= r ** 2, 2 * int(np.ceil(r)) + 1)
    k = ndimage.gaussian_filter(np.pad(disk, int(np.ceil(3 * s2))), s2)
    return k / k.sum()


def defocus_blur(x, rng, strength=None):
    sigma = rng.uniform(*RANGES["defocus_blur"]) if strength is None else strength
    shape = rng.choice(["disk", "gaussian", "blend"])
    y = _convolve(x, _blur_kernel(sigma, shape))
    return y, _edge_loss(x, y), {"sigma": sigma, "kernel": str(shape)}


def motion_blur(x, rng, strength=None):
    length = rng.uniform(*RANGES["motion_blur"]) if strength is None else strength
    t = np.deg2rad(rng.uniform(0, 180))
    dx, dy = np.cos(t), np.sin(t)
    k = _supersampled(lambda u, v: (np.abs(-dy * u + dx * v) <= 0.5) & (np.abs(dx * u + dy * v) <= length / 2),
                      (int(np.ceil(length)) | 1) + 2)
    y = _convolve(x, k)
    return y, _edge_loss(x, y), {"length": length, "angle_deg": float(np.rad2deg(t))}


# ---------------------------------------------------------------- exposure
def _blown_share(a):
    return float((a.max(axis=2) >= BLOWN).mean())


def _gain_rolloff(x, g):
    """Linear-light gain; a pixel whose brightest channel exceeds 1 is blended toward white by
    1 - 1/max (camera-like highlight roll-off; brightest channel ends exactly at 1)."""
    yl = lin(x) * g
    m = yl.max(axis=2, keepdims=True)
    w = np.clip(1 - 1 / np.maximum(m, 1e-9), 0, 1)
    return delin(np.clip(yl, 0, 1) * (1 - w) + w)


DETAIL_WIN = 7          # local window (px at 512) for "detail" = local spread of brightness
DETAIL_KEEP = 0.5       # detail counts as lost when less than half of it is left


def _local_detail(a):
    L = lum(a)
    m = ndimage.uniform_filter(L, DETAIL_WIN)
    return np.sqrt(np.maximum(ndimage.uniform_filter(L * L, DETAIL_WIN) - m * m, 0))


def overexposure_label(x, y):
    """v4 label: share of the image where brightening wiped out at least half of
    the local fine detail: pixels that got brighter (> +0.02), had detail before (local spread > 0.01) and kept
    < DETAIL_KEEP of it. Same meaning for white, blue and slit-beam photos (no brightness cut-off)."""
    dx, dy = _local_detail(x), _local_detail(y)
    lost = (lum(y) > lum(x) + 0.02) & (dx > 0.01) & (dy < DETAIL_KEEP * dx)
    return float(lost.mean())


def overexposure(x, rng, strength=None):
    """v4: one of the overexposure KINDS at random (EXPOSURE_KINDS), strength t = share of the way
    through that kind's range; label = detail lost."""
    t = rng.uniform(*RANGES["overexposure"]) if strength is None else strength
    kind = str(rng.choice(list(EXPOSURE_KINDS["overexposure"])))
    return exposure_kind("overexposure", kind, x, rng, t)


def underexposure(x, rng, strength=None):
    """v4: one of the underexposure KINDS at random; label = share of light lost (linear light)."""
    t = rng.uniform(*RANGES["underexposure"]) if strength is None else strength
    kind = str(rng.choice(list(EXPOSURE_KINDS["underexposure"])))
    return exposure_kind("underexposure", kind, x, rng, t)


def uneven_illumination(x, rng, strength=None):
    """v3c: the light falls off ACROSS the eye region, not gently over the whole picture.
    Light field G (linear light) = r + (1 - r) * s, with s = 1 where lit and 0 in full shadow:
      shadow_edge     soft edge (smoothstep, width 15-35% of the shorter side) in a random direction,
                      positioned inside the central box: part of the eye lit, part in shadow
      offcentre_spot  light spot (Gaussian, sigma 18-30% of the shorter side) displaced 15-30% from
                      the centre: beam not centred on the eye / vignetting reaching the eye
    Label v4  = 1 - min(Gd) / max(Gd) over the central box, Gd = G ** (1 / 2.2) = the field in DISPLAYED
    brightness: how much darker the darkest part of the eye region looks (v3 used linear light, which
    overstated mid-range labels)."""
    r = rng.uniform(*RANGES["uneven_illumination"]) if strength is None else strength
    h, w = x.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    m = min(h, w)
    u, v = (xx - w / 2) / m, (yy - h / 2) / m                # centred, in units of the shorter side
    if rng.uniform() < 0.5:
        t = rng.uniform(0, 2 * np.pi)
        q = u * np.cos(t) + v * np.sin(t)
        width = rng.uniform(0.10, 0.25)                      # v4b: whole transition fits in the eye region,
        lim = max(0.0, 0.23 - width / 2)                     # so the box sees both full light and full shadow
        c = rng.uniform(-lim, lim)                           # edge position, inside the central box
        z = np.clip((q - c) / width + 0.5, 0, 1)
        s = 1 - z * z * (3 - 2 * z)                          # smoothstep: 1 lit side, 0 shadow side
        kind = "shadow_edge"
    else:
        a = rng.uniform(0, 2 * np.pi)
        off = rng.uniform(0.15, 0.30)
        sig = rng.uniform(0.18, 0.30)
        s = np.exp(-((u - off * np.cos(a)) ** 2 + (v - off * np.sin(a)) ** 2) / (2 * sig ** 2))
        kind = "offcentre_spot"
    G = r + (1 - r) * s
    y = delin(lin(x) * G[..., None])
    ys, xs = _box_slices(x.shape)
    Gr = G[ys, xs] ** (1 / 2.2)
    return y, float(np.clip(1 - Gr.min() / Gr.max(), 0, 1)), {"dark_bright_ratio": r, "field": kind}


def _uneven_illumination_v3a(x, rng, strength=None):
    """v3a, superseded: gentle ramp / vignette over the whole picture; barely visible on the eye
    . Kept to reproduce 05_uneven_illumination_v3a.png;
    its range was (0.15, 0.8)."""
    r = strength
    h, w = x.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    if rng.uniform() < 0.5:                                  # linear ramp in a random direction
        t = rng.uniform(0, 2 * np.pi)
        p = (xx / w - 0.5) * np.cos(t) + (yy / h - 0.5) * np.sin(t)
        p = (p - p.min()) / (p.max() - p.min())
        kind = "ramp"
    else:                                                    # radial vignette, random centre near the middle
        cx, cy = rng.uniform(0.35, 0.65) * w, rng.uniform(0.35, 0.65) * h
        p = np.hypot((xx - cx) / w, (yy - cy) / h)
        p = p / p.max()
        kind = "vignette"
    G = 1 - (1 - r) * p                                      # 1 at the brightest point, r at the darkest
    y = delin(lin(x) * G[..., None])
    ys, xs = _box_slices(x.shape)
    Gr = G[ys, xs]
    return y, float(np.clip(1 - Gr.min() / Gr.max(), 0, 1)), {"dark_bright_ratio": r, "field": kind}


def haze(x, rng, strength=None):
    h = rng.uniform(*RANGES["haze"]) if strength is None else strength
    veil = 0.5 * x.reshape(-1, 3).mean(0) + 0.5 * 0.85       # milky, slightly tinted by the photo
    y = (1 - h) * x + h * veil[None, None, :]
    return y, haze_label(x, y), {"veil_weight": h}


def haze_label(x, y):
    """v4 haze label: share of LOCAL contrast lost = 1 - mean local detail(y) / mean local
    detail(x) (7 x 7 local brightness spread). Equals v3's global label for a uniform veil; robust to patchy veils."""
    dx, dy = _local_detail(x).mean(), _local_detail(y).mean()
    return float(np.clip(1 - dy / dx, 0, 1)) if dx > 0 else 0.0


# ---------------------------------------------------------------- glare
def glare(x, rng, strength=None):
    r_max = rng.uniform(*RANGES["glare"]) if strength is None else strength
    h, w = x.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    tint = 0.85 + 0.15 * x.reshape(-1, 3).mean(0) / max(x.mean(), 1e-6)
    tint = np.clip(tint / tint.max(), 0, 1)
    s = (r_max - RANGES["glare"][0]) / (RANGES["glare"][1] - RANGES["glare"][0])
    n = 1 + int(rng.integers(0, 1 + round(2 * s)))
    keep = np.ones((h, w))                                   # product of (1 - w_i)
    x0, y0, x1, y1 = eye_box()
    for i in range(n):
        r = r_max if i == 0 else r_max * rng.uniform(0.5, 0.9)
        cy, cx = rng.uniform(y0, y1) * h, rng.uniform(x0, x1) * w   # only inside the central box (point 8)
        aspect, rot = rng.uniform(1.0, 2.0), rng.uniform(0, np.pi)
        u = (xx - cx) * np.cos(rot) + (yy - cy) * np.sin(rot)
        v = (-(xx - cx) * np.sin(rot) + (yy - cy) * np.cos(rot)) * aspect
        wi = np.clip(1.6 * np.exp(-2.0 * (u ** 2 + v ** 2) / r ** 2), 0, 1) * rng.uniform(0.6, 1.0)
        keep *= 1 - wi
    wmap = 1 - keep
    y = x + wmap[..., None] * (tint - x)
    ys, xs = _box_slices(x.shape)
    return np.clip(y, 0, 1), float(wmap[ys, xs].mean()), {"radius": r_max, "spots": n}


# ---------------------------------------------------------------- occlusion (option B, point 10)
def _photo_sharpness_ratio(x):
    """How much detail a 1 px Gaussian blur removes (0 = very sharp photo, 1 = already soft)."""
    L = lum(x)
    gx, gy = ndimage.sobel(L, axis=1), ndimage.sobel(L, axis=0)
    t0 = (gx ** 2 + gy ** 2).mean()
    Lb = ndimage.gaussian_filter(L, 1.0)
    gx, gy = ndimage.sobel(Lb, axis=1), ndimage.sobel(Lb, axis=0)
    return float((gx ** 2 + gy ** 2).mean() / t0) if t0 > 0 else 1.0


def occlusion(x, rng, strength=None):
    depth_frac = rng.uniform(*RANGES["occlusion"]) if strength is None else strength
    h, w = x.shape[:2]
    from_top = rng.uniform() < 0.7
    depth = depth_frac * h                                   # depth at the arch's shallowest point
    x0 = rng.uniform(0.3, 0.7) * w
    curv = rng.uniform(0.06, 0.15) * h
    xs = np.arange(w)
    edge = np.minimum(depth + curv * (2 * (xs - x0) / w) ** 2, h)   # arched margin
    yy = np.arange(h)[:, None].astype(float)
    ydist = yy if from_top else (h - 1 - yy)
    inside = edge[None, :] - ydist                           # > 0 inside the lid (px from the margin)
    mask = np.clip(inside / 5.0 + 0.5, 0, 1)                 # soft 5 px margin
    # skin: colour from the photo's own border strip on the lid side (no tiling: v3a's tiled strip
    # left repeating stripes); smooth surface with soft large-scale shading and fine grain
    b = max(4, int(0.08 * h))
    strip = x[:b] if from_top else x[-b:]
    base = 0.75 * np.median(strip.reshape(-1, 3), axis=0)
    coarse = ndimage.gaussian_filter(rng.standard_normal((h, w)), 25)
    coarse = coarse / (np.abs(coarse).max() + 1e-9)
    grain = ndimage.gaussian_filter(rng.standard_normal((h, w)), 0.8)
    grain = grain / (np.abs(grain).max() + 1e-9)
    shade = 1.0 + 0.30 * np.clip(inside / (0.3 * h + 1e-9), 0, 1)   # lid brighter away from the margin
    lid = base[None, None, :] * (shade * (1 + 0.10 * coarse + 0.04 * grain))[..., None]
    # wet lid margin: a darker band with a thin highlight just inside it (tear meniscus reflection)
    lid = lid * (1 - 0.35 * np.exp(-((inside - 4) / 2.5) ** 2))[..., None]
    lid = lid + 0.12 * np.exp(-((inside - 1.5) / 1.0) ** 2)[..., None]
    lid = np.clip(lid, 0, 1)
    # match the photo's softness: softer photos get a softer lid
    sig = 0.4 + 2.0 * _photo_sharpness_ratio(x)
    lid = ndimage.gaussian_filter(lid, (sig, sig, 0))
    y = x * (1 - mask[..., None]) + lid * mask[..., None]
    # lashes: thin, semi-transparent, curved strokes from the margin
    lash = np.zeros((h, w))
    for _ in range(int(rng.integers(30, 70))):
        xp = rng.uniform(0, w)
        y_edge = edge[int(np.clip(xp, 0, w - 1))]
        if y_edge <= 2 or y_edge >= h - 2:
            continue
        length = rng.uniform(8, 22)
        ang = rng.uniform(-0.5, 0.5) + 0.4 * np.sign(xp - x0) * abs(xp - x0) / w
        bend = rng.uniform(0.01, 0.04)
        opacity = rng.uniform(0.5, 0.9)
        for t in np.linspace(0, length, int(length * 3)):
            px = xp + t * np.sin(ang + bend * t)
            py = y_edge + t * np.cos(ang + bend * t)
            pxi, pyi = int(round(px)), int(round(py if from_top else h - 1 - py))
            if 0 <= pxi < w and 0 <= pyi < h:
                lash[pyi, pxi] = max(lash[pyi, pxi], opacity)
    lash = np.clip(ndimage.gaussian_filter(lash, 0.5) * 1.6, 0, 0.9)
    y = y * (1 - lash[..., None])
    label = float(np.clip(mask.mean(), 0, 1))                # share of the whole image covered (point 8)
    return np.clip(y, 0, 1), label, {"depth_frac": depth_frac, "from_top": bool(from_top)}


# ---------------------------------------------------------------- noise (labelled; no nuisance)
def noise(x, rng, strength=None):
    """Label (c): the standard deviation of the noise actually added, y - x
    over all pixels and channels (after clipping), on the 0-1 brightness scale. The same grain gets
    the same label on every photo. Replaces (a) var(n) / (var(n) + var(L(x))), which depended on the
    photo's overall contrast."""
    sigma = rng.uniform(*RANGES["noise"]) if strength is None else strength
    y = np.clip(x + rng.normal(0, sigma, x.shape), 0, 1)
    return y, float(np.clip((y - x).std(), 0, 1)), {"sigma": sigma}


FUNCS = {"defocus_blur": defocus_blur, "motion_blur": motion_blur, "overexposure": overexposure,
         "underexposure": underexposure, "uneven_illumination": uneven_illumination, "haze": haze,
         "glare": glare, "occlusion": occlusion, "noise": noise}


# ---------------------------------------------------------------- mixing policy
LOW_LIGHT_P = 1 / 3      # share of underexposed samples that also get sensor noise ("low light")


def draw_strength(name, rng):
    """Strength draw for sample. Uneven illumination: the DISPLAYED darkness ratio r^(1/2.2) is drawn evenly
    , so labels spread over their range instead of piling up low; everything else uniform in RANGES."""
    lo, hi = RANGES[name]
    if name == "uneven_illumination":
        return float(rng.uniform(lo ** (1 / 2.2), hi ** (1 / 2.2)) ** 2.2)
    return float(rng.uniform(lo, hi))


def _is_extreme(name, strength):
    if name not in EXTREME:
        return False
    lo, hi = RANGES[name]
    return (strength - lo) / (hi - lo) >= EXTREME[name]


def sample(x, rng, min_defects=0):
    """Draw a training sample: 0-3 defects (N_DEFECTS_P), never two from NOT_TOGETHER, an extreme
    exposure / haze draw applied alone. Returns (image, labels {9 values}, applied {name: params}).
    min_defects=1 draws only from k >= 1 (probabilities renormalised), used for the stored
    Quality Data, where every source also has its clean copy ."""
    ks = [k for k in N_DEFECTS_P if k >= min_defects]
    p = np.array([N_DEFECTS_P[k] for k in ks])
    k = int(rng.choice(ks, p=p / p.sum()))
    chosen = []
    for name in rng.permutation(VALUES):
        if len(chosen) == k:
            break
        if any({name, c} in [set(p) for p in NOT_TOGETHER] for c in chosen):
            continue
        chosen.append(str(name))
    strengths = {n: draw_strength(n, rng) for n in chosen}
    extreme = [n for n in chosen if _is_extreme(n, strengths[n])]
    if extreme:
        chosen = [extreme[0]]
    elif ("underexposure" in chosen and "noise" not in chosen and len(chosen) < 3
          and rng.uniform() < LOW_LIGHT_P):                 # low light = dark + sensor noise
        chosen.append("noise")
        strengths["noise"] = draw_strength("noise", rng)
    labels = {v: 0.0 for v in VALUES}
    applied = {}
    y = x
    for name in ORDER:
        if name in chosen:
            y, labels[name], applied[name] = FUNCS[name](y, rng, strengths[name])
    return y, labels, applied


# ---------------------------------------------------------------- exposure kinds
# Training draws one kind at random per application (EXPOSURE_KINDS). Labels: overexposure = detail lost
# (overexposure_label); underexposure = share of light lost (linear light). Strength = share of the way
# through each kind's own range (0..1), so EXTREME / sampling stay kind-independent.
def _over_rolloff(x, rng, t):        # whole photo brighter, camera-like highlight roll-off (v4 overexposure)
    ev = 0.5 + 2.5 * t
    return _gain_rolloff(x, 2.0 ** ev), {"kind": "rolloff", "ev": ev}


def _over_clip(x, rng, t):           # whole photo brighter, colour channels clip separately (no roll-off)
    ev = 0.3 + 1.3 * t
    return np.clip(x * 2.0 ** ev, 0, 1), {"kind": "channel_clip", "ev": ev}


def _over_local(x, rng, t):          # only part of the photo blown: too-strong beam / local hot spot
    h, w = x.shape[:2]
    m = min(h, w)
    yy, xx = np.mgrid[0:h, 0:w]
    L = ndimage.gaussian_filter(lum(x), m / 12)
    cy, cx = np.unravel_index(np.argmax(L + 0.02 * rng.standard_normal(L.shape)), L.shape)  # near the lit area
    rad = rng.uniform(0.2, 0.45) * m
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rad ** 2))
    ev = 1.0 + 3.0 * t
    gain = 2.0 ** (ev * g)                                   # full gain at the hot spot, 1 far away
    yl = lin(x) * gain[..., None]
    mx = yl.max(axis=2, keepdims=True)
    wgt = np.clip(1 - 1 / np.maximum(mx, 1e-9), 0, 1)
    return delin(np.clip(yl, 0, 1) * (1 - wgt) + wgt), {"kind": "local", "ev": ev, "radius_px": float(rad)}


def _over_curve(x, rng, t):          # brightened by a tone curve after capture: washed out, low contrast
    gamma = 0.75 - 0.5 * t                                   # 0.75 .. 0.25
    lift = 0.10 * t                                          # raised black level
    return np.clip(lift + (1 - lift) * x ** gamma, 0, 1), {"kind": "tone_curve", "gamma": gamma, "lift": lift}


def _under_linear(x, rng, t):        # less light, uniformly (v3 / v4 underexposure)
    loss = 0.2 + 0.75 * t
    return delin(lin(x) * (1 - loss)), {"kind": "linear", "light_removed": loss}


def _under_gamma(x, rng, t):         # tone curve on BRIGHTNESS only: mid-tones darker, reflections stay bright,
    gamma = 1.15 + 2.35 * t            # colours keep their saturation (per-channel gamma over-saturated them)
    L = lum(x)
    ratio = np.where(L > 1e-4, L ** gamma / np.maximum(L, 1e-4), 1.0)
    return np.clip(x * ratio[..., None], 0, 1), {"kind": "gamma_brightness", "gamma": gamma}


def _under_crush(x, rng, t):         # shadows crushed to black + darker overall
    b = 0.05 + 0.30 * t                                      # black point
    k = 1.0 - 0.5 * t                                        # overall gain
    return np.clip((x - b) / (1 - b) * k, 0, 1), {"kind": "crushed_blacks", "black_point": b, "gain": k}


EXPOSURE_KINDS = {"overexposure": {"rolloff": _over_rolloff, "channel_clip": _over_clip, "local": _over_local,
                                   },              # tone_curve dropped: looks like haze
                  "underexposure": {"linear": _under_linear, "gamma": _under_gamma, "crushed_blacks": _under_crush}}


def exposure_kind(name, kind, x, rng, t):
    """Apply one exposure kind at strength t (0..1) and measure its label."""
    y, params = EXPOSURE_KINDS[name][kind](x, rng, t)
    if name == "overexposure":
        return y, overexposure_label(x, y), params
    m0 = lin(lum(x)).mean()
    return y, float(np.clip(1 - lin(lum(y)).mean() / m0, 0, 1)) if m0 > 0 else 0.0, params

"""Several implementations ("kinds") per defect for TRAINING. Exposure kinds live in degradations.py.

For each of the other 7 defects, three kinds:
  ours   = degradations.py (v4) function
  alt    = degradations_alt.py (v2) function (was the generalisation implementation; now a training kind)
  third  = new, modelled on a real slit-lamp situation (below)
Every kind returns (y, label, params); labels use the same measurement as degradations.py for that defect.
Strength t in [0, 1] = share of the way through the kind's own range.
"""
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

import degradations as D
import degradations_alt as A


def _range(R, name, t):
    lo, hi = R[name]
    return lo + (hi - lo) * t


def _arr(im):
    return np.asarray(im).astype(np.float64) / 255.0


def _gauss_pil(x, sigma):
    return _arr(D.to_pil(x).filter(ImageFilter.GaussianBlur(float(sigma)))) if sigma > 0.05 else x


# ---------------------------------------------------------------- third kinds
def defocus_depth(x, rng, t):
    """Shallow depth of field: sharp around a focus point inside the eye region, blur growing with distance."""
    h, w = x.shape[:2]
    m = min(h, w)
    smax = 2.0 + 8.0 * t
    x0, y0, x1, y1 = D.eye_box()
    fy, fx = rng.uniform(y0, y1) * h, rng.uniform(x0, x1) * w
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.clip(np.hypot(xx - fx, yy - fy) / (rng.uniform(0.25, 0.5) * m), 0, 1)   # 0 at focus, 1 far away
    levels = np.linspace(0, smax, 5)
    stack = np.stack([_gauss_pil(x, s) for s in levels])
    pos = d * (len(levels) - 1)
    i0 = np.clip(np.floor(pos).astype(int), 0, len(levels) - 2)
    f = (pos - i0)[..., None]
    y = np.take_along_axis(stack, i0[None, ..., None].repeat(3, -1), 0)[0] * (1 - f) +\
        np.take_along_axis(stack, (i0 + 1)[None, ..., None].repeat(3, -1), 0)[0] * f
    return np.clip(y, 0, 1), D._edge_loss(x, y), {"kind": "depth_of_field", "max_sigma": smax}


def motion_decel(x, rng, t):
    """Decelerating smear: a straight path whose weight decays along it (movement slowing during the exposure)."""
    length = 6.0 + 30.0 * t
    ang = rng.uniform(0, np.pi)
    n = int(np.ceil(length)) | 1
    k = np.zeros((n + 2, n + 2))
    c = (n + 1) / 2
    for s in np.linspace(0, length, int(length * 4)):
        r, q = c + (s - length / 2) * np.sin(ang), c + (s - length / 2) * np.cos(ang)
        k[int(round(r)), int(round(q))] += np.exp(-3.0 * s / length)
    k /= k.sum()
    y = np.stack([ndimage.convolve(x[..., i], k, mode="nearest") for i in range(3)], axis=2)
    return np.clip(y, 0, 1), D._edge_loss(x, y), {"kind": "decelerating", "length": length}


def haze_flare(x, rng, t):
    """Veiling flare: a milky veil strongest around the brightest area (stray light from the source)."""
    h, w = x.shape[:2]
    m = min(h, w)
    L = ndimage.gaussian_filter(D.lum(x), m / 10)
    cy, cx = np.unravel_index(np.argmax(L), L.shape)
    yy, xx = np.mgrid[0:h, 0:w]
    s = rng.uniform(0.35, 0.8) * m
    vmax = 0.15 + 0.65 * t
    v = vmax * (0.35 + 0.65 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s ** 2)))
    veil = np.array([0.92, 0.92, 0.9])
    y = np.clip(x * (1 - v[..., None]) + veil * v[..., None], 0, 1)
    return y, D.haze_label(x, y), {"kind": "veiling_flare", "veil_max": vmax}


def haze_local_label(x, y):
    """PROPOSED haze label: share of LOCAL contrast lost = 1 - mean local detail(y) / mean local detail(x)
    (7 x 7 local brightness spread, as in the overexposure 'detail lost'). For a uniform veil it equals the
    current global label exactly; a non-uniform veil no longer fools it."""
    dx, dy = D._local_detail(x).mean(), D._local_detail(y).mean()
    return float(np.clip(1 - dy / dx, 0, 1)) if dx > 0 else 0.0


def glare_ring(x, rng, t):
    """Ring-light reflection: a bright, soft-edged ring (annulus) over the eye region, as when a ring
    illuminator reflects off the cornea / tear film. Larger and more structured than a normal corneal reflex."""
    h, w = x.shape[:2]
    m = min(h, w)
    yy, xx = np.mgrid[0:h, 0:w]
    x0, y0, x1, y1 = D.eye_box()
    cy, cx = rng.uniform(y0 + 0.05, y1 - 0.05) * h, rng.uniform(x0 + 0.05, x1 - 0.05) * w
    R = (0.06 + 0.16 * t) * m * rng.uniform(0.8, 1.0)          # ring radius
    width = (0.012 + 0.05 * t) * m                              # ring thickness
    aspect = rng.uniform(1.0, 1.4)
    d = np.hypot(xx - cx, (yy - cy) * aspect)
    wmap = np.exp(-((d - R) / width) ** 2) * rng.uniform(0.75, 1.0)
    if rng.uniform() < 0.5:                                       # broken ring (partial reflection)
        a0 = rng.uniform(0, 2 * np.pi)
        ang = np.arctan2(yy - cy, xx - cx)
        wmap *= 0.5 * (1 + np.cos(ang - a0))
    y = x + wmap[..., None] * (0.97 - x)
    return np.clip(y, 0, 1), A.glare_label(x, y), {"kind": "ring_reflection", "radius": R, "width": width}


def occlusion_squint(x, rng, t):
    """Squint / half blink: upper AND lower lid close in (upper larger), leaving a narrow almond-shaped opening.
    Skin = the photo's darker border tones with smooth shading and fine horizontal creases; dark lid margins."""
    h, w = x.shape[:2]
    cover = 0.45 + 0.40 * t                                       # total covered share target (height)
    up = cover * rng.uniform(0.6, 0.75)
    lo = cover - up
    xs = (np.arange(w) - rng.uniform(0.4, 0.6) * w) / w
    bulge = rng.uniform(0.08, 0.16)
    top_edge = (up + bulge * (2 * xs) ** 2) * h                   # opening widest near the middle
    bot_edge = h - (lo + 0.6 * bulge * (2 * xs) ** 2) * h
    yy = np.arange(h)[:, None].astype(float)
    m_top = np.clip((top_edge[None, :] - yy) / 4.0 + 0.5, 0, 1)
    m_bot = np.clip((yy - bot_edge[None, :]) / 4.0 + 0.5, 0, 1)
    mask = np.maximum(m_top, m_bot)
    b = max(4, int(0.06 * min(h, w)))
    frame = np.concatenate([x[:b].reshape(-1, 3), x[-b:].reshape(-1, 3)])
    skin = np.clip(np.quantile(frame, 0.35, axis=0) * 0.9 + 0.06, 0, 1)
    shade = 0.85 + 0.25 * np.clip(np.minimum(top_edge[None, :] - yy, yy - bot_edge[None, :]) * -1 / (0.3 * h), 0, 1)
    shade = np.where(mask > 0, 0.85 + 0.25 * np.clip(np.maximum(top_edge[None, :] - yy, yy - bot_edge[None, :]) / (0.3 * h), 0, 1), 1.0)
    tex = ndimage.gaussian_filter(rng.standard_normal((h, w)), rng.uniform(6, 12))
    tex = 1 + 0.06 * tex / (np.abs(tex).max() + 1e-9) + 0.02 * ndimage.gaussian_filter(rng.standard_normal((h, w)), 0.8)
    lid = skin[None, None, :] * (shade * tex)[..., None]
    open_col = (top_edge < bot_edge)[None, :]                     # margins only where the eye is open
    dist = np.minimum(np.abs(yy - top_edge[None, :]), np.abs(yy - bot_edge[None, :]))
    margin = np.exp(-(dist / 2.5) ** 2) * open_col
    lid = lid * (1 - 0.45 * margin[..., None])
    y = x * (1 - mask[..., None]) + np.clip(lid, 0, 1) * mask[..., None]
    y = y * (1 - 0.45 * (margin * (1 - mask))[..., None])        # dark margin just inside the opening too
    return np.clip(y, 0, 1), float(mask.mean()), {"kind": "squint", "cover_target": cover}


def noise_colour(x, rng, t):
    """Correlated colour noise (blotchy chroma + luminance grain), like phone low-light noise."""
    sigma = 0.01 + 0.07 * t
    lum_n = rng.normal(0, 1, x.shape[:2])[..., None]
    chroma = np.stack([ndimage.gaussian_filter(rng.normal(0, 1, x.shape[:2]), 1.5) for _ in range(3)], axis=2)
    chroma /= chroma.std() + 1e-9
    n = 0.6 * lum_n + 0.8 * chroma
    n *= sigma / (n.std() + 1e-9)
    y = np.clip(x + n, 0, 1)
    return y, float(np.clip((y - x).std(), 0, 1)), {"kind": "colour_noise", "sigma": sigma}


def uneven_band(x, rng, t):
    """Slit-band light: a lit band crosses the eye region; outside it the light drops (linear light),
    label in displayed brightness over the central box as in degradations.py v4."""
    rho = (0.02 ** (1 / 2.2)) + ((0.8 ** (1 / 2.2)) - (0.02 ** (1 / 2.2))) * (1 - t)   # t = 1 strongest
    r = rho ** 2.2
    h, w = x.shape[:2]
    m = min(h, w)
    yy, xx = np.mgrid[0:h, 0:w]
    u, v = (xx - w / 2) / m, (yy - h / 2) / m
    a = rng.uniform(0, np.pi)
    q = u * np.cos(a) + v * np.sin(a)
    c, bw = rng.uniform(-0.1, 0.1), rng.uniform(0.08, 0.2)
    s = np.exp(-((q - c) ** 2) / (2 * bw ** 2))
    G = r + (1 - r) * s
    y = D.delin(D.lin(x) * G[..., None])
    ys, xs = D._box_slices(x.shape)
    Gr = G[ys, xs] ** (1 / 2.2)
    return y, float(np.clip(1 - Gr.min() / Gr.max(), 0, 1)), {"kind": "slit_band", "dark_ratio": r}


# ---------------------------------------------------------------- registry
def _wrap(F, R, name, kind, invert=False):
    def f(x, rng, t):
        s = _range(R, name, (1 - t) if invert else t)
        y, lab, p = F[name](x, rng, s)
        return y, lab, {"kind": kind, **p}
    return f


KINDS = {
    "defocus_blur": {"ours": _wrap(D.FUNCS, D.RANGES, "defocus_blur", "ours"),
                     "alt": _wrap(A.FUNCS, A.RANGES_ALT, "defocus_blur", "alt"), "depth": defocus_depth},
    "motion_blur": {"ours": _wrap(D.FUNCS, D.RANGES, "motion_blur", "ours"),
                    "alt": _wrap(A.FUNCS, A.RANGES_ALT, "motion_blur", "alt"), "decel": motion_decel},
    "haze": {"ours": _wrap(D.FUNCS, D.RANGES, "haze", "ours"),
             "alt": _wrap(A.FUNCS, A.RANGES_ALT, "haze", "alt"), "flare": haze_flare},
    "glare": {"ours": _wrap(D.FUNCS, D.RANGES, "glare", "ours"),
              "alt": _wrap(A.FUNCS, A.RANGES_ALT, "glare", "alt"), "ring": glare_ring},
    "occlusion": {"ours": _wrap(D.FUNCS, D.RANGES, "occlusion", "ours"),
                  "alt": _wrap(A.FUNCS, A.RANGES_ALT, "occlusion", "alt"), "squint": occlusion_squint},
    "noise": {"ours": _wrap(D.FUNCS, D.RANGES, "noise", "ours"),
              "alt": _wrap(A.FUNCS, A.RANGES_ALT, "noise", "alt"), "colour": noise_colour},
    # uneven: low r = strong, so t is inverted for the wrapped kinds (t = 1 strongest everywhere)
    "uneven_illumination": {"ours": _wrap(D.FUNCS, D.RANGES, "uneven_illumination", "ours", invert=True),
                            "alt": _wrap(A.FUNCS, A.RANGES_ALT, "uneven_illumination", "alt", invert=True),
                            "band": uneven_band},
}


# ---------------------------------------------------------------- training sampler
def _uneven_ours_display(x, rng, t):
    """Ours uneven illumination with the displayed darkness ratio drawn evenly: t = 1 strongest."""
    lo, hi = D.RANGES["uneven_illumination"]
    rho = lo ** (1 / 2.2) + (hi ** (1 / 2.2) - lo ** (1 / 2.2)) * (1 - t)
    y, lab, p = D.uneven_illumination(x, rng, rho ** 2.2)
    return y, lab, {"kind": "ours", **p}


KINDS["uneven_illumination"]["ours"] = _uneven_ours_display


def _exposure_kinds(name):
    return {k: (lambda kk: (lambda x, rng, t: D.exposure_kind(name, kk, x, rng, t)))(k) for k in D.EXPOSURE_KINDS[name]}


def kind_funcs(name):
    """All TRAINING kinds of one defect: {kind: f(x, rng, t) -> (y, label, params)}."""
    return _exposure_kinds(name) if name in D.EXPOSURE_KINDS else KINDS[name]


def sample_kinds(x, rng, min_defects=1):
    """Training sample with several implementations per defect. Same mixing policy as degradations.sample
    (N_DEFECTS_P, NOT_TOGETHER, an extreme exposure / haze draw applied alone, low light = underexposure + noise
    with p LOW_LIGHT_P, application ORDER); per applied defect a kind is drawn uniformly from kind_funcs(name) and
    a strength t ~ U(0, 1). The eye box must be set first (degradations.set_eye_box). Returns (y, labels, applied)."""
    ks = [k for k in D.N_DEFECTS_P if k >= min_defects]
    p = np.array([D.N_DEFECTS_P[k] for k in ks])
    k = int(rng.choice(ks, p=p / p.sum()))
    chosen = []
    for name in rng.permutation(D.VALUES):
        if len(chosen) == k:
            break
        if any({name, c} in [set(q) for q in D.NOT_TOGETHER] for c in chosen):
            continue
        chosen.append(str(name))
    t = {n: float(rng.uniform()) for n in chosen}
    extreme = [n for n in chosen if n in D.EXTREME and t[n] >= D.EXTREME[n]]
    if extreme:
        chosen = [extreme[0]]
    elif "underexposure" in chosen and "noise" not in chosen and len(chosen) < 3 and rng.uniform() < D.LOW_LIGHT_P:
        chosen.append("noise")
        t["noise"] = float(rng.uniform())
    labels, applied, y = {v: 0.0 for v in D.VALUES}, {}, x
    for name in D.ORDER:
        if name in chosen:
            funcs = kind_funcs(name)
            kind = str(rng.choice(sorted(funcs)))
            y, labels[name], params = funcs[kind](y, rng, t[name])
            applied[name] = {"impl": kind, "t": t[name], **params}
    return y, labels, applied


# ---------------------------------------------------------------- v3 additions (first)
def occlusion_side(x, rng, t):
    """Occluder entering from the LEFT / RIGHT or a CORNER (nose bridge, finger, lid speculum): a large rounded skin-toned
    shape with a moderately soft edge, shading toward its edge and fine texture; never darker than a share of the eye region's
    brightness (brightness floor, review). Label = covered share (as all occlusion)."""
    h, w = x.shape[:2]
    m = min(h, w)
    cover = 0.45 + 0.40 * t
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    side = str(rng.choice(["left", "right", "top_left", "top_right", "bottom_left", "bottom_right"]))
    # rounded front: ellipse whose far part lies outside the image; its depth is solved for the target cover
    ax, ay = (1.0, rng.uniform(0.9, 1.6)) if side in ("left", "right") else (1.0, rng.uniform(0.8, 1.2))
    cx = -0.0 if "left" in side else w
    cy = (h / 2 + rng.uniform(-0.2, 0.2) * h) if side in ("left", "right") else (0.0 if "top" in side else h)

    ang = np.arctan2(yy - cy, xx - cx)
    wob_f = rng.uniform(2, 5)                                       # irregular outline (low-frequency wobble)
    wob = 1 + rng.uniform(0.04, 0.12) * np.sin(wob_f * ang + rng.uniform(0, 2 * np.pi))             + 0.04 * np.sin(2.3 * wob_f * ang + rng.uniform(0, 2 * np.pi))
    rng_edge = rng.uniform(6.0, 30.0)                               # close to the lens: out of focus, soft edge

    def mask_for(r):
        d = np.sqrt(((xx - cx) / (ax * r)) ** 2 + ((yy - cy) / (ay * r)) ** 2) / wob
        return 1 / (1 + np.exp(-(1 - d) * r / rng_edge))
    lo, hi = 1.0, 3.0 * max(h, w)
    for _ in range(30):                                       # bisection on the size for the target covered share
        mid = (lo + hi) / 2
        if mask_for(mid).mean() < cover:
            lo = mid
        else:
            hi = mid
    mask = mask_for((lo + hi) / 2)
    b = max(4, int(0.08 * m))
    frame = np.concatenate([x[:b].reshape(-1, 3), x[-b:].reshape(-1, 3), x[:, :b].reshape(-1, 3), x[:, -b:].reshape(-1, 3)])
    skin = np.clip(np.quantile(frame, 0.5, axis=0) * rng.uniform(0.6, 0.95) + 0.04, 0, 1)
    # brightness floor (review): on dark-background photos the frame tone is near black and
    # the occluder looked like a shadow, not an object. Its colour is mixed with the eye region's (lit by the same light)
    # and its brightness is at least a share (0.5-0.8) of the eye region's mean brightness, and at least 0.1.
    ys, xs = D._box_slices(x.shape)
    eye = x[ys, xs].reshape(-1, 3).mean(0)
    floor = max(rng.uniform(0.5, 0.8) * eye.mean(), 0.1)
    if skin.mean() < floor:
        skin = 0.5 * skin + 0.5 * eye
        skin = np.clip(skin * floor / max(skin.mean(), 1e-6), 0, 1)
    inside = np.clip((mask - 0.5) * 2, 0, 1)
    shade = 1.15 - 0.45 * ndimage.gaussian_filter(inside, m / 8)            # lit rim, darker toward its far side
    rim = np.exp(-((mask - 0.6) / 0.12) ** 2)                               # bright rim just inside the edge (review)
    shade = shade + 0.35 * rim
    blotch = 1 + 0.06 * ndimage.gaussian_filter(rng.standard_normal((h, w)), m / 15) / 0.01
    occ = np.clip(skin[None, None, :] * (shade * np.clip(blotch, 0.85, 1.15))[..., None], 0, 1)
    occ = np.stack([ndimage.gaussian_filter(occ[..., c], rng_edge / 3) for c in range(3)], 2)   # whole object defocused
    y = x * (1 - mask[..., None]) + occ * mask[..., None]
    return np.clip(y, 0, 1), float(mask.mean()), {"kind": "side_corner", "side": side, "cover_target": cover}


def uneven_vignette(x, rng, t):
    """Smooth whole-frame falloff (lens vignetting / lamp off to one side): cos^4 falloff from an off-centre point plus a
    gentle linear tilt, in linear light. Label = displayed min / max over the eye box (as all uneven illumination)."""
    h, w = x.shape[:2]
    m = min(h, w)
    yy, xx = np.mgrid[0:h, 0:w]
    a = rng.uniform(0, 2 * np.pi)
    off = rng.uniform(0.1, 0.35) * m
    cx, cy = w / 2 + off * np.cos(a), h / 2 + off * np.sin(a)
    reach = (1.3 - 0.6 * t) * np.hypot(w, h) / 2                  # smaller reach = steeper falloff
    d = np.clip(np.hypot(xx - cx, yy - cy) / reach, 0, 1)
    depth = 0.15 + 0.83 * t                                       # how dark the far side gets (t = 1 strongest)
    G = 1 - depth * (1 - np.cos(d * np.pi / 2) ** 4)
    tilt = 1 - 0.2 * t * (((xx - w / 2) * np.cos(a) + (yy - h / 2) * np.sin(a)) / m + 0.5).clip(0, 1)
    G = np.clip(G * tilt, 0.02, 1)
    G = G / G.max()
    y = D.delin(D.lin(x) * G[..., None])
    ys, xs = D._box_slices(x.shape)
    Gr = G[ys, xs] ** (1 / 2.2)
    return y, float(np.clip(1 - Gr.min() / Gr.max(), 0, 1)), {"kind": "smooth_vignette", "reach": reach}


def defocus_astig_mild(x, rng, t):
    """Mild astigmatic defocus: elliptical Gaussian, minor / major axis ratio 0.6-0.9 (strongly elongated blur is left out:
    it is physically ambiguous with motion blur), spatial separable-rotated kernel."""
    s_major = 1.2 + 7.0 * t
    s_minor = s_major * rng.uniform(0.6, 0.9)
    ang = rng.uniform(0, np.pi)
    r = int(np.ceil(3 * s_major))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(float)
    u = xx * np.cos(ang) + yy * np.sin(ang)
    v = -xx * np.sin(ang) + yy * np.cos(ang)
    k = np.exp(-0.5 * ((u / s_major) ** 2 + (v / s_minor) ** 2))
    k /= k.sum()
    y = D._convolve(x, k)                                          # reflect-padded FFT convolution (as degradations.py)
    return y, D._edge_loss(x, y), {"kind": "astig_mild", "sigma_major": s_major, "sigma_minor": s_minor}


KINDS["occlusion"]["side"] = occlusion_side
KINDS["uneven_illumination"]["vignette"] = uneven_vignette
KINDS["defocus_blur"]["astig_mild"] = defocus_astig_mild

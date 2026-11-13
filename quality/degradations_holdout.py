"""HELD-OUT 4th implementation of the 9 defects. TEST ONLY: never used for training or model
selection. Each defect is modelled differently from every training kind (degradations.py v4, degradations_alt.py
v2, defect_kinds.py third kinds, exposure kinds ), and labels use the same measurements as training:

| Defect              | Held-out model                                                                     | Label                     |
| defocus_blur        | astigmatic (elliptical) Gaussian blur, FFT domain, random axis                    | 1 - T(y)/T(x)             |
| motion_blur         | rotational blur about a point near the eye (eye / head rotation)                  | 1 - T(y)/T(x)             |
| overexposure        | exposure up then a filmic S-curve tone map (phone-camera style)                   | detail lost               |
| underexposure       | low-light sensor response: exposure down, raised black level, colour cast        | share of light lost       |
| uneven_illumination | inverse-square falloff from a point lamp off to one side                          | displayed min/max, eye box|
| haze                | diffuse glow / dirty lens: blurred brightened copy blended over the image          | local contrast lost       |
| glare               | lens-flare ghosts: bright source in the eye box + hexagonal ghosts along a line   | pull toward white, eye box|
| occlusion           | out-of-focus dark foreground blob (lid / finger too close to the lens)             | covered share (soft)      |
| noise               | raw-space read + shot noise (strongest in shadows) + faint row banding            | std of added noise        |
Strength t in [0, 1], t = 1 strongest.
"""
import numpy as np
from scipy import ndimage

import degradations as D
import degradations_alt as A


def defocus_astig(x, rng, t):
    h, w = x.shape[:2]
    s_major = 1.5 + 7.0 * t
    s_minor = s_major * rng.uniform(0.25, 0.6)
    ang = rng.uniform(0, np.pi)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    u = fx * np.cos(ang) + fy * np.sin(ang)
    v = -fx * np.sin(ang) + fy * np.cos(ang)
    H = np.exp(-2 * np.pi ** 2 * ((s_major * u) ** 2 + (s_minor * v) ** 2))
    y = np.stack([np.real(np.fft.ifft2(np.fft.fft2(x[..., c]) * H)) for c in range(3)], axis=2)
    y = np.clip(y, 0, 1)
    return y, D._edge_loss(x, y), {"kind": "astigmatic", "sigma_major": s_major, "sigma_minor": s_minor}


def motion_rotational(x, rng, t):
    h, w = x.shape[:2]
    x0, y0, x1, y1 = D.eye_box()
    cy, cx = rng.uniform(y0, y1) * h, rng.uniform(x0, x1) * w
    total_deg = 1.0 + 7.0 * t
    n = 15
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    acc = np.zeros_like(x)
    for a in np.deg2rad(np.linspace(-total_deg / 2, total_deg / 2, n)):
        ca, sa = np.cos(a), np.sin(a)
        sx = cx + (xx - cx) * ca - (yy - cy) * sa
        sy = cy + (xx - cx) * sa + (yy - cy) * ca
        acc += np.stack([ndimage.map_coordinates(x[..., c], [sy, sx], order=1, mode="nearest") for c in range(3)], 2)
    y = np.clip(acc / n, 0, 1)
    return y, D._edge_loss(x, y), {"kind": "rotational", "degrees": total_deg}


def _filmic(v):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14              # ACES-style fitted curve
    return np.clip((v * (a * v + b)) / (v * (c * v + d) + e), 0, 1)


def over_filmic(x, rng, t):
    ev = 0.3 + 2.2 * t                                         # review: 1-4 stops was too strong
    xl = D.lin(x) * 2.0 ** ev
    y = D.delin(_filmic(xl) / _filmic(np.array(1.0)))           # normalised so exposure 0 maps white to white
    y = np.clip(y, 0, 1)
    return y, D.overexposure_label(x, y), {"kind": "filmic", "ev": ev}


def under_lowlight(x, rng, t):
    gain = 1.0 - (0.25 + 0.7 * t)                              # light kept
    black = 0.02 + 0.05 * t                                    # raised black level (sensor offset)
    cast = 1 + rng.uniform(-0.12, 0.12, 3) * t                 # low-light white-balance error
    yl = D.lin(x) * gain * cast[None, None, :]
    y = np.clip(black + (1 - black) * D.delin(np.clip(yl, 0, 1)), 0, 1)
    m0 = D.lin(D.lum(x)).mean()
    label = 1 - D.lin(D.lum(y)).mean() / m0 if m0 > 0 else 0.0
    return y, float(np.clip(label, 0, 1)), {"kind": "low_light", "gain": gain, "black": black}


def uneven_pointlamp(x, rng, t):
    h, w = x.shape[:2]
    m = min(h, w)
    a = rng.uniform(0, 2 * np.pi)
    dist = rng.uniform(0.35, 0.6) * m
    lx, ly = w / 2 + dist * np.cos(a), h / 2 + dist * np.sin(a)
    height = (0.9 - 0.75 * t) * m                              # lower lamp = steeper falloff (t = 1 strongest)
    yy, xx = np.mgrid[0:h, 0:w]
    G = height ** 2 / ((xx - lx) ** 2 + (yy - ly) ** 2 + height ** 2)
    G = G / G.max()
    y = D.delin(D.lin(x) * G[..., None])
    ys, xs = D._box_slices(x.shape)
    Gr = G[ys, xs] ** (1 / 2.2)
    return y, float(np.clip(1 - Gr.min() / Gr.max(), 0, 1)), {"kind": "point_lamp", "lamp_height": height}


def haze_glow(x, rng, t):
    a = 0.15 + 0.6 * t
    s = rng.uniform(15, 35)
    glow = np.stack([ndimage.gaussian_filter(x[..., c], s) for c in range(3)], 2)
    glow = np.clip(glow * 1.15 + 0.12, 0, 1)
    y = np.clip((1 - a) * x + a * glow, 0, 1)
    return y, D.haze_label(x, y), {"kind": "diffuse_glow", "weight": a, "sigma": s}


def glare_ghosts(x, rng, t):
    h, w = x.shape[:2]
    m = min(h, w)
    x0, y0, x1, y1 = D.eye_box()
    sy, sx = rng.uniform(y0, y1) * h, rng.uniform(x0, x1) * w
    yy, xx = np.mgrid[0:h, 0:w]
    src_r = (0.02 + 0.06 * t) * m
    wmap = np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * src_r ** 2))
    cxm, cym = w / 2, h / 2                                     # ghosts lie on the line source -> image centre
    for k in range(int(2 + 4 * t)):
        f = rng.uniform(-1.4, 0.6)
        gx, gy = cxm + (sx - cxm) * f, cym + (sy - cym) * f
        r = rng.uniform(0.02, 0.07) * m
        ang = np.arctan2(yy - gy, xx - gx)
        rr = r * np.cos(np.pi / 6) / np.cos(((ang + np.pi / 6) % (np.pi / 3)) - np.pi / 6)   # hexagon radius
        d = np.hypot(xx - gx, yy - gy)
        wmap = np.maximum(wmap, np.clip((rr - d) / 3 + 0.5, 0, 1) * rng.uniform(0.15, 0.4) * (0.5 + 0.5 * t))
    tint = np.array([1.0, 0.97, 0.9])
    y = np.clip(x + wmap[..., None] * (tint - x), 0, 1)
    return y, A.glare_label(x, y), {"kind": "flare_ghosts", "source_radius": src_r}


def occlusion_foreground(x, rng, t):
    h, w = x.shape[:2]
    m = min(h, w)
    cover = 0.45 + 0.4 * t
    side = rng.choice(["top", "bottom", "left", "right"], p=[0.45, 0.25, 0.15, 0.15])
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    wob = ndimage.gaussian_filter(rng.standard_normal(max(h, w)), 20) * 0.08 * m
    if side in ("top", "bottom"):
        edge = cover * h + wob[:w][None, :]
        dist = (edge - yy) if side == "top" else (edge - (h - 1 - yy))
    else:
        edge = cover * w + wob[:h][:, None]
        dist = (edge - xx) if side == "left" else (edge - (w - 1 - xx))
    blur = rng.uniform(0.012, 0.025) * m                       # out of focus but a clear edge (review)
    mask = 1 / (1 + np.exp(-dist / blur))
    b = max(4, int(0.08 * m))                                  # skin-like tone from the photo's frame, lit from the eye side
    frame = np.concatenate([x[:b].reshape(-1, 3), x[-b:].reshape(-1, 3), x[:, :b].reshape(-1, 3), x[:, -b:].reshape(-1, 3)])
    base = np.clip(np.quantile(frame, 0.6, axis=0) * rng.uniform(0.55, 0.85) + 0.05, 0, 1)
    shade = np.clip(1.15 - 0.5 * np.clip(dist / (0.4 * m), 0, 1), 0.6, 1.15)   # brighter near its edge (facing the light)
    tex = 1 + 0.05 * ndimage.gaussian_filter(rng.standard_normal((h, w)), 18) / 0.02
    colour = np.clip(base[None, None, :] * (shade * np.clip(tex, 0.85, 1.15))[..., None], 0, 1)
    y = x * (1 - mask[..., None]) + colour * mask[..., None]
    return np.clip(y, 0, 1), float(mask.mean()), {"kind": "foreground_blob", "side": str(side), "cover": cover}


def noise_raw(x, rng, t):
    read = 0.004 + 0.02 * t
    shot = 0.01 + 0.06 * t
    xl = D.lin(x)
    nl = rng.normal(0, 1, x.shape) * np.sqrt(read ** 2 + shot ** 2 * xl)
    band = rng.normal(0, 0.3 * read, (x.shape[0], 1, 1))       # faint row banding
    y = np.clip(D.delin(np.clip(xl + nl + band, 0, 1)), 0, 1)
    return y, float(np.clip((y - x).std(), 0, 1)), {"kind": "raw_sensor", "read": read, "shot": shot}


HOLDOUT = {"defocus_blur": defocus_astig, "motion_blur": motion_rotational, "overexposure": over_filmic,
           "underexposure": under_lowlight, "uneven_illumination": uneven_pointlamp, "haze": haze_glow,
           "glare": glare_ghosts, "occlusion": occlusion_foreground, "noise": noise_raw}

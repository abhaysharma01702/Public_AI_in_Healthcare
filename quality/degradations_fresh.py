"""FRESH held-out implementation of the 9 defects (plan step A3). TEST ONLY: never used for
training, model selection or development. Frozen after visual review (MD5 recorded); never edited
after that. Each defect is modelled differently from every training kind (degradations.py v4 + exposure kinds,
degradations_alt.py v2, defect_kinds.py incl. the kinds) AND from the development held-out set
(degradations_holdout.py). Labels use the same measurements as training:

| Defect              | Fresh model                                                                          | Label                      |
| defocus_blur        | wave-optics defocus PSF (pupil with defocus + spherical phase, |FFT|^2), slight    | 1 - T(y)/T(x)              |
|                     | longitudinal chromatic aberration (colour channels focused differently)              |                            |
| motion_blur         | zoom (radial) blur: head / eye moving toward or away from the camera                 | 1 - T(y)/T(x)              |
| overexposure        | raw-pipeline clipping: gain in sensor space, raw channels clip, then white balance   | detail lost                |
|                     | (coloured / hue-shifted highlights, no roll-off)                                     |                            |
| underexposure       | linear-light 8-bit capture: less light, quantised in linear light (posterised        | share of light lost        |
|                     | shadows), then gamma                                                                 |                            |
| uneven_illumination | ring light with a segment out: sum of the working ring lamps (inverse square)        | displayed min/max, eye box |
| haze                | condensation: droplets on the lens, each scattering (blurred + lifted) light         | local contrast lost        |
| glare               | window reflection: soft-edged rectangle (room window / lamp housing) on the cornea,   | pull toward white, eye box |
|                     | optionally with window bars                                                          |                            |
| occlusion           | gloved fingers (2-3 rounded fingers, nitrile blue / purple / latex white) from the   | covered share (soft)       |
|                     | top or bottom edge, holding the lid                                                  |                            |
| noise               | sensor noise + JPEG compression (phone pipeline: blocky, smeared noise)              | std of y - x               |
Strength t in [0, 1], t = 1 strongest. The eye box must be set first (degradations.set_eye_box).
"""
import io

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.signal import fftconvolve

import degradations as D
import degradations_alt as A


def _wave_psf(w20, w40, n=192, pupil=48, crop=71):
    """Incoherent PSF of a circular pupil with defocus (w20) and spherical (w40) aberration, in waves."""
    c = np.arange(n) - n / 2
    yy, xx = np.meshgrid(c, c, indexing="ij")
    rho2 = (xx ** 2 + yy ** 2) / pupil ** 2
    P = (rho2 <= 1).astype(float) * np.exp(2j * np.pi * (w20 * rho2 + w40 * rho2 ** 2))
    psf = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(P)))) ** 2
    h = crop // 2
    psf = psf[n // 2 - h:n // 2 + h + 1, n // 2 - h:n // 2 + h + 1]
    return psf / psf.sum()


def defocus_wave(x, rng, t):
    w20 = 0.1 + 3.4 * t ** 2                                      # defocus, waves at the pupil edge
    w40 = rng.uniform(-0.4, 0.4)                               # some spherical aberration
    lca = rng.uniform(0.15, 0.4) * rng.choice([-1, 1])         # longitudinal chromatic aberration, waves
    out = []
    for c, shift in enumerate([lca, 0.0, -lca]):              # R / G / B focused at slightly different depths
        k = _wave_psf(max(w20 + shift, 0.0), w40)
        p = k.shape[0] // 2
        a = np.pad(x[..., c], p, mode="reflect")
        out.append(fftconvolve(a, k, mode="same")[p:-p, p:-p])
    y = np.clip(np.stack(out, 2), 0, 1)
    return y, D._edge_loss(x, y), {"kind": "wave_optics", "w20": w20, "w40": w40, "lca": lca}


def motion_zoom(x, rng, t):
    h, w = x.shape[:2]
    x0, y0, x1, y1 = D.eye_box()
    cy, cx = rng.uniform(y0, y1) * h, rng.uniform(x0, x1) * w
    z = 0.01 + 0.12 * t                                        # total scale change during the exposure
    direction = rng.choice([-1, 1])                            # toward or away from the camera
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    n = 17
    acc = np.zeros_like(x)
    for s in np.linspace(0, z, n):
        f = 1 + direction * s
        sy, sx = cy + (yy - cy) / f, cx + (xx - cx) / f
        acc += np.stack([ndimage.map_coordinates(x[..., c], [sy, sx], order=1, mode="nearest") for c in range(3)], 2)
    y = np.clip(acc / n, 0, 1)
    return y, D._edge_loss(x, y), {"kind": "zoom", "scale_change": z, "direction": int(direction)}


def over_raw(x, rng, t):
    ev = 0.3 + 2.2 * t
    wb = np.array([rng.uniform(1.6, 2.4), 1.0, rng.uniform(1.3, 2.0)])   # camera white-balance gains (R, G, B)
    raw = D.lin(x) / wb[None, None, :]                         # back to sensor space
    raw = np.clip(raw * 2.0 ** ev, 0, 1)                       # sensor channels saturate separately
    y = np.clip(D.delin(np.clip(raw * wb[None, None, :], 0, 1)), 0, 1)   # white balance after clipping: hue shifts
    return y, D.overexposure_label(x, y), {"kind": "raw_clip", "ev": ev, "wb": wb.round(3).tolist()}


def under_linear8(x, rng, t):
    keep = 1.0 - (0.25 + 0.7 * t)                              # share of light kept
    yl = D.lin(x) * keep
    levels = 255 * rng.uniform(0.5, 1.0)                       # linear-light 8-bit (or coarser) capture
    yl = np.round(yl * levels) / levels                        # posterised shadows after the gamma below
    y = np.clip(D.delin(np.clip(yl, 0, 1)), 0, 1)
    m0 = D.lin(D.lum(x)).mean()
    label = 1 - D.lin(D.lum(y)).mean() / m0 if m0 > 0 else 0.0
    return y, float(np.clip(label, 0, 1)), {"kind": "linear_8bit", "keep": keep, "levels": levels}


HEIGHT_HI, GAMMA_LO, GAMMA_SPAN = 0.9, 0.3, 4.0      # calibrated on 10 val photos: labels 0.10 / 0.54 / 0.77 at t = 0 / 0.5 / 1


def uneven_ringgap(x, rng, t):
    """Ring light with a segment out (or partly blocked): the light at each point is the sum over the working lamps
    of the ring (inverse square, cosine-weighted); points on the side of the gap get less light. Smooth, no singular point."""
    h, w = x.shape[:2]
    m = min(h, w)
    x0, y0, x1, y1 = D.eye_box()
    cx, cy = (x0 + x1) / 2 * w, (y0 + y1) / 2 * h
    R = rng.uniform(0.45, 0.7) * m                             # ring radius
    H = HEIGHT_HI * m                                         # ring height above the eye (lower = steeper)
    gap_c, gap_w = rng.uniform(0, 2 * np.pi), rng.uniform(0.45, 0.75) * 2 * np.pi   # missing arc (centre, width)
    th = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    on = np.abs((th - gap_c + np.pi) % (2 * np.pi) - np.pi) > gap_w / 2
    yy, xx = np.mgrid[0:h:4, 0:w:4].astype(float)              # computed at 1/4 resolution (the field is smooth)
    G = np.zeros_like(xx)
    for a in th[on]:
        d2 = (xx - cx - R * np.cos(a)) ** 2 + (yy - cy - R * np.sin(a)) ** 2 + H ** 2
        G += H / d2 ** 1.5
    G = ndimage.zoom(G, (h / G.shape[0], w / G.shape[1]), order=1)[:h, :w]
    ys, xs = D._box_slices(x.shape)
    G = G / G[ys, xs].max()                                    # the brightest part of the eye region keeps its light
    G = np.clip(G ** (GAMMA_LO + GAMMA_SPAN * t), 0.02, 1)     # strength: how steeply the light falls toward the gap
    y = D.delin(D.lin(x) * G[..., None])
    Gr = G[ys, xs] ** (1 / 2.2)
    return y, float(np.clip(1 - Gr.min() / Gr.max(), 0, 1)), {"kind": "ring_gap", "ring_height": H, "gap_width": gap_w}


def haze_droplets(x, rng, t):
    h, w = x.shape[:2]
    m = min(h, w)
    n = int(300 + 900 * t)                                     # droplet clusters (mottle of the fog)
    cov = np.zeros((h, w))
    ys, xs = rng.uniform(0, h, n), rng.uniform(0, w, n)
    rs = rng.uniform(0.006, 0.03, n) * m
    yy, xx = np.ogrid[0:h, 0:w]
    for cy, cx, r in zip(ys, xs, rs):
        y0, y1 = int(max(cy - 2 * r, 0)), int(min(cy + 2 * r + 1, h))
        x0, x1 = int(max(cx - 2 * r, 0)), int(min(cx + 2 * r + 1, w))
        d = np.hypot(yy[y0:y1] - cy, xx[:, x0:x1] - cx)
        cov[y0:y1, x0:x1] = np.maximum(cov[y0:y1, x0:x1], np.clip((r - d) / 1.5 + 0.5, 0, 1))
    # the lens is out of focus: the droplet layer is seen as a smooth fog with a large-scale mottle (sharp droplet
    # edges would ADD local contrast; first draft, review)
    cov = ndimage.gaussian_filter(cov, rng.uniform(0.02, 0.035) * m)
    cov = cov / max(cov.max(), 1e-6)
    cov = (0.15 + 0.8 * t) * (0.5 + 0.5 * cov)
    s = rng.uniform(4, 10)
    scat = np.stack([ndimage.gaussian_filter(x[..., c], s) for c in range(3)], 2)
    veil = np.clip(0.6 * scat + 0.4 * (0.75 + 0.25 * x.reshape(-1, 3).mean(0)), 0, 1)
    y = np.clip(x * (1 - cov[..., None]) + veil * cov[..., None], 0, 1)
    return y, D.haze_label(x, y), {"kind": "droplets", "droplets": n, "sigma": s}


def glare_window(x, rng, t):
    h, w = x.shape[:2]
    m = min(h, w)
    x0, y0, x1, y1 = D.eye_box()
    cy, cx = rng.uniform(y0 + 0.2 * (y1 - y0), y1 - 0.2 * (y1 - y0)) * h, rng.uniform(x0 + 0.2 * (x1 - x0), x1 - 0.2 * (x1 - x0)) * w
    bw = (0.03 + 0.09 * t) * m                                 # half width of the reflection
    bh = bw * rng.uniform(0.5, 0.9)
    rot = rng.uniform(-0.4, 0.4)
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    u = (xx - cx) * np.cos(rot) + (yy - cy) * np.sin(rot)
    v = -(xx - cx) * np.sin(rot) + (yy - cy) * np.cos(rot)
    k = rng.uniform(0.1, 0.25)                                 # curved cornea: barrel-distorted window and bars
    u, v = u * (1 + k * (v / bh) ** 2) * (1 + 0.1 * (v / bh)), v * (1 + k * (u / bw) ** 2)
    edge = rng.uniform(2.5, 6.0)
    wmap = (np.clip((bw - np.abs(u)) / edge + 0.5, 0, 1) * np.clip((bh - np.abs(v)) / edge + 0.5, 0, 1))
    bars = bool(rng.uniform() < 0.5)
    if bars:                                                   # window frame: a cross of darker bars
        bar = max(1.5, 0.08 * bw)
        wmap *= 1 - 0.7 * np.clip((bar - np.abs(u)) / 1.5 + 0.5, 0, 1)
        wmap *= 1 - 0.7 * np.clip((bar - np.abs(v)) / 1.5 + 0.5, 0, 1)
    grad = 1 - 0.3 * np.clip((u / bw + 1) / 2, 0, 1) * rng.uniform(0.3, 1.0)   # brightness gradient across the window
    wmap = ndimage.gaussian_filter(wmap * grad, 1.2) * rng.uniform(0.8, 1.0)
    tint = np.array([0.97, 0.99, 1.0])                         # daylight
    y = np.clip(x + wmap[..., None] * (tint - x), 0, 1)
    return y, A.glare_label(x, y), {"kind": "window", "half_width": bw, "bars": bars}


GLOVES = {"nitrile_blue": (0.35, 0.55, 0.85), "nitrile_purple": (0.5, 0.4, 0.75), "latex_white": (0.88, 0.86, 0.8)}


def occlusion_glove(x, rng, t):
    h, w = x.shape[:2]
    m = min(h, w)
    cover = 0.45 + 0.4 * t
    from_top = bool(rng.uniform() < 0.6)
    k = int(rng.integers(2, 4))
    total = rng.uniform(0.85, 1.05) * w                        # fingers side by side, slightly overlapping
    widths = rng.dirichlet(np.full(k, 6.0)) * total
    centres = np.cumsum(widths) - widths / 2 + (w - total) / 2 + rng.uniform(-0.05, 0.05) * w
    offsets = rng.uniform(-0.12, 0.12, k) * h                  # fingertips at different heights
    tilts = rng.uniform(-0.25, 0.25) + rng.uniform(-0.08, 0.08, k)   # each finger at its own angle
    taper = rng.uniform(0.0, 0.2)                                    # fingers narrower toward the tip
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    depth_from_edge = yy if from_top else (h - 1 - yy)
    edge_blur = rng.uniform(2.0, 6.0)

    def mask_for(length):
        dist = np.full((h, w), -np.inf)                          # signed distance inside (> 0) the union of capsules
        depth = np.zeros((h, w))                                 # inside depth relative to that finger's radius (0..1)
        for c, wd, off, tl in zip(centres, widths, offsets, tilts):
            L = max(length + off - wd / 2, 0)
            r = wd / 2 * (1 - taper * np.clip(depth_from_edge / max(L, 1.0), 0, 1))
            px = xx - (c + tl * depth_from_edge)
            seg = np.clip(depth_from_edge, -np.inf, L)          # capsule from the edge to depth L, rounded tip
            d = r - np.hypot(px, depth_from_edge - seg)
            better = d > dist
            depth = np.where(better, np.clip(d / r, 0, 1), depth)
            dist = np.maximum(dist, d)
        return 1 / (1 + np.exp(-dist / edge_blur)), depth
    lo, hi = 0.0, 2.0 * h
    for _ in range(28):
        mid = (lo + hi) / 2
        if mask_for(mid)[0].mean() < cover:
            lo = mid
        else:
            hi = mid
    mask, depth = mask_for((lo + hi) / 2)
    name = str(rng.choice(sorted(GLOVES)))
    base = np.array(GLOVES[name])
    ys, xs = D._box_slices(x.shape)
    light = float(np.clip(D.lum(x[ys, xs]).mean() * 1.4 + 0.15, 0.25, 1.0))   # lit by the same lamp as the eye
    illum = x[ys, xs].reshape(-1, 3).mean(0)
    illum = 0.4 + 0.6 * illum / max(illum.max(), 1e-6)                        # colour of that light (cobalt blue -> bluish)
    base = np.clip(base * illum / max((base * illum).max(), 1e-6) * base.max(), 0, 1)
    shade = 0.55 + 0.5 * np.sin(np.pi / 2 * depth)                     # round fingers: bright along the middle, darker sides
    wrinkle = 1 + 0.02 * ndimage.gaussian_filter(rng.standard_normal((h, w)), 6) / 0.03
    colour = np.clip(base[None, None, :] * light * (shade * np.clip(wrinkle, 0.95, 1.05))[..., None], 0, 1)
    soft = rng.uniform(2.0, 5.0)                                          # close to the lens: the whole object is defocused
    colour = np.stack([ndimage.gaussian_filter(colour[..., c], soft) for c in range(3)], 2)
    y = x * (1 - mask[..., None]) + colour * mask[..., None]
    return np.clip(y, 0, 1), float(mask.mean()), {"kind": "glove", "glove": name, "fingers": k, "from_top": from_top}


def noise_jpeg(x, rng, t):
    sigma = 0.01 + 0.07 * t
    lumn = rng.normal(0, sigma, x.shape[:2])[..., None]         # luminance + weaker chroma noise
    chroma = ndimage.gaussian_filter(rng.normal(0, 0.6 * sigma, x.shape), (1.5, 1.5, 0))
    y0 = np.clip(x + lumn + chroma, 0, 1)
    q = int(round(85 - 40 * t))                                # stronger noise ~ cheaper / harder-compressed phone image
    buf = io.BytesIO()
    Image.fromarray((y0 * 255 + 0.5).astype(np.uint8)).save(buf, format="JPEG", quality=q)
    y = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), dtype=np.float64) / 255
    return y, float(np.clip((y - x).std(), 0, 1)), {"kind": "noise_jpeg", "sigma": sigma, "quality": q}


FRESH = {"defocus_blur": defocus_wave, "motion_blur": motion_zoom, "overexposure": over_raw,
         "underexposure": under_linear8, "uneven_illumination": uneven_ringgap, "haze": haze_droplets,
         "glare": glare_window, "occlusion": occlusion_glove, "noise": noise_jpeg}

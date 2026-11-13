"""Eye-region locator for wide-view cobalt-blue photos (SLID-E). The fluorescein tear meniscus is a bright green line along both lid margins, outlining
the eye opening. Green pixels (G clearly above B and bright) are found, cleaned, and their extent (robust 2-98%
percentiles) gives the eye-opening box. Returns None when too few green pixels (caller falls back)."""
import numpy as np
from scipy import ndimage

import degradations as D


def green_mask(x):
    R, G, B = x[..., 0], x[..., 1], x[..., 2]
    m = (G > 0.35) & (G > B * 0.95) & (G > R * 1.15)
    m = ndimage.binary_opening(m, iterations=1)
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    keep = np.isin(lab, 1 + np.where(sizes >= max(20, 0.05 * sizes.max()))[0])
    return keep


def eye_box(x, min_pixels=150):
    """(x0, y0, x1, y1) as fractions of width / height, or None."""
    m = green_mask(x)
    if m.sum() < min_pixels:
        return None
    ys, xs = np.nonzero(m)
    h, w = m.shape
    x0, x1 = np.percentile(xs, [2, 98]) / w
    y0, y1 = np.percentile(ys, [2, 98]) / h
    if (x1 - x0) < 0.15 or (y1 - y0) < 0.05:
        return None
    return float(x0), float(y0), float(x1), float(y1)


def reflex_point(x):
    """Corneal reflex: a SMALL bright spot (white top-hat of brightness, 15 px) on a DARK surround (the iris):
    score = top-hat x (1 - mean brightness of the 61 px neighbourhood); outer 10% border ignored. (cx, cy, score)."""
    L = D.lum(x)
    th = L - ndimage.grey_opening(L, size=(15, 15))
    th = ndimage.gaussian_filter(th, 1.5) * (L > 0.7)
    surround = ndimage.uniform_filter(L, 61)
    score = th * np.clip(1 - surround, 0, 1)
    h, w = L.shape
    bh, bw = int(0.10 * h), int(0.10 * w)
    score[:bh] = 0; score[-bh:] = 0; score[:, :bw] = 0; score[:, -bw:] = 0
    cy, cx = np.unravel_index(np.argmax(score), score.shape)
    return float(cx), float(cy), float(score.max())


def reflex_box(x, half_w=0.20, half_h=0.15):
    """Eye box centred on the corneal reflex: +-half_w of the width, +-half_h of the height (fractions)."""
    h, w = x.shape[:2]
    cx, cy, s = reflex_point(x)
    if s < 0.03:
        return None
    fx, fy = cx / w, cy / h
    return (max(0.0, fx - half_w), max(0.0, fy - half_h), min(1.0, fx + half_w), min(1.0, fy + half_h))


def reflex_point_v3(x, n_cand=8):
    """v3: candidates = the n strongest small bright spots (top-hat, non-maximum suppressed, outer 10% ignored);
    pick the one whose iris-scale neighbourhood is DARKEST (disc of radius 7% of the shorter side, excluding the
    spot itself). The iris is the darkest round region; lid-margin / corner highlights sit on brighter skin."""
    L = D.lum(x)
    h, w = L.shape
    m = min(h, w)
    th = L - ndimage.grey_opening(L, size=(15, 15))
    th = ndimage.gaussian_filter(th, 1.5) * (L > 0.6)
    bh, bw = int(0.10 * h), int(0.10 * w)
    th[:bh] = 0; th[-bh:] = 0; th[:, :bw] = 0; th[:, -bw:] = 0
    peaks = (th == ndimage.maximum_filter(th, size=25)) & (th > 0.05)
    ys, xs = np.nonzero(peaks)
    if len(ys) == 0:
        return float(w / 2), float(h / 2), 0.0
    order = np.argsort(-th[ys, xs])[:n_cand]
    r = int(0.07 * m)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disc = (xx ** 2 + yy ** 2 <= r ** 2) & (xx ** 2 + yy ** 2 > (0.25 * r) ** 2)
    Lp = np.pad(L, r, mode="edge")
    best, best_val = order[0], 9.0
    for k in order:
        cy, cx = ys[k], xs[k]
        val = Lp[cy:cy + 2 * r + 1, cx:cx + 2 * r + 1][disc].mean()
        if val < best_val:
            best, best_val = k, val
    return float(xs[best]), float(ys[best]), float(th[ys[best], xs[best]])

"""Glare placement check on SLID-E: 16 random validation photos (seed 14) with the CENTRAL_BOX used
for glare placement / label (red), an example glare from our generator, and a trial iris detector (Hough circle on
Canny edges of the blurred brightness; radius 6-20% of the shorter side) in yellow.
Output: decisions/evidence/generalisation/10_slide_glare_eye_check.png"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from PIL import Image
from skimage import feature, transform, filters
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import degradations as D
QD = os.path.join(ROOT, "Data", "Quality Data")
q = pd.read_csv(os.path.join(QD, "labels.csv"))
src = q[(q.split == "val") & (q.kind == "clean") & (q.dataset == "SLID-E")].sample(16, random_state=14)

def iris(x):
    L = filters.gaussian(D.lum(x), 3)
    h, w = L.shape; m = min(h, w); s = 0.5
    Ls = transform.rescale(L, s, anti_aliasing=True)
    edges = feature.canny(Ls, sigma=2)
    radii = np.arange(int(0.06 * m * s), int(0.20 * m * s), 2)
    hs = transform.hough_circle(edges, radii)
    _, cx, cy, r = transform.hough_circle_peaks(hs, radii, total_num_peaks=1)
    return cx[0] / s, cy[0] / s, r[0] / s

fig, axes = plt.subplots(4, 4, figsize=(20, 15))
for ax, (k, row) in zip(axes.ravel(), enumerate(src.itertuples())):
    x = D.to_float(Image.open(os.path.join(QD, row.path)))
    y, lab, _ = D.glare(x, np.random.default_rng([10, k]), 60)
    ax.imshow(D.to_pil(y)); h, w = x.shape[:2]
    x0, y0, x1, y1 = D.CENTRAL_BOX
    ax.add_patch(plt.Rectangle((x0 * w, y0 * h), (x1 - x0) * w, (y1 - y0) * h, fill=False, ec="red", lw=2))
    cx, cy, r = iris(x)
    ax.add_patch(plt.Circle((cx, cy), r, fill=False, ec="yellow", lw=2))
    ax.set_title(f"{row.source_image_id[:22]}  glare label {lab:.2f}", fontsize=11, loc="left"); ax.axis("off")
fig.tight_layout(pad=0.4)
out = os.path.join(ROOT, "decisions", "evidence", "generalisation", "10_slide_glare_eye_check.png")
fig.savefig(out, dpi=55, facecolor="white"); print(out)

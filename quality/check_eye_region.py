"""Sheet: green-meniscus eye box (yellow) vs. CENTRAL_BOX (red) on 32 random SLID-E validation photos (seed 21).
Output: decisions/evidence/generalisation/11_slide_eye_box_check.png"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import degradations as D
from eye_region import eye_box, green_mask
QD = os.path.join(ROOT, "Data", "Quality Data")
q = pd.read_csv(os.path.join(QD, "labels.csv"))
src = q[(q.split == "val") & (q.kind == "clean") & (q.dataset == "SLID-E")].sample(32, random_state=21)
fig, axes = plt.subplots(6, 6, figsize=(30, 20)); n_none = 0
for ax, row in zip(axes.ravel(), src.itertuples()):
    x = D.to_float(Image.open(os.path.join(QD, row.path))); h, w = x.shape[:2]
    ax.imshow(D.to_pil(x))
    x0, y0, x1, y1 = D.CENTRAL_BOX
    ax.add_patch(plt.Rectangle((x0 * w, y0 * h), (x1 - x0) * w, (y1 - y0) * h, fill=False, ec="red", lw=1.5, ls="--"))
    b = eye_box(x)
    if b is None:
        n_none += 1; ax.set_title("NO BOX (fallback)", fontsize=12, loc="left", color="red")
    else:
        ax.add_patch(plt.Rectangle((b[0] * w, b[1] * h), (b[2] - b[0]) * w, (b[3] - b[1]) * h, fill=False, ec="yellow", lw=2.5))
        ax.set_title(f"green px {int(green_mask(x).sum)}", fontsize=12, loc="left")
    ax.axis("off")
for ax in axes.ravel()[len(src):]: ax.axis("off")
fig.tight_layout(pad=0.3)
out = os.path.join(ROOT, "decisions", "evidence", "generalisation", "11_slide_eye_box_check.png")
fig.savefig(out, dpi=40, facecolor="white"); print(out, "no box:", n_none)

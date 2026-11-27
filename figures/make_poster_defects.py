"""Poster figure: the 9 synthetic defects on photos from all 5 datasets, 4 columns x 3 rows.
3 clean photos + 9 defects (current quality/degradations.py), each panel captioned with defect and dataset.
One clean SLID-style training photo per dataset (random_state 14); strength 85 % of each defect's range; of 5 random
draws per defect the one with the largest measured label is shown (most visible kind).
Usage: python figures/make_poster_defects.py  -> figures/poster/fig_defects_12.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "quality"))
import degradations as D  # noqa: E402

QD = os.path.join(ROOT, "Data", "Quality Data")
NAMES = {"defocus_blur": "Defocus blur", "motion_blur": "Motion blur", "overexposure": "Overexposure",
         "underexposure": "Underexposure", "uneven_illumination": "Uneven illumination", "haze": "Haze",
         "glare": "Glare", "occlusion": "Occlusion (eyelid)", "noise": "Noise"}
CLEAN = ("SLID", "Nuclear Cataract", "Infectious Keratitis")
DEFECT_SRC = {"defocus_blur": "SLID-E", "motion_blur": "Nuclear Cataract", "overexposure": "Infectious Keratitis",
              "underexposure": "SUSTech-SYSU", "uneven_illumination": "SLID", "haze": "SLID-E",
              "glare": "Nuclear Cataract", "occlusion": "SLID", "noise": "SUSTech-SYSU"}


def main():
    q = pd.read_csv(os.path.join(QD, "labels.csv"))
    q = q[(q.split == "train") & (q.kind == "clean")]
    photo = {ds: D.to_float(Image.open(os.path.join(QD, q[q.dataset == ds].sample(1, random_state=14).iloc[0].path)))
             for ds in set(CLEAN) | set(DEFECT_SRC.values())}
    panels = [("Clean", ds, photo[ds]) for ds in CLEAN]
    for k, v in enumerate(D.VALUES):
        lo, hi = D.RANGES[v]
        x = photo[DEFECT_SRC[v]]
        y = max((D.FUNCS[v](x, np.random.default_rng([14, k, i]), lo + (1.0 if v == "uneven_illumination" else 0.85) * (hi - lo)) for i in range(8)),
                key=lambda r: r[1])[0]
        panels.append((NAMES[v], DEFECT_SRC[v], y))
    fig, axes = plt.subplots(3, 4, figsize=(16, 10.2))
    # column 1 = the 3 clean photos, columns 2-4 = the 9 defects (row by row)
    order = [0, 3, 4, 5, 1, 6, 7, 8, 2, 9, 10, 11]
    for ax, (title, ds, im) in zip(axes.ravel(), [panels[i] for i in order]):
        ax.imshow(np.asarray(D.to_pil(im).resize((640, 480))))
        ax.set_title(title, fontsize=21, loc="left", fontweight="bold")
        ax.text(0.02, 0.03, ds, transform=ax.transAxes, fontsize=14, color="white",
                bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=4))
        ax.axis("off")
    fig.tight_layout(pad=1.0, h_pad=1.2)
    out = os.path.join(HERE, "poster", "fig_defects_12.png")
    fig.savefig(out, dpi=300, facecolor="white")
    print(out)


if __name__ == "__main__":
    main()

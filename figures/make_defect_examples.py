"""Figure for Project Update 2: one clean slit-lamp photo and each of the 9 synthetic defects (short captions, large type: v2 for the 4-slide deck)
(quality/degradations.py v3), with the measured label of each defect.

Source: a clean SLID training photo chosen with seed 14; strength = 65% of the way through each defect's
range; rng default_rng([14, k]) per defect. Output: figures/defect_examples.png
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
MEASURE = {"defocus_blur": "sharpness lost", "motion_blur": "sharpness lost",
           "overexposure": "newly blown out", "underexposure": "light lost",
           "uneven_illumination": "1 − dark / bright", "haze": "contrast lost",
           "glare": "glare in centre", "occlusion": "area covered", "noise": "noise std"}


def main():
    q = pd.read_csv(os.path.join(QD, "labels.csv"))
    src = q[(q.split == "train") & (q.kind == "clean") & (q.dataset == "SLID")].sample(1, random_state=14).iloc[0]
    x = D.to_float(Image.open(os.path.join(QD, src.path)))
    panels = [("Clean photo", "all 9 = 0", x)]
    for k, v in enumerate(D.VALUES):
        lo, hi = D.RANGES[v]
        y, label, _ = D.FUNCS[v](x, np.random.default_rng([14, k]), lo + 0.65 * (hi - lo))
        panels.append((NAMES[v], f"{MEASURE[v]}: {label:.2f}", y))
    fig, axes = plt.subplots(2, 5, figsize=(20, 7.9))
    for ax, (title, sub, im) in zip(axes.ravel(), panels):
        ax.imshow(np.asarray(D.to_pil(im)))
        ax.set_title(f"{title}\n{sub}", fontsize=22, loc="left")
        ax.axis("off")
    fig.tight_layout(pad=0.6)
    out = os.path.join(HERE, "defect_examples.png")
    fig.savefig(out, dpi=110, facecolor="white")
    print(out, src.sample_id)


if __name__ == "__main__":
    main()

"""Review sheets for the FRESH held-out implementation (plan A3): each defect at 30 / 65 / 100% on a SLID,
SLID-E, Nuclear Cataract and SUSTech-SYSU validation photo (eye boxes from labels/eye_regions.csv).
Output: decisions/evidence/generalisation/17_fresh_<defect>.png"""
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
sys.path.insert(0, HERE)
import degradations as D  # noqa: E402
from degradations_fresh import FRESH  # noqa: E402

V2 = os.path.join(ROOT, "Data", "Quality Data v2")
q = pd.read_csv(os.path.join(V2, "labels.csv"))
eye = pd.read_csv(os.path.join(ROOT, "labels", "eye_regions.csv")).set_index("source_image_id")[["x0", "y0", "x1", "y1"]]
cl = q[(q.kind == "clean") & (q.split == "val")]
src = [cl[cl.dataset == d].sample(1, random_state=5).iloc[0] for d in ["SLID", "SLID-E", "Nuclear Cataract", "SUSTech-SYSU"]]
for name in sys.argv[1:] or list(FRESH):
    fig, axes = plt.subplots(len(src), 4, figsize=(17, 3.4 * len(src)))
    for j, r in enumerate(src):
        x = D.to_float(Image.open(os.path.join(V2, r.path)))
        D.set_eye_box(tuple(eye.loc[r.source_image_id]))
        axes[j, 0].imshow(D.to_pil(x))
        axes[j, 0].set_title(f"{name} | clean {r.dataset}", fontsize=13, loc="left")
        for c, t in enumerate([0.3, 0.65, 1.0]):
            y, lab, p = FRESH[name](x, np.random.default_rng([62, j, c]), t)
            axes[j, c + 1].imshow(D.to_pil(y))
            axes[j, c + 1].set_title(f"{p['kind']} {int(t * 100)}%: label {lab:.3f}", fontsize=13, loc="left")
    D.set_eye_box()
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout(pad=0.4)
    out = os.path.join(ROOT, "decisions", "evidence", "generalisation", f"17_fresh_{name}.png")
    fig.savefig(out, dpi=60, facecolor="white")
    plt.close(fig)
    print(out)

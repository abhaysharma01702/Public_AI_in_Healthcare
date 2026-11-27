"""Review sheet for the held-out implementation: each defect at 30 / 65 / 100% on SLID, SLID-E, Nuclear Cataract
validation photos (eye boxes from labels/eye_regions.csv). Output: decisions/evidence/generalisation/13_holdout.png"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import degradations as D
from degradations_holdout import HOLDOUT
QD = os.path.join(ROOT, "Data", "Quality Data")
er = pd.read_csv(os.path.join(ROOT, "labels", "eye_regions.csv"))
src = [er[(er.split == "val") & (er.dataset == d)].sample(1, random_state=5).iloc[0] for d in ["SLID", "SLID-E", "Nuclear Cataract"]]
names = sys.argv[1:] or list(HOLDOUT)
for v in names:
  fig, axes = plt.subplots(3, 4, figsize=(17, 9.6)); i = 0
  if True:
    for j, r in enumerate(src):
        x = D.to_float(Image.open(os.path.join(QD, r.path))); D.set_eye_box((r.x0, r.y0, r.x1, r.y1))
        row = i * 3 + j
        axes[row, 0].imshow(D.to_pil(x)); axes[row, 0].set_title(f"{v} | clean {r.dataset}", fontsize=13, loc="left")
        for c, t in enumerate([0.3, 0.65, 1.0]):
            y, lab, p = HOLDOUT[v](x, np.random.default_rng([55, row, c]), t)
            axes[row, c + 1].imshow(D.to_pil(y)); axes[row, c + 1].set_title(f"{p['kind']} {int(t*100)}%: label {lab:.3f}", fontsize=13, loc="left")
  D.set_eye_box()
  for ax in axes.ravel(): ax.axis("off")
  fig.tight_layout(pad=0.4)
  out = os.path.join(ROOT, "decisions", "evidence", "generalisation", f"13_holdout_{v}.png"); fig.savefig(out, dpi=60, facecolor="white"); plt.close(fig); print(out)

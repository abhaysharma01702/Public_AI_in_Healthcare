"""Review sheets for the training kinds of the 7 non-exposure defects: each kind (ours / alt / third) at
30 / 65 / 100% on a white-light (SLID), blue-light (SLID-E) and slit-beam (Nuclear Cataract) validation photo.
Output: decisions/evidence/generalisation/09_kinds_<defect>.png"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import degradations as D
from defect_kinds import KINDS
QD = os.path.join(ROOT, "Data", "Quality Data"); OUT = os.path.join(ROOT, "decisions", "evidence", "generalisation")
q = pd.read_csv(os.path.join(QD, "labels.csv"))
src = [q[(q.split == "val") & (q.kind == "clean") & (q.dataset == d)].sample(1, random_state=5).iloc[0]
       for d in ["SLID", "SLID-E", "Nuclear Cataract"]]
X = [(s.dataset, D.to_float(Image.open(os.path.join(QD, s.path)))) for s in src]
only = sys.argv[1:] or list(KINDS)
for name in only:
    kinds = list(KINDS[name])
    fig, axes = plt.subplots(len(kinds) * len(X), 4, figsize=(17, 3.15 * len(kinds) * len(X)))
    for i, kind in enumerate(kinds):
        for j, (ds, x) in enumerate(X):
            r = i * len(X) + j
            axes[r, 0].imshow(D.to_pil(x)); axes[r, 0].set_title(f"{kind} | clean {ds}", fontsize=13, loc="left")
            for c, t in enumerate([0.3, 0.65, 1.0]):
                y, lab, p = KINDS[name][kind](x, np.random.default_rng([53, r, c]), t)
                axes[r, c + 1].imshow(D.to_pil(y)); axes[r, c + 1].set_title(f"{kind} {int(t*100)}%: label {lab:.3f}", fontsize=13, loc="left")
    for ax in axes.ravel(): ax.axis("off")
    fig.tight_layout(pad=0.4)
    out = os.path.join(OUT, f"09_kinds_{name}.png"); fig.savefig(out, dpi=45, facecolor="white"); plt.close(fig); print(out)

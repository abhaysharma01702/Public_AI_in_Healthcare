"""Review sheet for the generalisation test: our defect vs. the alternative implementation, side by side,
on validation photos (never test). Strength = middle of each range. Output: decisions/evidence/generalisation/01_ours_vs_alt_<id>.png"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import degradations as D, degradations_alt as A
QD = os.path.join(ROOT, "Data", "Quality Data")
q = pd.read_csv(os.path.join(QD, "labels.csv"))
for ds in ["SLID", "SLID-E"]:
    src = q[(q.split == "val") & (q.kind == "clean") & (q.dataset == ds)].sample(1, random_state=14).iloc[0]
    x = D.to_float(Image.open(os.path.join(QD, src.path)))
    fig, axes = plt.subplots(2, 10, figsize=(30, 5.6))
    for row, (name, R, F) in enumerate([("ours", D.RANGES, D.FUNCS), ("alternative", A.RANGES_ALT, A.FUNCS)]):
        axes[row, 0].imshow(np.asarray(D.to_pil(x))); axes[row, 0].set_title(f"clean ({name} row)", fontsize=13, loc="left")
        for k, v in enumerate(D.VALUES):
            lo, hi = R[v]
            y, lab, p = F[v](x, np.random.default_rng([51, k]), (lo + hi) / 2)
            kind = p.get("kernel") or p.get("path") or p.get("field") or p.get("kind") or ""
            axes[row, k + 1].imshow(np.asarray(D.to_pil(y)))
            axes[row, k + 1].set_title(f"{name}: {v}\n{kind} label {lab:.2f}", fontsize=12, loc="left")
    for ax in axes.ravel(): ax.axis("off")
    fig.tight_layout(pad=0.4)
    out = os.path.join(ROOT, "decisions", "evidence", "generalisation", f"01_ours_vs_alt_{src.source_image_id}.png")
    fig.savefig(out, dpi=72, facecolor="white"); print(out)

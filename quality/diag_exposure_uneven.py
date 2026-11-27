"""Diagnostic sheets: overexposure and uneven illumination, ours vs. other code, 4 strengths,
one validation photo per dataset. Also prints an alternative 'near-white' label for overexposure:
share of pixels with ALL channels >= 250/255 (newly), next to the current any-channel-at-255 label.
Output: decisions/evidence/generalisation/03_diag_<defect>_<impl>.png"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import degradations as D, degradations_alt as A
QD = os.path.join(ROOT, "Data", "Quality Data"); OUT = os.path.join(ROOT, "decisions", "evidence", "generalisation")
q = pd.read_csv(os.path.join(QD, "labels.csv"))
src = [q[(q.split == "val") & (q.kind == "clean") & (q.dataset == d)].sample(1, random_state=3).iloc[0]
       for d in ["SLID", "SLID-E", "Infectious Keratitis", "Nuclear Cataract", "SUSTech-SYSU"]]
def white(a): return float((a.min(axis=2) >= 250 / 255).mean())
for v in ["overexposure", "uneven_illumination"]:
    for impl, R, F in [("ours", D.RANGES, D.FUNCS), ("alt", A.RANGES_ALT, A.FUNCS)]:
        lo, hi = R[v]; fr = [0.1, 0.4, 0.7, 1.0]
        fig, axes = plt.subplots(len(src), 5, figsize=(22, 3.4 * len(src)))
        for r, s in enumerate(src):
            x = D.to_float(Image.open(os.path.join(QD, s.path)))
            axes[r, 0].imshow(D.to_pil(x)); axes[r, 0].set_title(f"clean {s.dataset}", fontsize=12, loc="left")
            for c, f in enumerate(fr):
                y, lab, p = F[v](x, np.random.default_rng([7, r, c]), lo + f * (hi - lo))
                extra = f"  x{p['gain']:.1f}" if v == "overexposure" else f"  {p.get('field', '')}"
                axes[r, c + 1].imshow(D.to_pil(y)); axes[r, c + 1].set_title(f"{impl} {int(f*100)}% of range: label {lab:.2f}{extra}", fontsize=12, loc="left")
        for ax in axes.ravel(): ax.axis("off")
        fig.tight_layout(pad=0.4); out = os.path.join(OUT, f"07_diag_v4c_{v}_{impl}.png"); fig.savefig(out, dpi=50, facecolor="white"); plt.close(fig); print(out)

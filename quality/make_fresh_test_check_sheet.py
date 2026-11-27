"""A4 check: 2 random gen_fresh images per defect next to their clean source. Output: decisions/evidence/generalisation/18_fresh_test_check_{1,2}.png"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

QD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data", "Quality Data v2")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "decisions", "evidence", "generalisation")
d = pd.read_csv(os.path.join(QD, "gen_fresh.csv"))
clean = d[d.impl == "clean"].set_index("source_image_id").path
f = d[d.impl == "fresh"]
defects = sys.argv[1:] or sorted(f.defects.unique())
for half in (defects[:5], defects[5:]):
    if not half:
        continue
    fig, axes = plt.subplots(len(half), 4, figsize=(17, 3.4 * len(half)))
    for i, v in enumerate(half):
        s = f[f.defects == v].sample(2, random_state=7)
        for j, r in enumerate(s.itertuples()):
            axes[i, 2 * j].imshow(Image.open(os.path.join(QD, clean[r.source_image_id])))
            axes[i, 2 * j].set_title(f"clean | {r.dataset}", fontsize=12, loc="left")
            p = json.loads(r.params)[v]
            axes[i, 2 * j + 1].imshow(Image.open(os.path.join(QD, r.path)))
            axes[i, 2 * j + 1].set_title(f"{v} t={p['t']:.2f}: label {getattr(r, v):.3f}", fontsize=12, loc="left")
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout(pad=0.4)
    out = os.path.join(OUT, f"18_fresh_test_check_{1 + (half[0] != defects[0])}.png")
    fig.savefig(out, dpi=60, facecolor="white")
    plt.close(fig)
    print(out)

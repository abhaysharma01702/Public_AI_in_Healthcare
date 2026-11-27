"""A2 gate: random sample of 24 extension (d4) training images, 8 per new kind, each next to its clean source,
with the defects applied and their labels. One sheet per new kind.
Output: decisions/evidence/generalisation/15_extension_sample_<kind>.png"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
V2 = os.path.join(ROOT, "Data", "Quality Data v2")
OUT = os.path.join(ROOT, "decisions", "evidence", "generalisation")
q = pd.read_csv(os.path.join(V2, "labels.csv")).set_index("sample_id")
ext = pd.read_csv(os.path.join(V2, "labels_ext_v3.csv"))
for kind, g in ext[ext.split == "train"].groupby("new_kind"):
    s = g.sample(8, random_state=60)
    fig, axes = plt.subplots(4, 4, figsize=(18, 17))
    for i, r in enumerate(s.itertuples()):
        a, b = axes[i // 2, 2 * (i % 2)], axes[i // 2, 2 * (i % 2) + 1]
        a.imshow(Image.open(os.path.join(V2, q.loc[f"{r.source_image_id}__clean", "path"])))
        a.set_title(f"clean | {r.dataset}", fontsize=12, loc="left")
        b.imshow(Image.open(os.path.join(V2, r.path)))
        p = json.loads(r.params)
        b.set_title("\n".join(f"{n} ({p[n]['impl']}): {getattr(r, n):.2f}" for n in r.defects.split(";")),
                    fontsize=11, loc="left", va="bottom")
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"Extension d4, new kind '{kind}': 8 random training images (left clean, right d4)", fontsize=14)
    fig.tight_layout(pad=0.5, h_pad=2.5, rect=(0, 0, 1, 0.97))
    out = os.path.join(OUT, f"15_extension_sample_{kind}.png")
    fig.savefig(out, dpi=60, facecolor="white")
    plt.close(fig)
    print(out)

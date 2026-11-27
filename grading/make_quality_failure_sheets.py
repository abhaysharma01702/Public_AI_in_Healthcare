"""Sheets of the grading test images on which the quality module (D3 Swin-T) measures haze, occlusion and glare badly
. Per defect and strength the worst case: the largest underestimation (true label minus
the matching output) for haze and occlusion; for glare (never detected) the most visible one (largest true label).
Each column: the clean original (top) and the degraded test image (bottom) with strength, true and predicted value;
glare sheets add a zoomed crop around the glare (found as the changed pixels, boxed in red).
Output: results/grading/<task>/quality_check/failures_<defect>.png
Usage: python grading/make_quality_failure_sheets.py [--task sustech_sysu]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFECTS = ["haze", "occlusion", "glare"]


def changed_box(x, y, pad=40):
    """Bounding box (x0, y0, x1, y1) of the pixels that differ clearly between clean x and degraded y, padded."""
    diff = np.abs(y.astype(int) - x.astype(int)).sum(2)
    ys, xs = np.nonzero(diff > 60)
    if not len(xs):
        return None
    h, w = diff.shape
    return max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad, w), min(ys.max() + pad, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="sustech_sysu")
    a = ap.parse_args()
    data = os.path.join(ROOT, "Data", "Grading Data", a.task)
    out = os.path.join(ROOT, "results", "grading", a.task, "quality_check")
    os.makedirs(out, exist_ok=True)
    d = pd.read_csv(os.path.join(data, "labels.csv")).merge(pd.read_csv(os.path.join(data, "quality_d3.csv")),
                                                            on="sample_id", validate="1:1")
    clean = d[d.kind == "clean"].set_index("image_id").path
    for v in DEFECTS:
        t = d[(d.kind == "test_fresh") & (d.defect == v)].copy()
        t["under"] = t[v] - t[f"q_{v}"]
        key = v if v == "glare" else "under"
        pick = t.sort_values(key, ascending=False).groupby("strength", sort=True).head(1).sort_values("strength")
        rows = 3 if v == "glare" else 2
        fig, ax = plt.subplots(rows, len(pick), figsize=(4.2 * len(pick), 3.5 * rows))
        for j, r in enumerate(pick.itertuples()):
            x = np.array(Image.open(os.path.join(data, clean[r.image_id])).convert("RGB"))
            y = np.array(Image.open(os.path.join(data, r.path)).convert("RGB"))
            ax[0, j].imshow(x)
            ax[0, j].set_title(f"{r.image_id}: clean original", fontsize=11)
            ax[1, j].imshow(y)
            ax[1, j].set_title(f"t = {r.strength:g}: true {getattr(r, v):.2f}, predicted {getattr(r, 'q_' + v):.3f}",
                               fontsize=12, color="#b00020")
            if v == "glare":
                b = changed_box(x, y)
                if b:
                    ax[1, j].add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="red", lw=2))
                    ax[2, j].imshow(y[b[1]:b[3], b[0]:b[2]])
                    ax[2, j].set_title("zoom on the glare", fontsize=11)
        for a_ in ax.flat:
            a_.axis("off")
        fig.suptitle(f"{v}: quality module (Swin-T) on the grading test set, worst case per strength", fontsize=15)
        fig.tight_layout(h_pad=2.0)
        path = os.path.join(out, f"failures_{v}.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(path)
        print(pick[["sample_id", "strength", v, f"q_{v}"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

"""Sheets for the round-1 grading diagnosis. Do the type labels look right, and
are micro / macro punctate dots still visible at the grading resolution?
1. types_overview.png: per type (rows 0-4) 3 val + 3 test photos from different eyes (groups), cornea region of the
   768 x 512 grading image (what the model gets, before the squeeze to 672 x 512).
2. resolution_zoom.png: per type 1-4 one test photo, the same 400 x 400 px patch at the cornea centre in the original
   (2592 x 1728, 1:1), in the 768 x 512 grading image and in the 672 x 512 training input (both enlarged to the same size).
3. test_worst_errors.png: the 8 clean test photos (different eyes) with the largest mean |predicted - true| type over
   the 12 round-1 runs, with the true type and the 12 predictions.
4. same_eye_types.png: the 3 test groups (same-eye photos) with the widest spread of type labels, all their photos
   in file order with type and category: are photos of one eye taken one after another labelled alike?
Cornea regions from the dataset's cornea masks (Data/Raw/SUSTech-SYSU/corneaLabels, 0 = cornea).
Output: results/grading/<task>/diagnosis/
Usage: python grading/make_type_sheets.py [--task sustech_sysu]
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TYPES = {0: "no ulcer", 1: "micro punctate", 2: "macro punctate", 3: "coalescent macro punctate", 4: "patch >= 1 mm"}
RAW = os.path.join(ROOT, "Data", "Raw", "SUSTech-SYSU")


def cornea_box(image_id, pad=0.05):
    """Cornea bounding box (x0, y0, x1, y1) in original pixels, padded by 5 % of the image size."""
    with Image.open(os.path.join(RAW, "corneaLabels", f"{int(image_id.split('_')[1])}.png")) as im:
        m = np.asarray(im) == 0
    ys, xs = np.nonzero(m)
    h, w = m.shape
    p = int(pad * w)
    return max(xs.min() - p, 0), max(ys.min() - p, 0), min(xs.max() + p, w), min(ys.max() + p, h)


def grading_image(data, labels, image_id):
    path = labels[(labels.image_id == image_id) & (labels.kind == "clean")].path.item()
    return Image.open(os.path.join(data, path)).convert("RGB")


def crop_scaled(im, box, orig_w=2592, orig_h=1728):
    sx, sy = im.size[0] / orig_w, im.size[1] / orig_h
    return im.crop((int(box[0] * sx), int(box[1] * sy), int(box[2] * sx), int(box[3] * sy)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="sustech_sysu")
    a = ap.parse_args()
    data = os.path.join(ROOT, "Data", "Grading Data", a.task)
    out = os.path.join(ROOT, "results", "grading", a.task, "diagnosis")
    os.makedirs(out, exist_ok=True)
    lab = pd.read_csv(os.path.join(data, "labels.csv"))
    clean = lab[lab.kind == "clean"]
    rng = np.random.default_rng(14)

    # 1. overview: 3 val + 3 test photos per type, one per eye
    fig, ax = plt.subplots(5, 6, figsize=(6 * 3.0, 5 * 3.1))
    for t in range(5):
        picks = []
        for s in ["val", "test"]:
            g = clean[(clean.split == s) & (clean.grade == t)].drop_duplicates("group_id")
            picks += [(s, i) for i in g.image_id.to_numpy()[rng.permutation(len(g))[:3]]]
        for j in range(6):
            ax[t, j].axis("off")
            if j < len(picks):
                s, i = picks[j]
                ax[t, j].imshow(crop_scaled(grading_image(data, lab, i), cornea_box(i)))
                ax[t, j].set_title(f"{i} ({s})", fontsize=10)
        ax[t, 0].text(-0.08, 0.5, f"type {t}\n{TYPES[t]}", transform=ax[t, 0].transAxes, ha="right", va="center",
                      fontsize=12)
    fig.suptitle("SUSTech-SYSU type grade: cornea region of the 768 x 512 grading images (3 val + 3 test eyes per type)",
                 fontsize=14)
    fig.tight_layout(rect=(0.07, 0, 1, 0.97))
    fig.savefig(os.path.join(out, "types_overview.png"), dpi=100)
    plt.close(fig)

    # 2. resolution: the same 400 x 400 patch at three resolutions
    fig, ax = plt.subplots(4, 3, figsize=(3 * 4.2, 4 * 4.4))
    for r, t in enumerate(range(1, 5)):
        i = clean[(clean.split == "test") & (clean.grade == t)].image_id.iloc[0]
        x0, y0, x1, y1 = cornea_box(i, pad=0)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        box = (cx - 200, cy - 200, cx + 200, cy + 200)
        raw = Image.open(os.path.join(RAW, "rawImages", f"{int(i.split('_')[1])}.jpg")).convert("RGB")
        g = grading_image(data, lab, i)
        tr = g.resize((672, 512), Image.LANCZOS)
        for c, (im, name) in enumerate([(raw, "original 2592 x 1728 (1:1)"), (g, "grading image 768 x 512"),
                                        (tr, "training input 672 x 512")]):
            patch = crop_scaled(im, box)
            ax[r, c].imshow(patch.resize((400, 400), Image.NEAREST))
            ax[r, c].set_title(f"{i}, type {t}: {name}\n({patch.size[0]} x {patch.size[1]} px shown at 400 x 400)",
                               fontsize=10)
            ax[r, c].axis("off")
    fig.suptitle("Same 400 x 400 px cornea patch (original pixels) at the three resolutions", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(out, "resolution_zoom.png"), dpi=100)
    plt.close(fig)

    # 3. worst clean-test errors over the 12 round-1 runs
    runs = sorted(glob.glob(os.path.join(ROOT, "runs", "grading", a.task, "*_seed*")))
    P = pd.concat([pd.read_csv(os.path.join(d, "predictions_test.csv")).query("kind == 'clean'")
                   .assign(run=os.path.basename(d)) for d in runs])
    err = P.assign(abs_err=(P.pred - P.grade).abs()).groupby(["image_id", "group_id", "grade"])\
           .agg(mean_abs_err=("abs_err", "mean"), preds=("pred", lambda s: "".join(map(str, s)))).reset_index()
    worst = err.sort_values("mean_abs_err", ascending=False).drop_duplicates("group_id").head(8)
    fig, ax = plt.subplots(2, 4, figsize=(4 * 3.6, 2 * 3.9))
    for k, r in enumerate(worst.itertuples()):
        ax.flat[k].imshow(crop_scaled(grading_image(data, lab, r.image_id), cornea_box(r.image_id)))
        ax.flat[k].set_title(f"{r.image_id}: true type {r.grade}\n12 runs: {r.preds} (mean |err| {r.mean_abs_err:.1f})",
                             fontsize=10)
        ax.flat[k].axis("off")
    fig.suptitle("Clean test photos with the largest round-1 errors (one per eye; runs A, B, C_oracle, C x seeds 14-16)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(out, "test_worst_errors.png"), dpi=100)
    plt.close(fig)
    print(worst.to_string(index=False))

    # 4. same-eye photos with different type labels
    m = pd.read_csv(os.path.join(ROOT, "labels", "sustech_master.csv"))
    t = m[(m.status == "usable") & (m.split == "test")]
    spread = t.groupby("group_id").type.agg(lambda s: (s.max() - s.min(), s.nunique(), len(s)))
    top = sorted(spread.index, key=lambda g: spread[g], reverse=True)[:3]
    ncol = int(t[t.group_id.isin(top)].group_id.value_counts().max())
    fig, ax = plt.subplots(len(top), ncol, figsize=(ncol * 2.9, len(top) * 2.6))
    for r, gid in enumerate(top):
        g = t[t.group_id == gid].sort_values("image_id")
        for c in range(ncol):
            ax[r, c].axis("off")
            if c < len(g):
                row = g.iloc[c]
                ax[r, c].imshow(crop_scaled(grading_image(data, lab, row.image_id), cornea_box(row.image_id)))
                ax[r, c].set_title(f"{row.image_id}: type {row.type}\n{row.category_name}", fontsize=9)
        ax[r, 0].text(-0.06, 0.5, gid, transform=ax[r, 0].transAxes, ha="right", va="center", fontsize=10)
    fig.suptitle("Same-eye test groups with the widest spread of type labels (photos in file order)", fontsize=13)
    fig.tight_layout(rect=(0.04, 0, 1, 0.94))
    fig.savefig(os.path.join(out, "same_eye_types.png"), dpi=100)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()

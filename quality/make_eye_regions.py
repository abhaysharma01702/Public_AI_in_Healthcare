"""Eye region per clean source. SLID-E (wide view): box around the corneal reflex when two detectors
(eye_region.reflex_point v2 and reflex_point_v3) agree within 5% and the reflex is not within 15% of the edge; else CENTRAL_BOX; all other datasets: CENTRAL_BOX
(close-ups). Manual corrections (after visual review of the sheets) are read from labels/eye_regions_manual.csv
(source_image_id, method) and override the automatic choice.
Output: labels/eye_regions.csv; review sheets figures/eye_regions/slide_<NN>.png (64 per sheet, numbered).
Usage: python make_eye_regions.py [--sheets]"""
import os, sys, zlib
from multiprocessing import Pool
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "data_prep"))
import degradations as D
from eye_region import reflex_point, reflex_point_v3
from run_log import RunLog
QD = os.path.join(ROOT, "Data", "Quality Data")
OUT_CSV = os.path.join(ROOT, "labels", "eye_regions.csv")
MANUAL = os.path.join(ROOT, "labels", "eye_regions_manual.csv")
SHEETS = os.path.join(ROOT, "figures", "eye_regions")
EDGE = 0.15; HALF_W, HALF_H = 0.20, 0.15
# tweak: SLID-E fallback = reflex-sized box at the typical cornea position (median reflex 0.59, 0.51 over the
# 1,702 agreed reflex boxes; 80% within x 0.53-0.66, y 0.42-0.59) instead of the large CENTRAL_BOX
SLIDE_FALLBACK = (0.39, 0.36, 0.79, 0.66)


def work(row):
    x = D.to_float(Image.open(os.path.join(QD, row["path"])))
    h, w = x.shape[:2]
    cx, cy, s = reflex_point_v3(x)
    cx2, cy2, s2 = reflex_point(x)
    fx, fy = cx / w, cy / h
    agree = abs(cx - cx2) / w < 0.05 and abs(cy - cy2) / h < 0.05      # two detectors pick the same spot
    if agree and EDGE <= fx <= 1 - EDGE and EDGE <= fy <= 1 - EDGE:
        box, method = (max(0, fx - HALF_W), max(0, fy - HALF_H), min(1, fx + HALF_W), min(1, fy + HALF_H)), "reflex"
    else:
        box, method = SLIDE_FALLBACK, ("fallback_disagree" if not agree else "fallback_edge")
    return dict(source_image_id=row["source_image_id"], dataset=row["dataset"], split=row["split"], path=row["path"],
                method=method, x0=box[0], y0=box[1], x1=box[2], y1=box[3], reflex_x=fx, reflex_y=fy, reflex_score=s)


def main():
    q = pd.read_csv(os.path.join(QD, "labels.csv"))
    clean = q[q.kind == "clean"]
    with RunLog("quality/make_eye_regions.py", notes="eye regions ") as log:
        se = clean[clean.dataset == "SLID-E"]
        with Pool(max(1, os.cpu_count() - 1)) as p:
            rows = p.map(work, se.to_dict("records"), chunksize=8)
        other = clean[clean.dataset != "SLID-E"]
        x0, y0, x1, y1 = D.CENTRAL_BOX
        rows += [dict(source_image_id=r.source_image_id, dataset=r.dataset, split=r.split, path=r.path, method="central",
                      x0=x0, y0=y0, x1=x1, y1=y1, reflex_x=np.nan, reflex_y=np.nan, reflex_score=np.nan)
                 for r in other.itertuples()]
        df = pd.DataFrame(rows).sort_values(["dataset", "source_image_id"]).reset_index(drop=True)
        if os.path.exists(MANUAL):
            man = pd.read_csv(MANUAL).set_index("source_image_id")
            for sid, r in man.iterrows():
                i = df.index[df.source_image_id == sid][0]
                if r.method == "manual_central":
                    df.loc[i, ["x0", "y0", "x1", "y1"]] = SLIDE_FALLBACK
                df.loc[i, "method"] = r.method
        df.to_csv(OUT_CSV, index=False, float_format="%.5g")
        print(df[df.dataset == "SLID-E"].method.value_counts().to_dict())
        log.add(images_in=len(se), slide_methods=df[df.dataset == "SLID-E"].method.value_counts().to_dict())
    if "--sheets" in sys.argv:
        os.makedirs(SHEETS, exist_ok=True)
        se = df[df.dataset == "SLID-E"].reset_index(drop=True)
        for s0 in range(0, len(se), 64):
            chunk = se.iloc[s0:s0 + 64]
            fig, axes = plt.subplots(8, 8, figsize=(32, 24))
            for ax, (i, r) in zip(axes.ravel(), chunk.iterrows()):
                im = Image.open(os.path.join(QD, r.path)); w, h = im.size
                ax.imshow(im.resize((w // 3, h // 3)))
                col = "yellow" if r.method == "reflex" else "red"
                ax.add_patch(plt.Rectangle((r.x0 * w / 3, r.y0 * h / 3), (r.x1 - r.x0) * w / 3, (r.y1 - r.y0) * h / 3, fill=False, ec=col, lw=2.5))
                ax.set_title(f"#{i}", fontsize=16, loc="left", color="black" if r.method == "reflex" else "red"); ax.axis("off")
            for ax in axes.ravel()[len(chunk):]: ax.axis("off")
            fig.tight_layout(pad=0.3)
            fig.savefig(os.path.join(SHEETS, f"slide_{s0 // 64:02d}.png"), dpi=32, facecolor="white"); plt.close(fig)
        print("sheets:", (len(se) + 63) // 64)


if __name__ == "__main__":
    main()

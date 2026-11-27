"""Example grids for degradations v3, for the user's review of the strength ranges.

Same five example images as v1/v2 (one clean train image per dataset, seed 14). For each of the 9
values: columns = strength at 0%, 25%, 50%, 75% and 100% of its range (RANGES in degradations.py);
under every tile the strength and the MEASURED label the model would be trained on.
Plus 09_mixed_samples.png: 15 training samples drawn by the mixing policy, with their label vectors.
Tiles at 224 px height (network input size).

Output: decisions/evidence/degradations/v3/
Usage:  python -I make_degradation_grids_v3.py
"""
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "data_prep"))
import degradations as D  # noqa: E402
from run_log import RunLog  # noqa: E402

OUT = os.path.join(ROOT, "decisions", "evidence", "degradations", "v3")
DATASETS = ["Nuclear Cataract", "SUSTech-SYSU", "SLID-E", "Infectious Keratitis", "SLID"]
FRACS = [0.0, 0.25, 0.5, 0.75, 1.0]
UNITS = {"defocus_blur": "sigma {:.1f}px", "motion_blur": "length {:.0f}px", "overexposure": "u {:.2f}",
         "underexposure": "light removed {:.2f}", "uneven_illumination": "dark/bright {:.2f}",
         "haze": "veil {:.2f}", "glare": "radius {:.0f}px", "occlusion": "depth {:.0%}",
         "noise": "sigma {:.3f}"}
TH = 224
try:
    FONT = ImageFont.truetype("arial.ttf", 13)
    BIG = ImageFont.truetype("arial.ttf", 15)
except OSError:
    FONT = BIG = ImageFont.load_default()


def tile(a):
    im = D.to_pil(a)
    return im.resize((round(im.width * TH / im.height), TH), Image.LANCZOS)


def strength_at(name, f):
    lo, hi = D.RANGES[name]
    return hi - (hi - lo) * f if name == "uneven_illumination" else lo + (hi - lo) * f  # lower ratio = stronger


def sheet(rows, captions, col_titles, row_titles, title, cap_h=34):
    tw = max(t.width for r in rows for t in r)
    left, top = 150, 52
    W = left + len(col_titles) * (tw + 8)
    H = top + len(rows) * (TH + cap_h + 6)
    s = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(s)
    d.text((8, 6), title, fill="black", font=BIG)
    for j, c in enumerate(col_titles):
        d.text((left + j * (tw + 8), 30), c, fill="black", font=FONT)
    for i, (r, caps, rt) in enumerate(zip(rows, captions, row_titles)):
        y = top + i * (TH + cap_h + 6)
        d.text((8, y + TH // 2 - 8), rt, fill="black", font=FONT)
        for j, (t, cap) in enumerate(zip(r, caps)):
            x = left + j * (tw + 8)
            s.paste(t, (x, y))
            d.text((x, y + TH + 2), cap, fill="black", font=FONT)
    return s


def main():
    os.makedirs(OUT, exist_ok=True)
    with RunLog("make_degradation_grids_v3.py", notes="v3 grids: ranges at 0/25/50/75/100 %, measured labels") as log:
        pool = pd.read_csv(os.path.join(ROOT, "labels", "quality_clean_pool.csv"))
        ex = pd.concat([pool[(pool.dataset == ds) & pool.clean & (pool.split == "train")].sample(1, random_state=14)
                        for ds in DATASETS])
        ex[["dataset", "image_id", "resized_path"]].to_csv(os.path.join(OUT, "examples.csv"), index=False)
        imgs = [D.to_float(Image.open(os.path.join(ROOT, p))) for p in ex.resized_path]
        labels = [f"{ds}\n{iid[-14:]}" for ds, iid in zip(ex.dataset, ex.image_id)]
        records = []
        for k, name in enumerate(D.VALUES):
            rows, caps = [], []
            for i, a in enumerate(imgs):
                r, c = [], []
                for f in FRACS:
                    st = strength_at(name, f)
                    y, lab, _ = D.FUNCS[name](a, np.random.default_rng([14, i]), st)
                    r.append(tile(y))
                    c.append(f"{UNITS[name].format(st)}\nlabel {lab:.3f}")
                    records.append(dict(value=name, dataset=ex.dataset.iloc[i], frac=f, strength=st, label=lab))
                rows.append(r)
                caps.append(c)
            cols = [f"{int(f * 100)}% of range" for f in FRACS]
            sheet(rows, caps, cols, labels, f"{name}  (v3; caption: strength and measured label)").save(
                os.path.join(OUT, f"{k + 1:02d}_{name}.png"))
        pd.DataFrame(records).to_csv(os.path.join(OUT, "labels_by_strength.csv"), index=False)
        # mixed training samples
        rng = np.random.default_rng(14)
        rows, caps = [], []
        for i in range(3):
            r, c = [], []
            for j in range(5):
                a = imgs[(i * 5 + j) % len(imgs)]
                y, lab, applied = D.sample(a, rng)
                r.append(tile(y))
                nz = [f"{n[:10]} {v:.2f}" for n, v in lab.items() if v > 0]
                c.append("\n".join(nz) if nz else "clean (all 0)")
            rows.append(r)
            caps.append(c)
        sheet(rows, caps, [f"sample {j + 1}" for j in range(5)], ["", "", ""],
              "15 training samples from the mixing policy, with their non-zero labels", cap_h=60).save(
            os.path.join(OUT, "10_mixed_samples.png"))
        log.add(images_in=len(imgs), grids=len(D.VALUES) + 1)


if __name__ == "__main__":
    main()

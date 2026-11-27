"""Example grids of the synthetic degradations for visual review.

One clean TRAIN image per quality-pool dataset (labels/quality_clean_pool.csv, sampled with
seed 14). For each degradation: rows = datasets, columns = severity 0, 0.2, 0.4, 0.6, 0.8, 1.0.
Within a row every tile uses the same random seed, so positions / directions stay the same and
only the severity changes. Tiles are shown at 224 px height, the size the network will see.

Output: decisions/evidence/degradations/v2/<NN>_<name>.png, 08_nuisance.png, 09_zoom_all_images.png,
examples.csv. (v1 grids: decisions/evidence/degradations/v1/, made with degradations_v1.py.)
Framing is shown at a fixed zoom z = 0.55 in every column (severity 0 = centred), so only the
position changes.
Usage:  python -I make_degradation_grids.py
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

VERSION = "v2"
OUT = os.path.join(ROOT, "decisions", "evidence", "degradations", VERSION)
DATASETS = ["Nuclear Cataract", "SUSTech-SYSU", "SLID-E", "Infectious Keratitis", "SLID"]
SEVERITIES = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
GRIDS = ["defocus_blur", "motion_blur", "overexposure", "underexposure", "glare", "occlusion", "framing"]
TH = 224

try:
    FONT = ImageFont.truetype("arial.ttf", 14)
except OSError:
    FONT = ImageFont.load_default()


def tile(a):
    im = D.to_pil(a)
    return im.resize((round(im.width * TH / im.height), TH), Image.LANCZOS)


def grid(rows, col_titles, row_titles, title):
    tw = max(t.width for r in rows for t in r)
    left, top = 150, 52
    sheet = Image.new("RGB", (left + len(col_titles) * (tw + 6), top + len(rows) * (TH + 6)), "white")
    d = ImageDraw.Draw(sheet)
    d.text((8, 6), title, fill="black", font=FONT)
    for j, c in enumerate(col_titles):
        d.text((left + j * (tw + 6), 30), c, fill="black", font=FONT)
    for i, (r, rt) in enumerate(zip(rows, row_titles)):
        d.text((8, top + i * (TH + 6) + TH // 2 - 8), rt, fill="black", font=FONT)
        for j, t in enumerate(r):
            sheet.paste(t, (left + j * (tw + 6), top + i * (TH + 6)))
    return sheet


def main():
    os.makedirs(OUT, exist_ok=True)
    with RunLog("make_degradation_grids.py", notes=f"example grids, degradations {VERSION}") as log:
        pool = pd.read_csv(os.path.join(ROOT, "labels", "quality_clean_pool.csv"))
        ex = pd.concat([pool[(pool.dataset == ds) & pool.clean & (pool.split == "train")].sample(1, random_state=14)
                        for ds in DATASETS])
        ex[["dataset", "image_id", "resized_path"]].to_csv(os.path.join(OUT, "examples.csv"), index=False)
        imgs = [D.to_float(Image.open(os.path.join(ROOT, p))) for p in ex.resized_path]
        labels = [f"{ds}\n{iid[-14:]}" for ds, iid in zip(ex.dataset, ex.image_id)]
        for k, name in enumerate(GRIDS, 1):
            rows = []
            for i, a in enumerate(imgs):
                rows.append([tile(D.FUNCS[name](a, s, np.random.default_rng([14, i]))) for s in SEVERITIES])
            g = grid(rows, [f"severity {s}" for s in SEVERITIES], labels, f"{name}  (tiles at {TH} px height = network input size)")
            g.save(os.path.join(OUT, f"{k:02d}_{name}.png"))
        # nuisance: noise and JPEG
        cols = ["clean", "noise 0.01", "noise 0.02 (max)", "JPEG q60", "JPEG q30 (min)"]
        rows = []
        for i, a in enumerate(imgs):
            rng = np.random.default_rng([14, i])
            rows.append([tile(a), tile(D.gaussian_noise(a, 0.01, rng)), tile(D.gaussian_noise(a, 0.02, rng)),
                         tile(D.jpeg(a, 60)), tile(D.jpeg(a, 30))])
        grid(rows, cols, labels, "nuisance (no label): mild Gaussian noise (sigma, 0-1 scale) and JPEG re-compression").save(os.path.join(OUT, "08_nuisance.png"))
        # zoom applied to every training image (clean ones too): label unchanged
        zs = [1.0, 0.9, 0.8, 0.7, 0.6, 0.55]
        rows = [[tile(D.centred_zoom(a, z)) for z in zs] for a in imgs]
        grid(rows, [f"zoom z={z} ({1 / z:.2f}x)" for z in zs], labels, "centred zoom applied to ALL training images (framing label stays 0)").save(os.path.join(OUT, "09_zoom_all_images.png"))
        log.add(images_in=len(imgs), grids=len(GRIDS) + 1, examples=ex.image_id.tolist())


if __name__ == "__main__":
    main()

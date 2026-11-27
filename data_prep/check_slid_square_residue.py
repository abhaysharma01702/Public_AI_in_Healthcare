"""Count SLID images whose inpainting mask does not fully cover the black square's dark rim.

For every SLID image in labels/resized_512_manifest.csv: rebuild the 512 px image BEFORE filling
(Lanczos resize of the Split copy) and look just outside the fill mask (detected box + GROW px):
  near line = the first column right of the mask (rows of the square) / first row below it
              (a 1 px line: the fill takes its boundary values from exactly these pixels)
  far band  = a 12 px strip starting 4 px beyond the mask
A residue = near line less than half as bright as the far band (median max-channel), with the far
band not itself dark (median >= 40), i.e. a dark rim left between the mask and the real image. Images where the photo
itself is dark next to the square (far band dark) are not counted.

Usage: python -I check_slid_square_residue.py [out.csv]
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROW = 5


def check(job):
    im = Image.open(os.path.join(ROOT, job["source_path"])).convert("RGB")
    w, h = int(job["resized_width"]), int(job["resized_height"])
    a = np.asarray(im.resize((w, h), Image.LANCZOS)).max(axis=2).astype(float)
    out = dict(image_id=job["image_id"], residue_right=False, residue_bottom=False, unfilled=False)
    if job["square_inpainted"] not in (True, "True"):
        out["unfilled"] = bool(np.median(a[2:40, 3:40]) <= 2)  # square present but not filled
        return out
    x1, y1 = int(job["square_x1"]) + GROW, int(job["square_y1"]) + GROW
    rows, cols = slice(0, max(y1 - GROW, 1)), slice(0, max(x1 - GROW, 1))
    near_r, far_r = a[rows, x1], a[rows, x1 + 4:x1 + 16]   # first column just outside the mask
    near_b, far_b = a[y1, cols], a[y1 + 4:y1 + 16, cols]   # first row just outside the mask
    out.update(ratio_right=float(np.median(near_r) / max(np.median(far_r), 1)),
               ratio_bottom=float(np.median(near_b) / max(np.median(far_b), 1)),
               far_right=float(np.median(far_r)), far_bottom=float(np.median(far_b)))
    # residue: the strip next to the mask is less than half as bright as the image beyond it,
    # and the image beyond it is not itself dark (dark background next to the square is not a residue)
    out["residue_right"] = bool(out["ratio_right"] < 0.5 and out["far_right"] >= 40)
    out["residue_bottom"] = bool(out["ratio_bottom"] < 0.5 and out["far_bottom"] >= 40)
    return out


def main():
    m = pd.read_csv(os.path.join(ROOT, "labels", "resized_512_manifest.csv"))
    s = m[(m.dataset == "SLID") & (m.status == "resized")]
    with Pool(max(1, os.cpu_count() - 1)) as pool:
        r = pd.DataFrame(pool.map(check, s.to_dict("records"), chunksize=16))
    r["residue"] = r.residue_right | r.residue_bottom
    print(f"SLID images: {len(r)}; square present but not filled: {int(r.unfilled.sum)} {r[r.unfilled].image_id.tolist}")
    print(f"dark rim left next to the fill: {int(r.residue.sum)} (right {int(r.residue_right.sum)}, bottom {int(r.residue_bottom.sum)})")
    print("examples:", r[r.residue].image_id.head(15).tolist())
    if len(sys.argv) > 1:
        r.to_csv(sys.argv[1], index=False)


if __name__ == "__main__":
    main()

"""Step 1 of quality-module data: screen the 512 px images for real defects.

For every resized image (labels/resized_512_manifest.csv, status == resized) compute simple,
standard no-reference measures on the 512 px copy:
  - lap_var        variance of the Laplacian of the grey image (sharpness; low = blurry)
  - tenengrad      mean squared Sobel gradient magnitude (second sharpness measure)
  - mean_lum       mean luminance, 0-255 (ITU-R BT.601 weights)
  - frac_over      share of pixels with luminance >= 250 (blown out)
  - frac_under     share of pixels with luminance <= 5 (crushed black)
SLID: the inpainted corner (square box + 5 px) is left out of every measure.

Images are then ranked WITHIN their dataset (slit-beam photos are dark by nature, blue-light
photos have a different contrast, ...): z-scores of log(lap_var), log(tenengrad), mean_lum,
frac_over, frac_under per dataset, and flags for the extreme tails (thresholds in FLAGS).
Flagged images are only CANDIDATES; the decision per image is made by visual review.

Output: labels/quality_screening_metrics.csv (one row per resized image).
Usage:  python -I screen_quality_metrics.py
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I does not add the script folder
from run_log import RunLog  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "labels", "resized_512_manifest.csv")
OUT = os.path.join(ROOT, "labels", "quality_screening_metrics.csv")
GROW = 5  # same as make_resized_512.py

# tail flags, per dataset (percentile of the within-dataset distribution)
FLAGS = {
    "flag_blur": ("sharp_z", "low", 3),       # lowest 3% combined sharpness
    "flag_dark": ("mean_lum", "low", 2),      # darkest 2%
    "flag_bright": ("mean_lum", "high", 2),   # brightest 2%
    "flag_over": ("frac_over", "high", 2),    # most blown-out pixels 2%
}


def measure(job):
    a = np.asarray(Image.open(os.path.join(ROOT, job["resized_path"])).convert("RGB")).astype(np.float64)
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    valid = np.ones(lum.shape, bool)
    if job["square_x1"] == job["square_x1"]:  # not NaN: SLID inpainted corner
        valid[: int(job["square_y1"]) + GROW, : int(job["square_x1"]) + GROW] = False
    lap = ndimage.laplace(lum)
    gx, gy = ndimage.sobel(lum, axis=1), ndimage.sobel(lum, axis=0)
    inner = ndimage.binary_erosion(valid, iterations=2)  # avoid filter edges at the masked corner
    return dict(
        image_id=job["image_id"],
        lap_var=float(lap[inner].var()),
        tenengrad=float((gx ** 2 + gy ** 2)[inner].mean()),
        mean_lum=float(lum[valid].mean()),
        frac_over=float((lum[valid] >= 250).mean()),
        frac_under=float((lum[valid] <= 5).mean()),
    )


def main():
    with RunLog("screen_quality_metrics.py", notes="no-reference quality measures on 512 px copies") as log:
        m = pd.read_csv(MANIFEST)
        m = m[m.status == "resized"].reset_index(drop=True)
        jobs = m[["image_id", "resized_path", "square_x1", "square_y1"]].to_dict("records")
        workers = max(1, os.cpu_count() - 1)
        log.add(images_in=len(jobs), images_out=len(jobs), workers=workers)
        with Pool(workers) as pool:
            res = pd.DataFrame(pool.map(measure, jobs, chunksize=16))
        df = m[["dataset", "source_set", "image_id", "split"]].merge(res, on="image_id", validate="1:1")

        # within-dataset standardisation
        g = df.groupby("dataset")
        for col, src in [("z_log_lap", np.log(df.lap_var + 1e-6)), ("z_log_ten", np.log(df.tenengrad + 1e-6))]:
            df[col] = (src - src.groupby(df.dataset).transform("mean")) / src.groupby(df.dataset).transform("std")
        df["sharp_z"] = (df.z_log_lap + df.z_log_ten) / 2
        for flag, (col, side, pct) in FLAGS.items():
            q = g[col].transform(lambda s: s.quantile(pct / 100 if side == "low" else 1 - pct / 100))
            df[flag] = (df[col] <= q) if side == "low" else (df[col] >= q)
        df["flag_any"] = df[list(FLAGS)].any(axis=1)
        df = df.sort_values(["dataset", "image_id"]).reset_index(drop=True)
        df.to_csv(OUT, index=False, float_format="%.6g")

        summary = df.groupby("dataset").agg(n=("image_id", "size"), lap_var_median=("lap_var", "median"),
                                            tenengrad_median=("tenengrad", "median"), mean_lum_median=("mean_lum", "median"),
                                            frac_over_median=("frac_over", "median"), frac_under_median=("frac_under", "median"),
                                            flagged=("flag_any", "sum"))
        print(summary.round(3).to_string())
        print("flagged total:", int(df.flag_any.sum()))
        log.add(flagged=int(df.flag_any.sum()), flags={k: list(v) for k, v in FLAGS.items()},
                flagged_per_dataset=summary.flagged.astype(int).to_dict())


if __name__ == "__main__":
    main()

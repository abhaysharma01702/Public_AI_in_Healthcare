"""Clean pool for the quality module.

Rule: an image of the five quality-pool datasets is CLEAN unless any screening flag is set
(labels/quality_screening_metrics.csv: flag_blur, flag_dark, flag_bright, flag_over).
Flagged images are not deleted; they are marked not clean, with the reason, and stay available
for the real check set. External test sets (SDCTrans, SLIT-Net-v2) are not part of the pool.

Output: labels/quality_clean_pool.csv (one row per quality-pool image).
Usage:  python -I make_clean_pool.py
"""
import hashlib
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I does not add the script folder
from run_log import RunLog  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL = ["Nuclear Cataract", "SUSTech-SYSU", "SLID-E", "Infectious Keratitis", "SLID"]
FLAGS = ["flag_blur", "flag_dark", "flag_bright", "flag_over"]
OUT = os.path.join(ROOT, "labels", "quality_clean_pool.csv")


def main():
    with RunLog("make_clean_pool.py", notes="option 1: all screening-flagged images leave the clean pool") as log:
        s = pd.read_csv(os.path.join(ROOT, "labels", "quality_screening_metrics.csv"))
        m = pd.read_csv(os.path.join(ROOT, "labels", "resized_512_manifest.csv"))
        s = s[s.dataset.isin(POOL)]
        df = s.merge(m[["image_id", "resized_path", "note"]], on="image_id", validate="1:1")
        df["clean"] = ~df.flag_any
        df["not_clean_reason"] = df[FLAGS].apply(lambda r: ";".join(f[5:] for f in FLAGS if r[f]), axis=1)
        cols = ["dataset", "source_set", "image_id", "split", "resized_path", "clean", "not_clean_reason",
                "sharp_z", "mean_lum", "frac_over", "note"]
        df = df[cols].sort_values(["dataset", "image_id"]).reset_index(drop=True)
        df.to_csv(OUT, index=False, float_format="%.6g")

        t = pd.crosstab([df.dataset], [df.split, df.clean])
        print(t.to_string())
        tot = pd.crosstab(df.split, df.clean, margins=True)
        print(tot.to_string())
        reasons = df[~df.clean].not_clean_reason.str.split(";").explode().value_counts()
        print("reasons (an image can have several):", reasons.to_dict())
        log.add(images_in=len(df), images_out=int(df.clean.sum()), not_clean=int((~df.clean).sum()),
                clean_by_split=df[df.clean].split.value_counts().to_dict(),
                output_md5=hashlib.md5(open(OUT, "rb").read()).hexdigest())


if __name__ == "__main__":
    main()

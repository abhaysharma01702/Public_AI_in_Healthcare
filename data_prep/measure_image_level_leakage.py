"""Measure how much an image-level random split leaks.

Published papers on these datasets split by image. This script applies such a split (80/10/10, random,
seed 14, scikit-learn train_test_split) to our usable images and reports the share of test images that
have at least one image of the SAME eye / group in the training set.

Eye / group definitions:
  - Nuclear Cataract: volunteer + eye (right / left), from the folder names.
  - SUSTech-SYSU: likely-same-eye group, the grouping our own split uses.

Usage: python data_prep/measure_image_level_leakage.py
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 14


def measure(name, df, eye):
    train, rest = train_test_split(df.index, test_size=0.2, random_state=SEED)
    _, test = train_test_split(rest, test_size=0.5, random_state=SEED)
    train_eyes = set(eye[train])
    leaked = sum(eye[i] in train_eyes for i in test)
    print(f"{name}: {len(df)} images, {eye.nunique} eyes/groups ({len(df) / eye.nunique:.1f} images each); "
          f"image-level random split -> {leaked} of {len(test)} test images ({100 * leaked / len(test):.0f}%) "
          f"have the same eye/group in training")


def main():
    nc = pd.read_csv(os.path.join(ROOT, "labels", "nuclear_cataract_master.csv"))
    nc = nc[nc["status"] == "graded"]
    measure("Nuclear Cataract", nc, nc["volunteer"].astype(str) + "_" + nc["eye"])

    su = pd.read_csv(os.path.join(ROOT, "labels", "sustech_master.csv"))
    su = su[su["status"] == "usable"]
    measure("SUSTech-SYSU", su, su["group_id"])


if __name__ == "__main__":
    main()

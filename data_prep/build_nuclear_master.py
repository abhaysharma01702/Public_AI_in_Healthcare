"""Build the master CSV and patient-level 80/10/10 split for the Nuclear Cataract Database.

Labels come from Table 2 of Cruz-Vega et al. 2023 (labels/nuclear_cataract_table2.csv).
Rules:
  - Task label: LOCS III nuclear color (NC), half grades rounded up (2.5 -> 3).
    label_grade_down (2.5 -> 2) is kept for the sensitivity analysis.
  - Gradable images: eyes with a numeric NC. IOL, blood and ungraded eyes are excluded
    from grading (status column) but still get their patient's split.
  - Split: by volunteer (both eyes of a patient in the same split), stratified by
    label_grade, 80/10/10, seed 14.

Usage: python data_prep/build_nuclear_master.py
"""
import math
import os
import re

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

SEED = 14
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_ROOT = os.path.join(ROOT, "Data", "Raw", "Nuclear Cataract Database", "Nuclear Cataract Dataset")
TABLE2 = os.path.join(ROOT, "labels", "nuclear_cataract_table2.csv")
OUT = os.path.join(ROOT, "labels", "nuclear_cataract_master.csv")
EYE_FOLDERS = {"DER": "RE", "IZQ": "LE"}  # Spanish: derecho = right, izquierdo = left


def round_half_up(x):
    return int(math.floor(x + 0.5))


def round_half_down(x):
    return int(math.ceil(x - 0.5))


def list_images():
    rows = []
    for vol in sorted(os.listdir(IMG_ROOT), key=lambda s: (not s.isdigit(), s)):
        vol_dir = os.path.join(IMG_ROOT, vol)
        if not (os.path.isdir(vol_dir) and vol.isdigit()):
            continue
        for eye_folder in sorted(os.listdir(vol_dir)):
            eye_dir = os.path.join(vol_dir, eye_folder)
            if not os.path.isdir(eye_dir):
                continue
            eye = EYE_FOLDERS[eye_folder.upper()]  # folder case varies (DER / Der)
            for fname in sorted(os.listdir(eye_dir)):
                if not fname.lower().endswith(".jpg"):
                    continue  # skips the camera's DATAFILE index
                path = os.path.join(eye_dir, fname)
                try:
                    with Image.open(path) as im:
                        width, height = im.size
                        im.verify()
                except Exception as e:  # corrupt file: log and drop
                    print(f"DROPPED corrupt image {path}: {e}")
                    continue
                rows.append({
                    "image_id": f"nc_{int(vol):03d}_{eye}_{os.path.splitext(fname)[0]}",
                    "dataset": "nuclear_cataract",
                    "raw_path": os.path.relpath(path, ROOT).replace(os.sep, "/"),
                    "volunteer": int(vol),
                    "eye": eye,
                    "orig_width": width,
                    "orig_height": height,
                })
    return pd.DataFrame(rows)


def attach_labels(images):
    table = pd.read_csv(TABLE2, dtype=str)
    table["volunteer"] = table["volunteer"].astype(int)
    df = images.merge(table, on=["volunteer", "eye"], how="left", validate="many_to_one")

    nc = pd.to_numeric(df["NC"], errors="coerce")
    df["status"] = np.select(
        [nc.notna(), df["NC"] == "IOL", df["NC"] == "blood"],
        ["graded", "iol", "blood"],
        default="ungraded",
    )
    df["label_grade"] = nc.map(lambda x: round_half_up(x) if pd.notna(x) else pd.NA).astype("Int64")
    df["label_grade_down"] = nc.map(lambda x: round_half_down(x) if pd.notna(x) else pd.NA).astype("Int64")
    df["group_id"] = df["volunteer"].map(lambda v: f"nc_{v:03d}")  # patient-level grouping
    df["group_source"] = "folder"
    return df


def make_splits(df):
    graded = df[df["status"] == "graded"]
    y, groups = graded["label_grade"].to_numpy(), graded["group_id"].to_numpy()

    # 10 stratified group folds: one fold = test (10%).
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    rest_idx, test_idx = next(sgkf.split(graded, y, groups))
    # Remaining 90% into 9 folds: one fold = val (10% of total), the rest = train (80%).
    sgkf = StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=SEED)
    tr, va = next(sgkf.split(graded.iloc[rest_idx], y[rest_idx], groups[rest_idx]))

    split_of_group = {}
    for idx, name in [(rest_idx[tr], "train"), (rest_idx[va], "val"), (test_idx, "test")]:
        for g in set(groups[idx]):
            split_of_group[g] = name
    # Patients with no gradable eye (e.g. both eyes IOL) never enter evaluation; put them in train.
    df["split"] = df["group_id"].map(split_of_group).fillna("train")
    return df


def report(df):
    assert df.groupby("group_id")["split"].nunique().max() == 1, "a patient appears in two splits"
    graded = df[df["status"] == "graded"]
    print(f"\nImages: {len(df)} total, {len(graded)} graded")
    print("Status counts:", df["status"].value_counts().to_dict())
    print("\nPatients per split (graded):", graded.groupby("split")["group_id"].nunique().to_dict())
    print("\nGraded images per split x grade (label_grade, NC rounded half up):")
    tab = pd.crosstab(graded["split"], graded["label_grade"], margins=True)
    print(tab.reindex(["train", "val", "test", "All"]))
    print("\nShare of each split (%):")
    print((pd.crosstab(graded["split"], graded["label_grade"], normalize="index") * 100).round(1)
          .reindex(["train", "val", "test"]))
    print("\nSplit share of graded images (%):", (graded["split"].value_counts(normalize=True) * 100).round(1).to_dict())


def main():
    df = make_splits(attach_labels(list_images()))
    cols = ["image_id", "dataset", "raw_path", "volunteer", "eye", "group_id", "group_source",
            "status", "NO", "NC", "C", "P", "label_grade", "label_grade_down",
            "orig_width", "orig_height", "split"]
    df[cols].to_csv(OUT, index=False)
    report(df)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()

"""Copy graded Nuclear Cataract images into split folders, following labels/nuclear_cataract_master.csv.

Layout (ImageFolder style): Data/Split/nuclear_cataract/<split>/<grade>/<image_id>.jpg
Original images are copied, never moved. Run build_nuclear_master.py first.

Usage: python data_prep/make_nuclear_split_folders.py
"""
import os
import shutil

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER = os.path.join(ROOT, "labels", "nuclear_cataract_master.csv")
OUT_DIR = os.path.join(ROOT, "Data", "Split", "nuclear_cataract")


def main():
    df = pd.read_csv(MASTER)
    graded = df[df["status"] == "graded"]

    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)  # rebuild from the CSV so folders never drift from it
    for row in graded.itertuples():
        dst_dir = os.path.join(OUT_DIR, row.split, str(int(row.label_grade)))
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(os.path.join(ROOT, row.raw_path), os.path.join(dst_dir, f"{row.image_id}.jpg"))

    for split in ["train", "val", "test"]:
        counts = {g: len(os.listdir(os.path.join(OUT_DIR, split, g)))
                  for g in sorted(os.listdir(os.path.join(OUT_DIR, split)))}
        print(f"{split:5s} {sum(counts.values):4d} images  per grade: {counts}")
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()

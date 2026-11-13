"""Extract SLID-E into split folders (official partitions) and build its master CSV.

SLID-E ships as train.zip / val.zip / test.zip; the official split is patient-level,
so it is used unchanged. Images are streamed from the zips straight to
Data/Split/SLID-E/<split>/<image_id>.jpg (no class subfolders: SLID-E only trains the quality
module, which does not use the epiphora labels). The zips stay the untouched raw copy.

File names: <10-hex patient code>[space]<YYYYMMDD><3-digit number>S_<2-digit shot><L|R>.jpg
image_id = SLID-E_<code>_<date>_<number>_<shot><eye>, e.g. SLID-E_0479b7713d_20220117_001_07R.

Usage: python data_prep/build_slide_master.py
"""
import io
import os
import re
import shutil
import zipfile

import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "Data", "Raw", "SLID-E")
INFO = os.path.join(RAW, "SLID_E_information.csv")
OUT_DIR = os.path.join(ROOT, "Data", "Split", "SLID-E")
OUT = os.path.join(ROOT, "labels", "SLID-E_master.csv")
NAME = re.compile(r"^([0-9a-f]{10}) ?(\d{8})(\d{3})S_(\d{2})([LR])\.jpg$")
SPLITS = ["train", "val", "test"]


def main():
    info = pd.read_csv(INFO).dropna(subset=["Filename"])  # 370 rows have no file name
    stage_of = dict(zip(info["Filename"], info["epiphora_stage"]))
    csv_split_of = dict(zip(info["Filename"], info["partition_group"]))

    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)  # rebuild from the zips so folders never drift
    rows = []
    for split in SPLITS:
        os.makedirs(os.path.join(OUT_DIR, split))
        with zipfile.ZipFile(os.path.join(RAW, f"{split}.zip")) as zf:
            for member in zf.namelist():
                if member.endswith("/"):
                    continue
                fname = member.split("/", 1)[1]
                code, date, number, shot, eye = NAME.match(fname).groups()
                image_id = f"SLID-E_{code}_{date}_{number}_{shot}{eye}"
                data = zf.read(member)
                with Image.open(io.BytesIO(data)) as im:  # corrupt files raise here
                    width, height = im.size
                    im.verify()
                with open(os.path.join(OUT_DIR, split, f"{image_id}.jpg"), "wb") as f:
                    f.write(data)
                rows.append({
                    "image_id": image_id,
                    "dataset": "SLID-E",
                    "raw_path": f"Data/Raw/SLID-E/{split}.zip/{member}",
                    "split_path": f"Data/Split/SLID-E/{split}/{image_id}.jpg",
                    "patient_code": code,
                    "group_id": f"SLID-E_{code}",
                    "group_source": "filename_patient_code",
                    "eye": "RE" if eye == "R" else "LE",
                    "date": date,
                    "visit_number": number,
                    "shot": shot,
                    "epiphora_stage": stage_of.get(fname),  # empty for the 370 unmatched rows
                    "orig_width": width,
                    "orig_height": height,
                    "split": split,
                })
                # The CSV's partition, where present, must agree with the zip.
                assert csv_split_of.get(fname, split) == split, f"partition mismatch for {fname}"

    df = pd.DataFrame(rows)
    assert df["image_id"].is_unique
    assert df.groupby("group_id")["split"].nunique().max() == 1, "a patient appears in two splits"
    df.to_csv(OUT, index=False)

    print("Images per split:", df["split"].value_counts().reindex(SPLITS).to_dict())
    print("Patients per split:", df.groupby("split")["group_id"].nunique().reindex(SPLITS).to_dict())
    print("Image sizes:", df.groupby(["orig_width", "orig_height"]).size().to_dict())
    print("Epiphora stage known:", int(df["epiphora_stage"].notna().sum()), "/", len(df))
    print(f"\nWrote {OUT_DIR}\nWrote {OUT}")


if __name__ == "__main__":
    main()

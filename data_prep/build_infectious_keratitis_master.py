"""Build the master CSV, leakage-free groups and 80/10/10 split for the Infectious Keratitis dataset,
and extract the images into split folders.

Source: Figshare 10.6084/m9.figshare.28578860.v1 (AGEH_Dataset.zip, ZEH_Dataset.zip), .
Zip layout: <hospital>_Dataset/<disease>/<1|2|3>/<prefix>_<patient number>_<shot><L|R>.jpg

Rules:
  - Patient = hospital + disease + number (numbers restart per disease folder).
  - Groups join images that could come from the same person:
      (a) same patient;
      (b) same hospital and number under two diseases (possibly one person);
      (c) exact duplicates between patients: pHash distance 0 or ResNet50 cosine >= 0.98.
    Files without a patient number start as their own group.
  - Byte-identical files (the same photo filed in two subfolders): first copy kept, the others get
    status `duplicate_copy` and are not written to the split folders.
  - Split: by group, stratified by hospital x disease (lighting is not labelled yet), 80/10/10, seed 14.
  - Images are written to Data/Split/Infectious_Keratitis/<split>/<image_id>.jpg (no class
    subfolders: this dataset only trains the quality module). The zips stay the raw copy.

Image IDs include the subfolder (`_s<1|2|3>`) because the same file name can occur in two subfolders.
Hashes and embeddings are cached in Data/derived/infectious_keratitis/.

Usage: python data_prep/build_infectious_keratitis_master.py
"""
import hashlib
import io
import os
import re
import shutil
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dct
from sklearn.model_selection import StratifiedGroupKFold

SEED = 14
DUP_COSINE = 0.98

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "Data", "Raw", "Infectious Keratitis Dataset (2025)")
ZIPS = ["AGEH_Dataset.zip", "ZEH_Dataset.zip"]
CACHE = os.path.join(ROOT, "Data", "derived", "infectious_keratitis")
OUT_DIR = os.path.join(ROOT, "Data", "Split", "Infectious_Keratitis")
OUT = os.path.join(ROOT, "labels", "Infectious_Keratitis_master.csv")
LINKS_OUT = os.path.join(ROOT, "labels", "Infectious_Keratitis_group_links.csv")
NAME = re.compile(r"^([A-Za-z]+)_(\d+)_(\d+)([LR])$")
DISEASE_CODES = {"BK": "BK", "FK": "FK", "HSK": "HSK", "other_corneal_abnormalities": "Others"}
SPLITS = ["train", "val", "test"]
FEATURES = ("sizes", "md5", "phash64", "resnet50_imagenet_v2")


def list_images():
    rows = []
    for z in ZIPS:
        with zipfile.ZipFile(os.path.join(RAW, z)) as zf:
            for member in zf.namelist():
                if member.endswith("/"):
                    continue
                hosp_dir, disease, subfolder, fname = member.split("/")
                hospital = hosp_dir.replace("_Dataset", "")
                stem = os.path.splitext(fname)[0].strip()  # one name has a space before ".jpg"
                m = NAME.match(stem)
                code = DISEASE_CODES[disease]
                if m:
                    assert m.group(1) == code, f"prefix does not match folder: {member}"
                    number, shot, eye = int(m.group(2)), m.group(3), m.group(4)
                    patient_id = f"IK_{hospital}_{code}_{number:03d}"
                    image_id = f"{patient_id}_{shot}{eye}_s{subfolder}"
                else:  # no patient number in the name
                    number, shot, eye = None, None, None
                    clean = re.sub(r"[^A-Za-z0-9]+", "-", stem).strip("-")
                    patient_id = f"IK_{hospital}_{code}_nopid-{clean}"
                    image_id = f"{patient_id}_s{subfolder}"
                rows.append({
                    "image_id": image_id, "zip": z, "member": member, "hospital": hospital,
                    "disease": code, "subfolder": int(subfolder), "patient_number": number,
                    "patient_id": patient_id, "patient_id_missing": m is None,
                    "eye": {"R": "RE", "L": "LE"}.get(eye), "shot": shot,
                })
    df = pd.DataFrame(rows)
    assert df["image_id"].is_unique
    return df


def read_member(zf_cache, row):
    if row.zip not in zf_cache:
        zf_cache[row.zip] = zipfile.ZipFile(os.path.join(RAW, row.zip))
    return zf_cache[row.zip].read(row.member)


def phash(im, hash_size=8, factor=4):
    """64-bit perceptual hash (same algorithm as imagehash.phash)."""
    a = np.asarray(im.convert("L").resize((hash_size * factor,) * 2, Image.LANCZOS), dtype=float)
    d = dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")[:hash_size, :hash_size]
    return (d > np.median(d)).flatten()


def compute_features(df):
    """Image size, MD5, pHash and ResNet50 embedding for every image (cached)."""
    paths = {k: os.path.join(CACHE, f"{k}.npy") for k in FEATURES}
    ids_path = os.path.join(CACHE, "image_ids.txt")
    if all(os.path.exists(p) for p in paths.values()) and os.path.exists(ids_path)\
            and open(ids_path).read().split("\n") == df["image_id"].tolist():
        return [np.load(paths[k]) for k in FEATURES]

    import torch
    import torchvision

    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    model = torchvision.models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    tf = weights.transforms()
    sizes, md5s, hashes, feats, batch, zf_cache = [], [], [], [], [], {}
    with torch.no_grad():
        for k, row in enumerate(df.itertuples()):
            data = read_member(zf_cache, row)
            md5s.append(hashlib.md5(data).hexdigest())
            with Image.open(io.BytesIO(data)) as im:
                sizes.append(im.size)
                im.draft("RGB", (800, 600))
                rgb = im.convert("RGB")
            hashes.append(phash(rgb))
            batch.append(tf(rgb))
            if len(batch) == 32 or k == len(df) - 1:
                feats.append(model(torch.stack(batch)).numpy())
                batch = []
    F = np.concatenate(feats)
    F /= np.linalg.norm(F, axis=1, keepdims=True)
    out = [np.array(sizes), np.array(md5s), np.array(hashes), F]
    os.makedirs(CACHE, exist_ok=True)
    for k, arr in zip(FEATURES, out):
        np.save(paths[k], arr)
    open(ids_path, "w").write("\n".join(df["image_id"]))
    return out


def find_groups(df, H, F):
    n = len(df)
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[root(i)] = root(j)

    links = []
    # (a) same patient
    for idx in df.groupby("patient_id").indices.values():
        for j in idx[1:]:
            union(idx[0], j)
    # (b) same hospital + number under two diseases
    numbered = df[df["patient_number"].notna()]
    for _, sub in numbered.groupby(["hospital", "patient_number"]):
        if sub["disease"].nunique() > 1:
            for j in sub.index[1:]:
                union(sub.index[0], j)
            links.append((sorted(sub["patient_id"].unique()), "same_number_two_diseases", None, None))
    # (c) exact duplicates between different patients
    C = F @ F.T
    D = (H[:, None, :] != H[None, :, :]).sum(-1)
    pid = df["patient_id"].to_numpy()
    for i in range(n):
        for j in range(i + 1, n):
            if pid[i] != pid[j] and (D[i, j] == 0 or C[i, j] >= DUP_COSINE):
                union(i, j)
                links.append(([df.at[i, "image_id"], df.at[j, "image_id"]], "exact_duplicate",
                              int(D[i, j]), round(float(C[i, j]), 4)))

    roots = [root(i) for i in range(n)]
    first = {}
    for i, r in enumerate(roots):
        first.setdefault(r, df.at[i, "patient_id"])
    links = pd.DataFrame([(" | ".join(a), rule, d, c) for a, rule, d, c in links],
                         columns=["members", "rule", "phash_distance", "cosine"])
    return [f"grp_{first[r]}" for r in roots], links


def make_splits(df):
    usable = df[df["status"] == "usable"]
    y = (usable["hospital"] + "_" + usable["disease"]).to_numpy()
    groups = usable["group_id"].to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    rest_idx, test_idx = next(sgkf.split(usable, y, groups))
    sgkf = StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=SEED)
    tr, va = next(sgkf.split(usable.iloc[rest_idx], y[rest_idx], groups[rest_idx]))
    split_of_group = {}
    for idx, name in [(rest_idx[tr], "train"), (rest_idx[va], "val"), (test_idx, "test")]:
        for g in set(groups[idx]):
            split_of_group[g] = name
    df["split"] = df["group_id"].map(split_of_group)  # duplicate copies follow their group
    return df


def write_folders(df):
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)  # rebuild from the zips so folders never drift from the CSV
    for s in SPLITS:
        os.makedirs(os.path.join(OUT_DIR, s))
    zf_cache = {}
    for row in df[df["status"] == "usable"].itertuples():
        data = read_member(zf_cache, row)
        Image.open(io.BytesIO(data)).verify()  # corrupt files raise here
        with open(os.path.join(ROOT, row.split_path), "wb") as f:
            f.write(data)


def report(df, links):
    assert df.groupby("group_id")["split"].nunique().max() == 1, "a group appears in two splits"
    assert df.groupby("patient_id")["split"].nunique().max() == 1, "a patient appears in two splits"
    print(f"Images: {len(df)}; status: {df['status'].value_counts.to_dict}; "
          f"patients: {df['patient_id'].nunique}; groups: {df['group_id'].nunique}")
    print("Links beyond same-patient:", links["rule"].value_counts().to_dict() if len(links) else {})
    u = df[df["status"] == "usable"]
    strata = u["hospital"] + "_" + u["disease"]
    print("\nUsable images per split:", u["split"].value_counts().reindex(SPLITS).to_dict(),
          "  (%):", (u["split"].value_counts(normalize=True) * 100).round(1).reindex(SPLITS).to_dict())
    print("Groups per split:", u.groupby("split")["group_id"].nunique().reindex(SPLITS).to_dict())
    print("\nUsable images per split x hospital/disease:")
    print(pd.crosstab(u["split"], strata, margins=True).reindex(SPLITS + ["All"]))
    print("\nShare per split (%):")
    print((pd.crosstab(u["split"], strata, normalize="index") * 100).round(1).reindex(SPLITS))
    print("\nImage sizes:", u.groupby(["orig_width", "orig_height"]).size().to_dict())


def main():
    df = list_images()
    sizes, md5, H, F = compute_features(df)
    df["orig_width"], df["orig_height"] = sizes[:, 0], sizes[:, 1]
    df["md5"] = md5
    df["status"] = np.where(df.duplicated("md5", keep="first"), "duplicate_copy", "usable")
    df["group_id"], links = find_groups(df, H, F)
    df["group_source"] = "patient_id+same_number+exact_duplicate"
    df["dataset"] = "Infectious_Keratitis"
    df["raw_path"] = "Data/Raw/Infectious Keratitis Dataset (2025)/" + df["zip"] + "/" + df["member"]
    df = make_splits(df)
    df["split_path"] = np.where(df["status"] == "usable",
                                "Data/Split/Infectious_Keratitis/" + df["split"] + "/" + df["image_id"] + ".jpg", "")
    write_folders(df)

    cols = ["image_id", "dataset", "raw_path", "split_path", "status", "hospital", "disease", "subfolder",
            "patient_number", "patient_id", "patient_id_missing", "eye", "shot", "group_id",
            "group_source", "md5", "orig_width", "orig_height", "split"]
    df[cols].to_csv(OUT, index=False)
    links.to_csv(LINKS_OUT, index=False)
    report(df, links)
    print(f"\nWrote {OUT_DIR}\nWrote {OUT}\nWrote {LINKS_OUT}")


if __name__ == "__main__":
    main()

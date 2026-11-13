"""Build the master CSV, likely-same-eye groups and 80/10/10 split for SLID, and copy the images into
split folders.

SLID (github.com/xumingyu-hub/SLID): 2,617 slit-lamp images `1.png … 2617.png` and Annotations.csv
(region outlines and lesion names per image). No patient or eye IDs are released.

Rules, same approach as SUSTech-SYSU:
  - Exact duplicates anywhere: identical file bytes (MD5), pHash distance 0, or ResNet50 cosine >= 0.98.
  - Consecutive file numbers with ResNet50 cosine >= a cut-off (files are roughly in capture order, so
    shots of one eye sit next to each other, often from different angles):
      0.85 when both images are healthy (files 1-245; healthy eyes of different people look alike),
      0.55 otherwise (different views of one lesion can score as low as ~0.5).
  Groups are the connected components of these links.
  - Split: by group, stratified by image resolution (acquisition device) x healthy / lesion,
    80/10/10, seed 14.
  - Images are copied to Data/Split/SLID/<split>/<image_id>.png (no class subfolders: SLID only
    trains the quality module). Originals stay in Data/Raw/SLID/Original_Slit-lamp_Images/.

Hashes and embeddings are cached in Data/derived/slid/.

Usage: python data_prep/build_slid_master.py
"""
import hashlib
import json
import os
import shutil

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dct
from sklearn.model_selection import StratifiedGroupKFold

SEED = 14
DUP_COSINE = 0.98
NEIGHBOUR_COSINE = 0.55          # lesion images
HEALTHY_NEIGHBOUR_COSINE = 0.85  # both images healthy (no lesion annotated)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "Data", "Raw", "SLID")
IMG_DIR = os.path.join(DATA, "Original_Slit-lamp_Images")
ANNOTATIONS = os.path.join(DATA, "Annotations.csv")
CACHE = os.path.join(ROOT, "Data", "derived", "slid")
OUT_DIR = os.path.join(ROOT, "Data", "Split", "SLID")
OUT = os.path.join(ROOT, "labels", "SLID_master.csv")
LINKS_OUT = os.path.join(ROOT, "labels", "SLID_group_links.csv")
SPLITS = ["train", "val", "test"]
FEATURES = ("sizes", "md5", "phash64", "resnet50_imagenet_v2")


def load_images():
    files = sorted(os.listdir(IMG_DIR), key=lambda f: int(f.split(".")[0]))
    df = pd.DataFrame({"file": files})
    df["number"] = df["file"].str.replace(".png", "", regex=False).astype(int)
    df["image_id"] = df["number"].map(lambda k: f"SLID_{k:04d}")

    ann = pd.read_csv(ANNOTATIONS, encoding="utf-8-sig")
    lesion = ann["attributes"].map(lambda s: json.loads(s).get("lesion", ""))
    lesions = lesion.groupby(ann["filename"]).agg(lambda s: "|".join(sorted({x for x in s if x})))
    df["lesions"] = df["file"].map(lesions).fillna("")
    assert df["file"].isin(ann["filename"]).all(), "image without annotations"
    return df


def phash(im, hash_size=8, factor=4):
    """64-bit perceptual hash (same algorithm as imagehash.phash)."""
    a = np.asarray(im.convert("L").resize((hash_size * factor,) * 2, Image.LANCZOS), dtype=float)
    d = dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")[:hash_size, :hash_size]
    return (d > np.median(d)).flatten()


def compute_features(files):
    paths = {k: os.path.join(CACHE, f"{k}.npy") for k in FEATURES}
    ids_path = os.path.join(CACHE, "files.txt")
    if all(os.path.exists(p) for p in paths.values()) and os.path.exists(ids_path)\
            and open(ids_path).read().split("\n") == files:
        return [np.load(paths[k]) for k in FEATURES]

    import torch
    import torchvision

    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    model = torchvision.models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    tf = weights.transforms()
    sizes, md5s, hashes, feats, batch = [], [], [], [], []
    with torch.no_grad():
        for k, f in enumerate(files):
            path = os.path.join(IMG_DIR, f)
            md5s.append(hashlib.md5(open(path, "rb").read()).hexdigest())
            with Image.open(path) as im:
                sizes.append(im.size)
                im.verify()
            with Image.open(path) as im:
                rgb = im.convert("RGB")
            hashes.append(phash(rgb))
            batch.append(tf(rgb))
            if len(batch) == 32 or k == len(files) - 1:
                feats.append(model(torch.stack(batch)).numpy())
                batch = []
    F = np.concatenate(feats)
    F /= np.linalg.norm(F, axis=1, keepdims=True)
    out = [np.array(sizes), np.array(md5s), np.array(hashes), F]
    os.makedirs(CACHE, exist_ok=True)
    for k, arr in zip(FEATURES, out):
        np.save(paths[k], arr)
    open(ids_path, "w").write("\n".join(files))
    return out


def find_groups(df, md5, H, F):
    n = len(df)
    healthy = (df["lesions"] == "").to_numpy()
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    C = F @ F.T
    D = (H[:, None, :] != H[None, :, :]).sum(-1)
    links = []
    for i in range(n):
        for j in range(i + 1, n):
            if md5[i] == md5[j] or D[i, j] == 0 or C[i, j] >= DUP_COSINE:
                rule = "exact_duplicate"
            elif j == i + 1 and healthy[i] and healthy[j] and C[i, j] >= HEALTHY_NEIGHBOUR_COSINE:
                rule = "neighbour_healthy"
            elif j == i + 1 and not (healthy[i] and healthy[j]) and C[i, j] >= NEIGHBOUR_COSINE:
                rule = "neighbour"
            else:
                continue
            links.append((df.at[i, "image_id"], df.at[j, "image_id"], rule, int(D[i, j]), round(float(C[i, j]), 4)))
            parent[root(i)] = root(j)
    roots = [root(i) for i in range(n)]
    first = {}
    for i, r in enumerate(roots):
        first.setdefault(r, df.at[i, "number"])
    links = pd.DataFrame(links, columns=["image_a", "image_b", "rule", "phash_distance", "cosine"])
    return [f"SLID_g{first[r]:04d}" for r in roots], links


def make_splits(df):
    usable = df[df["status"] == "usable"]
    y = (usable["resolution"] + "_" + np.where(usable["lesions"] == "", "healthy", "lesion")).to_numpy()
    groups = usable["group_id"].to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    rest_idx, test_idx = next(sgkf.split(usable, y, groups))
    sgkf = StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=SEED)
    tr, va = next(sgkf.split(usable.iloc[rest_idx], y[rest_idx], groups[rest_idx]))
    split_of_group = {}
    for idx, name in [(rest_idx[tr], "train"), (rest_idx[va], "val"), (test_idx, "test")]:
        for g in set(groups[idx]):
            split_of_group[g] = name
    df["split"] = df["group_id"].map(split_of_group).fillna("train")
    return df


def write_folders(df):
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)  # rebuild from the CSV so folders never drift from it
    for s in SPLITS:
        os.makedirs(os.path.join(OUT_DIR, s))
    for row in df[df["status"] == "usable"].itertuples():
        shutil.copy2(os.path.join(IMG_DIR, row.file), os.path.join(ROOT, row.split_path))


def report(df, links):
    assert df.groupby("group_id")["split"].nunique().max() == 1, "a group appears in two splits"
    u = df[df["status"] == "usable"]
    sizes = df.groupby("group_id").size()
    print(f"Images: {len(df)}; status: {df['status'].value_counts.to_dict}")
    print(f"Groups: {sizes.size} (single images {int((sizes == 1).sum)}, largest {sizes.max})")
    print("Links:", links["rule"].value_counts().to_dict())
    print("\nUsable images per split:", u["split"].value_counts().reindex(SPLITS).to_dict(),
          " (%):", (u["split"].value_counts(normalize=True) * 100).round(1).reindex(SPLITS).to_dict())
    print("Groups per split:", u.groupby("split")["group_id"].nunique().reindex(SPLITS).to_dict())
    print("\nImages per split x resolution:")
    print(pd.crosstab(u["split"], u["resolution"], margins=True).reindex(SPLITS + ["All"]))
    print("\nResolution share per split (%):")
    print((pd.crosstab(u["split"], u["resolution"], normalize="index") * 100).round(1).reindex(SPLITS))
    print("\nHealthy (no lesion) share per split (%):", (u.groupby("split")["lesions"].apply(lambda s: (s == "").mean()) * 100).round(1).reindex(SPLITS).to_dict())


def main():
    df = load_images()
    sizes, md5, H, F = compute_features(df["file"].tolist())
    df["orig_width"], df["orig_height"] = sizes[:, 0], sizes[:, 1]
    df["resolution"] = df["orig_width"].astype(str) + "x" + df["orig_height"].astype(str)
    df["md5"] = md5
    # Byte-identical files: keep the first copy only.
    df["status"] = np.where(df.duplicated("md5", keep="first"), "duplicate_copy", "usable")
    df["group_id"], links = find_groups(df, md5, H, F)
    df["group_source"] = "md5_phash_resnet50_neighbour"
    df["dataset"] = "SLID"
    df["raw_path"] = "Data/Raw/SLID/Original_Slit-lamp_Images/" + df["file"]
    df = make_splits(df)
    df["split_path"] = np.where(df["status"] == "usable",
                                "Data/Split/SLID/" + df["split"] + "/" + df["image_id"] + ".png", "")
    write_folders(df)

    cols = ["image_id", "dataset", "raw_path", "split_path", "status", "group_id", "group_source",
            "lesions", "resolution", "orig_width", "orig_height", "md5", "split"]
    df[cols].to_csv(OUT, index=False)
    links.to_csv(LINKS_OUT, index=False)
    report(df, links)
    print(f"\nWrote {OUT_DIR}\nWrote {OUT}\nWrote {LINKS_OUT}")


if __name__ == "__main__":
    main()

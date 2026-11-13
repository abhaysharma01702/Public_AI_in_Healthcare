"""Build the master CSV, near-duplicate groups and 80/10/10 split for SUSTech-SYSU.

Task: type grade (TG standard, 5 classes). Category and severity grade are kept as extra columns.
Rules:
  - Labels from `Category information.xlsx`: column 2 = category (general pattern),
    column 3 = type, column 4 = severity grade.
  - No patient IDs exist, so images are grouped as likely-same-eye:
      (a) exact duplicates anywhere: pHash distance 0 or ResNet50 cosine >= 0.98;
      (b) consecutive file numbers with ResNet50 cosine >= 0.70 (files are roughly in
          capture order, so repeat shots of one eye sit next to each other).
    Groups are the connected components of these links.
  - Exact duplicates with different type labels -> status `label_conflict`, excluded from
    training and evaluation (kept in the CSV).
  - Split: by group, stratified by type, 80/10/10, seed 14 (same method as Nuclear Cataract).

Embeddings and hashes are cached in Data/derived/sustech/ so re-runs give the same groups.

Usage: python data_prep/build_sustech_master.py
"""
import os

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dct
from sklearn.model_selection import StratifiedGroupKFold

SEED = 14
DUP_COSINE = 0.98       # rule (a): exact duplicate
NEIGHBOUR_COSINE = 0.70  # rule (b): consecutive files, likely same eye

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "Data", "Raw", "SUSTech-SYSU")
IMG_DIR = os.path.join(DATA, "rawImages")
LABELS = os.path.join(DATA, "Category information.xlsx")
CACHE = os.path.join(ROOT, "Data", "derived", "sustech")
OUT = os.path.join(ROOT, "labels", "sustech_master.csv")
PAIRS_OUT = os.path.join(ROOT, "labels", "sustech_group_links.csv")

CATEGORY_NAMES = {0: "point_like", 1: "point_flaky_mixed", 2: "flaky"}


def load_labels():
    df = pd.read_excel(LABELS, sheet_name="Sheet1")
    df.columns = ["file", "category", "type", "grade"]
    expected = {"category": [358, 263, 91], "type": [36, 98, 203, 273, 102], "grade": [36, 78, 40, 10, 548]}
    for col, counts in expected.items():  # guards against the type/severity swap seen in two papers
        assert df[col].value_counts().sort_index().tolist() == counts, f"unexpected counts in {col}"
    df["number"] = df["file"].str.replace(".jpg", "", regex=False).astype(int)
    return df.sort_values("number").reset_index(drop=True)


def image_info(files):
    rows = []
    for f in files:
        path = os.path.join(IMG_DIR, f)
        with Image.open(path) as im:
            width, height = im.size
            im.verify()
        rows.append((width, height))
    return rows


def phash(path, hash_size=8, factor=4):
    """64-bit perceptual hash (same algorithm as imagehash.phash)."""
    with Image.open(path) as im:
        im.draft("L", (hash_size * factor * 8, hash_size * factor * 8))
        a = np.asarray(im.convert("L").resize((hash_size * factor,) * 2, Image.LANCZOS), dtype=float)
    d = dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")[:hash_size, :hash_size]
    return (d > np.median(d)).flatten()


def resnet50_embeddings(files):
    import torch
    import torchvision

    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    model = torchvision.models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    tf = weights.transforms()
    feats = []
    with torch.no_grad():
        for k in range(0, len(files), 32):
            batch = []
            for f in files[k:k + 32]:
                with Image.open(os.path.join(IMG_DIR, f)) as im:
                    im.draft("RGB", (800, 600))
                    batch.append(tf(im.convert("RGB")))
            feats.append(model(torch.stack(batch)).numpy())
    F = np.concatenate(feats)
    return F / np.linalg.norm(F, axis=1, keepdims=True)


def cached(name, fn, files):
    path = os.path.join(CACHE, name)
    if os.path.exists(path):
        return np.load(path)
    os.makedirs(CACHE, exist_ok=True)
    arr = fn(files)
    np.save(path, arr)
    return arr


def find_groups(files, H, F):
    n = len(files)
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
            if D[i, j] == 0 or C[i, j] >= DUP_COSINE:
                links.append((files[i], files[j], "exact_duplicate", int(D[i, j]), float(C[i, j])))
            elif j == i + 1 and C[i, j] >= NEIGHBOUR_COSINE:
                links.append((files[i], files[j], "neighbour", int(D[i, j]), float(C[i, j])))
    index = {f: k for k, f in enumerate(files)}
    for a, b, *_ in links:
        parent[root(index[a])] = root(index[b])
    roots = [root(i) for i in range(n)]
    # Name each group after its lowest file number, so names are stable and readable.
    first = {}
    for i, r in enumerate(roots):
        first.setdefault(r, i)
    group_id = [f"sus_g{int(files[first[r]].split('.')[0]):03d}" for r in roots]
    links = pd.DataFrame(links, columns=["file_a", "file_b", "rule", "phash_distance", "cosine"])
    return group_id, links


def make_splits(df):
    usable = df[df["status"] == "usable"]
    y, groups = usable["type"].to_numpy(), usable["group_id"].to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    rest_idx, test_idx = next(sgkf.split(usable, y, groups))
    sgkf = StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=SEED)
    tr, va = next(sgkf.split(usable.iloc[rest_idx], y[rest_idx], groups[rest_idx]))
    split_of_group = {}
    for idx, name in [(rest_idx[tr], "train"), (rest_idx[va], "val"), (test_idx, "test")]:
        for g in set(groups[idx]):
            split_of_group[g] = name
    # Groups made only of excluded images never enter evaluation; put them in train.
    df["split"] = df["group_id"].map(split_of_group).fillna("train")
    return df


def report(df):
    assert df.groupby("group_id")["split"].nunique().max() == 1, "a group appears in two splits"
    u = df[df["status"] == "usable"]
    order = ["train", "val", "test"]
    print(f"Images: {len(df)} total, {len(u)} usable; status: {df['status'].value_counts.to_dict}")
    sizes = df.groupby("group_id").size()
    print(f"Groups: {sizes.size} (singletons {int((sizes == 1).sum)}, largest {sizes.max})")
    print("\nGroups per split:", u.groupby("split")["group_id"].nunique().reindex(order).to_dict())
    print("\nUsable images per split x type:")
    print(pd.crosstab(u["split"], u["type"], margins=True).reindex(order + ["All"]))
    print("\nType share per split (%):")
    print((pd.crosstab(u["split"], u["type"], normalize="index") * 100).round(1).reindex(order))
    print("\nCategory share per split (%):")
    print((pd.crosstab(u["split"], u["category_name"], normalize="index") * 100).round(1).reindex(order))
    print("\nCentre involved (grade 4) share per split (%):")
    print((u.groupby("split")["centre_involved"].mean() * 100).round(1).reindex(order).to_dict())
    print("\nSplit share of usable images (%):", (u["split"].value_counts(normalize=True) * 100).round(1).to_dict())


def main():
    df = load_labels()
    files = df["file"].tolist()
    df[["orig_width", "orig_height"]] = image_info(files)

    H = cached("phash64.npy", lambda fs: np.array([phash(os.path.join(IMG_DIR, f)) for f in fs]), files)
    F = cached("resnet50_imagenet_v2.npy", resnet50_embeddings, files)
    df["group_id"], links = find_groups(files, H, F)
    df["group_source"] = "phash_resnet50_neighbour"

    # Exact duplicates whose type labels disagree: true label unknown.
    df["status"] = "usable"
    for a, b in links.loc[links["rule"] == "exact_duplicate", ["file_a", "file_b"]].itertuples(index=False):
        ta, tb = (df.loc[df["file"] == f, "type"].item() for f in (a, b))
        if ta != tb:
            df.loc[df["file"].isin([a, b]), "status"] = "label_conflict"

    df["image_id"] = df["number"].map(lambda k: f"sus_{k:03d}")
    df["dataset"] = "sustech_sysu"
    df["raw_path"] = df["file"].map(lambda f: f"Data/Raw/SUSTech-SYSU/rawImages/{f}")
    df["category_name"] = df["category"].map(CATEGORY_NAMES)
    df["centre_involved"] = (df["grade"] == 4).astype(int)
    df["label_grade"] = df["type"]  # the task label

    df = make_splits(df)
    cols = ["image_id", "dataset", "raw_path", "group_id", "group_source", "status",
            "category", "category_name", "type", "grade", "centre_involved", "label_grade",
            "orig_width", "orig_height", "split"]
    df[cols].to_csv(OUT, index=False)
    links.to_csv(PAIRS_OUT, index=False)
    report(df)
    print(f"\nWrote {OUT}\nWrote {PAIRS_OUT} ({len(links)} links)")


if __name__ == "__main__":
    main()

"""Cross-dataset duplicate check.

Every image of every prepared dataset is compared with every image of every OTHER dataset, to find the
same photograph appearing in two datasets (e.g. a training image in one and a test image in another).
Duplicates within a dataset were handled by each dataset's own build script.

Features per image (as in the build scripts): MD5 of the file bytes, 64-bit pHash, ResNet50 (ImageNet,
torchvision IMAGENET1K_V2) embedding. Cached per dataset in Data/derived/<dataset>/; computed here for the
datasets whose build script did not compute them (Nuclear Cataract, SLID-E).

Because caches were built at slightly different decoding resolutions (some JPEGs decoded at reduced size
for speed), candidates are collected with a deliberately loose net and then reviewed by eye:
    MD5 equal, or pHash distance <= PHASH_MAX, or cosine >= COS_MIN.

Sanity check: SLIT-Net (reference) vs SLIT-Net v2 must rediscover the known 267 shared images.

Outputs: labels/cross_dataset_candidates.csv (all candidate pairs with both images' split and status).

Usage: python data_prep/cross_dataset_duplicates.py
"""
import hashlib
import os
from itertools import combinations

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dct

PHASH_MAX = 4
COS_MIN = 0.95
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DERIVED = os.path.join(ROOT, "Data", "derived")
OUT = os.path.join(ROOT, "labels", "cross_dataset_candidates.csv")

# name: (master CSV, path column, cache folder, cache id file, compute here?)
DATASETS = {
    "Nuclear_Cataract": ("nuclear_cataract_master.csv", "raw_path", "nuclear_cataract", "image_ids.txt", True),
    "SUSTech-SYSU": ("sustech_master.csv", "raw_path", "sustech", None, False),
    "SLID-E": ("SLID-E_master.csv", "split_path", "slid_e", "image_ids.txt", True),
    "Infectious_Keratitis": ("Infectious_Keratitis_master.csv", None, "infectious_keratitis", "image_ids.txt", False),
    "SLID": ("SLID_master.csv", "raw_path", "slid", None, False),
    "SLIT-Net": ("SLIT-Net_master.csv", None, "slit_net", "image_ids.txt", False),
    "SLIT-Net-v2": ("SLIT-Net-v2_master.csv", None, "slit_net_v2", "image_ids.txt", False),
    "SDCTrans": ("SDCTrans_master.csv", None, "sdctrans", "image_ids.txt", False),
}


def phash(im, hash_size=8, factor=4):
    """64-bit perceptual hash (same algorithm as imagehash.phash)."""
    a = np.asarray(im.convert("L").resize((hash_size * factor,) * 2, Image.LANCZOS), dtype=float)
    d = dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")[:hash_size, :hash_size]
    return (d > np.median(d)).flatten()


def compute_cache(df, path_col, cache):
    """MD5, pHash and ResNet50 embedding for every image of a dataset (JPEGs decoded at reduced size)."""
    import torch
    import torchvision

    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    model = torchvision.models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    tf = weights.transforms()
    md5s, hashes, feats, batch = [], [], [], []
    with torch.no_grad():
        for k, rel in enumerate(df[path_col]):
            path = os.path.join(ROOT, rel)
            md5s.append(hashlib.md5(open(path, "rb").read()).hexdigest())
            with Image.open(path) as im:
                im.draft("RGB", (800, 600))
                rgb = im.convert("RGB")
            hashes.append(phash(rgb))
            batch.append(tf(rgb))
            if len(batch) == 32 or k == len(df) - 1:
                feats.append(model(torch.stack(batch)).numpy())
                batch = []
    F = np.concatenate(feats)
    F /= np.linalg.norm(F, axis=1, keepdims=True)
    os.makedirs(cache, exist_ok=True)
    np.save(os.path.join(cache, "md5.npy"), np.array(md5s))
    np.save(os.path.join(cache, "phash64.npy"), np.array(hashes))
    np.save(os.path.join(cache, "resnet50_imagenet_v2.npy"), F)
    open(os.path.join(cache, "image_ids.txt"), "w").write("\n".join(df["image_id"]))


def load(name):
    csv, path_col, folder, id_file, compute = DATASETS[name]
    df = pd.read_csv(os.path.join(ROOT, "labels", csv))
    cache = os.path.join(DERIVED, folder)
    if compute and not os.path.exists(os.path.join(cache, "resnet50_imagenet_v2.npy")):
        print(f"computing features for {name} ({len(df)} images)")
        compute_cache(df, path_col, cache)
    if id_file:
        ids = open(os.path.join(cache, id_file)).read().split("\n")
    elif name == "SUSTech-SYSU":  # cache rows follow file order 1.jpg ... 712.jpg = master row order
        ids = df["image_id"].tolist()
    else:  # SLID: cache rows follow files.txt (1.png ... 2617.png) = master row order
        ids = df["image_id"].tolist()
    F = np.load(os.path.join(cache, "resnet50_imagenet_v2.npy"))
    H = np.load(os.path.join(cache, "phash64.npy"))
    md5_path = os.path.join(cache, "md5.npy")
    M = np.load(md5_path) if os.path.exists(md5_path) else np.array([""] * len(ids))
    assert len(ids) == len(F) == len(H) == len(M), name
    info = df.set_index("image_id").reindex(ids)
    split = info["split"].fillna("").astype(str).to_numpy()
    status = info["status"].fillna("").astype(str).to_numpy() if "status" in info else np.array([""] * len(ids))
    return {"ids": np.array(ids), "F": F, "H": H, "M": M, "split": split, "status": status}


def main():
    data = {name: load(name) for name in DATASETS}
    rows = []
    for a, b in combinations(DATASETS, 2):
        A, B = data[a], data[b]
        X = A["F"] @ B["F"].T
        D = (A["H"][:, None, :] != B["H"][None, :, :]).sum(-1)
        Mq = (A["M"][:, None] == B["M"][None, :]) & (A["M"][:, None] != "")
        for i, j in zip(*np.where(Mq | (D <= PHASH_MAX) | (X >= COS_MIN))):
            rows.append({
                "dataset_a": a, "image_a": A["ids"][i], "split_a": A["split"][i], "status_a": A["status"][i],
                "dataset_b": b, "image_b": B["ids"][j], "split_b": B["split"][j], "status_b": B["status"][j],
                "md5_equal": bool(Mq[i, j]), "phash_distance": int(D[i, j]), "cosine": round(float(X[i, j]), 4),
            })
        print(f"{a} vs {b}: {len(A['ids'])} x {len(B['ids'])} images, max cosine {X.max:.3f}, "
              f"min pHash {D.min}, MD5 matches {int(Mq.sum)}, candidates "
              f"{sum(1 for r in rows if r['dataset_a'] == a and r['dataset_b'] == b)}")
    cand = pd.DataFrame(rows).sort_values("cosine", ascending=False)
    cand.to_csv(OUT, index=False)
    ref = cand[(cand.dataset_a == "SLIT-Net") & (cand.dataset_b == "SLIT-Net-v2")]
    print(f"\nSanity check: SLIT-Net vs SLIT-Net-v2 candidates {len(ref)} "
          f"(distinct v2 images {ref.image_b.nunique}; expected 267 shared images)")
    other = cand[~((cand.dataset_a == "SLIT-Net") & (cand.dataset_b == "SLIT-Net-v2"))]
    print(f"All other cross-dataset candidates: {len(other)}")
    if len(other):
        print(other.groupby(["dataset_a", "dataset_b"]).size().to_string())
        print(other.head(40).to_string(index=False))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()

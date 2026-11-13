"""Catalogue SLIT-Net v2 as an external test set, extract its images and match it against SLIT-Net
.

Source: Data/Raw/SLIT-Net-v2/slitnetv2.zip (Duke Box), layout
    Datasets/Segmentation/<White_Light|Blue_Light>/Images/V###.tif   (1-based eye numbers)
    Datasets/Segmentation/<...>/annotations.mat, idxs.mat; Datasets/BCVA_Prediction/*.mat  (not used)

SLIT-Net v2 overlaps SLIT-Net (same lab): e.g. v2 white V001 is byte-identical to SLIT-Net white V000.
In v2, white and blue images are numbered INDEPENDENTLY: v2 white V001 and v2 blue V001 are different eyes
(SLIT-Net eye V000 is v2 white V001 but v2 blue V057). So the v2 number identifies an image within one lighting,
not an eye across lightings.
Role: external test set (never trained on), so not split: split = "external_test".

  - Images converted losslessly to PNG: Data/Split/SLIT-Net-v2/test/<white|blue>/SLIT-Net-v2_<eye>_<light>.png.
  - Match against SLIT-Net (MD5, pHash 0 or cosine >= 0.98): columns `slitnet_match` (matching SLIT-Net image),
    `slitnet_match_rule`, `in_SLIT-Net`.
  - group_id: an image matching SLIT-Net takes that SLIT-Net eye's group (so the white and blue photo of one eye
    are together, and v2 shares groups with SLIT-Net); every other image starts in its own group
    `SLIT-Net-v2_<light>_<number>`. Duplicates between v2 images (same rule) are flagged and their groups merged.
  - eval_include: SLIT-Net v2 is THE external test set (SLIT-Net itself is not evaluated
    separately). Each picture counts once: of each duplicate pair inside v2, the lower v2 number is kept.
  - MD5 of the original TIF bytes, pHash and ResNet50 embeddings cached in Data/derived/slit_net_v2/.

Usage: python data_prep/build_slitnetv2_master.py
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

DUP_COSINE = 0.98
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "Data", "Raw", "SLIT-Net-v2", "slitnetv2.zip")
LIGHTS = {"White_Light": ("white", "diffuse_white"), "Blue_Light": ("blue", "cobalt_blue")}
CACHE = os.path.join(ROOT, "Data", "derived", "slit_net_v2")
SLITNET_CACHE = os.path.join(ROOT, "Data", "derived", "slit_net")
OUT_DIR = os.path.join(ROOT, "Data", "Split", "SLIT-Net-v2", "test")
OUT = os.path.join(ROOT, "labels", "SLIT-Net-v2_master.csv")
LINKS_OUT = os.path.join(ROOT, "labels", "SLIT-Net-v2_duplicate_links.csv")


def phash(im, hash_size=8, factor=4):
    """64-bit perceptual hash (same algorithm as imagehash.phash)."""
    a = np.asarray(im.convert("L").resize((hash_size * factor,) * 2, Image.LANCZOS), dtype=float)
    d = dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")[:hash_size, :hash_size]
    return (d > np.median(d)).flatten()


def main():
    import torch
    import torchvision

    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    model = torchvision.models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    tf = weights.transforms()

    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)  # rebuild from the zip so folders never drift
    for folder, _ in LIGHTS.values():
        os.makedirs(os.path.join(OUT_DIR, folder))

    rows, md5s, hashes, feats = [], [], [], []
    with torch.no_grad(), zipfile.ZipFile(RAW) as zf:
        members = sorted(m for m in zf.namelist() if m.endswith(".tif") and "/Segmentation/" in m)
        for member in members:
            light_dir = member.split("/")[2]
            folder, illumination = LIGHTS[light_dir]
            eye = re.search(r"(V\d{3})\.tif$", member).group(1)
            image_id = f"SLIT-Net-v2_{eye}_{folder}"
            data = zf.read(member)
            md5s.append(hashlib.md5(data).hexdigest())
            with Image.open(io.BytesIO(data)) as im:
                im.load()
                rgb = im.convert("RGB")
            split_path = f"Data/Split/SLIT-Net-v2/test/{folder}/{image_id}.png"
            rgb.save(os.path.join(ROOT, split_path), compress_level=1)  # lossless
            hashes.append(phash(rgb))
            feats.append(model(tf(rgb).unsqueeze(0)).numpy()[0])
            rows.append({
                "image_id": image_id, "dataset": "SLIT-Net-v2", "raw_path": f"Data/Raw/SLIT-Net-v2/slitnetv2.zip/{member}",
                "split_path": split_path, "v2_number": eye, "group_id": f"SLIT-Net-v2_{folder}_{eye}",
                "group_source": "own_image", "illumination": illumination,
                "orig_width": rgb.size[0], "orig_height": rgb.size[1], "split": "external_test",
            })
    df = pd.DataFrame(rows)
    assert df["image_id"].is_unique
    F = np.array(feats)
    F /= np.linalg.norm(F, axis=1, keepdims=True)
    H = np.array(hashes)
    md5 = np.array(md5s)
    df["md5"] = md5
    os.makedirs(CACHE, exist_ok=True)
    for k, arr in (("md5", md5), ("phash64", H), ("resnet50_imagenet_v2", F)):
        np.save(os.path.join(CACHE, f"{k}.npy"), arr)
    open(os.path.join(CACHE, "image_ids.txt"), "w").write("\n".join(df["image_id"]))

    links = []
    # Duplicates between any two v2 images (all pairs; a white and a blue photo of one eye differ too much
    # to reach these thresholds, so in practice only same-lighting copies match).
    C = F @ F.T
    D = (H[:, None, :] != H[None, :, :]).sum(-1)
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if md5[i] == md5[j] or D[i, j] == 0 or C[i, j] >= DUP_COSINE:
                rule = "md5" if md5[i] == md5[j] else ("phash0" if D[i, j] == 0 else "cosine")
                links.append((df.at[i, "image_id"], df.at[j, "image_id"], "within_v2", rule, int(D[i, j]), round(float(C[i, j]), 4)))

    # Matches with SLIT-Net.
    s_ids = np.array(open(os.path.join(SLITNET_CACHE, "image_ids.txt")).read().split("\n"))
    sF = np.load(os.path.join(SLITNET_CACHE, "resnet50_imagenet_v2.npy"))
    sH = np.load(os.path.join(SLITNET_CACHE, "phash64.npy"))
    sM = np.load(os.path.join(SLITNET_CACHE, "md5.npy"))
    X = F @ sF.T
    XD = (H[:, None, :] != sH[None, :, :]).sum(-1)
    XM = md5[:, None] == sM[None, :]
    match, match_rule = [], []
    for i in range(len(df)):
        hits = np.where(XM[i] | (XD[i] == 0) | (X[i] >= DUP_COSINE))[0]
        for j in hits:
            rule = "md5" if XM[i, j] else ("phash0" if XD[i, j] == 0 else "cosine")
            links.append((df.at[i, "image_id"], s_ids[j], "with_SLIT-Net", rule, int(XD[i, j]), round(float(X[i, j]), 4)))
        best = hits[np.argmax(X[i, hits])] if len(hits) else None
        match.append(s_ids[best] if best is not None else "")
        match_rule.append(("md5" if XM[i, best] else ("phash0" if XD[i, best] == 0 else "cosine")) if best is not None else "")
    df["slitnet_match"], df["slitnet_match_rule"] = match, match_rule
    df["slitnet_best_cosine"] = X.max(axis=1).round(4)
    links = pd.DataFrame(links, columns=["image_a", "image_b", "scope", "rule", "phash_distance", "cosine"])

    within = links[links["scope"] == "within_v2"]
    flagged = set(within["image_a"]) | set(within["image_b"])
    df["status"] = np.where(df["image_id"].isin(flagged), "duplicate_other_image", "usable")
    df["in_SLIT-Net"] = df["slitnet_match"] != ""
    # Images that are SLIT-Net images join that SLIT-Net eye's group.
    s_group = dict(pd.read_csv(os.path.join(ROOT, "labels", "SLIT-Net_master.csv"))[["image_id", "group_id"]].values)
    df.loc[df["in_SLIT-Net"], "group_id"] = df.loc[df["in_SLIT-Net"], "slitnet_match"].map(s_group)
    df.loc[df["in_SLIT-Net"], "group_source"] = "SLIT-Net_eye"
    # Duplicates inside v2: merge their groups.
    for a, b in within[["image_a", "image_b"]].itertuples(index=False):
        group_of = dict(zip(df["image_id"], df["group_id"]))
        ga, gb = sorted((group_of[a], group_of[b]))
        if ga != gb:
            df.loc[df["group_id"] == gb, "group_id"] = ga
    # Evaluation set: every image once. Of each duplicate pair, keep the lower v2 number.
    df["eval_include"] = True
    for a, b in within[["image_a", "image_b"]].itertuples(index=False):
        df.loc[df["image_id"] == max(a, b), "eval_include"] = False

    df.to_csv(OUT, index=False)
    links.to_csv(LINKS_OUT, index=False)

    print("Images:", len(df), " by illumination:", df["illumination"].value_counts().to_dict(),
          " groups:", df["group_id"].nunique(), " (groups shared with SLIT-Net:",
          df.loc[df["in_SLIT-Net"], "group_id"].nunique(), ")")
    print("Image sizes:", df.groupby(["illumination", "orig_width", "orig_height"]).size().to_dict())
    print("Duplicates between v2 images:", len(within))
    print(within.to_string(index=False) if len(within) else "")
    print("\nImages matching SLIT-Net, by rule and illumination:")
    print(pd.crosstab(df["slitnet_match_rule"].replace("", "no match"), df["illumination"]))
    print("Images new vs SLIT-Net:", df.loc[~df["in_SLIT-Net"], "illumination"].value_counts().to_dict())
    print("Evaluation set (eval_include):", int(df["eval_include"].sum()), " by illumination:",
          df.loc[df["eval_include"], "illumination"].value_counts().to_dict(),
          " excluded:", df.loc[~df["eval_include"], "image_id"].tolist())
    m = df[df["in_SLIT-Net"]]
    s_eye = m["slitnet_match"].str.extract(r"_(V\d{3})_")[0]
    pairs = pd.DataFrame({"s_eye": s_eye, "light": m["illumination"], "v2": m["v2_number"]})
    w = pairs[pairs.light == "diffuse_white"].groupby("s_eye").v2.apply(set)
    b = pairs[pairs.light == "cobalt_blue"].groupby("s_eye").v2.apply(set)
    both = sorted(set(w.index) & set(b.index))
    print("SLIT-Net eyes with white and blue in v2:", len(both),
          " of which under the same v2 number:", sum(bool(w[e] & b[e]) for e in both))
    same_light = (m["illumination"].str[:4] == m["slitnet_match"].map(lambda s: "diff" if s.endswith("white") else "coba"))
    print("Matches with the same lighting:", int(same_light.sum()), "/", len(m))
    print("Unmatched images: best cosine to SLIT-Net max %.3f" % df.loc[df["slitnet_match"] == "", "slitnet_best_cosine"].max())
    print(f"\nWrote {OUT_DIR}\nWrote {OUT}\nWrote {LINKS_OUT}")


if __name__ == "__main__":
    main()

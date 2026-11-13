"""Catalogue SDCTrans as an external test set and extract its photographs.

Source: `Data/Raw/SDCTrans/SDCTrans Data.zip` (Duke Box), which holds two zips, unpacked once into Data/Raw/SDCTrans/:
White_light.zip and Blue+scs.zip. Layout inside:
    <Lighting>/<patient>/img/<patient>_<OD|OS>_D<day>_<B|W|ScS><magnification><letter>.jpg
    <Lighting>/<patient>/gt/...  (segmentation masks and ImageJ ROI sets; not used)

Role: external test set for the quality module (same lab as SLIT-Net, unseen hospitals). Not split:
every image gets split = "external_test".

  - Only the photographs (`img/`) are extracted, unchanged (JPG bytes copied as is), to
    Data/Split/SDCTrans/test/<white|blue|scs>/<image_id>.jpg.
  - Illumination from the file name: W = diffuse white, B = cobalt blue, ScS = sclerotic scatter.
  - group_id = patient (both eyes and all follow-up days of a patient together).
  - Duplicates between different patients (MD5, pHash distance 0 or ResNet50 cosine >= 0.98) are
    flagged and their patients merged into one group.
  - Overlap with SLIT-Net (same lab) is checked with the same rule against SLIT-Net's cached features.
  - MD5, pHash and ResNet50 embeddings are cached in Data/derived/sdctrans/.

Usage: python data_prep/build_sdctrans_master.py
"""
import hashlib
import io
import os
import re
import shutil
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from scipy.fft import dct

DUP_COSINE = 0.98
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "Data", "Raw", "SDCTrans")
ZIPS = ["White_light.zip", "Blue+scs.zip"]
NAME = re.compile(r"^(([A-Z]+)_(\d+))_(OD|OS)_D(\d+)_([A-Za-z]+?)([\d.]+)([a-z]*)\.jpg$")
LIGHT = {"W": ("white", "diffuse_white"), "B": ("blue", "cobalt_blue"), "ScS": ("scs", "sclerotic_scatter")}
CACHE = os.path.join(ROOT, "Data", "derived", "sdctrans")
SLITNET_CACHE = os.path.join(ROOT, "Data", "derived", "slit_net")
OUT_DIR = os.path.join(ROOT, "Data", "Split", "SDCTrans", "test")
OUT = os.path.join(ROOT, "labels", "SDCTrans_master.csv")
LINKS_OUT = os.path.join(ROOT, "labels", "SDCTrans_duplicate_links.csv")


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
        shutil.rmtree(OUT_DIR)  # rebuild from the zips so folders never drift
    for folder, _ in LIGHT.values():
        os.makedirs(os.path.join(OUT_DIR, folder))

    rows, md5s, hashes, feats = [], [], [], []
    with torch.no_grad():
        for z in ZIPS:
            with zipfile.ZipFile(os.path.join(RAW, z)) as zf:
                for member in sorted(m for m in zf.namelist() if "/img/" in m and not m.endswith("/")):
                    top, patient_folder, _, fname = member.split("/")
                    m = NAME.match(fname)
                    pid, site, _, eye, day, light, mag, letter = m.groups()
                    assert pid == patient_folder, member
                    folder, illumination = LIGHT[light]
                    image_id = "SDCTrans_" + fname[:-4]
                    data = zf.read(member)
                    md5s.append(hashlib.md5(data).hexdigest())
                    truncated = False
                    try:
                        with Image.open(io.BytesIO(data)) as im:
                            size = im.size
                            rgb = im.convert("RGB")
                    except OSError:  # JPEG ends early: decode what is there, flag the file
                        truncated = True
                        ImageFile.LOAD_TRUNCATED_IMAGES = True
                        with Image.open(io.BytesIO(data)) as im:
                            size = im.size
                            rgb = im.convert("RGB")
                        ImageFile.LOAD_TRUNCATED_IMAGES = False
                    split_path = f"Data/Split/SDCTrans/test/{folder}/{image_id}.jpg"
                    with open(os.path.join(ROOT, split_path), "wb") as f:
                        f.write(data)  # original JPG bytes, unchanged
                    hashes.append(phash(rgb))
                    feats.append(model(tf(rgb).unsqueeze(0)).numpy()[0])
                    rows.append({
                        "image_id": image_id, "dataset": "SDCTrans",
                        "raw_path": f"Data/Raw/SDCTrans/{z}/{member}", "split_path": split_path,
                        "patient_id": f"SDCTrans_{pid}", "site": site, "eye": "RE" if eye == "OD" else "LE",
                        "day": int(day), "magnification": mag, "shot": letter,
                        "group_id": f"SDCTrans_{pid}", "group_source": "patient_id",
                        "illumination": illumination, "orig_width": size[0], "orig_height": size[1],
                        "truncated_jpeg": truncated, "split": "external_test",
                    })
    df = pd.DataFrame(rows)
    assert df["image_id"].is_unique
    F = np.array(feats)
    F /= np.linalg.norm(F, axis=1, keepdims=True)
    H = np.array(hashes)
    df["md5"] = md5s
    os.makedirs(CACHE, exist_ok=True)
    for k, arr in (("md5", np.array(md5s)), ("phash64", H), ("resnet50_imagenet_v2", F)):
        np.save(os.path.join(CACHE, f"{k}.npy"), arr)
    open(os.path.join(CACHE, "image_ids.txt"), "w").write("\n".join(df["image_id"]))

    # Duplicates between different patients within SDCTrans.
    C = F @ F.T
    D = (H[:, None, :] != H[None, :, :]).sum(-1)
    pid = df["patient_id"].to_numpy()
    md5 = np.array(md5s)
    links = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if pid[i] != pid[j] and (md5[i] == md5[j] or D[i, j] == 0 or C[i, j] >= DUP_COSINE):
                links.append((df.at[i, "image_id"], df.at[j, "image_id"], "within_SDCTrans", int(D[i, j]), round(float(C[i, j]), 4)))

    # Overlap with SLIT-Net (same lab).
    s_ids = open(os.path.join(SLITNET_CACHE, "image_ids.txt")).read().split("\n")
    sF = np.load(os.path.join(SLITNET_CACHE, "resnet50_imagenet_v2.npy"))
    sH = np.load(os.path.join(SLITNET_CACHE, "phash64.npy"))
    sM = np.load(os.path.join(SLITNET_CACHE, "md5.npy"))
    X = F @ sF.T
    XD = (H[:, None, :] != sH[None, :, :]).sum(-1)
    for i, j in zip(*np.where((X >= DUP_COSINE) | (XD == 0) | (md5[:, None] == sM[None, :]))):
        links.append((df.at[i, "image_id"], s_ids[j], "with_SLIT-Net", int(XD[i, j]), round(float(X[i, j]), 4)))
    links = pd.DataFrame(links, columns=["image_a", "image_b", "rule", "phash_distance", "cosine"])

    within = links[links["rule"] == "within_SDCTrans"]
    flagged = set(within["image_a"]) | set(within["image_b"])
    df["status"] = np.where(df["image_id"].isin(flagged), "duplicate_other_patient", "usable")
    df.loc[df["truncated_jpeg"], "status"] = "truncated_jpeg"
    group_of = dict(zip(df["image_id"], df["group_id"]))
    for a, b in within[["image_a", "image_b"]].itertuples(index=False):
        ga, gb = sorted((group_of[a], group_of[b]))
        df.loc[df["group_id"] == gb, "group_id"] = ga
        group_of = dict(zip(df["image_id"], df["group_id"]))
    df["in_SLIT-Net"] = df["image_id"].isin(set(links.loc[links["rule"] == "with_SLIT-Net", "image_a"]))

    df.to_csv(OUT, index=False)
    links.to_csv(LINKS_OUT, index=False)

    print("Images:", len(df), " by illumination:", df["illumination"].value_counts().to_dict())
    print("Patients:", df["patient_id"].nunique(), " groups:", df["group_id"].nunique(),
          " by site:", df.groupby("site")["patient_id"].nunique().to_dict())
    print("Image sizes:", df.groupby(["orig_width", "orig_height"]).size().to_dict())
    print("Truncated JPEGs:", df.loc[df["truncated_jpeg"], "image_id"].tolist())
    print("Duplicates between SDCTrans patients:", len(within))
    print(within.to_string(index=False) if len(within) else "")
    xs = links[links["rule"] == "with_SLIT-Net"]
    print("Matches with SLIT-Net:", len(xs), " SDCTrans images involved:", int(df["in_SLIT-Net"].sum()))
    print(xs.head(30).to_string(index=False) if len(xs) else "")
    others = C[np.triu_indices(len(df), 1)][pid[np.triu_indices(len(df), 1)[0]] != pid[np.triu_indices(len(df), 1)[1]]]
    print("Highest cosine between different patients: %.3f; highest SDCTrans-SLIT-Net cosine: %.3f" % (others.max(), X.max()))
    print(f"\nWrote {OUT_DIR}\nWrote {OUT}\nWrote {LINKS_OUT}")


if __name__ == "__main__":
    main()

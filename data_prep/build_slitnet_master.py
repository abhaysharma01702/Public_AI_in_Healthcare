"""Catalogue SLIT-Net as an external test set and extract its images.

SLIT-Net (Harvard Dataverse doi:10.7910/DVN/BWJJRP): 133 eyes `V000 … V132`, each photographed once under
diffuse white light and once under cobalt-blue light with fluorescein, as 8-bit RGB TIFs
(White_Light_0-66.zip, White_Light_67-132.zip, Blue_Light.zip).

Role: external test set for the quality module (unseen hospitals). Never trained on, so not split:
every image gets split = "external_test".

  - Images are converted losslessly to PNG: Data/Split/SLIT-Net/test/<white|blue>/<image_id>.png.
    The zips stay the untouched raw copy.
  - Illumination comes from the zip / folder name.
  - group_id = eye (V number): the white and blue photo of one eye belong together.
  - Duplicate check between different eyes: pHash distance 0 or ResNet50 cosine >= 0.98. Duplicates are
    flagged (status `duplicate_other_eye`, not removed) and their eyes merged into one group.
  - MD5 of the original TIF bytes, pHash and ResNet50 embeddings are cached in Data/derived/slit_net/
    for the later cross-dataset duplicate check.

Usage: python data_prep/build_slitnet_master.py
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
RAW = os.path.join(ROOT, "Data", "Raw", "SLIT-Net")
ZIPS = {"White_Light_0-66.zip": "white", "White_Light_67-132.zip": "white", "Blue_Light.zip": "blue"}
ILLUMINATION = {"white": "diffuse_white", "blue": "cobalt_blue"}
CACHE = os.path.join(ROOT, "Data", "derived", "slit_net")
OUT_DIR = os.path.join(ROOT, "Data", "Split", "SLIT-Net", "test")
OUT = os.path.join(ROOT, "labels", "SLIT-Net_master.csv")
FEATURES = ("md5", "phash64", "resnet50_imagenet_v2")


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
    for light in ("white", "blue"):
        os.makedirs(os.path.join(OUT_DIR, light))

    rows, md5s, hashes, feats = [], [], [], []
    with torch.no_grad():
        for z, light in ZIPS.items():
            with zipfile.ZipFile(os.path.join(RAW, z)) as zf:
                for member in sorted(m for m in zf.namelist() if m.lower().endswith(".tif")):
                    eye = re.search(r"(V\d{3})\.tif$", member).group(1)
                    image_id = f"SLIT-Net_{eye}_{light}"
                    data = zf.read(member)
                    md5s.append(hashlib.md5(data).hexdigest())
                    with Image.open(io.BytesIO(data)) as im:
                        im.load()
                        rgb = im.convert("RGB")
                    split_path = f"Data/Split/SLIT-Net/test/{light}/{image_id}.png"
                    rgb.save(os.path.join(ROOT, split_path), compress_level=1)  # lossless
                    hashes.append(phash(rgb))
                    feats.append(model(tf(rgb).unsqueeze(0)).numpy()[0])
                    rows.append({
                        "image_id": image_id, "dataset": "SLIT-Net",
                        "raw_path": f"Data/Raw/SLIT-Net/{z}/{member}", "split_path": split_path,
                        "eye_id": eye, "group_id": f"SLIT-Net_{eye}", "group_source": "eye_number",
                        "illumination": ILLUMINATION[light], "orig_width": rgb.size[0], "orig_height": rgb.size[1],
                        "split": "external_test",
                    })
    df = pd.DataFrame(rows)
    F = np.array(feats)
    F /= np.linalg.norm(F, axis=1, keepdims=True)
    H = np.array(hashes)
    df["md5"] = md5s
    os.makedirs(CACHE, exist_ok=True)
    for k, arr in zip(FEATURES, (np.array(md5s), H, F)):
        np.save(os.path.join(CACHE, f"{k}.npy"), arr)
    open(os.path.join(CACHE, "image_ids.txt"), "w").write("\n".join(df["image_id"]))

    # Duplicates between different eyes.
    C = F @ F.T
    D = (H[:, None, :] != H[None, :, :]).sum(-1)
    eyes = df["eye_id"].to_numpy()
    dups = [(df.at[i, "image_id"], df.at[j, "image_id"], int(D[i, j]), round(float(C[i, j]), 4))
            for i in range(len(df)) for j in range(i + 1, len(df))
            if eyes[i] != eyes[j] and (md5s[i] == md5s[j] or D[i, j] == 0 or C[i, j] >= DUP_COSINE)]
    flagged = {a for a, b, *_ in dups} | {b for a, b, *_ in dups}
    df["status"] = np.where(df["image_id"].isin(flagged), "duplicate_other_eye", "usable")
    # Eyes sharing an identical photo are probably the same eye entered twice: one group.
    eye_of = dict(zip(df["image_id"], df["eye_id"]))
    for a, b, *_ in dups:
        ga, gb = sorted((f"SLIT-Net_{eye_of[a]}", f"SLIT-Net_{eye_of[b]}"))
        df.loc[df["group_id"] == gb, "group_id"] = ga

    assert df["image_id"].is_unique
    counts = df.groupby("illumination")["eye_id"].nunique()
    df.to_csv(OUT, index=False)

    print("Images:", len(df), " by illumination:", df["illumination"].value_counts().to_dict())
    print("Eyes per illumination:", counts.to_dict(),
          " eyes with both:", int((df.groupby("eye_id")["illumination"].nunique() == 2).sum()))
    print("Image sizes:", df.groupby(["orig_width", "orig_height"]).size().to_dict())
    print("Duplicates between different eyes:", dups if dups else "none")
    same_eye = [round(float(C[i, j]), 3) for i in range(len(df)) for j in range(i + 1, len(df)) if eyes[i] == eyes[j]]
    print("White-vs-blue cosine within an eye: median %.3f, max %.3f" % (np.median(same_eye), np.max(same_eye)))
    others = C[np.triu_indices(len(df), 1)][eyes[np.triu_indices(len(df), 1)[0]] != eyes[np.triu_indices(len(df), 1)[1]]]
    print("Highest cosine between different eyes: %.3f" % others.max())
    print(f"\nWrote {OUT_DIR}\nWrote {OUT}")


if __name__ == "__main__":
    main()

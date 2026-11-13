"""Frozen VisionFM slit-lamp features for every image of a grading data folder.
Reads <data_root>/labels.csv (all rows: clean + degraded, all splits) and writes <out_dir>/features_visionfm.npz
(sample_id, feats [N, 1536] float32), in labels.csv order, plus copies of labels.csv and quality_d3.csv, so <out_dir>
holds everything the heads need (train_frozen_head.py --data; on Ada the images stay on the node scratch).
<out_dir> defaults to <data_root>. A finished folder is skipped unless --force.
Preprocessing: grading/visionfm.py (resize to 224 x 224, PIL bilinear, VisionFM SlitLamp mean / SD); no augmentation
(the encoder is frozen, so the features are computed once, as Reti-Pioneer does).
Usage: python grading/extract_features.py --data_root "Data/Grading Data/nuclear_cataract" [--out_dir D] [--limit 64]
"""
import argparse
import hashlib
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "data_prep"))
import visionfm as V  # noqa: E402
from run_log import RunLog  # noqa: E402


class Images(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            x = np.array(im.convert("RGB").resize((V.SIZE, V.SIZE), Image.BILINEAR), dtype=np.uint8)
        return torch.from_numpy(x).permute(2, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N rows, written to a _smoke file")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out_dir", default=None)
    a = ap.parse_args()
    a.out_dir = a.out_dir or a.data_root
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, "features_visionfm" + ("_smoke" if a.limit else "") + ".npz")
    if os.path.exists(out) and not a.force:
        print("exists, skipped:", out)
        return
    torch.set_num_threads(a.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lab = pd.read_csv(os.path.join(a.data_root, "labels.csv"))
    if a.limit:
        lab = lab.head(a.limit)
    with RunLog("grading/extract_features.py", notes=f"VisionFM features {a.data_root}") as log:
        model = V.load_encoder().to(device)
        dl = torch.utils.data.DataLoader(Images([os.path.join(a.data_root, p) for p in lab.path]),
                                         batch_size=a.batch, num_workers=a.num_workers)
        mean, std = V.MEAN.to(device), V.STD.to(device)
        feats, t0 = [], time.time()
        with torch.inference_mode():
            for b, x in enumerate(dl):
                feats.append(model((x.to(device).float() / 255 - mean) / std).float().cpu().numpy())
                if b % 20 == 0:
                    n = sum(len(f) for f in feats)
                    print(f"{n}/{len(lab)}  {n / (time.time - t0):.1f} img/s", flush=True)
        if device.type == "cuda":
            print("gpu", torch.cuda.get_device_name(0))
        feats = np.concatenate(feats)
        assert len(feats) == len(lab) and np.isfinite(feats).all()
        np.savez(out, sample_id=lab.sample_id.to_numpy(str), feats=feats)
        if os.path.abspath(a.out_dir) != os.path.abspath(a.data_root):
            for f in ("labels.csv", "quality_d3.csv"):
                shutil.copy2(os.path.join(a.data_root, f), os.path.join(a.out_dir, f))
        md5 = hashlib.md5(open(out, "rb").read()).hexdigest()
        sec = time.time() - t0
        print(f"{out}: {feats.shape}, {sec:.0f} s, MD5 {md5}")
        log.add(out=out, n=len(feats), seconds=round(sec), md5=md5, device=str(device), torch=torch.__version__,
                weights_md5=hashlib.md5(open(V.WEIGHTS, "rb").read()).hexdigest())


if __name__ == "__main__":
    main()

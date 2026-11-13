"""Quality vector for every grading image, computed ONCE with the chosen quality module (section 6 point 3; D3 Swin-T). Uses the quality code's own loading, normalisation and prediction (quality/qm), so the numbers are what the
quality module would output in evaluate.py.
Output: <data_root>/quality_d3.csv = sample_id + q_<defect> (9 sigmoid outputs), and its MD5 printed / logged.
Grading runs read that file; they never rerun the quality module.
Usage: python grading/predict_quality.py --data_root "Data/Grading Data/sustech_sysu" --run_dir runs/quality/swin_t_ada
"""
import argparse
import hashlib
import os
import sys

import pandas as pd
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "quality"))
sys.path.insert(0, os.path.join(ROOT, "data_prep"))
from qm import VALUES  # noqa: E402
from qm import data as QD  # noqa: E402
from qm import engine as E  # noqa: E402
from qm import model as QM  # noqa: E402
from run_log import RunLog  # noqa: E402

EXPECTED_MD5 = {"swin_t_ada": "d8ef5199d1f0e33af5926043891470db"}   #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out_name", default="quality_d3.csv")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    with RunLog("grading/predict_quality.py", notes=f"quality vectors {a.data_root} with {a.run_dir}") as log:
        ck_path = os.path.join(a.run_dir, "best.pt")
        md5 = hashlib.md5(open(ck_path, "rb").read()).hexdigest()
        want = EXPECTED_MD5.get(os.path.basename(os.path.normpath(a.run_dir)))
        if want and md5 != want:
            raise SystemExit(f"{ck_path} MD5 {md5} != expected {want}")
        with open(os.path.join(a.run_dir, "config.yaml"), encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        amp = bool(cfg["optim"]["amp"]) and device.type == "cuda"
        mc = cfg["model"]
        model = QM.build_model(mc["name"], len(VALUES), pretrained=False, dropout=mc["dropout"]).to(device)
        model.load_state_dict(torch.load(ck_path, map_location=device, weights_only=False)["model"])
        df = pd.read_csv(os.path.join(a.data_root, "labels.csv"))
        ds = QD.QualityDataset(df, a.data_root, cfg["data"]["height"], cfg["data"]["width"], train=False)
        loader = QD.make_loader(ds, QD.sequential_batches(len(df), a.batch), a.num_workers, device.type == "cuda")
        p, _ = E.predict(model, loader, E.Batch(device, cfg["train"]["channels_last"]), device, amp)
        out = df[["sample_id"]].copy()
        for k, v in enumerate(VALUES):
            out[f"q_{v}"] = p[:, k]
        path = os.path.join(a.data_root, a.out_name)
        out.to_csv(path, index=False, float_format="%.9g")
        csv_md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
        print(f"{path}: {len(out)} rows, MD5 {csv_md5}, checkpoint MD5 {md5}, device {device}")
        log.add(out=path, n=len(out), md5=csv_md5, checkpoint_md5=md5, device=str(device))


if __name__ == "__main__":
    main()

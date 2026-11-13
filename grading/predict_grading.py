"""Predict an extra grading test set with already-trained runs. No training, no model choice.

For every run folder <runs>/<cond>_seed<seed> with best.pt: loads the checkpoint and run_info.json (condition, head,
label, input size, the training quality mean / SD), predicts every image of <data_root>/labels.csv (split test) exactly
as train_grading.py does at test time (same dataset class, normalisation, AMP on GPU), including C-shuffled for C /
C_oracle (quality vectors permuted across test images, seed 14), and writes <run>/predictions_<name>.csv with the
same columns as predictions_test.csv.
Check: the clean rows are the same images as in the original test set, so their predictions are compared with
predictions_test.csv (printed; expected to agree up to GPU / batch rounding).
Usage: python grading/predict_grading.py --runs runs/grading/sustech_sysu_category --data_root "Data/Grading Data/sustech_sysu_test_seen" --name test_seen
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_grading import ROOT, GradingDataset, GradingNet, quality_vectors, run_epoch  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "data_prep"))
from run_log import RunLog  # noqa: E402


def predict_run(run_dir, lab, a, device, amp):
    info = json.load(open(os.path.join(run_dir, "run_info.json")))
    cond, n_classes = info["cond"], info["n_classes"]
    te = lab.copy()
    if info.get("label", "grade") == "category":   # / 3: the general-pattern label
        cat = pd.read_csv(a.master).set_index("image_id").category
        assert te.image_id.isin(cat.index).all()
        te["grade"] = te.image_id.map(cat).astype(int)
    assert int(te.grade.max()) < n_classes
    model = GradingNet(n_classes, pretrained=False, head=info.get("head", "concat"),
                       grid=(info["height"] // 32, info["width"] // 32), arch=info.get("arch", "densenet121"),
                       size=(info["height"], info["width"])).to(device)
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    q_mu, q_sd = np.array(info["q_mean"]), np.array(info["q_sd"])
    qv0 = (quality_vectors(te, cond, a.data_root, info.get("q_file", "quality_d3.csv")) - q_mu) / q_sd
    variants = [("", qv0)]
    if cond in ("C", "C_oracle"):
        variants.append(("shuffled_", qv0[np.random.default_rng(14).permutation(len(qv0))]))
    out = te[["sample_id", "image_id", "group_id", "kind", "defect", "strength", "grade"]].copy()
    for prefix, qv in variants:
        ds = GradingDataset(te, a.data_root, qv, info["height"], info["width"], False, info["seed"])
        dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=a.num_workers,
                                         pin_memory=amp)
        _, lo, _, idx = run_epoch(model, dl, device, amp)
        assert (idx == np.arange(len(te))).all()
        pr = torch.softmax(lo, 1).numpy()
        out[f"{prefix}pred"] = pr.argmax(1)
        for c in range(n_classes):
            out[f"{prefix}p{c}"] = pr[:, c]
    out.to_csv(os.path.join(run_dir, f"predictions_{a.name}.csv"), index=False, float_format="%.6g")
    # check: clean rows = the original test images
    ref = pd.read_csv(os.path.join(run_dir, "predictions_test.csv")).set_index("sample_id")
    c = out[out.kind == "clean"]
    same = float((c.pred.to_numpy() == ref.loc[c.sample_id, "pred"].to_numpy()).mean())
    dp = float(np.abs(c[[f"p{k}" for k in range(n_classes)]].to_numpy()
                      - ref.loc[c.sample_id, [f"p{k}" for k in range(n_classes)]].to_numpy()).max())
    return dict(run=os.path.basename(run_dir), n=len(out), clean_same_pred=same, clean_max_dp=dp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="parent folders of <cond>_seed<seed> runs")
    ap.add_argument("--data_root", required=True, help="extra test set folder (labels.csv, quality_d3.csv, test)")
    ap.add_argument("--name", required=True, help="output predictions_<name>.csv in each run folder")
    ap.add_argument("--master", default=os.path.join(ROOT, "labels", "sustech_master.csv"))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: N images per kind")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    lab = pd.read_csv(os.path.join(a.data_root, "labels.csv"))
    lab = lab[lab.split == "test"].reset_index(drop=True)
    if a.limit:
        lab = lab.groupby("kind").head(a.limit).reset_index(drop=True)
    runs = sorted(d for r in a.runs for d in glob.glob(os.path.join(r, "*_seed*"))
                  if re.search(r"_seed\d+$", d) and os.path.exists(os.path.join(d, "best.pt")))
    with RunLog("grading/predict_grading.py", notes=f"{a.name}: {len(runs)} runs ") as log:
        res = []
        for d in runs:
            if os.path.exists(os.path.join(d, f"predictions_{a.name}.csv")) and not a.overwrite:
                print(f"skip {d} (done)")
                continue
            r = predict_run(d, lab, a, device, amp)
            print(f"{d}: {r['n']} rows; clean predictions = original test {r['clean_same_pred']:.3f}, "
                  f"max prob. difference {r['clean_max_dp']:.4f}", flush=True)
            res.append(r)
        log.add(name=a.name, data_root=a.data_root, n_runs=len(res), checks=res, device=str(device),
                gpu=torch.cuda.get_device_name(0) if amp else None, torch=torch.__version__)


if __name__ == "__main__":
    main()

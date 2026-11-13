"""Explainability, part 1: image features and the exact split of the head's output for trained runs.

The VisionFM grader's head is one linear layer on [1536 image features | 9 standardised quality values], so every
output is exactly  W_f. f  +  W_q. q  +  b. For each run folder (<cond>_seed<seed> with best.pt) this predicts the
test split of <data_root>/labels.csv as train_grading.py does at test time (no test-time augmentation) and writes
<run>/explain_test.npz:
  sample_id, feats [N, 1536] float16, contrib_f [N, K], contrib_q [N, K], bias [K], out [N, K]
(K = 1 for the Nuclear Cataract regression, else the number of classes). A / B runs feed q = 0, so contrib_q = 0.
Check: out is compared with predictions_test.csv (pred_raw for regression, argmax for classification).
Usage: python grading/explain_features.py --runs runs/grading/nuclear_cataract_reg_BC/C_seed14 --data_root "Data/Grading Data/nuclear_cataract"
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_grading import ROOT, GradingDataset, GradingNet, quality_vectors  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "data_prep"))
from run_log import RunLog  # noqa: E402


def explain_run(run_dir, lab, a, device, amp):
    info = json.load(open(os.path.join(run_dir, "run_info.json")))
    reg = info.get("label") == "nc_raw"
    model = GradingNet(1 if reg else info["n_classes"], pretrained=False, arch=info["arch"],
                       size=(info["height"], info["width"])).to(device).eval()
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=device,
                                     weights_only=False)["model"])
    qv = (quality_vectors(lab, info["cond"], a.data_root, info.get("q_file", "quality_d3.csv"))
          - np.array(info["q_mean"])) / np.array(info["q_sd"])
    ds = GradingDataset(lab, a.data_root, qv, info["height"], info["width"], False, info["seed"])
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=a.num_workers, pin_memory=amp)
    lin = model.head[1]
    wf, wq, b = (t.detach().float() for t in (lin.weight[:, :-9], lin.weight[:, -9:], lin.bias))
    feats, cf, cq = [], [], []
    with torch.no_grad():
        for x, q, _, _ in dl:
            x = (x.to(device, non_blocking=True).float() - model.mean) / model.std
            with torch.autocast(device.type, enabled=amp):
                f = model.features(x)
            f, q = f.float(), q.to(device).float()
            feats.append(f.half().cpu())
            cf.append((f @ wf.T).cpu())
            cq.append((q @ wq.T).cpu())
    feats, cf, cq = torch.cat(feats).numpy(), torch.cat(cf).numpy(), torch.cat(cq).numpy()
    bias = b.cpu().numpy()
    out = cf + cq + bias
    ref = pd.read_csv(os.path.join(run_dir, "predictions_test.csv")).set_index("sample_id").loc[lab.sample_id]
    check = (float(np.abs(out[:, 0] - ref.pred_raw.to_numpy()).max()) if reg
             else float((out.argmax(1) == ref.pred.to_numpy()).mean()))
    np.savez_compressed(os.path.join(run_dir, "explain_test.npz"), sample_id=lab.sample_id.to_numpy(), feats=feats,
                        contrib_f=cf, contrib_q=cq, bias=bias, out=out)
    return dict(run=run_dir, cond=info["cond"], n=len(lab), regression=reg,
                check=("max |out - pred_raw|" if reg else "argmax = pred"), check_value=check)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run folders <cond>_seed<seed>")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--master", default=None, help="accepted for symmetry with predict_grading.py; not used")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: N images per kind")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    lab = pd.read_csv(os.path.join(a.data_root, "labels.csv"))
    lab = lab[lab.split == "test"].reset_index(drop=True)
    if a.limit:
        lab = lab.groupby("kind").head(a.limit).reset_index(drop=True)
    with RunLog("grading/explain_features.py", notes=f"{len(a.runs)} runs") as log:
        res = []
        for d in a.runs:
            t = time.time()
            r = explain_run(d, lab, a, device, amp)
            r["seconds"] = round(time.time() - t, 1)
            print(r, flush=True)
            res.append(r)
        log.add(data_root=a.data_root, results=res, device=str(device),
                gpu=torch.cuda.get_device_name(0) if amp else None, torch=torch.__version__)


if __name__ == "__main__":
    main()

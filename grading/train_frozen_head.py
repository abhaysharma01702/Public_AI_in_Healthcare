"""Reti-Pioneer-style grader: frozen VisionFM slit-lamp features + a small head with
bilinear quality fusion. Conditions A / B / C / C_oracle (+ C_shuffled at test) exactly as train_grading.py:
same rows, same quality vectors (quality_vectors), same standardisation (--q_standardise of r2 / r2g, always on here),
same equal-steps rule (A sees each clean image 4x), same prediction files, so evaluate_grading.py is unchanged.

Head (Reti-Pioneer model/QualityAware.py + the ComplexModel tail, github.com/lyhyl/Reti-Pioneer):
  q (9 values) -> Linear(9, 4)  [Reti-Pioneer: Linear(3 quality probabilities, 1), fixed weights (1, .5, 0); ours are
                                 9 defect values, so the map is learned and wider (Quality plan 6.4: Linear(8 -> 4))]
  Bilinear([1, features] (1537), [1, q'] (5), 256, no bias) -> SELU -> Linear(256, 256) -> SELU -> Linear(256, K)
  Reti-Pioneer's metadata fusion (age / sex) and its ensemble over backbones are left out (no metadata; one backbone).
  A / B feed zeros, so q' = bias of Linear(9, 4): the same network, a plain linear map of the features.
Training (fixed before any result): AdamW lr 1e-4 (Reti-Pioneer), weight decay 1e-4 (as train_grading.py),
batch 256, 4 warm-up epochs then cosine, up to 200 epochs, early stop after 20 epochs without a lower validation loss,
checkpoint = lowest validation loss, label smoothing 0.1 (as r2 / r2g). Features are fixed, so no augmentation.
Inputs: <data>/<folder>/{features_visionfm.npz, labels.csv, quality_d3.csv} (extract_features.py) for the main folder
and the two extra test folders; <data> = Data/Grading Data on the laptop, $HOME/project/features on Ada.
Output: <out>/<cond>_seed<seed>/{best.pt, epochs.csv, run_info.json, predictions_test.csv, predictions_test_seen.csv,
predictions_test_holdout.csv}
Usage: python grading/train_frozen_head.py --task nuclear_cataract --out runs/grading/nuclear_cataract_grade_visionfm
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_grading import CONDS, ROOT, VALUES, qwk, quality_vectors  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "data_prep"))
from run_log import RunLog  # noqa: E402

DATA = os.path.join(ROOT, "Data", "Grading Data")
TASKS = {"sustech_sysu": "category", "nuclear_cataract": "grade"}   # label: r2 (category) / r2g (NC grade)
EXTRA = ["test_seen", "test_holdout"]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BilinearQualityHead(nn.Module):
    def __init__(self, in_dim, n_classes, n_q=len(VALUES), q_dim=4, fuse_dim=256):
        super().__init__()
        self.q_fc = nn.Linear(n_q, q_dim)
        self.q_fuse = nn.Bilinear(in_dim + 1, q_dim + 1, fuse_dim, bias=False)
        self.post = nn.Sequential(nn.SELU(), nn.Linear(fuse_dim, fuse_dim), nn.SELU(), nn.Linear(fuse_dim, n_classes))

    def forward(self, f, q):
        one = torch.ones(len(f), 1, dtype=f.dtype, device=f.device)
        return self.post(self.q_fuse(torch.cat([one, f], 1), torch.cat([one, self.q_fc(q)], 1)))


def load_split(data, folder, label):
    """labels.csv rows with their VisionFM features (labels.csv order = features order, checked)."""
    root = os.path.join(data, folder)
    lab = pd.read_csv(os.path.join(root, "labels.csv"))
    z = np.load(os.path.join(root, "features_visionfm.npz"))
    assert (z["sample_id"] == lab.sample_id.to_numpy(str)).all(), f"feature order differs from labels.csv in {folder}"
    if label == "category":
        cat = pd.read_csv(os.path.join(ROOT, "labels", "sustech_master.csv")).set_index("image_id").category
        lab["grade"] = lab.image_id.map(cat).astype(int)
    return root, lab, z["feats"].astype(np.float32)


def batches(n, size, rng=None):
    idx = rng.permutation(n) if rng is not None else np.arange(n)
    return [idx[i:i + size] for i in range(0, n, size)]


def predict(model, f, q):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(torch.from_numpy(f[b]).to(DEV), torch.from_numpy(q[b]).to(DEV)).cpu()
                          for b in batches(len(f), 1024)])


def run(a, cond, seed, data):
    root, lab, F = data["main"]
    run_dir = os.path.join(a.out, f"{cond}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_classes = int(lab.grade.max()) + 1
    keep = (lab.kind == "clean") | (cond != "A")
    tr_i = np.flatnonzero((lab.split == "train") & keep)
    va_i = np.flatnonzero((lab.split == "val") & keep)
    if cond == "A":   # equal steps per epoch (train_grading.py)
        tr_i = np.tile(tr_i, 4)
    q_all = quality_vectors(lab, cond, root, "quality_d3.csv").astype(np.float32)
    q_mu, q_sd = np.zeros(len(VALUES)), np.ones(len(VALUES))
    if cond in ("C", "C_oracle"):
        q_mu, q_sd = q_all[tr_i].mean(0), q_all[tr_i].std(0) + 1e-6
    qn = lambda q: ((q - q_mu) / q_sd).astype(np.float32)
    Q = qn(q_all)
    y = torch.from_numpy(lab.grade.to_numpy(np.int64))
    model = BilinearQualityHead(F.shape[1], n_classes).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    steps_ep = math.ceil(len(tr_i) / a.batch)
    warm, total = a.warmup * steps_ep, a.max_epochs * steps_ep
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (total - warm))))
    best, bad, hist, t0 = np.inf, 0, [], time.time()
    for ep in range(1, a.max_epochs + 1):
        model.train()
        tl = 0.0
        for b in batches(len(tr_i), a.batch, rng):
            i = tr_i[b]
            loss = nn.functional.cross_entropy(model(torch.from_numpy(F[i]).to(DEV), torch.from_numpy(Q[i]).to(DEV)),
                                               y[i].to(DEV),
                                               label_smoothing=a.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item() * len(i)
        lo = predict(model, F[va_i], Q[va_i])
        vl = nn.functional.cross_entropy(lo, y[va_i]).item()
        p = lo.argmax(1).numpy()
        k = qwk(y[va_i].numpy(), p, n_classes)
        star = vl < best
        if star:
            best, bad = vl, 0
            torch.save({"model": model.state_dict(), "epoch": ep, "val_qwk": k, "val_loss": vl},
                       os.path.join(run_dir, "best.pt"))
        else:
            bad += 1
        hist.append(dict(epoch=ep, train_loss=tl / len(tr_i), val_loss=vl, val_qwk=k,
                         val_acc=float((p == y[va_i].numpy()).mean())))
        if bad >= a.patience:
            break
    pd.DataFrame(hist).to_csv(os.path.join(run_dir, "epochs.csv"), index=False)
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model"])
    for name in ["test"] + EXTRA:
        r, l, f = data["main"] if name == "test" else data[name]
        te = np.flatnonzero(l.split == "test")
        lt = l.iloc[te].reset_index(drop=True)
        out = lt[["sample_id", "image_id", "group_id", "kind", "defect", "strength", "grade"]].copy()
        qv = qn(quality_vectors(lt, cond, r, "quality_d3.csv").astype(np.float32))
        variants = [("", qv)]
        if cond in ("C", "C_oracle"):   # C-shuffled: permutation seed 14, as train_grading.py
            variants.append(("shuffled_", qv[np.random.default_rng(14).permutation(len(qv))]))
        for prefix, q in variants:
            pr = torch.softmax(predict(model, f[te], q), 1).numpy()
            out[f"{prefix}pred"] = pr.argmax(1)
            for c in range(n_classes):
                out[f"{prefix}p{c}"] = pr[:, c]
        out.to_csv(os.path.join(run_dir, f"predictions_{name}.csv"), index=False, float_format="%.6g")
    info = dict(vars(a), cond=cond, seed=seed, best_epoch=int(ck["epoch"]), best_val_qwk=float(ck["val_qwk"]),
                best_val_loss=float(ck["val_loss"]), epochs_run=len(hist), n_classes=n_classes, n_train=len(tr_i),
                n_val=len(va_i), q_mean=[float(v) for v in q_mu], q_sd=[float(v) for v in q_sd],
                seconds=round(time.time() - t0, 1), torch=torch.__version__, device=str(DEV),
                gpu=torch.cuda.get_device_name(0) if DEV.type == "cuda" else None)
    json.dump(info, open(os.path.join(run_dir, "run_info.json"), "w"), indent=1)
    print(f"{cond} seed {seed}: best epoch {ck['epoch']} / {len(hist)}, val loss {ck['val_loss']:.4f}, "
          f"val QWK {ck['val_qwk']:.3f}, {info['seconds']:.0f} s", flush=True)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=DATA, help="parent of the feature folders")
    ap.add_argument("--conds", nargs="+", default=CONDS, choices=CONDS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[14, 15, 16])
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--max_epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    label = TASKS[a.task]
    with RunLog("grading/train_frozen_head.py", notes=f"VisionFM bilinear head {a.task}") as log:
        data = {"main": load_split(a.data, a.task, label)}
        for name in EXTRA:
            data[name] = load_split(a.data, f"{a.task}_{name}", label)
        infos = [run(a, c, s, data) for c in a.conds for s in a.seeds]
        log.add(task=a.task, out=a.out, runs=len(infos), seconds=sum(i["seconds"] for i in infos))


if __name__ == "__main__":
    main()

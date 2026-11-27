"""Diagnosis of grading round 1. Why is the clean-test QWK about 0.1 when the
validation QWK of the same runs is about 0.43?
1. The 12 saved round-1 models (best.pt) on the 70 clean val and 70 clean test photos, laptop CPU, fp32: QWK / accuracy
   per run on val vs test, summed confusions; the test predictions are compared with the Ada ones (AMP) as a check.
2. Eyes: val and test hold 70 photos each but only about 35 groups (same-eye photos). Per test / val group: photos,
   type, how often the 12 models are off by 2 or more grades, and the QWK of all runs without that group.
3. Model-free check with the cached ResNet50 embeddings: for every val / test photo the most similar
   training photo (cosine), and the QWK of a 1- and 5-nearest-training-photo vote. If val photos have closer training
   look-alikes than test photos, the val score is inflated by near-duplicates the grouping did not catch.
Outputs in results/grading/<task>/diagnosis/: predictions_clean_val_test.csv, per_run.csv, confusions.csv, groups.csv,
neighbours.csv (+ printed summary).
Usage: python grading/diagnose_round1.py [--task sustech_sysu] [--threads 4]
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from train_grading import GradingDataset, GradingNet, qwk, quality_vectors, run_epoch  # noqa: E402

MASTER = {"sustech_sysu": ("sustech_master.csv", "sustech")}


def predict(run_dir, df, data_root, n_classes):
    cond = os.path.basename(run_dir).rsplit("_seed", 1)[0]
    model = GradingNet(n_classes, pretrained=False, memory_efficient=False)
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu", weights_only=False)["model"])
    ds = GradingDataset(df, data_root, quality_vectors(df, cond, data_root, "quality_d3.csv"), 512, 672, False, 0)
    dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)
    _, lo, _, _ = run_epoch(model, dl, torch.device("cpu"), False)
    return torch.softmax(lo, 1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="sustech_sysu")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    data_root = os.path.join(ROOT, "Data", "Grading Data", a.task)
    runs = os.path.join(ROOT, "runs", "grading", a.task)
    out = os.path.join(ROOT, "results", "grading", a.task, "diagnosis")
    os.makedirs(out, exist_ok=True)
    lab = pd.read_csv(os.path.join(data_root, "labels.csv"))
    n_classes = int(lab.grade.max()) + 1
    clean = lab[(lab.kind == "clean") & lab.split.isin(["val", "test"])].reset_index(drop=True)

    # 1. models on clean val + test
    pred_file = os.path.join(out, "predictions_clean_val_test.csv")
    if os.path.exists(pred_file):
        P = pd.read_csv(pred_file)
    else:
        rows = []
        for d in sorted(glob.glob(os.path.join(runs, "*_seed*"))):
            t0 = time.time()
            pr = predict(d, clean, data_root, n_classes)
            r = clean[["sample_id", "image_id", "group_id", "split", "grade"]].assign(run=os.path.basename(d),
                                                                                   pred=pr.argmax(1))
            for c in range(n_classes):
                r[f"p{c}"] = pr[:, c]
            rows.append(r)
            print(f"{os.path.basename(d)}: {time.time - t0:.0f} s", flush=True)
        P = pd.concat(rows, ignore_index=True)
        P.to_csv(pred_file, index=False, float_format="%.6g")
    P["cond"] = P.run.str.rsplit("_seed", n=1).str[0]

    # check: laptop test predictions = the Ada ones
    chk = []
    for run, g in P[P.split == "test"].groupby("run"):
        ada = pd.read_csv(os.path.join(runs, run, "predictions_test.csv")).set_index("sample_id").loc[g.sample_id]
        chk.append(dict(run=run, same_pred=float((ada.pred.to_numpy() == g.pred.to_numpy()).mean()),
                        max_abs_dp=float(np.abs(ada[[f"p{c}" for c in range(n_classes)]].to_numpy()
                                                - g[[f"p{c}" for c in range(n_classes)]].to_numpy()).max())))
    chk = pd.DataFrame(chk)
    print("laptop vs Ada test predictions: same class", f"{chk.same_pred.min:.3f}-{chk.same_pred.max:.3f};",
          f"max |dp| {chk.max_abs_dp.max:.4f}")

    per_run = []
    for (run, split), g in P.groupby(["run", "split"]):
        per_run.append(dict(run=run, split=split, qwk=qwk(g.grade.to_numpy(), g.pred.to_numpy(), n_classes),
                            acc=float((g.grade == g.pred).mean())))
    per_run = pd.DataFrame(per_run).pivot(index="run", columns="split", values=["qwk", "acc"])
    per_run.columns = [f"{m}_{s}" for m, s in per_run.columns]
    per_run = per_run.reset_index().merge(chk, on="run")
    per_run.to_csv(os.path.join(out, "per_run.csv"), index=False, float_format="%.4f")
    print(per_run.round(3).to_string(index=False))
    print(per_run.drop(columns="run").mean().round(3).to_string())

    conf = []
    for (cond, split), g in P.groupby(["cond", "split"]):
        cm = pd.crosstab(g.grade, g.pred).reindex(index=range(n_classes), columns=range(n_classes), fill_value=0)
        conf.append(cm.assign(cond=cond, split=split).reset_index().rename(columns={"grade": "true"}))
    conf = pd.concat(conf, ignore_index=True)
    conf.to_csv(os.path.join(out, "confusions.csv"), index=False)
    print("\nsummed confusions over all 12 runs (rows true, columns predicted)")
    for split in ["val", "test"]:
        print(split)
        print(conf[conf.split == split].groupby("true")[list(range(n_classes))].sum().to_string())

    # 2. groups (eyes)
    grp = []
    for (split, gid), g in P.groupby(["split", "group_id"]):
        rest = P[(P.split == split) & (P.group_id != gid)]
        q_wo = np.mean([qwk(r.grade.to_numpy(), r.pred.to_numpy(), n_classes) for _, r in rest.groupby("run")])
        grp.append(dict(split=split, group_id=gid, photos=g.image_id.nunique(),
                        types=",".join(map(str, sorted(g.grade.unique()))),
                        mean_abs_err=float((g.pred - g.grade).abs().mean()),
                        off_by_2plus=float(((g.pred - g.grade).abs() >= 2).mean()),
                        qwk_without_group=float(q_wo)))
    grp = pd.DataFrame(grp)
    base = per_run[["qwk_val", "qwk_test"]].mean()
    grp["qwk_gain_if_removed"] = grp.qwk_without_group - grp.split.map({"val": base.qwk_val, "test": base.qwk_test})
    grp = grp.sort_values(["split", "qwk_gain_if_removed"], ascending=[True, False])
    grp.to_csv(os.path.join(out, "groups.csv"), index=False, float_format="%.4f")
    print("\ngroups whose removal changes the mean QWK most (top 6 per split)")
    print(grp.groupby("split").head(6).round(3).to_string(index=False))

    # 3. nearest training photos (cached ResNet50)
    csv, folder = MASTER[a.task]
    m = pd.read_csv(os.path.join(ROOT, "labels", csv))
    F = np.load(os.path.join(ROOT, "Data", "derived", folder, "resnet50_imagenet_v2.npy"))   # rows = master order
    assert len(F) == len(m) and m.image_id.is_monotonic_increasing
    m = m.assign(row=np.arange(len(m)))
    u = m[m.status == "usable"]
    tr = u[u.split == "train"]
    nb = []
    for split in ["val", "test"]:
        s = u[u.split == split]
        S = F[s.row.to_numpy()] @ F[tr.row.to_numpy()].T
        order = np.argsort(-S, 1)
        y_tr = tr.type.to_numpy()
        for i, r in enumerate(s.itertuples()):
            nb.append(dict(split=split, image_id=r.image_id, group_id=r.group_id, type=r.type,
                           max_cos=float(S[i, order[i, 0]]), nn1_type=int(y_tr[order[i, 0]]),
                           nn1_image=tr.image_id.iloc[order[i, 0]],
                           nn5_type=int(np.bincount(y_tr[order[i, :5]], minlength=n_classes).argmax())))
    nb = pd.DataFrame(nb)
    nb.to_csv(os.path.join(out, "neighbours.csv"), index=False, float_format="%.4f")
    print("\nnearest training photo (ResNet50 cosine) and nearest-neighbour votes")
    for split, g in nb.groupby("split"):
        print(f"{split}: max cosine median {g.max_cos.median:.3f}, >= 0.85: {(g.max_cos >= 0.85).sum}, "
              f">= 0.90: {(g.max_cos >= 0.90).sum}; 1-NN QWK {qwk(g.type.to_numpy, g.nn1_type.to_numpy, n_classes):.3f}"
              f" acc {(g.type == g.nn1_type).mean:.3f}; 5-NN QWK {qwk(g.type.to_numpy, g.nn5_type.to_numpy, n_classes):.3f}"
              f" acc {(g.type == g.nn5_type).mean:.3f}")


if __name__ == "__main__":
    main()

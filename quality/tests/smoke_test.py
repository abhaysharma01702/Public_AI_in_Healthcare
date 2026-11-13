"""CPU smoke test of the quality-module training code (section 5).

Checks:
  1. qm.VALUES matches quality/degradations.py
  2. ranking pair rule on a hand-made example (A) and the RankNet value
  3. source-grouped batches: complete groups (4 versions, 5 with a d4 version), every image once per epoch,
     reshuffled per epoch; the smoke training subset contains d4 versions
  4. a tiny training run (16 train / 8 val sources, 128 x 168, batch 4 x 4-5, 3 epochs): losses finite,
     training loss falls, checkpoints written
  5. resume: 1 epoch, stop, continue to 3 epochs == uninterrupted 3 epochs
  6. evaluation on val / test / stress / real check / gen_train / gen_holdout (subsets, 50 bootstrap samples):
     files and metrics written, test includes d4 versions, paired generalisation comparison written
  7. gen_fresh  is loaded and its files checked WITHOUT running a model on it (opened once, after the
     final models are trained)
Data: Quality Data v2 + the extension (labels_ext_v3.csv), as in the final training (B1).
Outputs go to runs/quality/_smoke_test/ (deleted and rebuilt on every run); other configs (MCF-style, Swin-T)
to runs/quality/_smoke_test_<config name>/.
Usage: python quality/tests/smoke_test.py [--config quality/configs/<name>.yaml]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
QUALITY = os.path.dirname(HERE)
ROOT = os.path.dirname(QUALITY)
sys.path.insert(0, QUALITY)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from qm import VALUES  # noqa: E402
from qm import data as QD  # noqa: E402
from qm import losses as QL  # noqa: E402

DATA = os.path.join(ROOT, "Data", "Quality Data v2")
EXT = "labels_ext_v3.csv"   #
OUT = os.path.join(ROOT, "runs", "quality", "_smoke_test")
CONFIG = os.path.join(QUALITY, "configs", "densenet121.yaml")
SMOKE = ["data.height=128", "data.width=168", "data.limit_sources_train=16", "data.limit_sources_val=8",
         "data.num_workers=2", "batch.sources_per_batch=4", "optim.ema_decay=0.5", "train.log_every=0",
         f"data.extension={EXT}"]
SETS = ["val", "test", "stress", "real_check", "gen_train", "gen_holdout"]   # not gen_fresh


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        raise SystemExit(1)


def train(run_dir, epochs, extra=()):
    cmd = [sys.executable, os.path.join(QUALITY, "train.py"), "--config", CONFIG, "--data_root", DATA,
           "--run_dir", run_dir, "--no_eval", "--device", "cpu", "--set", *SMOKE, f"train.max_epochs={epochs}", *extra]
    t0 = time.time()
    subprocess.run(cmd, check=True)
    return time.time() - t0


def epochs(run_dir):
    with open(os.path.join(run_dir, "epochs.jsonl")) as f:
        return [json.loads(line) for line in f]


def main():
    global CONFIG, OUT
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG)
    CONFIG = os.path.abspath(ap.parse_args().config)
    name = os.path.splitext(os.path.basename(CONFIG))[0]
    if name != "densenet121":
        OUT = f"{OUT}_{name}"
    print(f"config {CONFIG}, output {OUT}", flush=True)
    t_all = time.time()
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)

    # 1
    import degradations
    check(list(degradations.VALUES) == VALUES, "qm.VALUES == degradations.VALUES")

    # 2: one defect; images 0..3 with labels 0, 0.30, 0.32, 0.50; min_diff 0.05
    y = torch.tensor([[0.0], [0.30], [0.32], [0.50]])
    pairs = QL.rank_pairs(y, torch.tensor([0.05]))[:, :, 0]
    got = {(i, j) for i, j in zip(*np.nonzero(pairs.numpy()))}
    want = {(1, 0), (2, 0), (3, 0), (3, 1), (3, 2)}  # 0.32 vs 0.30 differs by < 0.05: not ranked
    check(got == want, f"rank pairs {sorted(got)}")
    s = torch.tensor([[0.0], [1.0], [0.5], [2.0]])
    manual = np.mean([np.log1p(np.exp(-(s[i, 0] - s[j, 0]).item())) for i, j in want])
    check(abs(QL.ranknet(s, y, torch.tensor([0.05])).item() - manual) < 1e-6, f"RankNet value {manual:.6f}")
    st = QL.label_stats(QD.load_split(DATA, "train", EXT), VALUES, 0.05)
    print("     min_diff per defect:", {v: round(d, 4) for v, d in zip(VALUES, st["min_diff"])})

    # 3
    val = QD.load_split(DATA, "val", EXT)
    full = val.groupby("source_image_id").size()
    check(set(full.unique()) == {4, 5}, f"val sources with 4 / 5 versions: {full.value_counts.sort_index.to_dict}")
    smp = QD.SourceGroupedBatchSampler(val, 8, shuffle=True, seed=14)
    seen, ok = [], True
    for b in smp:
        g = val.iloc[b].groupby("source_image_id").size()
        ok &= bool((g == full[g.index]).all()) and len(b) == g.sum()
        seen += b
    check(ok and sorted(seen) == list(range(len(val))),
          f"grouped batches: {len(smp)} batches, complete groups, every val image once")
    e0 = list(smp)
    smp.set_epoch(1)
    e1 = list(smp)
    smp.set_epoch(0)
    check(e0 == list(smp) and e0 != e1, "shuffle reproducible per epoch, different between epochs")
    sub = QD.limit_sources(QD.load_split(DATA, "train", EXT), 16, 14)
    n5 = int((sub.groupby("source_image_id").size() == 5).sum())
    check(n5 > 0, f"smoke training subset: {len(sub)} images, {n5} of 16 sources with a d4 version")

    # 4
    a = os.path.join(OUT, "A")
    secs_a = train(a, 3)
    ea = epochs(a)
    losses = [r["train_loss"] for r in ea]
    check(all(np.isfinite([r[k] for r in ea for k in ("train_loss", "val_loss")])), "losses finite")
    check(losses[-1] < losses[0], f"training loss falls: {[round(v, 4) for v in losses]}")
    check(all(os.path.exists(os.path.join(a, f)) for f in ("last.pt", "best.pt", "config.yaml", "label_stats.json")),
          "checkpoints and config written")

    # 5
    b = os.path.join(OUT, "B")
    secs_b1 = train(b, 1)
    secs_b2 = train(b, 3)
    eb = epochs(b)
    keys = ["train_loss", "train_mse", "train_rank", "val_loss", "val_mean_auroc"]
    diff = max(abs(ra[k] - rb[k]) for ra, rb in zip(ea, eb) for k in keys)
    wa = torch.load(os.path.join(a, "last.pt"), weights_only=False)["model"]
    wb = torch.load(os.path.join(b, "last.pt"), weights_only=False)["model"]
    wdiff = max((wa[k].float() - wb[k].float()).abs().max().item() for k in wa)
    check(len(eb) == 3 and diff < 1e-5 and wdiff < 1e-5,
          f"resume == uninterrupted (max metric diff {diff:.2e}, max weight diff {wdiff:.2e})")

    # 6
    t0 = time.time()
    subprocess.run([sys.executable, os.path.join(QUALITY, "evaluate.py"), "--run_dir", a, "--data_root", DATA,
                    "--device", "cpu", "--num_workers", "2", "--limit_sources", "8", "--bootstrap", "50",
                    "--sets", *SETS], check=True)
    secs_eval = time.time() - t0
    files = [f"predictions_{s}.csv" for s in SETS] + [f"metrics_{s}.csv" for s in SETS] +\
            ["cross_talk_val.csv", "cross_talk_test.csv", "cross_talk_gen_train.csv", "cross_talk_gen_holdout.csv",
             "generalisation_comparison.csv", "metrics_summary.json"]
    missing = [f for f in files if not os.path.exists(os.path.join(a, f))]
    check(not missing, f"evaluation files written ({len(files)}; missing {missing})")
    pt = pd.read_csv(os.path.join(a, "predictions_test.csv"))
    n_d4 = int(pt.sample_id.str.endswith("__d4").sum())
    check(n_d4 > 0, f"test predictions include d4 versions ({n_d4})")
    m = pd.read_csv(os.path.join(a, "metrics_test.csv"))
    ov = m[(m.group_type == "overall") & (m.defect == "mean")]
    check(len(ov) == 5 and ov.value.notna().all() and ov.ci_low.notna().all(), "test metrics + CIs computed")
    rc = pd.read_csv(os.path.join(a, "predictions_real_check.csv"))
    check(int((rc.real_defect.fillna("") != "").sum()) == 30, "all 30 real check photos predicted")
    subprocess.run([sys.executable, os.path.join(QUALITY, "evaluate.py"), "--run_dir", a, "--data_root", DATA,
                    "--metrics_only", "--sets", "test", "--bootstrap", "50"], check=True)
    m2 = pd.read_csv(os.path.join(a, "metrics_test.csv"))
    check(np.allclose(m.value.fillna(-1), m2.value.fillna(-1)), "metrics recomputed from the prediction CSV match")

    # 7: no model is run on gen_fresh
    gf = QD.load_generalisation(DATA, "fresh")
    gone = [p for p in gf.path if not os.path.exists(os.path.join(DATA, p))]
    check(len(gf) == 3324 and int((gf.impl == "fresh").sum()) == 2493 and not gone,
          f"gen_fresh loads: {len(gf)} rows, all files present")
    x, _, _ = QD.QualityDataset(gf.iloc[:1], DATA, 128, 168)[0]
    check(tuple(x.shape) == (3, 128, 168), "gen_fresh image decodes")
    check(not os.path.exists(os.path.join(a, "predictions_gen_fresh.csv")), "no gen_fresh predictions exist")

    res = {"config": name, "train_A_3_epochs_s": round(secs_a, 1), "train_B_1_epoch_s": round(secs_b1, 1),
           "train_B_resume_s": round(secs_b2, 1), "evaluate_s": round(secs_eval, 1),
           "total_s": round(time.time() - t_all, 1), "train_losses": losses}
    print(json.dumps(res))
    with open(os.path.join(OUT, "smoke_result.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

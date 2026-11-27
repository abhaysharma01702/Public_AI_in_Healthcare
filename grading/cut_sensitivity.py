"""Cut-point sensitivity of the final Nuclear Cataract regression grader (report, reviewer request).

The final grader (5-seed regression ensemble, condition A) turns its continuous output into grades with cut points tuned
on the 121 clean validation photographs (each seed tuned separately, cuts averaged). This script asks how much the test
result depends on that small tuning set:
  1. re-runs the five saved checkpoints on the validation photographs (no TTA, as in training) -> val_raw.csv;
  2. bootstrap: resample validation eyes (1,000 times), re-tune every seed's cuts on the resample, average, apply to the
     saved test ensemble output and record degraded / clean QWK and the gain over the 3-seed classifier ensemble;
  3. 5-fold cross-fitting by eye: tune on four fifths of the validation eyes, score the test set; repeat for each fold.
Writes results/report_tables/nc_val_raw.csv, nc_cut_sensitivity.csv, nc_cut_sensitivity_boot.csv and
nc_cut_sensitivity_boot_summary.csv.
Usage (from the project root): python grading/cut_sensitivity.py [--skip_predict] [--summary_only]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_grading import GradingDataset, GradingNet, qwk, run_epoch, to_class, tune_cuts  # noqa: E402

ROOT = os.path.dirname(HERE)
G = os.path.join(ROOT, "runs", "grading")
REG = os.path.join(G, "nuclear_cataract_reg")
CLF = os.path.join(G, "nuclear_cataract_grade_visionfm_ft")
DATA = os.path.join(ROOT, "Data", "Grading Data", "nuclear_cataract")
OUT = os.path.join(ROOT, "results", "report_tables")
SEEDS = (14, 15, 16, 17, 18)
N = 6


def predict_val():
    lab = pd.read_csv(os.path.join(DATA, "labels.csv"))
    va = lab[(lab.split == "val") & (lab.kind == "clean")].reset_index(drop=True)
    out = va[["image_id", "group_id", "grade"]].copy()
    for s in SEEDS:
        info = json.load(open(os.path.join(REG, f"A_seed{s}", "run_info.json")))
        model = GradingNet(1, head=info["head"], grid=(info["height"] // 32, info["width"] // 32), arch="visionfm",
                           size=(info["height"], info["width"]), pretrained=False)
        ck = torch.load(os.path.join(REG, f"A_seed{s}", "best.pt"), map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ds = GradingDataset(va, DATA, np.zeros((len(va), 9), np.float32), info["height"], info["width"], False, s)
        dl = torch.utils.data.DataLoader(ds, batch_size=8, num_workers=0)
        _, lo, _, _ = run_epoch(model, dl, torch.device("cpu"), False, regression=True)
        out[f"raw{s}"] = lo.squeeze(1).numpy()
        _, val_k = tune_cuts(va.grade.to_numpy(), out[f"raw{s}"].to_numpy(), N)
        print(f"seed {s}: tuned val QWK {val_k:.3f} (run_info {info['best_val_qwk']:.3f}); cuts reproduced:",
              np.allclose(tune_cuts(va.grade.to_numpy(), out[f'raw{s}'].to_numpy(), N)[0], info["cuts"], atol=1e-6),
              flush=True)
    out.to_csv(os.path.join(OUT, "nc_val_raw.csv"), index=False, float_format="%.6g")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_predict", action="store_true", help="reuse results/report_tables/nc_val_raw.csv")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--summary_only", action="store_true", help="reuse nc_cut_sensitivity_boot.csv")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    val = pd.read_csv(os.path.join(OUT, "nc_val_raw.csv")) if a.skip_predict else predict_val()

    # test side: the saved TTA outputs of the five seeds and the 3-seed classifier ensemble
    clf_ps = [pd.read_csv(os.path.join(CLF, f"A_seed{s}", "predictions_test.csv")) for s in (14, 15, 16)]
    order = clf_ps[0].sample_id
    clf = np.mean([p[[f"p{k}" for k in range(N)]].to_numpy() for p in clf_ps], 0).argmax(1)
    reg = [pd.read_csv(os.path.join(REG, f"A_seed{s}", "predictions_test.csv")).set_index("sample_id").loc[order]
           for s in SEEDS]
    y = clf_ps[0].grade.to_numpy()
    deg = (clf_ps[0].kind != "clean").to_numpy()
    raw_te = np.mean([r.tta_raw.to_numpy() for r in reg], 0)
    clf_deg, clf_cln = qwk(y[deg], clf[deg], N), qwk(y[~deg], clf[~deg], N)

    def score(cuts):
        p = to_class(raw_te, cuts)
        d, c = qwk(y[deg], p[deg], N), qwk(y[~deg], p[~deg], N)
        return dict(deg_qwk=d, clean_qwk=c, gain_deg=d - clf_deg, gain_clean=c - clf_cln, cuts=list(np.round(cuts, 3)))

    def tuned(rows):
        yv = val.grade.to_numpy()[rows]
        return np.mean([tune_cuts(yv, val[f"raw{s}"].to_numpy()[rows], N)[0] for s in SEEDS], 0)

    allrows = np.arange(len(val))
    res = [dict(method="plain rounding (k + 0.5)", **score([k + 0.5 for k in range(N - 1)])),
           dict(method="tuned on all 121 validation photographs (reported)", **score(tuned(allrows)))]

    eyes = val.group_id.unique()
    rng = np.random.default_rng(14)
    folds = np.array_split(rng.permutation(eyes), 5)
    for i, f in enumerate(folds):
        rows = np.where(~val.group_id.isin(f))[0]
        res.append(dict(method=f"5-fold cross-fit, fold {i + 1} ({len(rows)} photographs)", **score(tuned(rows))))

    boot_file = os.path.join(OUT, "nc_cut_sensitivity_boot.csv")
    if a.summary_only:   # reuse the saved bootstrap draws (same seed, so identical to a re-run)
        b = pd.read_csv(boot_file)
    else:
        boot = []
        by_eye = {e: np.where(val.group_id == e)[0] for e in eyes}
        for r in range(a.reps):
            rows = np.concatenate([by_eye[e] for e in rng.choice(eyes, len(eyes))])
            boot.append(score(tuned(rows)))
            if (r + 1) % 100 == 0:
                print(f"bootstrap {r + 1}/{a.reps}", flush=True)
        b = pd.DataFrame(boot)
        b.to_csv(boot_file, index=False, float_format="%.4f")
    df = pd.DataFrame(res)
    df.to_csv(os.path.join(OUT, "nc_cut_sensitivity.csv"), index=False, float_format="%.4f")
    summ = pd.DataFrame([dict(stat=col, median=np.median(b[col]), p2_5=np.percentile(b[col], 2.5),
                              p97_5=np.percentile(b[col], 97.5), share_above_0=float((b[col] > 0).mean()),
                              n_resamples=len(b)) for col in ["deg_qwk", "clean_qwk", "gain_deg", "gain_clean"]])
    summ.to_csv(os.path.join(OUT, "nc_cut_sensitivity_boot_summary.csv"), index=False, float_format="%.4f")
    print(f"classifier x3: degraded {clf_deg:.3f}, clean {clf_cln:.3f}")
    with pd.option_context("display.width", 220, "display.max_colwidth", 70):
        print(df.round(3).to_string(index=False), "\n", summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

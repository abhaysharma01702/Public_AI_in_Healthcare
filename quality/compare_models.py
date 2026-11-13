"""Phase E model-vs-model comparison (section 4 and addendum A).

Reads the sealed gen_fresh pass (runs/quality/fresh_pass/<name>) and the real-check predictions of the original runs.
Writes to results/quality_phase_e/:
  summary.csv            per model: gen_fresh and gen_train mean metrics (+ CIs), fresh - train gap, real "any" AUROC
  per_defect_fresh.csv   per model and defect: gen_fresh AUROC / Spearman_present / MAE_present
  paired_fresh.csv       paired bootstrap B - A on gen_fresh (same source photos), all metrics, per defect + mean
  paired_real_any.csv    paired bootstrap B - A of the real "any" AUROC (real and clean photos resampled separately)
Single seed per model (D4 not run): "better" = the paired 95% CI excludes 0; the report says single seed.
Usage: python quality/compare_models.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "quality"))
from qm import metrics as QX  # noqa: E402

RUNS = os.path.join(ROOT, "runs", "quality")
OUT = os.path.join(ROOT, "results", "quality_phase_e")
VALUES = ["defocus_blur", "motion_blur", "overexposure", "underexposure", "uneven_illumination", "haze", "glare",
          "occlusion", "noise"]
# name -> original run folder (real-check predictions; gen_fresh comes from fresh_pass/<name>)
MODELS = {"v1": "kaggle_v1/densenet121_seed14", "v2": "ada_v2/densenet121_v2_ada", "D1": "ada_v3/densenet121_v3_ada",
          "D2": "mcf_densenet121_ada", "D3": "swin_t_ada"}
# (A, B): reported as B - A. Primary (section 4) first, then the data ablation (addendum A).
PAIRS = [("D1", "D2", "primary"), ("D1", "D3", "primary"), ("D2", "D3", "extra"),
         ("v1", "v2", "data ablation"), ("v2", "D1", "data ablation")]
B, SEED = 1000, 14


def fresh(name, s):
    return pd.read_csv(os.path.join(RUNS, "fresh_pass", name, f"predictions_{s}.csv"))


def real(name):
    return pd.read_csv(os.path.join(RUNS, MODELS[name], "predictions_real_check.csv"))


def real_any_paired(da, db):
    """Paired bootstrap of the 'any' AUROC (max of 9 outputs): same resampled real / clean photos for both models."""
    cols = QX.pred_cols(VALUES)
    m = da[["sample_id", "real_defect"]].merge(db[["sample_id"]], on="sample_id")
    a = da.set_index("sample_id").loc[m.sample_id, cols].max(1).to_numpy()
    b = db.set_index("sample_id").loc[m.sample_id, cols].max(1).to_numpy()
    pos = (m.real_defect.fillna("") != "").to_numpy()
    ip, ineg = np.flatnonzero(pos), np.flatnonzero(~pos)
    va, vb = QX.auroc(a, pos), QX.auroc(b, pos)
    rng = np.random.default_rng(SEED)
    d = []
    for _ in range(B):
        idx = np.r_[rng.choice(ip, len(ip)), rng.choice(ineg, len(ineg))]
        d.append(QX.auroc(b[idx], pos[idx]) - QX.auroc(a[idx], pos[idx]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return va, vb, lo, hi, int(pos.sum()), int((~pos).sum())


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, per = [], []
    for n in MODELS:
        s = json.load(open(os.path.join(RUNS, "fresh_pass", n, "metrics_summary.json")))
        mf = pd.read_csv(os.path.join(RUNS, "fresh_pass", n, "metrics_gen_fresh.csv"))
        ov = mf[mf.group_type == "overall"]
        r = {"model": n}
        for st in ["gen_fresh", "gen_train"]:
            for k in ["auroc", "spearman_present", "mae_present", "false_alarm"]:
                r[f"{st}_{k}"] = s[st][k]
        for k in ["auroc", "spearman_present"]:
            x = ov[(ov.defect == "mean") & (ov.metric == k)].iloc[0]
            r[f"gen_fresh_{k}_ci"] = f"[{x.ci_low:.3f}, {x.ci_high:.3f}]"
            r[f"gap_fresh_minus_train_{k}"] = s["generalisation_fresh_minus_train"][k]
        rc = QX.real_check_metrics(real(n), VALUES, bootstrap=B, seed=SEED).set_index("real_defect").loc["any"]
        r.update(real_any_auroc=rc.auroc, real_any_ci=f"[{rc.ci_low:.3f}, {rc.ci_high:.3f}]")
        rows.append(r)
        for k in ["auroc", "spearman_present", "mae_present"]:
            for x in ov[ov.metric == k].itertuples():
                per.append({"model": n, "defect": x.defect, "metric": k, "value": x.value, "ci_low": x.ci_low,
                            "ci_high": x.ci_high})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "summary.csv"), index=False, float_format="%.4f")
    pd.DataFrame(per).to_csv(os.path.join(OUT, "per_defect_fresh.csv"), index=False, float_format="%.4f")
    pf, pr = [], []
    for a, b, role in PAIRS:
        c = QX.paired_comparison(fresh(a, "gen_fresh"), fresh(b, "gen_fresh"), VALUES, bootstrap=B, seed=SEED,
                                 names=(a, b))
        c = c.rename(columns={a: "value_A", b: "value_B"}).assign(role=role)
        pf.append(c)
        va, vb, lo, hi, npos, nneg = real_any_paired(real(a), real(b))
        pr.append({"first": a, "second": b, "role": role, "value_A": va, "value_B": vb, "difference": vb - va,
                   "diff_ci_low": lo, "diff_ci_high": hi, "n_real": npos, "n_clean": nneg})
    pf = pd.concat(pf)
    pf.to_csv(os.path.join(OUT, "paired_fresh.csv"), index=False, float_format="%.4f")
    pd.DataFrame(pr).to_csv(os.path.join(OUT, "paired_real_any.csv"), index=False, float_format="%.4f")
    print(pd.DataFrame(rows).to_string(index=False))
    m = pf[(pf.defect == "mean") & pf.metric.isin(["auroc", "spearman_present", "mae_present"])]
    print(m[["first", "second", "role", "metric", "value_A", "value_B", "difference", "diff_ci_low",
             "diff_ci_high"]].to_string(index=False))
    print(pd.DataFrame(pr).to_string(index=False))


if __name__ == "__main__":
    main()

"""How well the chosen quality module (D3 Swin-T) measures defects on the grading data. Same metric code as the quality evaluation (quality/qm/metrics.py), so the numbers compare with.
Sets:
  val   clean val + degraded val (TRAINING-style defect code; photos the quality module never trained on)
  test  clean test + fresh test (9 defects x 5 strengths, UNSEEN defect code; compare with gen_fresh )
  test per defect x strength: true vs predicted value of the matching output, and detection AUROC vs clean test
  clean originals: largest of the 9 outputs on every original photo, split into screened-clean and screening-flagged
    , and AUROC of the matching output(s) flagged vs screened-clean (indicative: flags are our own screening)
95% CIs: bootstrap over photos (1,seed 14). Output: results/grading/<task>/quality_check/.
Usage: python grading/check_quality_on_grading.py [--task sustech_sysu]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "quality"))
from qm import VALUES  # noqa: E402
from qm import metrics as QX  # noqa: E402

# screening flag -> outputs that should flag it (blur / dark / bright / blown out)
FLAG_MAP = {"blur": ["defocus_blur", "motion_blur"], "dark": ["underexposure"], "bright": ["overexposure"],
            "over": ["overexposure"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="sustech_sysu")
    ap.add_argument("--bootstrap", type=int, default=1000)
    a = ap.parse_args()
    data = os.path.join(ROOT, "Data", "Grading Data", a.task)
    out = os.path.join(ROOT, "results", "grading", a.task, "quality_check")
    os.makedirs(out, exist_ok=True)
    d = pd.read_csv(os.path.join(data, "labels.csv")).merge(pd.read_csv(os.path.join(data, "quality_d3.csv")),
                                                            on="sample_id", validate="1:1")
    d = d.rename(columns={f"q_{v}": f"pred_{v}" for v in VALUES}).assign(
        source_image_id=lambda x: x.image_id, dataset=a.task, lighting="")
    d["lighting"] = np.nan
    summary = []
    for name, sub in [("val", d[d.split == "val"]), ("test", d[d.split == "test"])]:
        m = QX.evaluate_predictions(sub.reset_index(drop=True), VALUES, a.bootstrap)
        m = m[m.group_type == "overall"].drop(columns=["group_type", "group"])
        m.to_csv(os.path.join(out, f"metrics_{name}.csv"), index=False, float_format="%.4f")
        piv = m.pivot(index="defect", columns="metric", values="value")[["auroc", "spearman_present", "mae_present",
                                                                         "false_alarm"]]
        print(f"\n== {name}: {len(sub)} images ({(sub.kind == 'clean').sum} clean)")
        print(piv.loc[VALUES + ["mean"]].round(3).to_string())
        for r in m[m.defect == "mean"].itertuples():
            summary.append(dict(set=name, metric=r.metric, value=r.value, ci_low=r.ci_low, ci_high=r.ci_high))
    pd.DataFrame(summary).to_csv(os.path.join(out, "summary.csv"), index=False, float_format="%.4f")
    # test: per defect x strength
    t, clean_t = d[d.kind == "test_fresh"], d[(d.kind == "clean") & (d.split == "test")]
    rows = []
    for v in VALUES:
        for s, g in t[t.defect == v].groupby("strength"):
            lab = np.r_[np.ones(len(g), bool), np.zeros(len(clean_t), bool)]
            rows.append(dict(defect=v, strength=s, n=len(g), true_mean=g[v].mean(), pred_mean=g[f"pred_{v}"].mean(),
                             auroc_vs_clean=QX.auroc(np.r_[g[f"pred_{v}"], clean_t[f"pred_{v}"]], lab)))
    ps = pd.DataFrame(rows)
    ps.to_csv(os.path.join(out, "per_defect_strength_test.csv"), index=False, float_format="%.4f")
    print("\n== test, matching output per defect x strength (true mean / predicted mean)")
    print(ps.assign(cell=[f"{a_:.2f} / {b_:.2f}" for a_, b_ in zip(ps.true_mean, ps.pred_mean)])
          .pivot(index="defect", columns="strength", values="cell").loc[VALUES].to_string())
    # clean originals: screened vs flagged
    pool = pd.read_csv(os.path.join(ROOT, "labels", "quality_clean_pool.csv")).set_index("image_id")
    c = d[d.kind == "clean"].copy()
    c["screened"] = c.image_id.map(pool.clean).astype(bool)
    c["reason"] = c.image_id.map(pool.not_clean_reason).fillna("")
    c["q_max"] = c[[f"pred_{v}" for v in VALUES]].max(1)
    print("\n== clean originals: largest output")
    print(c.groupby("screened").q_max.describe(percentiles=[0.5, 0.95]).round(4).to_string())
    neg = c[c.screened]
    rows = []
    for f, outs in FLAG_MAP.items():
        pos = c[c.reason.str.split(";").apply(lambda r: f in r)]
        sp, sn = pos[[f"pred_{o}" for o in outs]].max(1), neg[[f"pred_{o}" for o in outs]].max(1)
        lab = np.r_[np.ones(len(sp), bool), np.zeros(len(sn), bool)]
        rows.append(dict(flag=f, outputs="+".join(outs), n_flagged=len(pos), n_screened=len(neg),
                         auroc=QX.auroc(np.r_[sp, sn], lab), flagged_mean_q=sp.mean(), screened_mean_q=sn.mean()))
    fl = pd.DataFrame(rows)
    fl.to_csv(os.path.join(out, "flagged_originals.csv"), index=False, float_format="%.4f")
    c[["sample_id", "split", "screened", "reason", "q_max"]].to_csv(os.path.join(out, "clean_originals_qmax.csv"),
                                                                    index=False, float_format="%.6g")
    print("\n== screening-flagged originals vs screened-clean (matching outputs; indicative)")
    print(fl.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

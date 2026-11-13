"""Explainability, part 2: analyses X1-X4 and X6 on the explain_test.npz files and the predictions.

X1 head split: share of the head output that comes from the quality vector (C runs): SD over test images of W_q.q
                 vs W_f.f, and how many predictions change when W_q.q is removed.
X2 probe: can the image features alone tell the defect? Logistic / ridge on standardised PCA-256 features,
                 GroupKFold(5) by eye (group_id); targets: degraded or not, defect type (10 incl. clean), strength.
X3 drift: cosine distance between a degraded photo's features and its own clean photo's, by defect x strength,
                 and the change of the prediction; Spearman correlation of the two.
X4 errors: final graders (5-seed regression for Nuclear Cataract, 3-seed baseline for SUSTech-SYSU) per
                 defect x strength: QWK, accuracy (and MAE for Nuclear Cataract).
X6 per class: train / test class counts and the confusion matrices of the final graders (clean, degraded).
Writes runs/explain_082/<task>_*.csv and summary.json.
Usage: python grading/explain_analysis.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import cohen_kappa_score, confusion_matrix, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "data_prep"))
from run_log import RunLog  # noqa: E402

G = os.path.join(ROOT, "runs", "grading")
OUT = os.path.join(ROOT, "runs", "explain_082")
NC_CUTS = [0.63, 1.46, 2.35, 2.99, 3.95]   # final Nuclear Cataract grader
TASKS = {
    "nuclear_cataract": dict(n=6, A=f"{G}/nuclear_cataract_reg/A_seed14", B=f"{G}/nuclear_cataract_reg_BC/B_seed14",
                             C=f"{G}/nuclear_cataract_reg_BC/C_seed14",
                             final=[f"{G}/nuclear_cataract_reg/A_seed{s}" for s in range(14, 19)]),
    "sustech_sysu": dict(n=3, A=f"{G}/sustech_sysu_category_visionfm_ft/A_seed14",
                         B=f"{G}/sustech_sysu_category_visionfm_ft/B_seed14",
                         C=f"{G}/sustech_sysu_category_visionfm_ft/C_seed14",
                         final=[f"{G}/sustech_sysu_category_visionfm_ft/A_seed{s}" for s in (14, 15, 16)]),
}


def qwk(y, p):
    return float(cohen_kappa_score(y, p, weights="quadratic")) if len(set(y)) > 1 else float("nan")


def load(run):
    z = np.load(os.path.join(run, "explain_test.npz"), allow_pickle=True)
    pr = pd.read_csv(os.path.join(run, "predictions_test.csv")).set_index("sample_id").loc[z["sample_id"]]
    return z, pr.reset_index()


def final_predictions(t, cfg):
    ps = [pd.read_csv(os.path.join(r, "predictions_test.csv")) for r in cfg["final"]]
    df = ps[0][["sample_id", "image_id", "group_id", "kind", "defect", "strength", "grade"]].copy()
    if t == "nuclear_cataract":
        df["raw"] = np.mean([p.tta_raw.to_numpy() for p in ps], 0)
        df["pred"] = np.searchsorted(NC_CUTS, df.raw)
    else:
        df["pred"] = np.mean([p[[f"p{k}" for k in range(cfg["n"])]].to_numpy() for p in ps], 0).argmax(1)
    df["defect"] = df.defect.fillna("clean")
    return df


def x1(z, reg):
    cf, cq = z["contrib_f"], z["contrib_q"]
    r = dict(sd_contrib_image=float(cf.std(0).mean()), sd_contrib_quality=float(cq.std(0).mean()))
    r["quality_share_of_variance"] = float(cq.var(0).sum() / (cq.var(0).sum() + cf.var(0).sum()))
    if reg:
        r["mean_abs_shift_from_quality"] = float(np.abs(cq[:, 0]).mean())
        r["changed_class_without_q"] = float((np.searchsorted(NC_CUTS, z["out"][:, 0])
                                              != np.searchsorted(NC_CUTS, z["out"][:, 0] - cq[:, 0])).mean())
    else:
        r["changed_class_without_q"] = float((z["out"].argmax(1) != (z["out"] - cq).argmax(1)).mean())
    return r


def x2(z, pr):
    X = z["feats"].astype(np.float32)
    g = pr.group_id.to_numpy()
    defect = pr.defect.fillna("clean").to_numpy()
    targets = dict(degraded=(defect != "clean").astype(int), defect_type=defect)
    res = {}
    cv = GroupKFold(5)
    for name, y in targets.items():
        pred = np.empty(len(y), dtype=object)
        for tr, te in cv.split(X, y, g):
            m = make_pipeline(StandardScaler(), PCA(256, random_state=14), LogisticRegression(max_iter=2000, C=0.1))
            m.fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        acc = float((pred == y).mean())
        chance = float(pd.Series(y).value_counts(normalize=True).max())
        res[name] = dict(accuracy=acc, majority_baseline=chance)
    d = defect != "clean"
    y = pr.strength.to_numpy()[d]
    pred = np.empty(d.sum())
    for tr, te in cv.split(X[d], y, g[d]):
        m = make_pipeline(StandardScaler(), PCA(256, random_state=14), Ridge(10.0))
        m.fit(X[d][tr], y[tr])
        pred[te] = m.predict(X[d][te])
    res["strength_r2"] = float(r2_score(y, pred))
    return res


def x3(z, pr, reg):
    X = z["feats"].astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    out = z["out"][:, 0] if reg else z["out"].argmax(1)
    clean = {iid: i for i, (iid, k) in enumerate(zip(pr.image_id, pr.kind)) if k == "clean"}
    rows = []
    for i, (iid, k) in enumerate(zip(pr.image_id, pr.kind)):
        if k == "clean" or iid not in clean:
            continue
        j = clean[iid]
        rows.append(dict(defect=pr.defect[i], strength=pr.strength[i], cos_dist=float(1 - X[i] @ X[j]),
                         pred_change=float(abs(out[i] - out[j])) if reg else float(out[i] != out[j])))
    df = pd.DataFrame(rows)
    rho = spearmanr(df.cos_dist, df.pred_change).correlation
    return df.groupby(["defect", "strength"]).mean().reset_index(), float(rho)


def x4(df, reg):
    rows = []
    for (d, s), g in df.groupby(["defect", "strength"]):
        r = dict(defect=d, strength=s, n=len(g), qwk=qwk(g.grade, g.pred), accuracy=float((g.grade == g.pred).mean()))
        if reg:
            r["mae"] = float(np.abs(g.raw - g.grade).mean())
        rows.append(r)
    return pd.DataFrame(rows)


def x6(t, df, n):
    lab = pd.read_csv(os.path.join(ROOT, "Data", "Grading Data", t, "labels.csv"))
    if t == "sustech_sysu":   # final grader uses the category label; predictions_test.csv already holds it
        cat = pd.read_csv(os.path.join(ROOT, "labels", "sustech_master.csv")).set_index("image_id").category
        lab["grade"] = lab.image_id.map(cat)
    counts = lab[lab.kind == "clean"].groupby(["split", "grade"]).size().unstack(fill_value=0)
    cms = {k: confusion_matrix(g.grade, g.pred, labels=range(n)).tolist()
           for k, g in (("clean", df[df.kind == "clean"]), ("degraded", df[df.kind != "clean"]))}
    return counts, cms


def main():
    os.makedirs(OUT, exist_ok=True)
    summary = {}
    with RunLog("grading/explain_analysis.py", notes="X1-X4, X6") as log:
        for t, cfg in TASKS.items():
            reg = t == "nuclear_cataract"
            s = summary[t] = {}
            zc, _ = load(cfg["C"])
            s["X1_head_split_C"] = x1(zc, reg)
            s["X2_probe"] = {}
            for c in ("A", "B"):
                z, pr = load(cfg[c])
                s["X2_probe"][c] = x2(z, pr)
            z, pr = load(cfg["A"])
            drift, rho = x3(z, pr, reg)
            drift.to_csv(os.path.join(OUT, f"{t}_x3_drift.csv"), index=False, float_format="%.4f")
            s["X3_spearman_drift_vs_pred_change"] = rho
            fin = final_predictions(t, cfg)
            x4(fin, reg).to_csv(os.path.join(OUT, f"{t}_x4_errors.csv"), index=False, float_format="%.4f")
            counts, cms = x6(t, fin, cfg["n"])
            counts.to_csv(os.path.join(OUT, f"{t}_x6_class_counts.csv"))
            s["X6_confusion"] = cms
            print(t, json.dumps({k: v for k, v in s.items() if k != "X6_confusion"}, indent=1), flush=True)
        json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
        log.add(out=OUT, summary=summary)


if __name__ == "__main__":
    main()

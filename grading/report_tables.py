"""Report tables: recomputes from the saved test predictions the grader-improvement numbers quoted in the
report: seed ensembles, class weights (3 and 5 seeds, per seed), augmentation "all" on condition B, NC regression vs
classifier (plain rounding / tuned cuts, 3 and 5 seeds) with eye-level bootstrap CIs, regression on B / C, per-class
recall and per-class (one-vs-rest) AUROC.

Conventions (as eval_regression.py): classifier ensemble = mean softmax over seeds; regression ensemble = mean TTA
output over seeds, classified with the mean of the seeds' validation-tuned cut points ("tuned") or k + 0.5 ("plain").
Clean = kind "clean", degraded = every other test row (fresh code). CIs: 2,000 bootstrap resamples of eyes
(group_id), seed 14, paired. Writes results/report_tables/*.csv.
Usage (from the project root): python grading/report_tables.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_grading import qwk, to_class  # noqa: E402

ROOT = os.path.dirname(HERE)
G = os.path.join(ROOT, "runs", "grading")
OUT = os.path.join(ROOT, "results", "report_tables")
N_CLS = {"nuclear_cataract": 6, "sustech_sysu": 3}


def preds(run, cond, seeds):
    return [pd.read_csv(os.path.join(G, run, f"{cond}_seed{s}", "predictions_test.csv")) for s in seeds]


def cuts(run, cond, seeds):
    return np.mean([json.load(open(os.path.join(G, run, f"{cond}_seed{s}", "run_info.json")))["cuts"] for s in seeds], 0)


def softmax_ens(ps, n, prefix="p"):
    return np.mean([p[[f"{prefix}{k}" for k in range(n)]].to_numpy() for p in ps], 0)


def split_masks(df):
    clean = (df.kind == "clean").to_numpy()
    return {"clean": clean, "degraded": ~clean}


def boot_diff(y, g, pa, pb, n, m, reps=2000):
    eyes = np.unique(g[m])
    idx = {e: np.where(m & (g == e))[0] for e in eyes}
    rng = np.random.default_rng(14)
    out = []
    for _ in range(reps):
        ii = np.concatenate([idx[e] for e in rng.choice(eyes, len(eyes))])
        out.append(qwk(y[ii], pa[ii], n) - qwk(y[ii], pb[ii], n))
    return np.percentile(out, [2.5, 97.5])


def scores(y, p, n, m):
    return dict(qwk=qwk(y[m], p[m], n), acc=float((y[m] == p[m]).mean()), mae=float(np.abs(y[m] - p[m]).mean()))


def recall(y, p, n, m):
    return [float((p[m][y[m] == c] == c).mean()) if (y[m] == c).any() else np.nan for c in range(n)]


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []

    def add(table, task, run, test, **kw):
        rows.append(dict(table=table, task=task, run=run, test=test, **kw))

    # 1. seed ensembles of the fine-tuned VisionFM classifiers (A, B; seeds 14-16)
    for task, run in [("nuclear_cataract", "nuclear_cataract_grade_visionfm_ft"),
                      ("sustech_sysu", "sustech_sysu_category_visionfm_ft")]:
        n = N_CLS[task]
        for cond in "AB":
            ps = preds(run, cond, (14, 15, 16))
            y = ps[0].grade.to_numpy()
            ens = softmax_ens(ps, n).argmax(1)
            for test, m in split_masks(ps[0]).items():
                single = np.mean([qwk(y[m], p.pred.to_numpy()[m], n) for p in ps])
                single_acc = np.mean([(p.pred.to_numpy()[m] == y[m]).mean() for p in ps])
                add("seed_ensemble", task, f"{run}/{cond}", test, single_qwk=single, single_acc=single_acc,
                    **{f"ens_{k}": v for k, v in scores(y, ens, n, m).items()})

    # 2. class weights: per seed, 3- and 5-seed ensembles, against the baseline A ensemble
    for task, run, seeds_list in [("sustech_sysu", "sustech_sysu_cw", [(14, 15, 16), (14, 15, 16, 17, 18)]),
                                  ("nuclear_cataract", "nuclear_cataract_cw", [(14, 15, 16)])]:
        n = N_CLS[task]
        base_run = "sustech_sysu_category_visionfm_ft" if task == "sustech_sysu" else "nuclear_cataract_grade_visionfm_ft"
        base = softmax_ens(preds(base_run, "A", (14, 15, 16)), n).argmax(1)
        for seeds in seeds_list:
            ps = preds(run, "A", seeds)
            y, g = ps[0].grade.to_numpy(), ps[0].group_id.to_numpy()
            ens = softmax_ens(ps, n).argmax(1)
            for test, m in split_masks(ps[0]).items():
                lo, hi = boot_diff(y, g, ens, base, n, m)
                add("class_weights", task, f"{run}/A x{len(seeds)}", test, **scores(y, ens, n, m),
                    diff_vs_base=qwk(y[m], ens[m], n) - qwk(y[m], base[m], n), ci_low=lo, ci_high=hi,
                    recall=recall(y, ens, n, m))
                if len(seeds) == max(len(s) for s in seeds_list):
                    for s, p in zip(seeds, ps):
                        add("class_weights_seed", task, f"{run}/A_seed{s}", test, qwk=qwk(y[m], p.pred.to_numpy()[m], n),
                            recall=recall(y, p.pred.to_numpy(), n, m))

    # 3. class weights on B and C (SUSTech), incl. C shuffled
    ps_by = {c: preds("sustech_sysu_cw_BC", c, (14, 15, 16)) for c in "BC"}
    y = ps_by["B"][0].grade.to_numpy()
    for name, ens in [("B", softmax_ens(ps_by["B"], 3).argmax(1)), ("C", softmax_ens(ps_by["C"], 3).argmax(1)),
                      ("C_shuffled", softmax_ens(ps_by["C"], 3, "shuffled_p").argmax(1))]:
        for test, m in split_masks(ps_by["B"][0]).items():
            add("class_weights_BC", "sustech_sysu", f"sustech_sysu_cw_BC/{name}", test, **scores(y, ens, 3, m),
                recall=recall(y, ens, 3, m))

    # 4. augmentation "all" on condition B (3-seed softmax ensembles)
    for task in N_CLS:
        n = N_CLS[task]
        ps = preds(f"{task}_aug_all_B", "B", (14, 15, 16))
        y = ps[0].grade.to_numpy()
        ens = softmax_ens(ps, n).argmax(1)
        for test, m in split_masks(ps[0]).items():
            single = np.mean([qwk(y[m], p.pred.to_numpy()[m], n) for p in ps])
            add("aug_all_B", task, f"{task}_aug_all_B/B", test, single_qwk=single, **scores(y, ens, n, m),
                recall=recall(y, ens, n, m))

    # 5. NC regression vs classifier ensemble (3 and 5 seeds, plain / tuned cuts)
    n = 6
    clf_ps = preds("nuclear_cataract_grade_visionfm_ft", "A", (14, 15, 16))
    clf = softmax_ens(clf_ps, n).argmax(1)
    for seeds in [(14, 15, 16), (14, 15, 16, 17, 18)]:
        ps = preds("nuclear_cataract_reg", "A", seeds)
        ps = [p.set_index("sample_id").loc[clf_ps[0].sample_id].reset_index() for p in ps]
        y, g = ps[0].grade.to_numpy(), ps[0].group_id.to_numpy()
        raw = np.mean([p.tta_raw.to_numpy() for p in ps], 0)
        for cut_name, c in [("plain", [k + 0.5 for k in range(n - 1)]), ("tuned", cuts("nuclear_cataract_reg", "A", seeds))]:
            pred = to_class(raw, c)
            for test, m in split_masks(ps[0]).items():
                lo, hi = boot_diff(y, g, pred, clf, n, m)
                single = np.mean([qwk(y[m], p.pred.to_numpy()[m], n) for p in ps])
                add("nc_regression", "nuclear_cataract", f"reg A x{len(seeds)} {cut_name}", test, **scores(y, pred, n, m),
                    diff_vs_classifier=qwk(y[m], pred[m], n) - qwk(y[m], clf[m], n), ci_low=lo, ci_high=hi,
                    single_seed_qwk_tuned=single, cuts=list(np.round(c, 3)), recall=recall(y, pred, n, m))
    for test, m in split_masks(clf_ps[0]).items():
        y = clf_ps[0].grade.to_numpy()
        add("nc_regression", "nuclear_cataract", "classifier A x3", test, **scores(y, clf, n, m), recall=recall(y, clf, n, m))

    # 6. regression on B and C (3-seed TTA ensembles), plain / tuned cuts
    for cond in "BC":
        ps = preds("nuclear_cataract_reg_BC", cond, (14, 15, 16))
        y = ps[0].grade.to_numpy()
        raw = np.mean([p.tta_raw.to_numpy() for p in ps], 0)
        for cut_name, c in [("plain", [k + 0.5 for k in range(n - 1)]), ("tuned", cuts("nuclear_cataract_reg_BC", cond, (14, 15, 16)))]:
            pred = to_class(raw, c)
            for test, m in split_masks(ps[0]).items():
                add("nc_regression_BC", "nuclear_cataract", f"reg {cond} x3 {cut_name}", test, **scores(y, pred, n, m))

    # 7. per-class one-vs-rest AUROC of the condition-A classifier ensembles
    for task, run in [("nuclear_cataract", "nuclear_cataract_grade_visionfm_ft"),
                      ("sustech_sysu", "sustech_sysu_category_visionfm_ft")]:
        n = N_CLS[task]
        ps = preds(run, "A", (14, 15, 16))
        y = ps[0].grade.to_numpy()
        P = softmax_ens(ps, n)
        for test, m in split_masks(ps[0]).items():
            per = [roc_auc_score(y[m] == k, P[m, k]) if 0 < (y[m] == k).sum() < m.sum() else np.nan for k in range(n)]
            add("per_class_auroc", task, f"{run}/A x3", test, auroc=[round(a, 3) for a in per],
                macro=roc_auc_score(y[m], P[m], multi_class="ovr", average="macro"))

    # 8. per-class accuracy (recall), the three seeds pooled, conditions A / B / C / C_oracle, both backbones
    for task, runs in [("nuclear_cataract", ["nuclear_cataract_grade_visionfm_ft", "nuclear_cataract_grade"]),
                       ("sustech_sysu", ["sustech_sysu_category_visionfm_ft", "sustech_sysu_category"])]:
        n = N_CLS[task]
        for run in runs:
            for cond in ["A", "B", "C", "C_oracle"]:
                d = pd.concat(preds(run, cond, (14, 15, 16)))
                y, p = d.grade.to_numpy(), d.pred.to_numpy()
                for test, m in split_masks(d).items():
                    add("per_class_recall", task, f"{run}/{cond}", test, recall=[round(r, 2) for r in recall(y, p, n, m)],
                        overall=float((y[m] == p[m]).mean()))

    df = pd.DataFrame(rows)
    for t, d in df.groupby("table"):
        d.dropna(axis=1, how="all").to_csv(os.path.join(OUT, f"{t}.csv"), index=False, float_format="%.4f")
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60):
        for t, d in df.groupby("table", sort=False):
            print(f"\n== {t}\n", d.dropna(axis=1, how="all").drop(columns=["table"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()

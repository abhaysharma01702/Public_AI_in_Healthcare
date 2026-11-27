"""Metrics of the grading runs. Reads <runs>/<cond>_seed<s>/predictions_test.csv.

Subsets of the test set: clean (the original test photos), split as well into clean_screened (passed the quality
screening) and clean_flagged (flagged as blurry / dark / bright; 7 photos for SUSTech-SYSU: indicative only),
degraded (all 9 fresh defects x 5 strengths), and each defect, each strength, each defect x strength.
Metrics: quadratic weighted kappa (primary), accuracy, macro F1.
Seeds: the metric is computed per seed and averaged (mean, and SD across seeds).
95% CIs: bootstrap over test EYES (group_id, same-eye photos all versions resampled together; fix, 2026-11-01: photos of one eye are not independent; round 1 first used image_id, `--boot_unit image_id`),
1,000 resamples, seed 14; each resample recomputes the seed-averaged metric. These CIs keep the trained models fixed;
the spread across seeds is the separate column sd_seeds.
Per-category subsets (dataset general pattern from --master; no CI): clean_cat=<name>, degraded_cat=<name>.
Paired differences (B - A, C - B, C_oracle - B, C - C_shuffled, C_oracle - C) use the same resampled eyes for both
conditions; "better" = CI excludes 0.
Outputs in <runs>/: metrics.csv (condition x subset), paired.csv, and a printed summary.
Extra test sets: --pred test_seen | test_holdout reads predictions_<pred>.csv (predict_grading.py) and writes
metrics_<pred>.csv / paired_<pred>.csv; "degraded" = every test_<impl> image. For the holdout set also
degraded_unrelated6: without defocus, occlusion and uneven illumination, whose holdout models have close relatives
among the.
Usage: python grading/evaluate_grading.py --runs runs/grading/sustech_sysu [--pred test_seen]
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

B, SEED = 1000, 14
PAIRS = [("A", "B"), ("B", "C"), ("B", "C_oracle"), ("C_shuffled", "C"), ("C_oracle_shuffled", "C_oracle"),
         ("C", "C_oracle"), ("A", "C"), ("A", "C_oracle")]   # A -> C added 2026-11-02
RELATED_059 = ["defocus_blur", "occlusion", "uneven_illumination"]   # holdout models with training relatives


def confusion(y, p, k):
    O = np.zeros((k, k))
    np.add.at(O, (np.asarray(y, int), np.asarray(p, int)), 1)
    return O


def qwk(y, p, k):
    """= sklearn cohen_kappa_score(y, p, weights="quadratic", labels=range(k)), NaN if undefined (numpy only,
    sklearn is not in the Ada environment; checked equal on random and edge cases)."""
    O = confusion(y, p, k)
    i = np.arange(k)
    W = (i[:, None] - i[None, :]) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    den = (W * E).sum()
    return float(1 - (W * O).sum() / den) if den > 0 else float("nan")


def macro_f1(y, p, k):
    """= sklearn f1_score(y, p, average="macro", labels=range(k), zero_division=0)."""
    O = confusion(y, p, k)
    tp = np.diag(O)
    den = 2 * tp + (O.sum(0) - tp) + (O.sum(1) - tp)
    return float(np.mean(np.divide(2 * tp, den, out=np.zeros(k), where=den > 0)))


def metric_fns(n_classes):
    return {"qwk": lambda y, p: qwk(y, p, n_classes),
            "acc": lambda y, p: float(np.mean(y == p)),
            "macro_f1": lambda y, p: macro_f1(y, p, n_classes)}


def load(runs, pred="test"):
    """{condition: [predictions per seed]}; C / C_oracle runs also give C_shuffled / C_oracle_shuffled."""
    out = {}
    for d in sorted(glob.glob(os.path.join(runs, "*_seed*"))):
        cond, seed = re.match(r"(.+)_seed(\d+)$", os.path.basename(d)).groups()
        f = os.path.join(d, f"predictions_{pred}.csv")
        if not os.path.exists(f):
            continue
        p = pd.read_csv(f).sort_values("sample_id").reset_index(drop=True)
        out.setdefault(cond, []).append(p.assign(seed=int(seed)))
        if "shuffled_pred" in p:
            out.setdefault(cond + "_shuffled", []).append(p.assign(pred=p.shuffled_pred, seed=int(seed)))
    return out


def subsets(df):
    yield "clean", df.kind == "clean"
    yield "clean_screened", (df.kind == "clean") & df.screened
    yield "clean_flagged", (df.kind == "clean") & ~df.screened
    deg = df.kind.str.startswith("test_")   # test_fresh  or an extra test set
    yield "degraded", deg
    # decided before any grading result: strong occlusion hides the cornea, so also without it
    yield "degraded_no_occlusion", deg & (df.defect != "occlusion")
    if (df.kind == "test_holdout").any():
        yield "degraded_unrelated6", deg & ~df.defect.isin(RELATED_059)
    for v in sorted(df.defect.dropna().unique()):
        yield f"defect={v}", df.defect == v
    for s in sorted(df.strength[deg].unique()):
        yield f"strength={s:g}", deg & (df.strength == s)
    for v in sorted(df.defect.dropna().unique()):
        for s in sorted(df.strength[deg].unique()):
            yield f"{v}@{s:g}", (df.defect == v) & (df.strength == s)
    for c in sorted(df.category_name.dropna().unique()):
        yield f"clean_cat={c}", (df.kind == "clean") & (df.category_name == c)
        yield f"degraded_cat={c}", deg & (df.category_name == c)


def seed_mean(preds, mask, fn, rows=None):
    vals = []
    for p in preds:
        q = p[mask] if rows is None else p[mask].iloc[rows]
        vals.append(fn(q.grade.to_numpy(), q.pred.to_numpy()))
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan


def boot_rows(df_masked, rng, unit="group_id"):
    """Row positions of one bootstrap resample of eyes / photos (all rows of a picked unit, repeated as often as picked)."""
    groups = df_masked.groupby(unit).indices
    ids = np.array(sorted(groups))
    pick = rng.integers(0, len(ids), len(ids))
    return np.concatenate([groups[ids[i]] for i in pick])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--bootstrap", type=int, default=B)
    ap.add_argument("--pool", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                   "labels", "quality_clean_pool.csv"), help="screening result ")
    ap.add_argument("--master", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                     "labels", "sustech_master.csv"), help="category per image")
    ap.add_argument("--boot_unit", choices=["group_id", "image_id"], default="group_id")
    ap.add_argument("--pred", default="test", help="predictions_<pred>.csv; test = the original fresh test ")
    a = ap.parse_args()
    P = load(a.runs, a.pred)
    sfx = "" if a.pred == "test" else f"_{a.pred}"
    ref = next(iter(P.values()))[0]
    n_classes = int(ref.grade.max()) + 1
    for c in P:   # every run must have the same test rows in the same order
        for p in P[c]:
            assert (p.sample_id.to_numpy() == ref.sample_id.to_numpy()).all()
    pool = pd.read_csv(a.pool).set_index("image_id").clean
    assert ref.image_id.isin(pool.index).all(), "test photo missing from the screening table"
    ref = ref.assign(screened=ref.image_id.map(pool).astype(bool).to_numpy())
    mst = pd.read_csv(a.master).set_index("image_id")
    if "category_name" in mst:   # SUSTech-SYSU general pattern; other tasks have none
        assert ref.image_id.isin(mst.index).all()
        ref = ref.assign(category_name=ref.image_id.map(mst.category_name).to_numpy())
    else:
        ref = ref.assign(category_name=np.nan)
    fns = metric_fns(n_classes)
    main_subsets = {"clean", "clean_screened", "clean_flagged", "degraded", "degraded_no_occlusion", "degraded_unrelated6"}
    rows, prs = [], []
    for name, mask in subsets(ref):
        mask = mask.to_numpy()
        if not mask.any():   # e.g. no flagged photo in a smoke subset
            continue
        sub = ref[mask].reset_index(drop=True)
        boot = name in main_subsets or name.startswith(("defect=", "strength="))
        samples = []
        if boot:
            rng = np.random.default_rng(SEED)
            samples = [boot_rows(sub, rng, a.boot_unit) for _ in range(a.bootstrap)]
        for cond, preds in P.items():
            for m, fn in fns.items():
                mean, sd = seed_mean(preds, mask, fn)
                r = dict(subset=name, condition=cond, metric=m, n_images=int(mask.sum()), n_seeds=len(preds),
                         mean=mean, sd_seeds=sd, ci_low=np.nan, ci_high=np.nan)
                if boot and m == "qwk":
                    bs = [seed_mean(preds, mask, fn, rows=s)[0] for s in samples]
                    r["ci_low"], r["ci_high"] = np.nanpercentile(bs, [2.5, 97.5])
                rows.append(r)
        if boot:
            for first, second in PAIRS:
                if first not in P or second not in P:
                    continue
                fn = fns["qwk"]
                d0 = seed_mean(P[second], mask, fn)[0] - seed_mean(P[first], mask, fn)[0]
                bs = [seed_mean(P[second], mask, fn, rows=s)[0] - seed_mean(P[first], mask, fn, rows=s)[0]
                      for s in samples]
                lo, hi = np.nanpercentile(bs, [2.5, 97.5])
                prs.append(dict(subset=name, first=first, second=second, metric="qwk", difference=d0, ci_low=lo,
                                ci_high=hi, excludes_0=bool(lo > 0 or hi < 0)))
    M, Q = pd.DataFrame(rows), pd.DataFrame(prs)
    M.to_csv(os.path.join(a.runs, f"metrics{sfx}.csv"), index=False, float_format="%.4f")
    Q.to_csv(os.path.join(a.runs, f"paired{sfx}.csv"), index=False, float_format="%.4f")
    main_list = [s for s in ["clean", "clean_screened", "clean_flagged", "degraded", "degraded_no_occlusion",
                             "degraded_unrelated6"] if s in set(M.subset)]
    show = M[M.subset.isin(main_list) & (M.metric == "qwk")].pivot_table(index="condition", columns="subset",
                                                                          values="mean").reindex(columns=main_list)
    print(show.round(3).to_string())
    if len(Q):   # no pairs when only one condition was run (E3)
        print(Q[Q.subset.isin(main_list)].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

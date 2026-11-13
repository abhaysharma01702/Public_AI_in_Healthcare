"""Filtering check (rule fixed ): model B grades only the test images the quality module does not
rate "poor"; is discarding bad images better than using the quality values (C)?
"Poor" = the largest of the 9 D3 quality outputs >= threshold. Main threshold 0.55 (the "poor" boundary fixed
before any filtering result); 0.25 and 0.75 as a sensitivity check. No retraining, no tuning on test results.
Per task, test set (clean / degraded of the fresh, seen and holdout code) and threshold:
  rejected share; B on kept images; B on rejected images; B and C on all images; C on the kept images.
Metrics: QWK (primary), accuracy; mean of 3 seeds. 95% CIs: bootstrap over groups (SUSTech-SYSU eyes, Nuclear Cataract
volunteers; 1,resamples, seed 14), as evaluate_grading.py. Paired differences on the same resamples:
  B_kept - B_all (what filtering buys on the images it keeps), B_kept - C_kept (B vs C on the same kept images).
Output: results/grading/filtering/filtering.csv (+ printed summary).
Usage: python grading/filtering_grading.py
"""
import os

import numpy as np
import pandas as pd

from evaluate_grading import load, boot_rows

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = os.path.join(ROOT, "Data", "Grading Data")
TASKS = {"sustech_sysu": "runs/grading/sustech_sysu_category",   # round 2
         "nuclear_cataract": "runs/grading/nuclear_cataract_grade"}
SETS = {"test": "", "test_seen": "_test_seen", "test_holdout": "_test_holdout"}   # pred name -> data folder suffix
THRESHOLDS = [0.55, 0.25, 0.75]
B, SEED = 1000, 14


def qwk(y, p, k):
    """Quadratic weighted kappa (same as sklearn cohen_kappa_score(weights="quadratic", labels=range(k)))."""
    O = np.zeros((k, k))
    np.add.at(O, (y, p), 1)
    i = np.arange(k)
    W = (i[:, None] - i[None, :]) ** 2 / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return 1 - (W * O).sum() / (W * E).sum() if (W * E).sum() > 0 else 0.0


def score(preds, rows, k):
    """Seed-mean QWK and accuracy on the given row positions."""
    q, a = [], []
    for y, p in preds:
        q.append(qwk(y[rows], p[rows], k))
        a.append(float(np.mean(y[rows] == p[rows])))
    return float(np.mean(q)), float(np.mean(a))


def main():
    out = []
    for task, runs in TASKS.items():
        for pred, sfx in SETS.items():
            P = load(os.path.join(ROOT, runs), pred)
            ref = P["B"][0]
            qv = pd.read_csv(os.path.join(G, task + sfx, "quality_d3.csv")).set_index("sample_id")
            qmax = qv.loc[ref.sample_id, [c for c in qv.columns if c.startswith("q_")]].max(axis=1).to_numpy()
            k = int(max(p.grade.max() for c in P for p in P[c])) + 1
            arr = {c: [(p.grade.to_numpy(), p.pred.to_numpy()) for p in P[c]] for c in ["B", "C"]}
            for c in ["B", "C"]:   # same rows, same order in every run
                assert all((p.sample_id.to_numpy() == ref.sample_id.to_numpy()).all() for p in P[c])
            for name, mask in [("clean", ref.kind == "clean"), ("degraded", ref.kind.str.startswith("test_"))]:
                sub = ref[mask.to_numpy()].reset_index(drop=True)
                idx = np.flatnonzero(mask.to_numpy())
                for t in THRESHOLDS:
                    keep = qmax[idx] < t
                    r = dict(task=task, set=pred, subset=name, threshold=t, n=len(idx),
                             rejected=float(1 - keep.mean()))
                    for lab, rows in [("B_all", idx), ("C_all", idx), ("B_kept", idx[keep]), ("C_kept", idx[keep]),
                                      ("B_rejected", idx[~keep])]:
                        if len(rows) == 0:
                            continue
                        r[f"{lab}_qwk"], r[f"{lab}_acc"] = score(arr[lab[0]], rows, k)
                    # bootstrap over groups: CIs of B_kept and of the paired differences
                    rng = np.random.default_rng(SEED)
                    bk, d1, d2 = [], [], []
                    for _ in range(B if keep.any() else 0):
                        pos = boot_rows(sub, rng)
                        kp = pos[keep[pos]]
                        if len(kp) == 0:
                            continue
                        b_kept = score(arr["B"], idx[kp], k)[0]
                        bk.append(b_kept)
                        d1.append(b_kept - score(arr["B"], idx[pos], k)[0])
                        d2.append(b_kept - score(arr["C"], idx[kp], k)[0])
                    for lab, v in [("B_kept_qwk", bk), ("Bkept_minus_Ball", d1), ("Bkept_minus_Ckept", d2)]:
                        if v:
                            r[f"{lab}_lo"], r[f"{lab}_hi"] = np.percentile(v, [2.5, 97.5])
                    r["Bkept_minus_Ball"] = r.get("B_kept_qwk", np.nan) - r["B_all_qwk"]
                    r["Bkept_minus_Ckept"] = r.get("B_kept_qwk", np.nan) - r.get("C_kept_qwk", np.nan)
                    out.append(r)
                    print(f"{task:16s} {pred:12s} {name:8s} t={t:.2f} rejected {r['rejected']:.1%}  "
                          f"B all {r['B_all_qwk']:.3f}  B kept {r.get('B_kept_qwk', np.nan):.3f}  "
                          f"C all {r['C_all_qwk']:.3f}  C kept {r.get('C_kept_qwk', np.nan):.3f}  "
                          f"B rejected {r.get('B_rejected_qwk', np.nan):.3f}", flush=True)
    os.makedirs(os.path.join(ROOT, "results", "grading", "filtering"), exist_ok=True)
    pd.DataFrame(out).round(4).to_csv(os.path.join(ROOT, "results", "grading", "filtering", "filtering.csv"),
                                      index=False)


if __name__ == "__main__":
    main()

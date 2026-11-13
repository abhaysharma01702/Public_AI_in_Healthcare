"""Post-hoc uses of the quality module with the trained graders.
No training: reads predictions_<set>.csv of finished runs and the D3 quality values of the same test images.
q = max of the 9 D3 defect probabilities (higher = worse photo).
1. Gating (referral): keep the fraction c of test photos with the lowest q, grade only those; QWK of the kept photos vs
   coverage c. Baselines: random referral (= QWK of all photos in expectation, 50 random draws averaged), the
   grader's own confidence (keep the highest max-probability photos), and combined (mean of the quality and uncertainty ranks).
2. Routing: photo with q < T (T = 0.55, the filtering rule) graded by model A, else by model B (or C).
3. Ensemble: mean of the DenseNet121 and fine-tuned VisionFM probabilities, same condition and seed.
QWK per seed, mean over seeds 14 / 15 / 16; 95 % CIs: bootstrap over eyes (group_id), as evaluate_grading.py.
Usage: python quality_posthoc.py --dense <run folder> --vfm <run folder> --quality <quality_d3.csv> --pred test --out <dir>
"""
import argparse
import os

import numpy as np
import pandas as pd

from evaluate_grading import qwk

SEEDS = (14, 15, 16)
COVERAGE = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)
T = 0.55


def load(folder, cond, pred, q):
    out = {}
    for s in SEEDS:
        d = pd.read_csv(os.path.join(folder, f"{cond}_seed{s}", f"predictions_{pred}.csv"))
        out[s] = d.merge(q, on="sample_id", how="left", validate="1:1")
        assert out[s].q.notna().all(), f"missing quality values in {folder} {cond}"
    return out


def probs(d):
    return d[[c for c in d.columns if c.startswith("p") and c[1:].isdigit()]].to_numpy()


def mean_qwk(frames, rows, k, pred_fn):
    """QWK of the rows (positions) averaged over seeds; pred_fn(seed) -> predicted classes of all rows."""
    return float(np.mean([qwk(frames[s].grade.to_numpy()[rows], pred_fn(s)[rows], k) for s in SEEDS]))


def rank(x):
    """0 .. 1 rank, ties by position (as argsort stable)."""
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r / max(1, len(x) - 1)


def boot(groups, n, rng):
    ids = np.array(sorted(groups))
    return [np.concatenate([groups[ids[i]] for i in rng.integers(0, len(ids), len(ids))]) for _ in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", required=True)
    ap.add_argument("--vfm", required=True)
    ap.add_argument("--quality", required=True)
    ap.add_argument("--pred", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    q = pd.read_csv(a.quality)
    q = pd.DataFrame({"sample_id": q.sample_id, "q": q.filter(like="q_").max(1)})
    rng = np.random.default_rng(14)
    runs = {(m, c): load(f, c, a.pred, q) for m, f in (("DenseNet121", a.dense), ("VisionFM_ft", a.vfm))
            for c in ("A", "B", "C")}
    ref = runs[("DenseNet121", "A")][14]
    for v in runs.values():   # same photos in the same order in every run
        for d in v.values():
            assert (d.sample_id.to_numpy() == ref.sample_id.to_numpy()).all()
    k = int(probs(ref).shape[1])
    n = len(ref)
    groups = ref.groupby("group_id").indices
    samples = boot(groups, a.bootstrap, rng)
    subsets = {"all": np.arange(n), "clean": np.flatnonzero(ref.kind == "clean"),
               "degraded": np.flatnonzero(ref.kind != "clean")}

    # 1. gating
    rows = []
    qv = ref.q.to_numpy()
    for (m, c), fr in runs.items():
        pred = lambda s, fr=fr: fr[s].pred.to_numpy()
        conf = {s: probs(fr[s]).max(1) for s in SEEDS}
        for cov in COVERAGE:
            m_keep = max(1, int(round(cov * n)))
            keep_q = np.argsort(qv, kind="stable")[:m_keep]
            keep_c = {s: np.argsort(-conf[s], kind="stable")[:m_keep] for s in SEEDS}
            r = dict(model=m, condition=c, coverage=cov, n_kept=m_keep,
                     kept_clean=int((ref.kind.to_numpy()[keep_q] == "clean").sum()),
                     qwk_quality=mean_qwk(fr, keep_q, k, pred),
                     qwk_confidence=float(np.mean([qwk(fr[s].grade.to_numpy()[keep_c[s]], pred(s)[keep_c[s]], k)
                                                   for s in SEEDS])),
                     qwk_random=float(np.mean([mean_qwk(fr, np.sort(rng.permutation(n)[:m_keep]), k, pred)
                                               for _ in range(50)])))
            # combined: mean of the quality rank and the uncertainty rank, keep the lowest
            comb = {s: np.argsort(rank(qv) + rank(-conf[s]), kind="stable")[:m_keep] for s in SEEDS}
            r["qwk_combined"] = float(np.mean([qwk(fr[s].grade.to_numpy()[comb[s]], pred(s)[comb[s]], k)
                                               for s in SEEDS]))
            if cov < 1:   # CI of combined minus confidence-only referral (paired, eyes resampled)
                d = []
                for b in samples:
                    mb = max(1, int(round(cov * len(b))))
                    v = []
                    for s in SEEDS:
                        y, p, cb = fr[s].grade.to_numpy()[b], pred(s)[b], conf[s][b]
                        kc = np.argsort(rank(qv[b]) + rank(-cb), kind="stable")[:mb]
                        kf = np.argsort(-cb, kind="stable")[:mb]
                        v.append(qwk(y[kc], p[kc], k) - qwk(y[kf], p[kf], k))
                    d.append(np.mean(v))
                r["comb_vs_conf"] = r["qwk_combined"] - r["qwk_confidence"]
                r["comb_vs_conf_ci_low"], r["comb_vs_conf_ci_high"] = np.percentile(d, [2.5, 97.5])
            if cov < 1:   # CI of quality-gated minus all photos (paired, eyes resampled)
                d = []
                for b in samples:
                    kb = b[qv[b] <= np.sort(qv[b])[max(1, int(round(cov * len(b)))) - 1]]
                    d.append(mean_qwk(fr, kb, k, pred) - mean_qwk(fr, b, k, pred))
                r["gain_vs_all"] = r["qwk_quality"] - mean_qwk(fr, np.arange(n), k, pred)
                r["gain_ci_low"], r["gain_ci_high"] = np.percentile(d, [2.5, 97.5])
            rows.append(r)
            print(m, c, cov, {x: round(y, 3) for x, y in r.items() if isinstance(y, float)}, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, f"gating_{a.pred}.csv"), index=False, float_format="%.4f")

    # 2. routing and 3. ensemble: QWK per subset with CIs; paired difference to the best single model
    def evaluate(name, pred_fn, base_fn, base_name):
        out = []
        for sub, idx in subsets.items():
            v = mean_qwk(runs[("DenseNet121", "A")], idx, k, pred_fn)
            bset = [b[np.isin(b, idx)] for b in samples]
            vals = np.array([mean_qwk(runs[("DenseNet121", "A")], b, k, pred_fn) for b in bset])
            diff = np.array([vals[i] - mean_qwk(runs[("DenseNet121", "A")], b, k, base_fn) for i, b in enumerate(bset)])
            out.append(dict(method=name, subset=sub, qwk=v, ci_low=np.percentile(vals, 2.5),
                            ci_high=np.percentile(vals, 97.5), vs=base_name,
                            difference=v - mean_qwk(runs[("DenseNet121", "A")], idx, k, base_fn),
                            diff_ci_low=np.percentile(diff, 2.5), diff_ci_high=np.percentile(diff, 97.5),
                            routed_to_second=float((qv[idx] >= T).mean())))
            print(out[-1], flush=True)
        return out

    rows = []
    for m in ("DenseNet121", "VisionFM_ft"):
        for c in ("A", "B", "C"):
            p = lambda s, m=m, c=c: runs[(m, c)][s].pred.to_numpy()
            rows += evaluate(f"{m} {c}", p, p, f"{m} {c}")   # single models (difference 0)
        for second in ("B", "C"):
            route = lambda s, m=m, second=second: np.where(qv < T, runs[(m, "A")][s].pred.to_numpy(),
                                                           runs[(m, second)][s].pred.to_numpy())
            rows += evaluate(f"{m} route A/{second}", route, lambda s, m=m, second=second:
                             runs[(m, second)][s].pred.to_numpy(), f"{m} {second}")
    for c in ("A", "B", "C"):
        ens = lambda s, c=c: (probs(runs[("DenseNet121", c)][s]) + probs(runs[("VisionFM_ft", c)][s])).argmax(1)
        rows += evaluate(f"ensemble {c}", ens, lambda s, c=c: runs[("VisionFM_ft", c)][s].pred.to_numpy(),
                         f"VisionFM_ft {c}")
    pd.DataFrame(rows).to_csv(os.path.join(a.out, f"routing_ensemble_{a.pred}.csv"), index=False, float_format="%.4f")
    print("done", a.pred)


if __name__ == "__main__":
    main()

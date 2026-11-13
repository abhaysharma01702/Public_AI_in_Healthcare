"""Threshold-free metrics of the quality module (section 2 and B).

Per defect (present = measured label > 0):
  auroc            detection: present vs. absent, from the predicted value
  mae_present      strength error on images that have the defect
  spearman_present strength ranking on images that have the defect
  mae_all          error over all images (absent ones should predict 0)
  false_alarm      mean prediction on images without the defect
'mean' rows average the 9 defects. Groups: overall, per dataset, per lighting.
95% CIs (B): percentile bootstrap over source photos (all versions of a photo resampled together).
Cross-talk: mean prediction of each output on clean images and on images with exactly one defect.
Real check: AUROC of the matching output(s), real defect photos vs. clean val / test photos.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata

METRICS = ["auroc", "mae_present", "spearman_present", "mae_all", "false_alarm"]

# Real-check defect types noted by eye  -> outputs that should flag them.
# 'framing' was dropped as a defect, so it only enters the 'any' row (max over all 9 outputs).
REAL_MAP = {
    "blur": ["defocus_blur", "motion_blur"],
    "underexposure": ["underexposure"],
    "overexposure": ["overexposure"],
    "overexposure_glare": ["overexposure", "glare"],
    "occlusion": ["occlusion"],
    "framing_occlusion": ["occlusion"],
    "occlusion_blur": ["occlusion", "defocus_blur", "motion_blur"],
    "framing": [],
}


def _nanmean(a, axis=None):
    """nanmean without the warning for all-NaN slices (small groups); returns NaN there."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def pred_cols(values):
    return [f"pred_{v}" for v in values]


def auroc(score, pos):
    n1 = int(pos.sum())
    n0 = len(pos) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def spearman(a, b):
    if len(a) < 3:
        return np.nan
    ra, rb = rankdata(a), rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else np.nan


def defect_metrics(y, p):
    """y, p: [N, K]. Returns {metric: array[K]}."""
    K = y.shape[1]
    out = {m: np.full(K, np.nan) for m in METRICS}
    for k in range(K):
        pres = y[:, k] > 0
        out["auroc"][k] = auroc(p[:, k], pres)
        if pres.any():
            out["mae_present"][k] = np.abs(p[pres, k] - y[pres, k]).mean()
            out["spearman_present"][k] = spearman(y[pres, k], p[pres, k])
        out["mae_all"][k] = np.abs(p[:, k] - y[:, k]).mean()
        if (~pres).any():
            out["false_alarm"][k] = p[~pres, k].mean()
    return out


def summary(y, p, values):
    """Flat dict for the per-epoch log: mean over defects + per defect."""
    m = defect_metrics(y, p)
    d = {f"mean_{name}": float(_nanmean(arr)) for name, arr in m.items()}
    for name in ("auroc", "spearman_present", "mae_present"):
        d.update({f"{name}/{v}": float(m[name][k]) for k, v in enumerate(values)})
    return d


def _clusters(ids):
    codes, inv = np.unique(ids, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.cumsum(np.bincount(inv, minlength=len(codes)))[:-1]
    return np.split(order, bounds)


def _rows(group_type, group, values, point, boot, y):
    rows = []
    for metric in METRICS:
        per = list(enumerate(values)) + [(None, "mean")]
        for k, name in per:
            val = point[metric][k] if k is not None else _nanmean(point[metric])
            lo = hi = np.nan
            if boot is not None:
                b = boot[metric][:, k] if k is not None else _nanmean(boot[metric], axis=1)
                b = b[~np.isnan(b)]
                if len(b):
                    lo, hi = np.percentile(b, [2.5, 97.5])
            n_present = int((y[:, k] > 0).sum()) if k is not None else int((y > 0).any(1).sum())
            rows.append(dict(group_type=group_type, group=group, defect=name, metric=metric, value=val,
                             ci_low=lo, ci_high=hi, n_images=len(y), n_present=n_present))
    return rows


def evaluate_predictions(df, values, bootstrap=0, seed=14):
    """Long table of metrics (with bootstrap CIs if bootstrap > 0) for overall / dataset / lighting groups."""
    groups = [("overall", "all", np.ones(len(df), bool))]
    for col in ("dataset", "lighting"):
        for g in sorted(df[col].dropna().unique()):
            groups.append((col, g, (df[col] == g).to_numpy()))
    rng = np.random.default_rng(seed)
    rows = []
    for gt, g, mask in groups:
        sub = df[mask]
        y, p = sub[values].to_numpy(float), sub[pred_cols(values)].to_numpy(float)
        point = defect_metrics(y, p)
        boot = None
        if bootstrap:
            cl = _clusters(sub.source_image_id.to_numpy())
            reps = {m: np.empty((bootstrap, len(values))) for m in METRICS}
            for b in range(bootstrap):
                idx = np.concatenate([cl[c] for c in rng.integers(0, len(cl), len(cl))])
                mb = defect_metrics(y[idx], p[idx])
                for m in METRICS:
                    reps[m][b] = mb[m]
            boot = reps
        rows += _rows(gt, g, values, point, boot, y)
    return pd.DataFrame(rows)


def paired_comparison(df_a, df_b, values, bootstrap=1000, seed=14, names=("ours", "alt")):
    """Generalisation test: the same source photos with defects from two implementations.
    Per defect and metric: value on A, value on B, B - A, and a 95% CI of B - A from a PAIRED bootstrap
    (the same resampled source photos in both sets)."""
    ids = np.intersect1d(df_a.source_image_id.unique(), df_b.source_image_id.unique())
    sets = []
    for df in (df_a, df_b):
        d = df[df.source_image_id.isin(ids)]
        groups = d.groupby("source_image_id").indices
        sets.append((d[values].to_numpy(float), d[pred_cols(values)].to_numpy(float), [groups[i] for i in ids]))
    point = [defect_metrics(y, p) for y, p, _ in sets]
    rng = np.random.default_rng(seed)
    diffs = {m: np.empty((bootstrap, len(values))) for m in METRICS}
    for b in range(bootstrap):
        pick = rng.integers(0, len(ids), len(ids))
        res = []
        for y, p, g in sets:
            idx = np.concatenate([g[i] for i in pick])
            res.append(defect_metrics(y[idx], p[idx]))
        for m in METRICS:
            diffs[m][b] = res[1][m] - res[0][m]
    rows = []
    for m in METRICS:
        for k, v in list(enumerate(values)) + [(None, "mean")]:
            a = point[0][m][k] if k is not None else _nanmean(point[0][m])
            c = point[1][m][k] if k is not None else _nanmean(point[1][m])
            d = diffs[m][:, k] if k is not None else _nanmean(diffs[m], axis=1)
            d = d[~np.isnan(d)]
            lo, hi = np.percentile(d, [2.5, 97.5]) if len(d) else (np.nan, np.nan)
            rows.append({"defect": v, "metric": m, "first": names[0], "second": names[1], names[0]: a, names[1]: c, "difference": c - a,
                         "diff_ci_low": lo, "diff_ci_high": hi, "n_sources": len(ids)})
    return pd.DataFrame(rows)


def cross_talk(df, values):
    """Rows: clean images, then images with exactly one defect; columns: mean prediction per output."""
    P = pred_cols(values)
    rows = [("clean", df[df.n_defects == 0])] + [(v, df[(df.n_defects == 1) & (df.defects == v)]) for v in values]
    out = pd.DataFrame([sub[P].mean().to_numpy() for _, sub in rows], columns=values,
                       index=[name for name, _ in rows])
    out.insert(0, "n_images", [len(sub) for _, sub in rows])
    out.index.name = "applied"
    return out


def real_check_metrics(df, values, bootstrap=0, seed=14):
    """AUROC per real defect type: score = max of the matching outputs; negatives = clean val / test photos.
    Bootstrap resamples real photos and clean photos separately."""
    neg = df[df.real_defect.fillna("") == ""]
    pos_all = df[df.real_defect.fillna("") != ""]
    rng = np.random.default_rng(seed)
    rows = []
    types = sorted(pos_all.real_defect.unique()) + ["any"]
    for t in types:
        outs = values if t == "any" else REAL_MAP.get(t, [])
        pos = pos_all if t == "any" else pos_all[pos_all.real_defect == t]
        if not outs:
            rows.append(dict(real_defect=t, outputs="", auroc=np.nan, ci_low=np.nan, ci_high=np.nan,
                             n_real=len(pos), n_clean=len(neg)))
            continue
        cols = [f"pred_{o}" for o in outs]
        sp, sn = pos[cols].max(1).to_numpy(), neg[cols].max(1).to_numpy()
        lab = np.r_[np.ones(len(sp), bool), np.zeros(len(sn), bool)]
        val = auroc(np.r_[sp, sn], lab)
        lo = hi = np.nan
        if bootstrap:
            bs = [auroc(np.r_[rng.choice(sp, len(sp)), rng.choice(sn, len(sn))], lab) for _ in range(bootstrap)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append(dict(real_defect=t, outputs="+".join(outs), auroc=val, ci_low=lo, ci_high=hi,
                         n_real=len(pos), n_clean=len(neg)))
    return pd.DataFrame(rows)

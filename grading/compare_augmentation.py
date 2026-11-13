import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "../../grading"); from train_grading import qwk
def load(run, col="pred"):
    rows = []
    for f in sorted(glob.glob(f"{run}/A_seed*/predictions_test.csv")):
        d = pd.read_csv(f); d["p"] = d[col]; d["seed"] = f; rows.append(d)
    return pd.concat(rows) if rows else None
for task, base in [("sustech_sysu", "sustech_sysu_category_visionfm_ft"), ("nuclear_cataract", "nuclear_cataract_grade_visionfm_ft")]:
    runs = [(base, "pred", "baseline")] + [(f"{task}_aug_{a}", c, a + (" +TTA" if c == "tta_pred" else ""))
             for a in ["geo", "mix", "bal", "all"] for c in ["pred", "tta_pred"]]
    out = []
    for run, col, name in runs:
        d = load(run, col)
        if d is None: continue
        n = int(d.grade.max()) + 1
        for sub in ["clean", "degraded"]:
            s = d[(d.kind == "clean") == (sub == "clean")]
            k = np.mean([qwk(g.grade.values, g.p.values, n) for _, g in s.groupby("seed")])
            r = dict(task=task, run=name, test=sub, QWK=round(k, 3), acc=round((s.p == s.grade).mean(), 3))
            r.update({f"c{c}": round((s[s.grade == c].p == c).mean(), 2) for c in range(n)})
            out.append(r)
    print(pd.DataFrame(out).to_string(index=False))

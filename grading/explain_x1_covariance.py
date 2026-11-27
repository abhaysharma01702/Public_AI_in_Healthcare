"""Explainability X1, full variance split (report table): for the seed-14 condition-C graders, Var(output) =
Var(image part) + Var(quality part) + 2 Cov(image part, quality part), per output (NC: 1; SUSTech-SYSU: one per class),
the quality share with and without the covariance, the image/quality correlation, and the share of test predictions
whose class changes when the quality part is removed. Reads <run>/explain_test.npz from explain_features.py.
Writes results/explainability/x1_variance_split.csv.
Usage (from the project root): python grading/explain_x1_covariance.py
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = os.path.join(ROOT, "runs", "grading")
NC_CUTS = [0.63, 1.46, 2.35, 2.99, 3.95]   # final Nuclear Cataract grader
RUNS = {"nuclear_cataract": f"{G}/nuclear_cataract_reg_BC/C_seed14",
        "sustech_sysu": f"{G}/sustech_sysu_category_visionfm_ft/C_seed14"}


def main():
    rows = []
    for task, run in RUNS.items():
        z = np.load(os.path.join(run, "explain_test.npz"), allow_pickle=True)
        f, q, out = z["contrib_f"].astype(float), z["contrib_q"].astype(float), z["out"].astype(float)
        if task == "nuclear_cataract":
            changed = float((np.searchsorted(NC_CUTS, out[:, 0]) != np.searchsorted(NC_CUTS, out[:, 0] - q[:, 0])).mean())
        else:
            changed = float((out.argmax(1) != (out - q).argmax(1)).mean())
        for k in range(f.shape[1]):
            vf, vq = f[:, k].var(), q[:, k].var()
            cov = float(np.cov(f[:, k], q[:, k], bias=True)[0, 1])
            vo = (f[:, k] + q[:, k]).var()
            rows.append(dict(task=task, output=k, sd_image=np.sqrt(vf), sd_quality=np.sqrt(vq), var_image=vf,
                             var_quality=vq, two_cov=2 * cov, var_output=vo, sum_check=vf + vq + 2 * cov,
                             share_without_cov=vq / (vf + vq), share_with_cov=(vq + abs(2 * cov)) / vo,
                             corr=np.corrcoef(f[:, k], q[:, k])[0, 1], changed_without_q=changed))
    df = pd.DataFrame(rows)
    out_dir = os.path.join(ROOT, "results", "explainability")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "x1_variance_split.csv"), index=False, float_format="%.6f")
    print(df.round(5).to_string(index=False))


if __name__ == "__main__":
    main()

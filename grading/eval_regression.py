"""regression vs classifier baseline (condition A), single seeds, ensembles, eye-level bootstrap CI."""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "../../grading"); from train_grading import qwk, to_class
S = [14, 15, 16]
R = {s: pd.read_csv(f"nuclear_cataract_reg/A_seed{s}/predictions_test.csv").set_index("sample_id") for s in S}
B = {s: pd.read_csv(f"nuclear_cataract_grade_visionfm_ft/A_seed{s}/predictions_test.csv").set_index("sample_id").loc[R[14].index] for s in S}
cuts = {s: json.load(open(f"nuclear_cataract_reg/A_seed{s}/run_info.json"))["cuts"] for s in S}
d0 = R[14]; y = d0.grade.to_numpy(); g = d0.group_id.to_numpy()
# ensembles: classifier = mean softmax; regression = mean TTA raw output classified with the mean of the 3 tuned cut sets
pb = sum(B[s][[f"p{c}" for c in range(6)]].to_numpy() for s in S) / 3
ens_b = pb.argmax(1)
ens_r = to_class(sum(R[s].tta_raw.to_numpy() for s in S) / 3, np.mean([cuts[s] for s in S], 0))
def boot(m, pa, pb_, n=2000):
    eyes = np.unique(g[m]); rng = np.random.default_rng(14); out = []
    idx = {e: np.where(m & (g == e))[0] for e in eyes}
    for _ in range(n):
        ii = np.concatenate([idx[e] for e in rng.choice(eyes, len(eyes))])
        out.append(qwk(y[ii], pa[ii], 6) - qwk(y[ii], pb_[ii], 6))
    return np.percentile(out, [2.5, 97.5])
for sub in ["clean", "degraded"]:
    m = (d0.kind == "clean").to_numpy() == (sub == "clean")
    sb = np.mean([qwk(y[m], B[s].pred.to_numpy()[m], 6) for s in S]); sr = np.mean([qwk(y[m], R[s].pred.to_numpy()[m], 6) for s in S])
    st = np.mean([qwk(y[m], R[s].tta_pred.to_numpy()[m], 6) for s in S])
    eb, er = qwk(y[m], ens_b[m], 6), qwk(y[m], ens_r[m], 6); lo, hi = boot(m, ens_r, ens_b)
    accb, accr = (ens_b[m] == y[m]).mean(), (ens_r[m] == y[m]).mean()
    mae_b, mae_r = np.abs(ens_b[m] - y[m]).mean(), np.abs(ens_r[m] - y[m]).mean()
    rec = lambda p: [round((p[m][y[m] == c] == c).mean(), 2) for c in range(6)]
    print(f"{sub}: single-seed QWK baseline {sb:.3f} reg {sr:.3f} reg+TTA {st:.3f} | ensemble baseline {eb:.3f} reg {er:.3f} "
          f"diff {er-eb:+.3f} CI [{lo:+.3f}, {hi:+.3f}] | acc {accb:.3f}->{accr:.3f} | mean abs error {mae_b:.3f}->{mae_r:.3f}")
    print(f"   recall NC1-6 baseline {rec(ens_b)}  reg {rec(ens_r)}")

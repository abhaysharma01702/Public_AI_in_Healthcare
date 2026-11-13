"""Evaluate a trained quality module.

Loads run_dir/best.pt (EMA weights) and, for each set, writes
  predictions_<set>.csv   sample id, dataset, lighting, labels, pred_<value> (sigmoid outputs)
  metrics_<set>.csv       per defect + mean: AUROC, MAE / Spearman on present, MAE all, false alarm;
                          overall / per dataset / per lighting; 95% bootstrap CIs over source photos
  cross_talk_<set>.csv    (val, test) mean outputs on clean and single-defect images
  metrics_real_check.csv  (real_check) AUROC per real defect type vs. clean val / test photos
Sets: val, test (Quality Data), stress (4 defects), real_check (30 real photos + clean val / test),
gen_ours / gen_alt (single defects on test photos by our generator / a different implementation;
evaluating both also writes generalisation_comparison.csv, a paired comparison with bootstrap CIs);
gen_train / gen_holdout / gen_fresh (training kinds / development held-out / fresh held-out on the same test
photos; gen_train + gen_fresh also writes generalisation_comparison_fresh.csv).
Metrics are computed from the prediction CSVs; --metrics_only recomputes them without the model / GPU.

Usage:
  python quality/evaluate.py --run_dir runs/quality/densenet121_seed14 --data_root "Data/Quality Data"
      [--sets val test stress real_check] [--bootstrap 1000] [--metrics_only] [--limit_sources N]
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from qm import VALUES  # noqa: E402
from qm import data as QD  # noqa: E402
from qm import engine as E  # noqa: E402
from qm import metrics as QX  # noqa: E402
from qm import model as QM  # noqa: E402

KEEP = ["sample_id", "split", "dataset", "lighting", "source_image_id", "kind", "n_defects", "defects", "real_defect"]


def load_set(root, name, extension=None):
    if name in ("val", "test", "train"):
        return QD.load_split(root, name, extension)
    if name == "stress":
        return QD.load_stress(root)
    if name == "real_check":
        return QD.load_real_check(root)
    if name.startswith("gen_"):
        return QD.load_generalisation(root, name[4:])
    raise ValueError(name)


def run(run_dir, root, sets, bootstrap, device=None, num_workers=4, limit_sources=None, metrics_only=False,
        logger=print):
    with open(os.path.join(run_dir, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed, H, W = cfg["seed"], cfg["data"]["height"], cfg["data"]["width"]
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = bool(cfg["optim"]["amp"]) and device.type == "cuda"
    model = None
    summary, frames = {}, {}
    for name in sets:
        t0 = time.time()
        pred_path = os.path.join(run_dir, f"predictions_{name}.csv")
        if metrics_only:
            df = pd.read_csv(pred_path)
            df["real_defect"] = df["real_defect"].fillna("")
        else:
            if model is None:
                ck = torch.load(os.path.join(run_dir, "best.pt"), map_location=device, weights_only=False)
                mc = cfg["model"]
                model = QM.build_model(mc["name"], len(VALUES), pretrained=False, dropout=mc["dropout"]).to(device)
                model.load_state_dict(ck["model"])
                if cfg["train"]["channels_last"]:
                    model = model.to(memory_format=torch.channels_last)
                logger(f"evaluate: best.pt from epoch {ck['epoch']} (val loss {ck['val_loss']:.4f})")
            df = load_set(root, name, cfg["data"].get("extension"))
            if "real_defect" not in df:
                df["real_defect"] = ""
            if limit_sources:  # smoke tests: keep all real photos, limit the rest
                real = df.real_defect != ""
                df = pd.concat([df[real], QD.limit_sources(df[~real], limit_sources, seed)], ignore_index=True)
            ds = QD.QualityDataset(df, root, H, W, train=False)
            loader = QD.make_loader(ds, QD.sequential_batches(len(df), 4 * cfg["batch"]["sources_per_batch"]),
                                    num_workers, device.type == "cuda")
            p, _ = E.predict(model, loader, E.Batch(device, cfg["train"]["channels_last"]), device, amp)
            df = df[KEEP + VALUES + ["path"]].copy()
            for k, v in enumerate(VALUES):
                df[f"pred_{v}"] = p[:, k]
            df.to_csv(pred_path, index=False, float_format="%.9g")  # 9 digits = float32 exactly (metrics from the CSV == in memory)
        if name == "real_check":
            m = QX.real_check_metrics(df, VALUES, bootstrap, seed)
            m.to_csv(os.path.join(run_dir, "metrics_real_check.csv"), index=False, float_format="%.6g")
            summary[name] = {r.real_defect: r.auroc for r in m.itertuples()}
            logger(f"evaluate {name}: {len(df)} images, AUROC any {summary[name].get('any', np.nan):.3f}  "
                   f"({time.time - t0:.0f} s)")
            continue
        m = QX.evaluate_predictions(df, VALUES, bootstrap, seed)
        m.to_csv(os.path.join(run_dir, f"metrics_{name}.csv"), index=False, float_format="%.6g")
        if name in ("val", "test") or name.startswith("gen_"):
            QX.cross_talk(df, VALUES).to_csv(os.path.join(run_dir, f"cross_talk_{name}.csv"), float_format="%.6g")
        frames[name] = df
        ov = m[(m.group_type == "overall") & (m.defect == "mean")].set_index("metric")
        summary[name] = {k: float(v) for k, v in ov.value.items()}
        logger(f"evaluate {name}: {len(df)} images, mean AUROC {ov.value['auroc']:.3f} "
               f"[{ov.ci_low['auroc']:.3f}, {ov.ci_high['auroc']:.3f}], Spearman {ov.value['spearman_present']:.3f}, "
               f"MAE {ov.value['mae_present']:.4f}  ({time.time - t0:.0f} s)")
    pairs = [(("gen_ours", "gen_alt") if "gen_ours" in frames else ("gen_train", "gen_holdout"),   #
              "generalisation_comparison.csv", "generalisation_alt_minus_ours"),
             (("gen_train", "gen_fresh"), "generalisation_comparison_fresh.csv", "generalisation_fresh_minus_train")]  #
    for pair, fname, key in pairs:
        if not (pair[0] in frames and pair[1] in frames):    # paired, same source photos
            continue
        cmp = QX.paired_comparison(frames[pair[0]], frames[pair[1]], VALUES, bootstrap, seed, names=(pair[0][4:], pair[1][4:]))
        cmp.to_csv(os.path.join(run_dir, fname), index=False, float_format="%.6g")
        r = cmp[cmp.defect == "mean"].set_index("metric")
        summary[key] = {m: float(r.difference[m]) for m in r.index}
        logger(f"generalisation ({pair[1][4:]} - {pair[0][4:]}, mean over defects): AUROC {r.difference['auroc']:+.3f} "
               f"[{r.diff_ci_low['auroc']:+.3f}, {r.diff_ci_high['auroc']:+.3f}], Spearman "
               f"{r.difference['spearman_present']:+.3f} [{r.diff_ci_low['spearman_present']:+.3f}, "
               f"{r.diff_ci_high['spearman_present']:+.3f}]")
    with open(os.path.join(run_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--sets", nargs="*", default=None)
    ap.add_argument("--bootstrap", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--limit_sources", type=int, default=None)
    ap.add_argument("--metrics_only", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    with open(os.path.join(a.run_dir, "config.yaml"), encoding="utf-8") as f:
        ev = yaml.safe_load(f)["eval"]
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "data_prep"))
    import run_log  # timing / environment record
    sets = a.sets or ev["sets"]
    with run_log.RunLog("quality/evaluate.py", notes=f"evaluate {os.path.basename(a.run_dir)}: {', '.join(sets)}") as rl:
        summary = run(os.path.abspath(a.run_dir), os.path.abspath(a.data_root), sets,
                      ev["bootstrap"] if a.bootstrap is None else a.bootstrap, a.device, a.num_workers, a.limit_sources,
                      a.metrics_only)
        rl.add(run_dir=os.path.abspath(a.run_dir), sets=sets, device=a.device or "auto", metrics_only=a.metrics_only,
               summary={k: v for k, v in summary.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()

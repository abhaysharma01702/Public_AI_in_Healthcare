"""Screening distributions for the report. Descriptive only: the screening is
NOT redone.

For the 5 quality-pool datasets, one histogram per measure and dataset:
  sharpness (sharp_z), mean brightness (mean_lum), blown-out share (frac_over, log scale)
with the percentile cut actually used (blur: lowest 3%; dark / bright / over: 2%) as a solid line, and for comparison
a robust outlier cut (median +- 3 x 1.4826 x MAD; frac_over on log10 of frac_over + 1e-4) as a dashed line.
Outputs: figures/report/screening_distributions.png, figures/report/screening_cut_comparison.csv
Usage: python figures/make_screening_distributions.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures", "report")
POOL = ["SLID", "SLID-E", "Nuclear Cataract", "SUSTech-SYSU", "Infectious Keratitis"]
INK, MUTED, TEAL, ORANGE = "#16202e", "#5b6670", "#1f6b80", "#c8702a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": MUTED, "savefig.dpi": 200})


def mad_cut(x, side):
    med = np.median(x)
    s = 1.4826 * np.median(np.abs(x - med))
    return med - 3 * s if side == "low" else med + 3 * s


def main():
    os.makedirs(OUT, exist_ok=True)
    d = pd.read_csv(os.path.join(ROOT, "labels", "quality_screening_metrics.csv"))
    d = d[d.dataset.isin(POOL)]
    d["log_over"] = np.log10(d.frac_over + 1e-4)
    # (column, label, flag columns with their side)
    rows = [("sharp_z", "sharpness (sharp_z)", [("flag_blur", "low")]),
            ("mean_lum", "mean brightness (0–255)", [("flag_dark", "low"), ("flag_bright", "high")]),
            ("log_over", "blown-out share (log10)", [("flag_over", "high")])]
    fig, ax = plt.subplots(3, 5, figsize=(22, 11))
    table = []
    for j, ds in enumerate(POOL):
        g = d[d.dataset == ds]
        for i, (col, lab, flags) in enumerate(rows):
            a = ax[i, j]
            x = g[col].to_numpy()
            a.hist(x, bins=60, color=TEAL, alpha=0.85)
            for fl, side in flags:
                f = g[fl].to_numpy()
                used = x[f].max() if side == "low" else x[f].min()   # the percentile cut actually applied
                mc = mad_cut(x, side)
                n_mad = int((x < mc).sum() if side == "low" else (x > mc).sum())
                a.axvline(used, color=ORANGE, lw=2)
                a.axvline(mc, color=INK, lw=1.5, ls="--")
                table.append({"dataset": ds, "flag": fl, "n_images": len(x), "n_flagged_percentile": int(f.sum()),
                              "n_flagged_mad3": n_mad, "cut_percentile": round(float(used), 4),
                              "cut_mad3": round(float(mc), 4)})
            if i == 0:
                a.set_title(f"{ds} (n = {len(g)})", fontsize=14)
            if j == 0:
                a.set_ylabel(lab)
            a.spines[["top", "right"]].set_visible(False)
    fig.legend(handles=[plt.Line2D([], [], color=ORANGE, lw=2, label="cut used (percentile tail)"),
                        plt.Line2D([], [], color=INK, lw=1.5, ls="--", label="robust cut for comparison (3 MAD)")],
               loc="upper center", ncol=2, frameon=False, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(OUT, "screening_distributions.png"))
    t = pd.DataFrame(table)
    t.to_csv(os.path.join(OUT, "screening_cut_comparison.csv"), index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()

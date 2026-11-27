"""Poster prototype figure: mock quality-check screen with the real D3 (Swin-T) outputs for one clean and
one defocused test photo of the fresh defect code. Retake rule: any score >= 0.55 (the filtering threshold).
Usage: python figures/make_poster_prototype.py  -> figures/poster/fig_prototype.png
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "Data", "Quality Data v2")
PRED = os.path.join(ROOT, "runs", "quality", "fresh_pass", "D3", "predictions_gen_fresh.csv")
SAMPLES = ("SLID_1450__clean", "SLID_1010__fresh_defocus_blur")
T = 0.55
NAMES = {"defocus_blur": "defocus", "motion_blur": "motion blur", "overexposure": "overexposure",
         "underexposure": "underexposure", "uneven_illumination": "uneven light", "haze": "haze", "glare": "glare",
         "occlusion": "occlusion", "noise": "noise"}
INK, MUTED, GRID = "#1f2328", "#59636e", "#d8dee4"
OK, BAD, BAR = "#1a7f37", "#cf222e", "#2f6fb3"

plt.rcParams.update({"font.size": 22, "text.color": INK, "xtick.color": MUTED, "ytick.color": INK,
                     "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False})


def main():
    d = pd.read_csv(PRED).set_index("sample_id")
    fig, axes = plt.subplots(2, 2, figsize=(17, 11), gridspec_kw={"width_ratios": [1.45, 1]})
    for row, sid in enumerate(SAMPLES):
        r = d.loc[sid]
        q = pd.Series({NAMES[k]: r["pred_" + k] for k in NAMES})
        bad = q.max() >= T
        ax = axes[row, 0]
        ax.imshow(Image.open(os.path.join(DATA, r.path)))
        ax.axis("off")
        ax.set_title("photo just taken" if row == 0 else "", loc="left", fontsize=22, color=MUTED)
        ax = axes[row, 1]
        ax.barh(range(9), q.values, color=[BAD if v >= T else BAR for v in q.values], height=0.65)
        ax.axvline(T, color=MUTED, ls="--", lw=2)
        ax.set_yticks(range(9), q.index)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        for i, v in enumerate(q.values):
            ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=18, color=MUTED)
        verdict = f"✗  RETAKE: {q.idxmax}" if bad else "✓  OK: send to grader"
        ax.set_title(verdict, loc="left", fontsize=28, fontweight="bold", color=BAD if bad else OK, pad=14)
    axes[1, 1].set_xlabel(f"defect score (retake if ≥ {T})")
    fig.tight_layout(h_pad=3)
    fig.savefig(os.path.join(ROOT, "figures", "poster", "fig_prototype.png"), dpi=200)
    for sid in SAMPLES:
        print(sid, d.loc[sid, [c for c in d.columns if c.startswith("pred_")]].astype(float).round(2).to_dict())


if __name__ == "__main__":
    main()

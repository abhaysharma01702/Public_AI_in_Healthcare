"""Poster figures: hypothesis test (A / B / C / C_shuffled) and quality gating.
Numbers: fine-tuned VisionFM (mean of seeds 14 / 15 / 16), gating CSVs.
Usage: python figures/make_poster_figures.py  (writes figures/poster/*.png)
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures", "poster")
INK, MUTED, GRID = "#1f2328", "#59636e", "#d8dee4"
BLUE, ORANGE, GREY = "#2f6fb3", "#d9822b", "#9aa4ae"
TASKS = (("nuclear_cataract", "Nuclear cataract grade"), ("sustech_sysu", "Corneal ulcer type"))

# table (fresh test code, degraded photos)
HYP = {"nuclear_cataract": dict(A=0.542, B=0.582, C=0.572, C_shuffled=0.570),
       "sustech_sysu": dict(A=0.482, B=0.504, C=0.500, C_shuffled=0.503)}

plt.rcParams.update({"font.size": 26, "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False})


def hypothesis():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    labels = ["A", "B", "C", "C\nshuffled"]
    for ax, (t, name) in zip(axes, TASKS):
        v = list(HYP[t].values())
        bars = ax.bar(range(4), v, width=0.6, color=[GREY, BLUE, ORANGE, GREY], edgecolor="white", linewidth=2)
        for b, x in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, x + 0.01, f"{x:.2f}", ha="center", fontsize=24)
        ax.set_xticks(range(4), labels, fontsize=22)
        ax.set_title(name, fontsize=26, loc="left")
        ax.set_ylim(0, 0.7)
        ax.yaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("QWK on degraded test photos")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_hypothesis.png"), dpi=200)


def gating():
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=False)
    for ax, (t, name) in zip(axes, TASKS):
        d = pd.read_csv(os.path.join(ROOT, "results", "grading", "posthoc", t, "gating_test_holdout.csv"))
        d = d[(d.model == "VisionFM_ft") & (d.condition == "B")].sort_values("coverage")
        x = d.coverage * 100
        ax.plot(x, d.qwk_quality, color=ORANGE, lw=3, marker="o", ms=9, label="quality module")
        ax.plot(x, d.qwk_confidence, color=BLUE, lw=3, marker="s", ms=9, label="grader confidence")
        ax.plot(x, d.qwk_random, color=GREY, lw=3, ls="--", label="random")
        ax.set_xlim(102, 28)
        ax.set_xlabel("% of photos kept")
        ax.set_title(name, fontsize=26, loc="left")
        ax.yaxis.grid(True, color=GRID)
    axes[0].set_ylabel("QWK of kept photos")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=23, loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(OUT, "fig_gating.png"), dpi=200)


def gating_report():
    """Report version of the gating figure (Fig. 5): one column wide, panels stacked, printed at 1:1 (fonts 8-9 pt)."""
    out = os.path.join(ROOT, "figures", "report")
    os.makedirs(out, exist_ok=True)
    with plt.rc_context({"font.size": 8.5}):
        fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.3), sharex=True)
        for ax, (t, name) in zip(axes, TASKS):
            d = pd.read_csv(os.path.join(ROOT, "results", "grading", "posthoc", t, "gating_test_holdout.csv"))
            d = d[(d.model == "VisionFM_ft") & (d.condition == "B")].sort_values("coverage")
            x = d.coverage * 100
            ax.plot(x, d.qwk_quality, color=ORANGE, lw=1.6, marker="o", ms=3.5, label="quality module")
            ax.plot(x, d.qwk_confidence, color=BLUE, lw=1.6, marker="s", ms=3.5, label="grader confidence")
            ax.plot(x, d.qwk_random, color=GREY, lw=1.6, ls="--", label="random")
            ax.set_xlim(102, 28)
            ax.set_title(name, fontsize=9, loc="left", pad=3)
            ax.set_ylabel("QWK of kept photos")
            ax.yaxis.grid(True, color=GRID)
            ax.tick_params(length=2, pad=2)
        axes[-1].set_xlabel("% of photos kept")
        fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, loc="upper center", ncol=3,
                   handlelength=1.6, columnspacing=1.0)
        fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=0.8)
        fig.savefig(os.path.join(out, "fig_gating.png"), dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    hypothesis()
    gating()
    gating_report()

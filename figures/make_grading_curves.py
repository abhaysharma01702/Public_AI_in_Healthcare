"""Degradation curves of the grading experiments (E2 ).
QWK (mean of 3 seeds) against defect strength; strength 0 = the clean test photos. Lines: A, B, C, C_oracle.
Reads the evaluation outputs (metrics*.csv in the run folders); nothing is recomputed.
  grading_curves_<task>.png        one panel per defect, fresh test code
  grading_curves_by_code_<task>.png all 9 defects pooled per strength, one panel per defect code (seen, holdout, fresh)
Palette: reference categorical slots 1-4, checked with the dataviz validator (light; contrast WARN -> legend and a
different marker per condition, numbers in the metrics tables).
Usage: python figures/make_grading_curves.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures", "report")
TASKS = {"sustech_sysu": ("SUSTech-SYSU (ulcer pattern, 3 classes)", "runs/grading/sustech_sysu_category"),
         "nuclear_cataract": ("Nuclear Cataract (NC grade, 6 classes)", "runs/grading/nuclear_cataract_grade")}
CONDS = {"A": ("#2a78d6", "o", "A: clean training"), "B": ("#eb6834", "s", "B: + degraded copies"),
         "C": ("#1baf7a", "^", "C: + quality values"), "C_oracle": ("#eda100", "D", "C_oracle: + true values")}
DEFECTS = ["defocus_blur", "motion_blur", "noise", "haze", "glare", "occlusion", "overexposure", "underexposure",
           "uneven_illumination"]
CODES = {"test_seen": "seen (training code)", "test_holdout": "holdout code", "test": "fresh code"}
STRENGTHS = [0.2, 0.4, 0.6, 0.8, 1.0]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRID, lw=0.6)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xticks([0] + STRENGTHS)
    ax.set_xticklabels(["clean", ".2", ".4", ".6", ".8", "1"])


def curve(m, cond, key):
    q = m[(m.metric == "qwk") & (m.condition == cond)].set_index("subset")["mean"]
    return [q["clean"]] + [q[key(s)] for s in STRENGTHS]


def legend(fig):
    h = [plt.Line2D([], [], color=c, marker=mk, lw=2, ms=6, label=lab) for c, mk, lab in CONDS.values()]
    fig.legend(handles=h, loc="upper center", ncol=4, frameon=False, fontsize=9, labelcolor=INK,
               bbox_to_anchor=(0.5, 0.945))


def main():
    os.makedirs(OUT, exist_ok=True)
    for task, (title, runs) in TASKS.items():
        m = pd.read_csv(os.path.join(ROOT, runs, "metrics.csv"))
        fig, axes = plt.subplots(3, 3, figsize=(11, 9), sharex=True, sharey=True, facecolor="#fcfcfb")
        for ax, v in zip(axes.flat, DEFECTS):
            style(ax)
            for c, (col, mk, _) in CONDS.items():
                ax.plot([0] + STRENGTHS, curve(m, c, lambda s: f"{v}@{s:g}"), color=col, marker=mk, lw=2, ms=5)
            ax.set_title(v.replace("_", " "), fontsize=10, color=INK)
        for ax in axes[:, 0]:
            ax.set_ylabel("QWK", color=MUTED, fontsize=9)
        for ax in axes[-1]:
            ax.set_xlabel("defect strength", color=MUTED, fontsize=9)
        fig.suptitle(f"{title}: QWK by defect and strength, fresh test code (mean of 3 seeds)", color=INK,
                     fontsize=12, y=0.99)
        legend(fig)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(os.path.join(OUT, f"grading_curves_{task}.png"), dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True, facecolor="#fcfcfb")
        for ax, (pred, lab) in zip(axes, CODES.items()):
            mm = pd.read_csv(os.path.join(ROOT, runs, "metrics.csv" if pred == "test" else f"metrics_{pred}.csv"))
            style(ax)
            for c, (col, mk, _) in CONDS.items():
                ax.plot([0] + STRENGTHS, curve(mm, c, lambda s: f"strength={s:g}"), color=col, marker=mk, lw=2, ms=5)
            ax.set_title(lab, fontsize=10, color=INK)
            ax.set_xlabel("defect strength (9 defects pooled)", color=MUTED, fontsize=9)
        axes[0].set_ylabel("QWK", color=MUTED, fontsize=9)
        fig.suptitle(f"{title}: QWK by strength for three defect implementations", color=INK, fontsize=12, y=0.99)
        legend(fig)
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        fig.savefig(os.path.join(OUT, f"grading_curves_by_code_{task}.png"), dpi=150)
        plt.close(fig)
        print("written", task)


if __name__ == "__main__":
    main()

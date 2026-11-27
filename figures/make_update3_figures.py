"""Figures for Project Update 3, written to figures/update3/.

  fig_real_problem.png   4 real defective photos from the real-photo check set (one per tag, random_state 14)
  fig_training_groups.png 3 training sources (3 datasets, random_state 14): clean + 3 degraded versions, labels shown
  fig_generalisation.png per-defect strength ranking (Spearman, present only) on held-out defect code: v1 / v2 / D1
  fig_defects.png        the 9 defects, each on a different training photo: single-defect training samples with a
                         label between the 50th and 85th percentile (occlusion 0-20th, mildest), one fixed dataset per defect (random_state 14)
  fig_synthetic_vs_real.png largest of the 9 outputs of D1: synthetic degraded test vs. real defective vs. real clean
Usage: python figures/make_update3_figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QD = os.path.join(ROOT, "Data", "Quality Data v2")
RUNS = os.path.join(ROOT, "runs", "quality")
OUT = os.path.join(ROOT, "figures", "update3")
V = ["defocus_blur", "motion_blur", "overexposure", "underexposure", "uneven_illumination", "haze", "glare",
     "occlusion", "noise"]
NAME = {"defocus_blur": "defocus", "motion_blur": "motion blur", "overexposure": "overexposure",
        "underexposure": "underexposure", "uneven_illumination": "uneven light", "haze": "haze", "glare": "glare",
        "occlusion": "occlusion", "noise": "noise"}
INK, MUTED, TEAL, ORANGE, GREY = "#16202e", "#5b6670", "#1f6b80", "#c8702a", "#b8c0c6"
RUN = {"v1": os.path.join(RUNS, "kaggle_v1_on_v2"), "v2": os.path.join(RUNS, "ada_v2", "densenet121_v2_ada"),
       "D1": os.path.join(RUNS, "ada_v3", "densenet121_v3_ada")}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": MUTED, "savefig.dpi": 200,
                     "figure.facecolor": "#f7f6f2", "axes.facecolor": "#f7f6f2", "savefig.facecolor": "#f7f6f2"})


def crop_box(im, aspect=4 / 3):
    w, h = im.size
    if w / h > aspect:
        nw = int(h * aspect)
        return im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    nh = int(w / aspect)
    return im.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))


def real_problem():
    p = pd.read_csv(os.path.join(RUN["D1"], "predictions_real_check.csv"))
    r = p[p.real_defect.notna()]
    tags = [("blur", "blur"), ("underexposure", "underexposed"), ("overexposure", "overexposed"),
            ("occlusion_blur", "lid in view + blur")]
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
    for a, (tag, label) in zip(ax, tags):
        row = r[r.real_defect == tag].sample(1, random_state=14).iloc[0]
        a.imshow(crop_box(Image.open(os.path.join(QD, row.path)).convert("RGB")))
        a.set_title(f"{label}  ·  {row.dataset}", fontsize=15, loc="left")
        a.axis("off")
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT, "fig_real_problem.png"))
    plt.close(fig)


def training_groups():
    q = pd.read_csv(os.path.join(QD, "labels.csv"))
    tr = q[q.split == "train"]
    srcs = [tr[tr.dataset == d].source_image_id.drop_duplicates().sample(1, random_state=14).iloc[0]
            for d in ["SLID", "Nuclear Cataract", "Infectious Keratitis"]]
    fig, ax = plt.subplots(3, 4, figsize=(16, 9.6))
    for i, s in enumerate(srcs):
        g = tr[tr.source_image_id == s].sort_values("kind")  # clean first
        for j, row in enumerate(g.itertuples()):
            a = ax[i, j]
            a.imshow(crop_box(Image.open(os.path.join(QD, row.path)).convert("RGB")))
            a.axis("off")
            if row.kind == "clean":
                txt = f"clean  ·  {row.dataset}"
            else:
                vals = sorted(((getattr(row, v), v) for v in V if getattr(row, v) > 0), reverse=True)
                txt = ",  ".join(f"{NAME[v]} {x:.2f}" for x, v in vals[:2]) + ("  …" if len(vals) > 2 else "")  # top 2
            a.set_title(txt, fontsize=13, loc="left", color=MUTED if row.kind == "clean" else INK)
    fig.tight_layout(pad=0.4, h_pad=0.8)
    fig.savefig(os.path.join(OUT, "fig_training_groups.png"))
    plt.close(fig)
    return srcs


def generalisation():
    rows = {}
    for k, d in RUN.items():
        m = pd.read_csv(os.path.join(d, "metrics_gen_holdout.csv"))
        o = m[(m.group_type == "overall") & (m.metric == "spearman_present")].set_index("defect").value
        rows[k] = o
    t = pd.DataFrame(rows).loc[V[::-1]]
    fig, a = plt.subplots(figsize=(9, 7.2))
    y = np.arange(len(t))
    for i in y:
        a.plot([t.iloc[i].min(), t.iloc[i].max()], [i, i], color=GREY, lw=2, zorder=1)
    for k, c, mk in [("v1", GREY, "o"), ("v2", ORANGE, "s"), ("D1", TEAL, "D")]:
        a.scatter(t[k], y, s=90, color=c, marker=mk, label=k, zorder=2, edgecolor="#f7f6f2", linewidth=1)
    a.set_yticks(y, [NAME[v] for v in t.index])
    a.set_xlim(0.3, 1.0)
    a.set_xlabel("Strength ranking on held-out defect code (Spearman)")
    a.grid(axis="x", color="#dfe3e5")
    a.spines[["top", "right", "left"]].set_visible(False)
    a.tick_params(axis="y", length=0)
    a.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_generalisation.png"))
    plt.close(fig)
    return t


def defect_gallery():
    q = pd.read_csv(os.path.join(QD, "labels.csv"))
    one = q[(q.split == "train") & (q.n_defects == 1)]
    fig, ax = plt.subplots(3, 3, figsize=(12, 9.9))
    used = set()
    # fixed dataset per defect so all 5 datasets / lightings appear (random picks favour SLID-E, the largest)
    src = {"defocus_blur": "SLID", "motion_blur": "Nuclear Cataract", "overexposure": "SUSTech-SYSU",
           "underexposure": "SLID", "uneven_illumination": "Infectious Keratitis", "haze": "Nuclear Cataract",
           "glare": "SLID", "occlusion": "Infectious Keratitis", "noise": "SUSTech-SYSU"}
    for a, v in zip(ax.ravel(), V):
        d = one[(one.defects == v) & (one.dataset == src[v])]
        lo, hi = d[v].quantile([0.0, 0.2] if v == "occlusion" else [0.5, 0.85])  # occlusion: keep the eye visible
        d = d[(d[v] >= lo) & (d[v] <= hi) & ~d.source_image_id.isin(used)]
        row = d.sample(1, random_state=14).iloc[0]
        used.add(row.source_image_id)
        a.imshow(crop_box(Image.open(os.path.join(QD, row.path)).convert("RGB")))
        a.set_title(f"{NAME[v]}   label {row[v]:.2f}", fontsize=16, loc="left")
        a.axis("off")
    fig.tight_layout(pad=0.4, h_pad=0.9, w_pad=0.5)
    fig.savefig(os.path.join(OUT, "fig_defects.png"))
    plt.close(fig)


def synthetic_vs_real():
    cols = [f"pred_{v}" for v in V]
    test = pd.read_csv(os.path.join(RUN["D1"], "predictions_test.csv"))
    real = pd.read_csv(os.path.join(RUN["D1"], "predictions_real_check.csv"))
    groups = [("synthetic\ndegraded (test)", test[test.kind == "degraded"][cols].max(1), TEAL),
              ("real defective\n(n = 30)", real[real.real_defect.notna()][cols].max(1), ORANGE),
              ("real clean\n(val / test)", real[real.real_defect.isna()][cols].max(1), GREY)]
    rng = np.random.default_rng(14)
    fig, a = plt.subplots(figsize=(9, 7.2))
    for i, (lab, s, c) in enumerate(groups):
        s = s.to_numpy()
        if len(s) > 600:
            s = rng.choice(s, 600, replace=False)
        a.scatter(i + rng.uniform(-0.28, 0.28, len(s)), s, s=14 if len(s) > 100 else 40, color=c, alpha=0.6,
                  edgecolor="none")
        med = np.median(groups[i][1])
        a.plot([i - 0.35, i + 0.35], [med, med], color=INK, lw=2.5)
        a.text(i - 0.35, med + 0.035, f"median {med:.3f}" if med < 0.01 else f"median {med:.2f}", fontsize=14,
               fontweight="bold")
    a.set_xticks(range(3), [g[0] for g in groups])
    a.set_xlim(-0.5, 2.5)
    a.set_ylim(-0.03, 1.0)
    a.set_ylabel("Largest of the 9 predicted values (D1)")
    a.grid(axis="y", color="#dfe3e5")
    a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_synthetic_vs_real.png"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    real_problem()
    defect_gallery()
    print("training sources:", training_groups())
    print(generalisation().round(2))
    synthetic_vs_real()
    print("written to", OUT)

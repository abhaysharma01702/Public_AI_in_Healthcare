"""Review sheets of the generalisation test images actually generated: per defect, 4 random test photos
(seed 14): clean | ours | alternative, with measured label and the baseline's prediction (kaggle_v1, epoch 38).
Output: decisions/evidence/generalisation/02_review_<defect>.png"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
QD = os.path.join(ROOT, "Data", "Quality Data")
RUN = os.path.join(ROOT, "runs", "quality", "kaggle_v1", "densenet121_seed14")
OUT = os.path.join(ROOT, "decisions", "evidence", "generalisation")
sys.path.insert(0, HERE)
from qm import VALUES
P = {i: pd.read_csv(os.path.join(RUN, f"predictions_gen_{i}.csv")).set_index("sample_id") for i in ("ours", "alt")}
G = {i: pd.read_csv(os.path.join(QD, f"gen_{i}.csv")).set_index("sample_id") for i in ("ours", "alt")}
lab = pd.read_csv(os.path.join(QD, "labels.csv")).set_index("source_image_id")
for v in VALUES:
    alt = G["alt"][G["alt"].defects == v]
    pick = alt.sample(4, random_state=14)
    fig, axes = plt.subplots(4, 3, figsize=(15, 13.2))
    for r, (sid, row) in enumerate(pick.iterrows()):
        src = row.source_image_id
        clean_path = lab[(lab.kind == "clean")].loc[src, "path"] if False else lab.loc[[src]].query("kind == 'clean'").path.iloc[0]
        osid = f"{src}__ours_{v}"
        panels = [("clean", os.path.join(QD, clean_path), ""),
                  ("ours", os.path.join(QD, G["ours"].loc[osid, "path"]),
                   f"label {G['ours'].loc[osid, v]:.2f}  pred {P['ours'].loc[osid, 'pred_' + v]:.2f}"),
                  ("other code", os.path.join(QD, row.path),
                   f"{json.loads(row.params).get('kind') or json.loads(row.params).get('kernel') or json.loads(row.params).get('field') or ''} "
                   f"label {row[v]:.2f}  pred {P['alt'].loc[sid, 'pred_' + v]:.2f}")]
        for c, (name, path, txt) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(Image.open(path)); ax.axis("off")
            ax.set_title(f"{name} ({row.dataset})\n{txt}" if c else f"clean ({row.dataset})", fontsize=13, loc="left")
    fig.suptitle(f"{v}: clean | our generator | other code   (label = measured, pred = baseline model)", fontsize=15, x=0.01, ha="left")
    fig.tight_layout(pad=0.5)
    out = os.path.join(OUT, f"02_review_{v}.png"); fig.savefig(out, dpi=60, facecolor="white"); plt.close(fig); print(out)

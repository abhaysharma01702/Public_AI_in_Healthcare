"""Explainability, part 3 (X5): where does the grader look, and does a defect move it?

SmoothGrad (8 noisy copies, noise SD 0.1 of the normalised input) of the grader output with respect to the input:
regression output for Nuclear Cataract, the predicted class's logit for SUSTech-SYSU. |gradient| is summed over colour
and pooled to the 16 x 16 patch grid. For N test eyes with a clean photo: the clean photo and its 9 defects at one
strength. Metric per defect: Pearson correlation of the degraded map with the clean map of the same eye (1 = the
grader looks at the same places), plus the share of saliency in the central half of the image (where the lens is).
Writes runs/explain_082/<task>_x5_saliency.csv and <task>_x5_saliency.png (2 eyes, photo + map overlay).
Usage: python grading/explain_saliency.py --task nuclear_cataract --run runs/grading/nuclear_cataract_reg/A_seed14
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_grading import ROOT, GradingDataset, GradingNet, VALUES  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "data_prep"))
from run_log import RunLog  # noqa: E402


def saliency(model, x, reg, n=8, sd=0.1):
    x = x.unsqueeze(0)
    q = torch.zeros(1, 9, device=x.device)
    total = torch.zeros_like(x)
    with torch.no_grad():
        k = 0 if reg else int(model(x, q).argmax(1))
    for _ in range(n):
        xi = (x + sd * torch.randn_like(x)).requires_grad_(True)
        model(xi, q)[0, k].backward()
        total += xi.grad.abs()
    m = total[0].sum(0)
    return torch.nn.functional.avg_pool2d(m[None, None], 16)[0, 0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--eyes", type=int, default=20)
    ap.add_argument("--strength", type=float, default=0.6)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = json.load(open(os.path.join(a.run, "run_info.json")))
    reg = info.get("label") == "nc_raw"
    model = GradingNet(1 if reg else info["n_classes"], pretrained=False, arch=info["arch"],
                       size=(info["height"], info["width"])).to(device).eval()
    model.load_state_dict(torch.load(os.path.join(a.run, "best.pt"), map_location=device, weights_only=False)["model"])
    root = os.path.join(ROOT, "Data", "Grading Data", a.task)
    lab = pd.read_csv(os.path.join(root, "labels.csv"))
    lab = lab[lab.split == "test"]
    eyes = lab[lab.kind == "clean"].image_id.drop_duplicates().sample(frac=1, random_state=14).head(a.eyes)
    sel = lab[lab.image_id.isin(eyes) & ((lab.kind == "clean") | np.isclose(lab.strength, a.strength))]
    sel = sel.assign(defect=sel.defect.fillna("clean")).reset_index(drop=True)
    ds = GradingDataset(sel, root, np.zeros((len(sel), 9)), info["height"], info["width"], False, 14)
    maps, imgs = {}, {}
    out = os.path.join(ROOT, "runs", "explain_082")
    os.makedirs(out, exist_ok=True)
    with RunLog("grading/explain_saliency.py", notes=f"X5: {a.task}") as log:
        for i in range(len(sel)):
            x8 = ds[i][0]
            x = (x8.to(device).float()[None] - model.mean) / model.std
            maps[i] = saliency(model, x[0], reg)
            imgs[i] = x8.permute(1, 2, 0).numpy()
        gh, gw = maps[0].shape
        centre = np.zeros((gh, gw), bool)
        centre[gh // 4:3 * gh // 4, gw // 4:3 * gw // 4] = True
        rows = []
        for iid, g in sel.groupby("image_id"):
            c = g.index[g.defect == "clean"][0]
            for i, d in zip(g.index, g.defect):
                m = maps[i]
                rows.append(dict(image_id=iid, defect=d, corr_with_clean=float(np.corrcoef(m.ravel(),
                                 maps[c].ravel())[0, 1]), centre_share=float(m[centre].sum() / m.sum())))
        df = pd.DataFrame(rows)
        res = df.groupby("defect")[["corr_with_clean", "centre_share"]].agg(["mean", "std"]).round(3)
        res.columns = ["_".join(c) for c in res.columns]
        res.to_csv(os.path.join(out, f"{a.task}_x5_saliency.csv"))
        print(res, flush=True)
        order = ["clean"] + VALUES
        show = list(sel.image_id.drop_duplicates()[:2])
        fig, ax = plt.subplots(2 * len(show), len(order), figsize=(2.2 * len(order), 3.4 * len(show)))
        for r, iid in enumerate(show):
            for col, d in enumerate(order):
                i = sel.index[(sel.image_id == iid) & (sel.defect == d)][0]
                ax[2 * r, col].imshow(imgs[i])
                ax[2 * r + 1, col].imshow(imgs[i])
                ax[2 * r + 1, col].imshow(np.kron(maps[i], np.ones((16, 16))), cmap="magma", alpha=0.6)
                ax[2 * r, col].set_title(d.replace("_", " "), fontsize=8)
        for x in ax.ravel():
            x.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{a.task}_x5_saliency.png"), dpi=110)
        log.add(task=a.task, run=a.run, n_images=len(sel), strength=a.strength,
                result=res.reset_index().to_dict("records"), device=str(device))


if __name__ == "__main__":
    main()

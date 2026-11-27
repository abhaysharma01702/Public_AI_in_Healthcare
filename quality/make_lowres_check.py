"""Low-resolution check (agreed; plan step B4). Does a low-resolution (downscaled, then upscaled) photo read
as defocus? Not a training defect and not a primary result: descriptive only, reported either way.

10 clean TEST photos (2 per dataset, random_state 61); each downscaled by 2 and by 4 (box filter) and upscaled back
to its size (bicubic), plus the clean copy: 30 images. Labels are all 0 (no defect is claimed); column edge_loss holds
the training blur measurement (1 - T(y)/T(x), degradations._edge_loss) for reference: what a defocus label would be.
Output: lowres_check/<source_set>/<id>__lowres<f>.png, gen_lowres.csv (evaluate.py set name: gen_lowres).

Usage: python -I make_lowres_check.py                         build
       python -I make_lowres_check.py --summary <run_dir>     table from <run_dir>/predictions_gen_lowres.csv
"""
import argparse
import hashlib
import io
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "data_prep"))
import degradations as D  # noqa: E402

QD = os.path.join(ROOT, "Data", "Quality Data v2")
FACTORS = (2, 4)


def build():
    from run_log import RunLog
    with RunLog("quality/make_lowres_check.py", notes="low-resolution check set for the B4 pilot ") as log:
        q = pd.read_csv(os.path.join(QD, "labels.csv"))
        out_dir = os.path.join(QD, "lowres_check")
        if os.path.exists(out_dir):
            raise SystemExit(f"{out_dir} exists: refusing to overwrite")
        cl = q[(q.split == "test") & (q.kind == "clean")]
        src = pd.concat([g.sample(2, random_state=61) for _, g in cl.groupby("dataset")])
        rows = []
        for r in src.itertuples():
            os.makedirs(os.path.join(out_dir, r.source_set), exist_ok=True)
            im = Image.open(os.path.join(QD, r.path)).convert("RGB")
            x = D.to_float(im)
            base = {c: getattr(r, c) for c in q.columns}
            rows.append({**base, "impl": "clean", "factor": 1, "edge_loss": 0.0})
            for f in FACTORS:
                w, h = im.size
                y = im.resize((round(w / f), round(h / f)), Image.BOX).resize((w, h), Image.BICUBIC)
                buf = io.BytesIO()
                y.save(buf, format="PNG")
                data = buf.getvalue()
                sid = f"{r.source_image_id}__lowres{f}"
                rel = f"lowres_check/{r.source_set}/{sid}.png"
                with open(os.path.join(QD, rel), "wb") as fh:
                    fh.write(data)
                rows.append({**base, "sample_id": sid, "kind": "lowres", "n_defects": 0, "defects": "",
                             **{v: 0.0 for v in D.VALUES}, "params": json.dumps({"factor": f}), "path": rel,
                             "md5": hashlib.md5(data).hexdigest(), "impl": "lowres", "factor": f,
                             "edge_loss": round(D._edge_loss(x, D.to_float(y)), 6)})
        df = pd.DataFrame(rows)
        path = os.path.join(QD, "gen_lowres.csv")
        df.to_csv(path, index=False, float_format="%.6g")
        print(df.groupby("factor").edge_loss.describe()[["mean", "min", "max"]])
        log.add(images_in=len(src), images_out=int((df.impl == "lowres").sum()),
                gen_lowres_md5=hashlib.md5(open(path, "rb").read()).hexdigest())


def summary(run_dir):
    p = pd.read_csv(os.path.join(run_dir, "predictions_gen_lowres.csv"))
    ref = pd.read_csv(os.path.join(QD, "gen_lowres.csv"))[["sample_id", "factor", "edge_loss"]]
    p = p.merge(ref, on="sample_id")
    cols = ["edge_loss"] + [f"pred_{v}" for v in D.VALUES]
    t = p.groupby("factor")[cols].mean().T
    print("mean over the 10 photos (factor 1 = clean copy)")
    print(t.round(3).to_string())
    clean = p[p.factor == 1].set_index("source_image_id").pred_defocus_blur
    for f in FACTORS:
        s = p[p.factor == f].set_index("source_image_id")
        print(f"factor {f}: defocus output above the clean copy in {int((s.pred_defocus_blur > clean[s.index]).sum)} of {len(s)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=None)
    a = ap.parse_args()
    summary(a.summary) if a.summary else build()

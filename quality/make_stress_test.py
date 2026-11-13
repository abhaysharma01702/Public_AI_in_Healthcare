"""4-defect stress-test set.

For every clean TEST source of Data/Quality Data (831 images), one image with EXACTLY 4 defects
(training data has 1-3). Test split only, never used for training or model selection.
Same degradation code and measured labels as the training data (quality/degradations.py).
Rules: the 4 defects respect NOT_TOGETHER (no defocus + motion, no over + under exposure);
"extreme" strengths (EXTREME in degradations.py) are re-drawn from the non-extreme part of the range,
so the 4 defects stay visible together (the fix-4 masking rule) instead of collapsing to 1.
Seed: default_rng([14, crc32(image_id), 100]) (j = 100 is distinct from the training versions 1-3).

Output: Data/Quality Data/stress_4defects/test/<source_set>/<image_id>__s4.png and
        Data/Quality Data/stress_4defects.csv (same columns as labels.csv).
Usage:  python -I make_stress_test.py
"""
import hashlib
import io
import json
import os
import sys
import zlib
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "data_prep"))
import degradations as D  # noqa: E402
from run_log import RunLog  # noqa: E402

QD = os.path.join(ROOT, "Data", "Quality Data")
OUT = os.path.join(QD, "stress_4defects")
K, SEED, J = 4, 14, 100


def draw(rng):
    while True:
        chosen = [str(v) for v in rng.choice(D.VALUES, size=K, replace=False)]
        if not any(set(p) <= set(chosen) for p in D.NOT_TOGETHER):
            break
    strengths = {}
    for n in chosen:
        lo, hi = D.RANGES[n]
        s = rng.uniform(lo, hi)
        while D._is_extreme(n, s):
            s = rng.uniform(lo, hi)
        strengths[n] = s
    return chosen, strengths


def work(job):
    x = D.to_float(Image.open(os.path.join(QD, job["path"])))
    rng = np.random.default_rng([SEED, zlib.crc32(job["source_image_id"].encode()), J])
    chosen, strengths = draw(rng)
    labels, applied, y = {v: 0.0 for v in D.VALUES}, {}, x
    for name in D.ORDER:
        if name in chosen:
            y, labels[name], applied[name] = D.FUNCS[name](y, rng, strengths[name])
    buf = io.BytesIO()
    D.to_pil(y).save(buf, format="PNG")
    data = buf.getvalue()
    sid = f"{job['source_image_id']}__s4"
    rel = f"stress_4defects/test/{job['source_set']}/{sid}.png"
    with open(os.path.join(QD, rel), "wb") as f:
        f.write(data)
    return dict(sample_id=sid, split="test", dataset=job["dataset"], source_set=job["source_set"],
                source_image_id=job["source_image_id"], source_group=job["source_group"], kind="stress_4",
                n_defects=len(applied), defects=";".join(applied),
                **{v: round(float(labels[v]), 6) for v in D.VALUES},
                params=json.dumps(applied, default=float, sort_keys=True), path=rel,
                md5=hashlib.md5(data).hexdigest())


def main():
    with RunLog("make_stress_test.py", notes="4-defect stress test, test split only") as log:
        q = pd.read_csv(os.path.join(QD, "labels.csv"))
        src = q[(q.split == "test") & (q.kind == "clean")]
        if os.path.exists(OUT):
            raise SystemExit(f"{OUT} exists: refusing to overwrite")
        for ss in src.source_set.unique():
            os.makedirs(os.path.join(OUT, "test", ss), exist_ok=True)
        with Pool(max(1, os.cpu_count() - 1)) as p:
            rows = p.map(work, src.to_dict("records"), chunksize=4)
        df = pd.DataFrame(rows)[q.columns].sort_values(["dataset", "sample_id"]).reset_index(drop=True)
        assert (df.n_defects == K).all(), df.n_defects.value_counts()
        assert set(df.source_group) <= set(q[q.split == "test"].source_group)
        assert not set(df.source_group) & set(q[q.split != "test"].source_group)
        df.to_csv(os.path.join(QD, "stress_4defects.csv"), index=False, float_format="%.6g")
        print(df.dataset.value_counts().to_string())
        print("defect frequency:", {v: int((df[v] > 0).sum()) for v in D.VALUES})
        log.add(images_in=len(src), images_out=len(df),
                csv_md5=hashlib.md5(open(os.path.join(QD, "stress_4defects.csv"), "rb").read()).hexdigest())


if __name__ == "__main__":
    main()

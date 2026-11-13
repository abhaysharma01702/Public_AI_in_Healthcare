"""Generalisation test set: the same defects made by OUR generator and by a DIFFERENT
implementation (quality/degradations_alt.py), paired on the same test photos.

For every clean TEST source (831): 3 different defects drawn at random (seed [14, crc32(id), 200]); each
defect is applied ALONE, once with degradations.py ("ours") and once with degradations_alt.py ("alt"),
strengths drawn uniformly from each implementation's range (seeds [14, crc32, 201+k] and [14, crc32, 301+k]).
Labels: measured with the same formulas (degradations_alt.py docstring).
Test split only: never used for training or model selection.

Output: Data/Quality Data/generalisation_test/{ours,alt}/<source_set>/<image_id>__<defect>.png
        Data/Quality Data/gen_ours.csv, gen_alt.csv (labels.csv columns + impl; each also lists the 831 clean
        test copies, so detection AUROC has clean negatives as in the normal test set)
Usage:  python make_generalisation_test.py
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
import degradations_alt as A  # noqa: E402
from run_log import RunLog  # noqa: E402

QD = os.path.join(ROOT, "Data", "Quality Data")
OUT = os.path.join(QD, "generalisation_test")
K, SEED = 3, 14
IMPLS = {"ours": (D.RANGES, D.FUNCS, 201), "alt": (A.RANGES_ALT, A.FUNCS, 301)}


def work(job):
    x = D.to_float(Image.open(os.path.join(QD, job["path"])))
    crc = zlib.crc32(job["source_image_id"].encode())
    defects = [str(v) for v in np.random.default_rng([SEED, crc, 200]).choice(D.VALUES, size=K, replace=False)]
    rows = []
    for impl, (ranges, funcs, base) in IMPLS.items():
        for v in defects:
            k = D.VALUES.index(v)
            rng = np.random.default_rng([SEED, crc, base + k])
            y, label, params = funcs[v](x, rng, rng.uniform(*ranges[v]))
            buf = io.BytesIO()
            D.to_pil(y).save(buf, format="PNG")
            data = buf.getvalue()
            sid = f"{job['source_image_id']}__{impl}_{v}"
            rel = f"generalisation_test/{impl}/{job['source_set']}/{sid}.png"
            with open(os.path.join(QD, rel), "wb") as f:
                f.write(data)
            labels = {u: 0.0 for u in D.VALUES}
            labels[v] = round(float(label), 6)
            rows.append(dict(sample_id=sid, split="test", dataset=job["dataset"], source_set=job["source_set"],
                             source_image_id=job["source_image_id"], source_group=job["source_group"],
                             kind=f"gen_{impl}", n_defects=1, defects=v, **labels,
                             params=json.dumps(params, default=float, sort_keys=True), path=rel,
                             md5=hashlib.md5(data).hexdigest(), impl=impl))
    return rows


def main():
    with RunLog("make_generalisation_test.py", notes="generalisation test: ours vs alternative implementation ") as log:
        q = pd.read_csv(os.path.join(QD, "labels.csv"))
        src = q[(q.split == "test") & (q.kind == "clean")]
        if os.path.exists(OUT):
            raise SystemExit(f"{OUT} exists: refusing to overwrite")
        for impl in IMPLS:
            for ss in src.source_set.unique():
                os.makedirs(os.path.join(OUT, impl, ss), exist_ok=True)
        with Pool(max(1, os.cpu_count() - 1)) as p:
            rows = [r for rs in p.map(work, src.to_dict("records"), chunksize=4) for r in rs]
        df = pd.DataFrame(rows)
        assert set(df.source_group) <= set(q[q.split == "test"].source_group)
        assert not set(df.source_group) & set(q[q.split != "test"].source_group)
        clean = src.assign(impl="clean")
        cols = list(q.columns) + ["impl"]
        out = {}
        for impl in IMPLS:
            d = pd.concat([df[df.impl == impl], clean], ignore_index=True)[cols]
            d = d.sort_values(["dataset", "sample_id"]).reset_index(drop=True)
            path = os.path.join(QD, f"gen_{impl}.csv")
            d.to_csv(path, index=False, float_format="%.6g")
            out[f"gen_{impl}_csv_md5"] = hashlib.md5(open(path, "rb").read()).hexdigest()
            print(impl, len(d), "rows;", {v: int((d[v] > 0).sum()) for v in D.VALUES})
        for v in D.VALUES:
            a, b = df[(df.impl == "ours") & (df.defects == v)][v], df[(df.impl == "alt") & (df.defects == v)][v]
            print(f"{v:20s} n {len(a)}  label median ours {a.median:.3f} alt {b.median:.3f}  "
                  f"q05-q95 ours {a.quantile(.05):.3f}-{a.quantile(.95):.3f} alt {b.quantile(.05):.3f}-{b.quantile(.95):.3f}")
        log.add(images_in=len(src), images_out=len(df), **out)


if __name__ == "__main__":
    main()

"""Test-only sets for Quality Data v2. Uses only the 831 clean TEST sources of Data/Quality Data v2.

1. stress_4defects/  : one image per source with EXACTLY 4 defects, training kinds (defect_kinds.sample-style:
                       NOT_TOGETHER respected, extreme strengths re-drawn so all 4 stay visible), seed [14, crc, 100]
2. generalisation_test/{train,holdout}/ : 3 different defects per source (seed [14, crc, 200]), each applied ALONE
                       once with a random TRAINING kind (seed [14, crc, 201+k]) and once with the HELD-OUT implementation
                       (degradations_holdout.py, seed [14, crc, 301+k]); strength t ~ U(0, 1) each.
CSVs (labels.csv columns + impl): stress_4defects.csv, gen_train.csv, gen_holdout.csv (the gen CSVs also list the 831 clean
test copies as negatives). Eye boxes from labels/eye_regions.csv.
Usage: python make_test_sets_v2.py
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
import defect_kinds as K  # noqa: E402
from degradations_holdout import HOLDOUT  # noqa: E402
from run_log import RunLog  # noqa: E402

QD = os.path.join(ROOT, "Data", "Quality Data v2")
EYE = pd.read_csv(os.path.join(ROOT, "labels", "eye_regions.csv")).set_index("source_image_id")[["x0", "y0", "x1", "y1"]]
SEED = 14


def _save(y, rel):
    buf = io.BytesIO()
    D.to_pil(y).save(buf, format="PNG")
    data = buf.getvalue()
    with open(os.path.join(QD, rel), "wb") as f:
        f.write(data)
    return hashlib.md5(data).hexdigest()


def _row(job, sid, rel, md5, kind, labels, applied, impl):
    return dict(sample_id=sid, split="test", dataset=job["dataset"], source_set=job["source_set"],
                source_image_id=job["source_image_id"], source_group=job["source_group"], kind=kind,
                n_defects=len(applied), defects=";".join(applied), **{v: round(float(labels[v]), 6) for v in D.VALUES},
                params=json.dumps(applied, default=float, sort_keys=True), path=rel, md5=md5, impl=impl)


def work(job):
    x = D.to_float(Image.open(os.path.join(QD, job["path"])))
    D.set_eye_box(tuple(EYE.loc[job["source_image_id"]]))
    crc = zlib.crc32(job["source_image_id"].encode())
    rows = []
    # 1. stress: 4 defects
    rng = np.random.default_rng([SEED, crc, 100])
    while True:
        chosen = [str(v) for v in rng.choice(D.VALUES, size=4, replace=False)]
        if not any(set(p) <= set(chosen) for p in D.NOT_TOGETHER):
            break
    labels, applied, y = {v: 0.0 for v in D.VALUES}, {}, x
    for name in D.ORDER:
        if name in chosen:
            t = rng.uniform()
            while name in D.EXTREME and t >= D.EXTREME[name]:
                t = rng.uniform()
            funcs = K.kind_funcs(name)
            kind = str(rng.choice(sorted(funcs)))
            y, labels[name], p = funcs[kind](y, rng, t)
            applied[name] = {"impl": kind, "t": t, **p}
    sid = f"{job['source_image_id']}__s4"
    rel = f"stress_4defects/test/{job['source_set']}/{sid}.png"
    rows.append(_row(job, sid, rel, _save(y, rel), "stress_4", labels, applied, "stress"))
    # 2. generalisation pair
    defects = [str(v) for v in np.random.default_rng([SEED, crc, 200]).choice(D.VALUES, size=3, replace=False)]
    for v in defects:
        k = D.VALUES.index(v)
        for impl, base in (("train", 201), ("holdout", 301)):
            rng = np.random.default_rng([SEED, crc, base + k])
            t = rng.uniform()
            if impl == "train":
                funcs = K.kind_funcs(v)
                kind = str(rng.choice(sorted(funcs)))
                y, lab, p = funcs[kind](x, rng, t)
            else:
                kind = "holdout"
                y, lab, p = HOLDOUT[v](x, rng, t)
            labels = {u: 0.0 for u in D.VALUES}
            labels[v] = lab
            sid = f"{job['source_image_id']}__{impl}_{v}"
            rel = f"generalisation_test/{impl}/{job['source_set']}/{sid}.png"
            rows.append(_row(job, sid, rel, _save(y, rel), f"gen_{impl}", labels, {v: {"impl": kind, "t": t, **p}}, impl))
    D.set_eye_box()
    return rows


def main():
    with RunLog("quality/make_test_sets_v2.py", notes="stress + generalisation (train kinds vs held-out) test sets ") as log:
        q = pd.read_csv(os.path.join(QD, "labels.csv"))
        src = q[(q.split == "test") & (q.kind == "clean")]
        for sub in ["stress_4defects/test"] + [f"generalisation_test/{i}" for i in ("train", "holdout")]:
            if os.path.exists(os.path.join(QD, sub)):
                raise SystemExit(f"{sub} exists: refusing to overwrite")
            for ss in src.source_set.unique():
                os.makedirs(os.path.join(QD, sub, ss), exist_ok=True)
        with Pool(max(1, os.cpu_count() - 1)) as p:
            rows = [r for rs in p.map(work, src.to_dict("records"), chunksize=4) for r in rs]
        df = pd.DataFrame(rows)
        assert not set(df.source_group) & set(q[q.split != "test"].source_group)
        cols = list(q.columns) + ["impl"]
        out = {}
        stress = df[df.impl == "stress"][cols].sort_values(["dataset", "sample_id"])
        stress.to_csv(os.path.join(QD, "stress_4defects.csv"), index=False, float_format="%.6g")
        assert (stress.n_defects == 4).all()
        clean = src.assign(impl="clean")
        for impl in ("train", "holdout"):
            d = pd.concat([df[df.impl == impl], clean], ignore_index=True)[cols].sort_values(["dataset", "sample_id"])
            path = os.path.join(QD, f"gen_{impl}.csv")
            d.to_csv(path, index=False, float_format="%.6g")
            out[f"gen_{impl}_md5"] = hashlib.md5(open(path, "rb").read()).hexdigest()
        g = df[df.impl != "stress"]
        for v in D.VALUES:
            a = g[(g.impl == "train") & (g.defects == v)][v]
            b = g[(g.impl == "holdout") & (g.defects == v)][v]
            print(f"{v:20s} n {len(a)}  median train {a.median:.3f} holdout {b.median:.3f}  "
                  f"q05-q95 train {a.quantile(.05):.3f}-{a.quantile(.95):.3f} holdout {b.quantile(.05):.3f}-{b.quantile(.95):.3f}")
        log.add(images_in=len(src), images_out=len(df), **out)


if __name__ == "__main__":
    main()

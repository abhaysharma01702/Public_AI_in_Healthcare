"""Fresh held-out test set (plan step A4). Uses only the 831 clean TEST sources of Data/Quality Data v2.

Paired with the generalisation sets: for every source the SAME 3 defects as gen_train / gen_holdout (seed
[14, crc, 200]); each applied ALONE with the FROZEN fresh implementation (degradations_fresh.py), strength
t ~ U(0, 1), seed [14, crc, 401+k]. Labels: the training measurements (degradations_fresh.py docstring).
Test only: never used for training, model selection or development.

Checks: code MD5 = frozen value; eye box present for every source; test source groups only; same sources and
defects as gen_train.csv / gen_holdout.csv; determinism re-check of 30 images (MD5).
Output: generalisation_test/fresh/<source_set>/<image_id>__fresh_<defect>.png, gen_fresh.csv (labels.csv columns + impl;
also lists the 831 clean test copies as negatives).
Usage: python -I make_fresh_test.py
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
from degradations_fresh import FRESH  # noqa: E402
from run_log import RunLog  # noqa: E402

QD = os.path.join(ROOT, "Data", "Quality Data v2")
EYE = pd.read_csv(os.path.join(ROOT, "labels", "eye_regions.csv")).set_index("source_image_id")[["x0", "y0", "x1", "y1"]]
FROZEN_MD5 = "141d4590d70ba44ef70587820da828ac"   #
SEED = 14


def work(job, write=True):
    x = D.to_float(Image.open(os.path.join(QD, job["path"])))
    D.set_eye_box(tuple(EYE.loc[job["source_image_id"]]))
    crc = zlib.crc32(job["source_image_id"].encode())
    defects = [str(v) for v in np.random.default_rng([SEED, crc, 200]).choice(D.VALUES, size=3, replace=False)]
    rows = []
    for v in defects:
        k = D.VALUES.index(v)
        rng = np.random.default_rng([SEED, crc, 401 + k])
        t = rng.uniform()
        y, lab, p = FRESH[v](x, rng, t)
        buf = io.BytesIO()
        D.to_pil(y).save(buf, format="PNG")
        data = buf.getvalue()
        sid = f"{job['source_image_id']}__fresh_{v}"
        rel = f"generalisation_test/fresh/{job['source_set']}/{sid}.png"
        if write:
            with open(os.path.join(QD, rel), "wb") as f:
                f.write(data)
        labels = {u: 0.0 for u in D.VALUES}
        labels[v] = round(float(lab), 6)
        rows.append(dict(sample_id=sid, split="test", dataset=job["dataset"], source_set=job["source_set"],
                         source_image_id=job["source_image_id"], source_group=job["source_group"], kind="gen_fresh",
                         n_defects=1, defects=v, **labels,
                         params=json.dumps({v: {"impl": "fresh", "t": t, **p}}, default=float, sort_keys=True),
                         path=rel, md5=hashlib.md5(data).hexdigest(), impl="fresh"))
    D.set_eye_box()
    return rows


def _redo(job):
    return work(job, write=False)


def main():
    with RunLog("quality/make_fresh_test.py", notes="fresh held-out test set, paired with gen_train / gen_holdout ") as log:
        code = hashlib.md5(open(os.path.join(HERE, "degradations_fresh.py"), "rb").read()).hexdigest()
        if code != FROZEN_MD5:
            raise SystemExit(f"degradations_fresh.py MD5 {code} != frozen {FROZEN_MD5}")
        q = pd.read_csv(os.path.join(QD, "labels.csv"))
        src = q[(q.split == "test") & (q.kind == "clean")]
        assert len(src) == 831 and src.source_image_id.isin(EYE.index).all()
        out_dir = os.path.join(QD, "generalisation_test", "fresh")
        if os.path.exists(out_dir):
            raise SystemExit(f"{out_dir} exists: refusing to overwrite")
        for ss in src.source_set.unique():
            os.makedirs(os.path.join(out_dir, ss), exist_ok=True)
        jobs = src.to_dict("records")
        workers = max(1, os.cpu_count() - 1)
        with Pool(workers) as p:
            rows = [r for rs in p.map(work, jobs, chunksize=4) for r in rs]
        df = pd.DataFrame(rows)
        # checks
        assert not set(df.source_group) & set(q[q.split != "test"].source_group), "leakage: non-test source group"
        for ref in ("train", "holdout"):
            g = pd.read_csv(os.path.join(QD, f"gen_{ref}.csv"))
            g = g[g.impl == ref]
            assert sorted(zip(g.source_image_id, g.defects)) == sorted(zip(df.source_image_id, df.defects)),\
                f"pairing with gen_{ref} broken"
        pick = np.random.default_rng(63).choice(len(jobs), size=10, replace=False)   # 10 sources = 30 images
        with Pool(workers) as p:
            redo = [r for rs in p.map(_redo, [jobs[i] for i in pick]) for r in rs]
        ref_md5 = df.set_index("sample_id").md5
        mism = sum(ref_md5[r["sample_id"]] != r["md5"] for r in redo)
        assert mism == 0, f"determinism: {mism} MD5 mismatches"
        on_disk = sum(hashlib.md5(open(os.path.join(QD, r.path), "rb").read()).hexdigest() != r.md5
                      for r in df.sample(30, random_state=63).itertuples())
        assert on_disk == 0, f"{on_disk} files differ from the recorded MD5"
        # CSV
        cols = list(q.columns) + ["impl"]
        d = pd.concat([df, src.assign(impl="clean")], ignore_index=True)[cols].sort_values(["dataset", "sample_id"])
        path = os.path.join(QD, "gen_fresh.csv")
        d.to_csv(path, index=False, float_format="%.6g")
        csv_md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
        # label distributions: fresh vs the paired training kinds and the development held-out
        tr = pd.read_csv(os.path.join(QD, "gen_train.csv"))
        ho = pd.read_csv(os.path.join(QD, "gen_holdout.csv"))
        print(f"gen_fresh.csv: {len(d)} rows ({len(df)} fresh + {len(src)} clean), MD5 {csv_md5}")
        print(f"{'defect':20s} {'n':>4s}  median / q05-q95: train | holdout (dev) | fresh;  fresh label 0")
        for v in D.VALUES:
            a, b, c = (f[(f.impl == i) & (f.defects == v)][v] for f, i in ((tr, "train"), (ho, "holdout"), (df, "fresh")))
            fmt = lambda s: f"{s.median:.3f} {s.quantile(.05):.3f}-{s.quantile(.95):.3f}"  # noqa: E731
            print(f"{v:20s} {len(c):4d}  {fmt(a)} | {fmt(b)} | {fmt(c)};  {int((c == 0).sum)}")
        n_bytes = sum(os.path.getsize(os.path.join(QD, r)) for r in df.path)
        log.add(images_in=len(src), images_out=len(df), gen_fresh_md5=csv_md5, code_md5=code,
                determinism_mismatches=int(mism), output_bytes=n_bytes, workers=workers)


if __name__ == "__main__":
    main()

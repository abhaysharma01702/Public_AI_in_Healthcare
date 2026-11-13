"""Grading data for one task (first task SUSTech-SYSU type grade).

Sources: the usable images of the task's master CSV, 512 px PNGs: Data/Resized 512/<folder>/<split>/<grade>/<id>.png.
Output: Data/Grading Data/<folder>/ with labels.csv (one row per image) and:
  <split>/<id>__clean.png          every usable image (A trains on these only)
  train|val/<id>__d1..d3.png       3 degraded copies per train / val image with the TRAINING generator of the quality
                                   data (defect_kinds.sample_kinds, >= 1 defect), seed [14, crc32(id), 700+j]
                                   (not the quality data's seeds, so these exact images were never seen by the quality module)
  test/<id>__<defect>_s<k>.png     degraded test: every test image x 9 defects x 5 strengths t = 0.2 .. 1.0, each defect
                                   ALONE, made with the FRESH implementation (degradations_fresh.py, never used for
                                   training), seed [14, crc32(id), 800 + 10*defect + k]
Columns: sample_id, image_id, split, group_id, grade, kind (clean / train_degraded / test_fresh), defect, strength,
the 9 measured defect values (C-oracle input; 0 for clean), params, path, md5.
Eye box: labels/eye_regions.csv; images without one get CENTRAL_BOX, the rule for every non-SLID-E dataset.
Checks: degradations_fresh.py MD5 = frozen value; no group in two splits; determinism re-check (30 images).
Extra test sets: --test_impl seen | holdout builds ONLY the test split (the same clean
test originals + 9 defects x 5 strengths, each defect alone) into Data/Grading Data/<task>_test_<impl>/, kind test_<impl>:
  seen     the TRAINING kinds (defect_kinds.kind_funcs, kind drawn uniformly per image, as gen_train ), seed 900 + ...
  holdout  the development held-out implementation (degradations_holdout.py), seed 1000 + ...
Usage: python -I grading/make_grading_data.py [--task sustech_sysu|nuclear_cataract] [--test_impl fresh|seen|holdout] [--limit N]
       (--limit: quick test into a _test folder)
"""
import argparse
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
sys.path.insert(0, os.path.join(ROOT, "quality"))
sys.path.insert(0, os.path.join(ROOT, "data_prep"))
import degradations as D  # noqa: E402
import defect_kinds as K  # noqa: E402
from degradations_fresh import FRESH  # noqa: E402
from degradations_holdout import HOLDOUT  # noqa: E402
from run_log import RunLog  # noqa: E402

TASKS = {"sustech_sysu": ("sustech_master.csv", "usable", 0),             # task: (master, status kept, grade offset)
         "nuclear_cataract": ("nuclear_cataract_master.csv", "graded", 1)}  # NC grade 1-6 stored as 0-5
FROZEN_FRESH_MD5 = "141d4590d70ba44ef70587820da828ac"   #
CODE_MD5 = {"defect_kinds.py": "661b9239234feab1429fe4e050e567da",        # (training kinds)
            "degradations_holdout.py": "1c0255894d1c91b0ea77a1d40b421c88"}  # (checked 2026-11-02)
TEST_SEED_BASE = {"fresh": 800, "seen": 900, "holdout": 1000}
STRENGTHS = [0.2, 0.4, 0.6, 0.8, 1.0]
N_TRAIN_COPIES = 3
SEED = 14
EYE = pd.read_csv(os.path.join(ROOT, "labels", "eye_regions.csv")).set_index("source_image_id")[["x0", "y0", "x1", "y1"]]


def _save(y, out_root, rel):
    buf = io.BytesIO()
    D.to_pil(y).save(buf, format="PNG")
    data = buf.getvalue()
    if out_root is not None:
        with open(os.path.join(out_root, rel), "wb") as f:
            f.write(data)
    return hashlib.md5(data).hexdigest()


def _test_defect(impl, v, x, rng, t):
    """One test defect alone at strength t with the given implementation -> (y, label, params)."""
    if impl == "fresh":
        y, lab, p = FRESH[v](x, rng, t)
        return y, lab, {"impl": "fresh", "t": t, **p}
    if impl == "holdout":
        y, lab, p = HOLDOUT[v](x, rng, t)
        return y, lab, {"impl": "holdout", "t": t, **p}
    funcs = K.kind_funcs(v)                      # seen: a training kind, drawn uniformly (as gen_train)
    kind = str(rng.choice(sorted(funcs)))
    y, lab, p = funcs[kind](x, rng, t)
    return y, lab, {"impl": kind, "t": t, **p}


def work(job, out_root=None, impl="fresh", offset=0):
    """All samples of one source image; out_root None = do not write (determinism re-check)."""
    x = D.to_float(Image.open(job["src"]).convert("RGB"))
    iid, split = job["image_id"], job["split"]
    D.set_eye_box(tuple(EYE.loc[iid]) if iid in EYE.index else None)
    crc = zlib.crc32(iid.encode())
    base = dict(image_id=iid, split=split, group_id=job["group_id"], grade=int(job["label_grade"]) - offset)
    zero = {v: 0.0 for v in D.VALUES}
    rel = f"{split}/{iid}__clean.png"
    rows = [dict(base, sample_id=f"{iid}__clean", kind="clean", defect="", strength=0.0, **zero, params="{}",
                 path=rel, md5=_save(x, out_root, rel))]
    if split in ("train", "val"):
        for j in range(1, N_TRAIN_COPIES + 1):
            y, lab, applied = K.sample_kinds(x, np.random.default_rng([SEED, crc, 700 + j]), min_defects=1)
            rel = f"{split}/{iid}__d{j}.png"
            rows.append(dict(base, sample_id=f"{iid}__d{j}", kind="train_degraded", defect="+".join(sorted(applied)),
                             strength=np.nan, **{v: round(float(lab[v]), 6) for v in D.VALUES},
                             params=json.dumps(applied, default=float, sort_keys=True), path=rel,
                             md5=_save(y, out_root, rel)))
    else:
        for d_i, v in enumerate(D.VALUES):
            for k, t in enumerate(STRENGTHS):
                rng = np.random.default_rng([SEED, crc, TEST_SEED_BASE[impl] + 10 * d_i + k])
                y, lab, p = _test_defect(impl, v, x, rng, t)
                rel = f"test/{iid}__{v}_s{k + 1}.png"
                labels = dict(zero, **{v: round(float(lab), 6)})
                rows.append(dict(base, sample_id=f"{iid}__{v}_s{k + 1}", kind=f"test_{impl}", defect=v, strength=t,
                                 **labels, params=json.dumps({v: p}, default=float, sort_keys=True), path=rel,
                                 md5=_save(y, out_root, rel)))
    D.set_eye_box()
    return rows


def _work_write(args):
    return work(*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="sustech_sysu", choices=sorted(TASKS))
    ap.add_argument("--test_impl", default="fresh", choices=sorted(TEST_SEED_BASE),
                    help="fresh = full grading data; seen / holdout = extra test set only ")
    ap.add_argument("--limit", type=int, default=None, help="only N images per split (quick test, separate folder)")
    a = ap.parse_args()
    name = a.task + ("" if a.test_impl == "fresh" else f"_test_{a.test_impl}")
    out = os.path.join(ROOT, "Data", "Grading Data", name + ("_test" if a.limit else ""))
    with RunLog("grading/make_grading_data.py", notes=f"grading data {name} " + (f", limit {a.limit}" if a.limit else "")) as log:
        code = hashlib.md5(open(os.path.join(ROOT, "quality", "degradations_fresh.py"), "rb").read()).hexdigest()
        if code != FROZEN_FRESH_MD5:
            raise SystemExit(f"degradations_fresh.py MD5 {code} != frozen {FROZEN_FRESH_MD5}")
        for f, want in CODE_MD5.items():
            got = hashlib.md5(open(os.path.join(ROOT, "quality", f), "rb").read()).hexdigest()
            if got != want:
                raise SystemExit(f"{f} MD5 {got} != recorded {want}")
        master, status, offset = TASKS[a.task]
        m = pd.read_csv(os.path.join(ROOT, "labels", master))
        m = m[m.status == status].copy()
        assert m.groupby("group_id").split.nunique().max() == 1, "leakage: a group in two splits"
        m["src"] = [os.path.join(ROOT, "Data", "Resized 512", a.task, r.split, str(int(r.label_grade)), f"{r.image_id}.png")
                    for r in m.itertuples()]
        assert all(os.path.exists(p) for p in m.src), "missing 512 px source PNG"
        if a.test_impl != "fresh":
            m = m[m.split == "test"]
        if a.limit:
            m = m.groupby("split").head(a.limit)
        if os.path.exists(out):
            raise SystemExit(f"{out} exists: refusing to overwrite")
        for s in m.split.unique():
            os.makedirs(os.path.join(out, s))
        jobs = m.to_dict("records")
        workers = max(1, os.cpu_count() - 1)
        with Pool(workers) as p:
            rows = [r for rs in p.map(_work_write, [(j, out, a.test_impl, offset) for j in jobs], chunksize=2) for r in rs]
        df = pd.DataFrame(rows)
        # determinism re-check: 30 images from 10 sources
        pick = np.random.default_rng(71).choice(len(jobs), size=min(10, len(jobs)), replace=False)
        with Pool(workers) as p:
            redo = [r for rs in p.map(_work_write, [(jobs[i], None, a.test_impl, offset) for i in pick]) for r in rs]
        ref = df.set_index("sample_id").md5
        sub = pd.DataFrame(redo).sample(min(30, len(redo)), random_state=71)
        assert (sub.md5.to_numpy() == ref[sub.sample_id].to_numpy()).all(), "determinism: MD5 mismatch"
        df.to_csv(os.path.join(out, "labels.csv"), index=False)
        csv_md5 = hashlib.md5(open(os.path.join(out, "labels.csv"), "rb").read()).hexdigest()
        counts = df.groupby(["split", "kind"]).size()
        print(counts.to_string())
        print(f"labels.csv: {len(df)} rows, MD5 {csv_md5}")
        log.add(task=a.task, test_impl=a.test_impl, out=out, n_rows=len(df), labels_md5=csv_md5, workers=workers,
                counts={f"{s}/{k}": int(n) for (s, k), n in counts.items()})


if __name__ == "__main__":
    main()

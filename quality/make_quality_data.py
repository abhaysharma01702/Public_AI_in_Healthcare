"""Build Data/Quality Data/: the stored training / validation / test data of the quality module.

Sources: the clean pool (labels/quality_clean_pool.csv, clean == True), 512 px PNGs.
For every clean source image, in the source's own split:
    <image_id>__clean.png            the clean image (label vector all 0)
    <image_id>__d1/d2/d3.png         3 degraded versions from the mixing policy with >= 1 defect
                                     (quality/degradations.py v3, sample(min_defects=1))
Labels: the 9 measured values; the clean copy of every source is in the same split, so
ranking pairs (degraded > its own clean source) are always available.
Real check set: only the val / test photos are copied (real_check); the 16 train-split photos
are excluded, because their patients' clean photos are in quality training (patient overlap).

Leakage checks run FIRST (the build aborts if any fails); report in Data/Quality Data/leakage_report.md:
  1. every source's split equals its master-CSV split
  2. no patient / eye group (master group_id) in more than one split, in any of the 5 datasets
  3. no identical file (MD5 of the 512 px PNG) in two splits, across all datasets
  4. no duplicate photo in two splits, across all datasets, by the project's duplicate rules
     (pHash distance 0 or ResNet50 cosine >= 0.98; cached features, Data/derived), plus the closest
     cross-split pairs for transparency
  5. real check: no real-check photo shares a group with a quality-training source
After generation: every sample in its source's split; determinism re-check of 30 samples.

Seeds: sample j of source s uses numpy default_rng([14, crc32(image_id), j]) (order- and worker-independent).

Usage:
  python -I make_quality_data.py --check     # leakage checks only (writes the report)
  python -I make_quality_data.py             # checks, then build (refuses if the folder exists)
  python -I make_quality_data.py --v2        # Data/Quality Data v2 with several training implementations per
                                             # defect (defect_kinds.sample_kinds) and per-source eye boxes
"""
import hashlib
import io
import json
import os
import shutil
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
import cross_dataset_duplicates as CDD  # noqa: E402
from run_log import RunLog  # noqa: E402

V2 = "--v2" in sys.argv
OUT = os.path.join(ROOT, "Data", "Quality Data v2" if V2 else "Quality Data")
LABELS_COPY = os.path.join(ROOT, "labels", "quality_data_v2_labels.csv" if V2 else "quality_data_labels.csv")
if V2:
    import defect_kinds as K  # noqa: E402
    EYE = pd.read_csv(os.path.join(ROOT, "labels", "eye_regions.csv")).set_index("source_image_id")[["x0", "y0", "x1", "y1"]]
SEED = 14
K_DEGRADED = 3
COS_DUP, PHASH_DUP = 0.98, 0
# pool dataset -> (master CSV, key in cross_dataset_duplicates.DATASETS)
MASTERS = {
    "Nuclear Cataract": ("nuclear_cataract_master.csv", "Nuclear_Cataract"),
    "SUSTech-SYSU": ("sustech_master.csv", "SUSTech-SYSU"),
    "SLID-E": ("SLID-E_master.csv", "SLID-E"),
    "Infectious Keratitis": ("Infectious_Keratitis_master.csv", "Infectious_Keratitis"),
    "SLID": ("SLID_master.csv", "SLID"),
}


def md5_bytes(b):
    return hashlib.md5(b).hexdigest()


# ---------------------------------------------------------------- leakage checks
def leakage_checks():
    lines, ok = [], True
    pool = pd.read_csv(os.path.join(ROOT, "labels", "quality_clean_pool.csv"))
    pool = pool[pool.clean].reset_index(drop=True)
    man = pd.read_csv(os.path.join(ROOT, "labels", "resized_512_manifest.csv")).set_index("image_id")
    group = {}
    for ds, (csv, _) in MASTERS.items():
        m = pd.read_csv(os.path.join(ROOT, "labels", csv)).set_index("image_id")
        sub = pool[pool.dataset == ds]
        # 1 split consistency
        bad = sub[sub.split.values != m.loc[sub.image_id, "split"].values]
        lines.append(f"- [{'ok' if bad.empty else 'FAIL'}] 1. {ds}: {len(sub)} clean sources, split = master split for all"
                     + ("" if bad.empty else f"; mismatches: {bad.image_id.tolist[:10]}"))
        ok &= bad.empty
        # 2 groups within one split (whole master, not only the pool)
        g = m.groupby("group_id").split.nunique()
        lines.append(f"- [{'ok' if (g > 1).sum == 0 else 'FAIL'}] 2. {ds}: {len(g)} patient/eye groups, "
                     f"{int((g > 1).sum)} in more than one split")
        ok &= int((g > 1).sum()) == 0
        for iid in sub.image_id:
            group[iid] = f"{ds}|{m.at[iid, 'group_id']}"
    pool["group"] = pool.image_id.map(group)
    # 3 identical files across splits
    pool["md5"] = man.loc[pool.image_id, "md5"].values
    per = pool.groupby("md5").split.nunique()
    n3 = int((per > 1).sum())
    lines.append(f"- [{'ok' if n3 == 0 else 'FAIL'}] 3. identical 512 px files (MD5) in two splits: {n3} "
                 f"({pool.md5.nunique} distinct files for {len(pool)} sources)")
    ok &= n3 == 0
    # 4 duplicates across splits by pHash / ResNet50 cosine (all datasets pooled)
    feats, hashes, ids = [], [], []
    for ds, (_, key) in MASTERS.items():
        c = CDD.load(key)
        pos = {iid: i for i, iid in enumerate(c["ids"])}
        sub = pool[pool.dataset == ds]
        idx = [pos[i] for i in sub.image_id]
        feats.append(c["F"][idx]); hashes.append(c["H"][idx]); ids += sub.image_id.tolist()
    F = np.concatenate(feats).astype(np.float32)
    H = np.concatenate(hashes).astype(np.float32)
    split = pool.set_index("image_id").loc[ids, "split"].to_numpy()
    worst, n4 = [], 0
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        ia, ib = np.where(split == a)[0], np.where(split == b)[0]
        cos = F[ia] @ F[ib].T
        ham = H[ia] @ (1 - H[ib]).T + (1 - H[ia]) @ H[ib].T
        dup = (cos >= COS_DUP) | (ham <= PHASH_DUP)
        n4 += int(dup.sum())
        flat = np.argsort(cos, axis=None)[::-1][:5]
        for f in flat:
            r, c_ = np.unravel_index(f, cos.shape)
            worst.append((float(cos[r, c_]), int(ham[r, c_]), ids[ia[r]], a, ids[ib[c_]], b))
        lines.append(f"- {a} vs {b}: {len(ia)} × {len(ib)} pairs; cosine ≥ {COS_DUP} or pHash = {PHASH_DUP}: {int(dup.sum)}; "
                     f"cosine ≥ 0.95: {int((cos >= 0.95).sum)}")
    lines.insert(len(lines) - 3, f"- [{'ok' if n4 == 0 else 'FAIL'}] 4. duplicate photos across splits (pHash 0 or cosine ≥ {COS_DUP}): {n4}")
    ok &= n4 == 0
    # 6 datasets without patient IDs: the same-eye rule (file numbers <= 10 apart and cosine >= 0.85,
    # or cosine >= 0.92) must find no pair across splits
    import re
    offs = np.cumsum([0] + [len(f) for f in feats])
    for k, ds in enumerate(MASTERS):
        if ds not in ("SLID", "SUSTech-SYSU"):
            continue
        sl = slice(offs[k], offs[k + 1])
        Fd, sd = F[sl], split[sl]
        nums = np.array([int(re.search(r"(\d+)$", i).group(1)) for i in ids[sl]])
        iu = np.triu_indices(len(sd), 1)
        cc, gap = (Fd @ Fd.T)[iu], np.abs(nums[iu[0]] - nums[iu[1]])
        n6 = int(((sd[iu[0]] != sd[iu[1]]) & (((gap <= 10) & (cc >= 0.85)) | (cc >= 0.92))).sum())
        lines.append(f"- [{'ok' if n6 == 0 else 'FAIL'}] 6. {ds} (no patient IDs): cross-split pairs under the same-eye rule "
                     f"(≤ 10 files apart and cosine ≥ 0.85, or cosine ≥ 0.92): {n6}")
        ok &= n6 == 0
    worst.sort(reverse=True)
    lines.append("  - closest cross-split pairs (cosine, pHash distance, image, split, image, split):")
    lines += [f"    - {w[0]:.4f}, {w[1]}, {w[2]} ({w[3]}), {w[4]} ({w[5]})" for w in worst[:10]]
    # 5 real check set
    rc = pd.read_csv(os.path.join(ROOT, "labels", "real_check_set.csv"))
    rc_groups = {}
    for ds, (csv, _) in MASTERS.items():
        m = pd.read_csv(os.path.join(ROOT, "labels", csv)).set_index("image_id")
        for iid in rc[rc.dataset == ds].image_id:
            rc_groups[iid] = f"{ds}|{m.at[iid, 'group_id']}"
    rc["group"] = rc.image_id.map(rc_groups)
    train_groups = set(pool[pool.split == "train"].group)
    kept = rc[rc.split.isin(["val", "test"])]
    overlap_kept = int(kept.group.isin(train_groups).sum())
    overlap_train = int(rc[rc.split == "train"].group.isin(train_groups).sum())
    lines.append(f"- [{'ok' if overlap_kept == 0 else 'FAIL'}] 5. real check set: {len(kept)} val/test photos kept, "
                 f"{overlap_kept} share a group with a quality-training source; the {int((rc.split == 'train').sum)} "
                 f"train-split photos are excluded ({overlap_train} of them share a group with training sources)")
    ok &= overlap_kept == 0
    return ok, lines, pool, kept


# ---------------------------------------------------------------- generation
def work(job):
    """One source: clean copy + K degraded versions. Returns rows (and writes files unless dry)."""
    src = os.path.join(ROOT, job["resized_path"])
    folder = os.path.join(OUT, job["split"], job["source_set"])
    base = dict(split=job["split"], dataset=job["dataset"], source_set=job["source_set"],
                source_image_id=job["image_id"], source_group=job["group"])
    rows = []
    sid = f"{job['image_id']}__clean"
    rel = f"{job['split']}/{job['source_set']}/{sid}.png"
    if not job.get("dry"):
        shutil.copyfile(src, os.path.join(folder, sid + ".png"))
    rows.append(dict(sample_id=sid, kind="clean", n_defects=0, defects="", **{v: 0.0 for v in D.VALUES},
                     params="{}", path=rel, md5=job["md5"], **base))
    x = D.to_float(Image.open(src))
    crc = zlib.crc32(job["image_id"].encode())
    if V2:
        D.set_eye_box(tuple(EYE.loc[job["image_id"]]))
    for j in range(1, K_DEGRADED + 1):
        rng = np.random.default_rng([SEED, crc, j])
        y, labels, applied = K.sample_kinds(x, rng, min_defects=1) if V2 else D.sample(x, rng, min_defects=1)
        buf = io.BytesIO()
        D.to_pil(y).save(buf, format="PNG")
        data = buf.getvalue()
        sid = f"{job['image_id']}__d{j}"
        rel = f"{job['split']}/{job['source_set']}/{sid}.png"
        if not job.get("dry"):
            with open(os.path.join(folder, sid + ".png"), "wb") as f:
                f.write(data)
        rows.append(dict(sample_id=sid, kind="degraded", n_defects=len(applied), defects=";".join(applied),
                         **{v: round(float(labels[v]), 6) for v in D.VALUES},
                         params=json.dumps(applied, default=float, sort_keys=True), path=rel,
                         md5=md5_bytes(data), **base))
    return rows


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # the report contains "≥" (Windows console is cp1252)
    check_only = "--check" in sys.argv
    with RunLog("make_quality_data.py", notes="leakage checks only" if check_only else ("build Data/Quality Data v2 " if V2 else "build Data/Quality Data")) as log:
        ok, lines, pool, rc_kept = leakage_checks()
        report = "# Quality Data: leakage checks\n\n" + "\n".join(lines) + "\n"
        print(report)
        log.add(leakage_checks_passed=bool(ok))
        if check_only:
            os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
            open(os.path.join(ROOT, "logs", "quality_data_leakage_precheck.md"), "w", encoding="utf-8").write(report)
            return
        if not ok:
            raise SystemExit("leakage checks failed: nothing built")
        if os.path.exists(OUT):
            raise SystemExit(f"{OUT} exists: refusing to overwrite")
        for (sp, ss) in pool[["split", "source_set"]].drop_duplicates().itertuples(index=False):
            os.makedirs(os.path.join(OUT, sp, ss), exist_ok=True)
        jobs = pool[["image_id", "dataset", "source_set", "split", "resized_path", "md5", "group"]].to_dict("records")
        workers = max(1, os.cpu_count() - 1)
        rows = []
        with Pool(workers) as p:
            for i, r in enumerate(p.imap(work, jobs, chunksize=4), 1):
                rows += r
                if i % 1000 == 0:
                    print(f"  {i} / {len(jobs)} sources", flush=True)
        cols = ["sample_id", "split", "dataset", "source_set", "source_image_id", "source_group", "kind", "n_defects",
                "defects"] + D.VALUES + ["params", "path", "md5"]
        df = pd.DataFrame(rows)[cols].sort_values(["split", "dataset", "sample_id"]).reset_index(drop=True)
        # post-checks
        assert df.sample_id.is_unique
        assert (df.groupby("source_group").split.nunique() == 1).all(), "a group in two splits"
        src_split = pool.set_index("image_id").split
        assert (df.split.values == src_split.loc[df.source_image_id].values).all()
        rng = np.random.default_rng(SEED)
        recheck = pool.iloc[rng.choice(len(pool), 30, replace=False)]
        mism = 0
        for job in recheck[["image_id", "dataset", "source_set", "split", "resized_path", "md5", "group"]].to_dict("records"):
            job["dry"] = True
            for r in work(job):
                mism += int(df.loc[df.sample_id == r["sample_id"], "md5"].iloc[0] != r["md5"])
        report += f"- [{'ok' if mism == 0 else 'FAIL'}] determinism: 30 random sources (120 samples) regenerated in memory, MD5 mismatches: {mism}\n"
        # real check copies
        rc_rows = []
        for r in rc_kept.itertuples():
            rel = f"real_check/{r.split}/{r.image_id}.png"
            os.makedirs(os.path.join(OUT, "real_check", r.split), exist_ok=True)
            shutil.copyfile(os.path.join(ROOT, r.resized_path), os.path.join(OUT, rel))
            rc_rows.append(dict(image_id=r.image_id, dataset=r.dataset, split=r.split, defect_type=r.defect_type,
                                comment=r.comment, path=rel))
        pd.DataFrame(rc_rows).to_csv(os.path.join(OUT, "real_check.csv"), index=False)
        df.to_csv(os.path.join(OUT, "labels.csv"), index=False, float_format="%.6g")
        shutil.copyfile(os.path.join(OUT, "labels.csv"), LABELS_COPY)
        open(os.path.join(OUT, "leakage_report.md"), "w", encoding="utf-8").write(report)
        code_md5 = md5_bytes(open(os.path.join(HERE, "degradations.py"), "rb").read())
        if V2:
            code_md5 += " + defect_kinds.py " + md5_bytes(open(os.path.join(HERE, "defect_kinds.py"), "rb").read()) +                 " + degradations_alt.py " + md5_bytes(open(os.path.join(HERE, "degradations_alt.py"), "rb").read())
        n_files = sum(len(f) for _, _, f in os.walk(OUT))
        n_bytes = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(OUT) for f in fs)
        counts = df.groupby(["split", "kind"]).size().unstack().to_dict()
        open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write(
            "# Quality Data\n\nTraining / validation / test data of the quality module (project ).\n\n"
            "- `train/`, `val/`, `test/`: per clean source image, `<id>__clean.png` (all labels 0) and "
            f"`<id>__d1..d{K_DEGRADED}.png` (synthetic defects), 512 px PNG\n"
            "- `labels.csv`: one row per image: split, dataset, source image and patient group, defects applied, "
            "the 9 measured labels (" + ", ".join(D.VALUES) + "), parameters (JSON), path, MD5\n"
            "- `real_check/` + `real_check.csv`: real photos with clearly visible defects (val / test only), "
            "for evaluation only, no labels\n"
            "- `leakage_report.md`: leakage checks\n\n"
            f"Generator: `quality/make_quality_data.py`; degradations: `quality/degradations.py` (MD5 {code_md5}); seed {SEED}.\n")
        print(df.groupby(["split", "kind"]).size().to_string())
        print(report)
        log.add(images_in=len(pool), images_out=len(df), samples_per_split=df.split.value_counts().to_dict(),
                counts=str(counts), real_check=len(rc_rows), output_files=n_files, output_bytes=n_bytes,
                determinism_mismatches=mism, degradations_md5=code_md5,
                labels_md5=md5_bytes(open(os.path.join(OUT, "labels.csv"), "rb").read()), workers=workers)


if __name__ == "__main__":
    main()

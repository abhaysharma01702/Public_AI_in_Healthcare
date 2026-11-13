"""Build the v3 extension of Quality Data v2 (step A2, coverage audit ).

For about half of the clean sources (all splits, each in its own split) one extra degraded version
    ext_v3/<split>/<source_set>/<image_id>__d4.png
containing ONE of the 3 new training kinds (occlusion "side", uneven illumination "vignette",
defocus "astig_mild"), plus 0-2 other defects by the normal mixing policy (sample_extension below).
Quality Data v2 itself is not changed; the extension rows go to Data/Quality Data v2/labels_ext_v3.csv
with paths relative to the v2 root, so the loader can append them to labels.csv.

Seeds: source s uses numpy default_rng([14, crc32(image_id), 4]) (j = 4: independent of d1-d3, which use
j = 1..3). The first draw decides whether the source gets a d4 (p = 0.5), so the choice is per source and
order- / worker-independent.

Checks (the build aborts if one fails; report in ext_v3/extension_report.md):
  1-5. the leakage checks of make_quality_data.py, re-run on the current files
  6.   every extension source is a v2 clean source, in the same split; no group in two splits over v2 + extension
  after generation: determinism re-check of 30 extension samples (MD5).

Usage:
  python -I make_quality_extension.py --dry    # draws only (no files): kind shares and the planned count
  python -I make_quality_extension.py          # checks, then build (refuses if ext_v3/ exists)
"""
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
import degradations as D  # noqa: E402
import defect_kinds as K  # noqa: E402
import make_quality_data as MQD  # noqa: E402  (V2 off: only its leakage checks and md5 helper are used)
from run_log import RunLog  # noqa: E402

V2_ROOT = os.path.join(ROOT, "Data", "Quality Data v2")
EXT = "ext_v3"
OUT = os.path.join(V2_ROOT, EXT)
LABELS_EXT = os.path.join(V2_ROOT, "labels_ext_v3.csv")
EYE = pd.read_csv(os.path.join(ROOT, "labels", "eye_regions.csv")).set_index("source_image_id")[["x0", "y0", "x1", "y1"]]
SEED, J = 14, 4
P_EXTEND = 0.5
NEW_KINDS = [("occlusion", "side"), ("uneven_illumination", "vignette"), ("defocus_blur", "astig_mild")]


def sample_extension(x, rng):
    """One new kind (uniform over NEW_KINDS, strength t ~ U(0, 1)) plus k - 1 other defects, k drawn from
    N_DEFECTS_P with k >= 1 (as d1-d3). The others follow the normal policy: never with a NOT_TOGETHER partner,
    kind uniform over kind_funcs, t ~ U(0, 1), low light (noise added to underexposure, p = LOW_LIGHT_P).
    An extreme exposure / haze draw among the others is dropped (extremes are applied alone, and the new kind
    must stay). Application in the usual ORDER. Returns (y, labels, applied)."""
    forced, forced_kind = NEW_KINDS[int(rng.integers(len(NEW_KINDS)))]
    ks = [k for k in D.N_DEFECTS_P if k >= 1]
    p = np.array([D.N_DEFECTS_P[k] for k in ks])
    k = int(rng.choice(ks, p=p / p.sum()))
    chosen = [forced]
    for name in rng.permutation(D.VALUES):
        if len(chosen) == k:
            break
        if name in chosen or any({name, c} in [set(q) for q in D.NOT_TOGETHER] for c in chosen):
            continue
        chosen.append(str(name))
    t = {n: float(rng.uniform()) for n in chosen}
    chosen = [n for n in chosen if n == forced or not (n in D.EXTREME and t[n] >= D.EXTREME[n])]
    if "underexposure" in chosen and "noise" not in chosen and len(chosen) < 3 and rng.uniform() < D.LOW_LIGHT_P:
        chosen.append("noise")
        t["noise"] = float(rng.uniform())
    labels, applied, y = {v: 0.0 for v in D.VALUES}, {}, x
    for name in D.ORDER:
        if name in chosen:
            funcs = K.kind_funcs(name)
            kind = forced_kind if name == forced else str(rng.choice(sorted(funcs)))
            y, labels[name], params = funcs[kind](y, rng, t[name])
            applied[name] = {"impl": kind, "t": t[name], "new_kind": name == forced, **params}
    return y, labels, applied


def work(job):
    """One source: decide (p = 0.5) and, if chosen, make d4. Returns [] or [row]; writes the file unless dry."""
    crc = zlib.crc32(job["source_image_id"].encode())
    rng = np.random.default_rng([SEED, crc, J])
    if rng.uniform() >= P_EXTEND:
        return []
    if job.get("plan_only"):
        name, kind = NEW_KINDS[int(rng.integers(len(NEW_KINDS)))]
        return [dict(source_image_id=job["source_image_id"], split=job["split"], new_defect=name, new_kind=kind)]
    x = D.to_float(Image.open(os.path.join(V2_ROOT, job["path"])))
    D.set_eye_box(tuple(EYE.loc[job["source_image_id"]]))
    y, labels, applied = sample_extension(x, rng)
    buf = io.BytesIO()
    D.to_pil(y).save(buf, format="PNG")
    data = buf.getvalue()
    sid = f"{job['source_image_id']}__d{J}"
    rel = f"{EXT}/{job['split']}/{job['source_set']}/{sid}.png"
    if not job.get("dry"):
        with open(os.path.join(V2_ROOT, rel), "wb") as f:
            f.write(data)
    new = [n for n, a in applied.items() if a["new_kind"]][0]
    return [dict(sample_id=sid, split=job["split"], dataset=job["dataset"], source_set=job["source_set"],
                 source_image_id=job["source_image_id"], source_group=job["source_group"], kind="degraded",
                 n_defects=len(applied), defects=";".join(applied),
                 **{v: round(float(labels[v]), 6) for v in D.VALUES},
                 params=json.dumps(applied, default=float, sort_keys=True), path=rel, md5=MQD.md5_bytes(data),
                 new_defect=new, new_kind=applied[new]["impl"])]


def _block(text):
    return "```\n" + text + "\n```"


def sources():
    q = pd.read_csv(os.path.join(V2_ROOT, "labels.csv"))
    return q, q[q.kind == "clean"][["source_image_id", "dataset", "source_set", "split", "source_group", "path"]]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    q, src = sources()
    jobs = src.to_dict("records")
    if "--dry" in sys.argv:
        plan = pd.DataFrame([r for j in jobs for r in work({**j, "plan_only": True})])
        print(f"{len(plan)} of {len(src)} sources get a d4 ({len(plan) / len(src):.1%})")
        print(plan.groupby(["split", "new_kind"]).size().unstack().to_string())
        return
    with RunLog("make_quality_extension.py", notes="build Quality Data v2 extension ext_v3 (A2)") as log:
        ok, lines, pool, _ = MQD.leakage_checks()
        same = pool.set_index("image_id").split
        in_pool = src.source_image_id.isin(same.index)
        same_split = bool(in_pool.all()) and bool((same.loc[src.source_image_id].values == src.split.values).all())
        lines.append(f"- [{'ok' if same_split else 'FAIL'}] 6a. all {len(src)} v2 clean sources are in the current "
                     f"clean pool with the same split ({int((~in_pool).sum)} missing)")
        ok &= same_split
        report = "# Quality Data v2 extension (ext_v3): checks\n\n" + "\n".join(lines) + "\n"
        print(report)
        if not ok:
            raise SystemExit("checks failed: nothing built")
        if os.path.exists(OUT) or os.path.exists(LABELS_EXT):
            raise SystemExit(f"{OUT} or {LABELS_EXT} exists: refusing to overwrite")
        for (sp, ss) in src[["split", "source_set"]].drop_duplicates().itertuples(index=False):
            os.makedirs(os.path.join(OUT, sp, ss), exist_ok=True)
        workers = max(1, os.cpu_count() - 1)
        rows = []
        with Pool(workers) as p:
            for i, r in enumerate(p.imap(work, jobs, chunksize=4), 1):
                rows += r
                if i % 1000 == 0:
                    print(f"  {i} / {len(jobs)} sources", flush=True)
        cols = ["sample_id", "split", "dataset", "source_set", "source_image_id", "source_group", "kind", "n_defects",
                "defects"] + D.VALUES + ["params", "path", "md5", "new_defect", "new_kind"]
        df = pd.DataFrame(rows)[cols].sort_values(["split", "dataset", "sample_id"]).reset_index(drop=True)
        # 6b. combined: unique ids, one split per group, split = source split
        both = pd.concat([q, df[q.columns]], ignore_index=True)
        c6 = bool(both.sample_id.is_unique and (both.groupby("source_group").split.nunique() == 1).all()
                  and (df.split.values == same.loc[df.source_image_id].values).all())
        report += (f"- [{'ok' if c6 else 'FAIL'}] 6b. v2 + extension ({len(both)} images): sample ids unique, every "
                   f"group in one split, every d4 in its source's split\n")
        rng = np.random.default_rng(SEED)
        recheck = df.iloc[rng.choice(len(df), 30, replace=False)]
        mism = 0
        for r0 in recheck.itertuples():
            job = src[src.source_image_id == r0.source_image_id].iloc[0].to_dict()
            job["dry"] = True
            mism += int(work(job)[0]["md5"] != r0.md5)
        report += f"- [{'ok' if mism == 0 else 'FAIL'}] determinism: 30 random d4 samples regenerated in memory, MD5 mismatches: {mism}\n"
        # kind shares per defect over v2 train + extension train
        shares = []
        for frame in (q, df):
            for r0 in frame[frame.split == "train"].itertuples():
                if isinstance(r0.params, str) and r0.params != "{}":
                    for name, a in json.loads(r0.params).items():
                        shares.append((name, a.get("impl", "?")))
        sh = pd.DataFrame(shares, columns=["defect", "kind"]).value_counts().rename("n").reset_index()
        sh["share"] = sh.n / sh.groupby("defect").n.transform("sum")
        new_sh = sh.merge(pd.DataFrame(NEW_KINDS, columns=["defect", "kind"]))
        report += ("\n## Training kind shares (v2 train + extension train)\n\n"
                   + _block(sh.sort_values(["defect", "kind"]).to_string(index=False, float_format="%.3f")) + "\n")
        report += "\n## Label ranges of the new kinds (the new defect's own label)\n\n"
        lab = pd.DataFrame({"new_kind": df.new_kind, "label": [df.loc[i, df.loc[i, "new_defect"]] for i in df.index]})
        report += _block(lab.groupby("new_kind").label.describe(percentiles=[0.05, 0.5, 0.95]).to_string(float_format="%.3f")) + "\n"
        report += "\n## Counts\n\n" + _block(df.groupby(["split", "new_kind"]).size().unstack().to_string()) + "\n"
        df.to_csv(LABELS_EXT, index=False, float_format="%.6g")
        open(os.path.join(OUT, "extension_report.md"), "w", encoding="utf-8").write(report)
        print(report)
        code = {f: MQD.md5_bytes(open(os.path.join(HERE, f), "rb").read())
                for f in ["degradations.py", "defect_kinds.py", "degradations_alt.py", "make_quality_extension.py"]}
        n_bytes = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(OUT) for f in fs)
        log.add(sources=len(src), extension_images=len(df), per_split=df.split.value_counts().to_dict(),
                per_new_kind=df.new_kind.value_counts().to_dict(),
                new_kind_train_share=new_sh.set_index("kind").share.round(3).to_dict(),
                determinism_mismatches=mism, checks_passed=bool(ok and c6 and mism == 0), output_bytes=n_bytes,
                code_md5=code, labels_ext_md5=MQD.md5_bytes(open(LABELS_EXT, "rb").read()), workers=workers)
        if not (c6 and mism == 0):
            raise SystemExit("post-build checks FAILED (see report)")


if __name__ == "__main__":
    main()

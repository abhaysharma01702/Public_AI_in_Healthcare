"""Dataset composition tables of the report: Quality Data v2 per split / dataset, defects per degraded image, per-defect
frequency and label range, the 9 x 9 co-occurrence matrix (diagonal = defect alone), the number of distinct triples,
implementations per defect, the coverage extension, the evaluation sets (gen_train / gen_holdout / gen_fresh / stress /
low resolution), the grading training copies and test sets, the photograph accounting (release -> usable ->
quality pool -> clean pool -> split) and the split of every grading photograph in the quality data (independence).
Writes results/data_composition/*.csv.
Usage (from the project root): python quality/data_composition.py
"""
import collections
import itertools
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QD = os.path.join(ROOT, "Data", "Quality Data v2")
GD = os.path.join(ROOT, "Data", "Grading Data")
OUT = os.path.join(ROOT, "results", "data_composition")
D = ["defocus_blur", "motion_blur", "overexposure", "underexposure", "uneven_illumination", "haze", "glare",
     "occlusion", "noise"]


def present(df):
    return df[D].gt(0)


def main():
    os.makedirs(OUT, exist_ok=True)
    L = pd.read_csv(os.path.join(QD, "labels.csv"))
    deg = L[L.kind != "clean"]
    pd.crosstab(L.split, L.kind, margins=True).to_csv(os.path.join(OUT, "v2_split_by_kind.csv"))
    pd.crosstab(L.dataset, L.split, margins=True).to_csv(os.path.join(OUT, "v2_dataset_by_split.csv"))
    pd.crosstab(deg.split, deg.n_defects, margins=True).to_csv(os.path.join(OUT, "v2_defects_per_image.csv"))

    p = present(deg)
    freq = pd.DataFrame({s: p[deg.split == s].sum() for s in ["train", "val", "test"]})
    freq["share_pct"] = (p.mean() * 100).round(1)
    for stat, fn in [("label_min", "min"), ("label_median", "median"), ("label_max", "max")]:
        freq[stat] = [getattr(deg.loc[p[d], d], fn)() for d in D]
    freq.to_csv(os.path.join(OUT, "v2_defect_frequency.csv"), float_format="%.3f")

    co = pd.DataFrame(0, index=D, columns=D)
    for row, k in zip(p.values, deg.n_defects):
        on = [D[i] for i in range(9) if row[i]]
        for a, b in itertools.combinations(on, 2):
            co.loc[a, b] += 1
            co.loc[b, a] += 1
        if k == 1 and len(on) == 1:   # diagonal: exactly one defect was applied
            co.loc[on[0], on[0]] += 1
    co.to_csv(os.path.join(OUT, "v2_cooccurrence.csv"))
    triples = collections.Counter(tuple(sorted(D[i] for i in range(9) if r[i])) for r in p[deg.n_defects == 3].values)
    pd.DataFrame(triples.most_common(), columns=["triple", "images"]).to_csv(os.path.join(OUT, "v2_triples.csv"), index=False)

    impl = collections.Counter()
    for params in deg.params:
        for d, v in json.loads(params).items():
            impl[(d, v.get("kind", v.get("impl", "?")))] += 1
    pd.DataFrame([(d, k, n) for (d, k), n in sorted(impl.items())], columns=["defect", "implementation", "images"]).to_csv(
        os.path.join(OUT, "v2_implementations.csv"), index=False)

    E = pd.read_csv(os.path.join(QD, "labels_ext_v3.csv"))
    pd.crosstab(E.split, E.n_defects, margins=True).to_csv(os.path.join(OUT, "extension_defects_per_image.csv"))
    pd.crosstab(E.dataset, E.split, margins=True).to_csv(os.path.join(OUT, "extension_dataset_by_split.csv"))

    rows = []
    for name in ["gen_train", "gen_holdout", "gen_fresh", "stress_4defects", "gen_lowres"]:
        S = pd.read_csv(os.path.join(QD, f"{name}.csv"))
        rows.append(dict(set=name, images=len(S), clean=int((S.kind == "clean").sum()),
                         defects_per_image=sorted(S.n_defects.unique().tolist()),
                         **{d: int((S[d] > 0).sum()) for d in D}))
    S = pd.read_csv(os.path.join(QD, "stress_4defects.csv"))
    sp = present(S).values
    pair = collections.Counter(pr for r in sp for pr in itertools.combinations([D[i] for i in range(9) if r[i]], 2))
    rows.append(dict(set="stress_zero_pairs", images=0,
                     defects_per_image=[f"{a}+{b}" for a, b in itertools.combinations(D, 2) if pair[(a, b)] == 0]))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "evaluation_sets.csv"), index=False)

    g = []
    for task in ["nuclear_cataract", "sustech_sysu"]:
        G = pd.read_csv(os.path.join(GD, task, "labels.csv"))
        tr = G[G.kind == "train_degraded"]
        n = present(tr).sum(1)
        te = G[(G.split == "test") & (G.kind != "clean")]
        g.append(dict(task=task, **{f"clean_{s}": int(((G.kind == "clean") & (G.split == s)).sum()) for s in ["train", "val", "test"]},
                      degraded_train=int(((G.kind == "train_degraded") & (G.split == "train")).sum()),
                      degraded_val=int(((G.kind == "train_degraded") & (G.split == "val")).sum()),
                      **{f"copies_{k}_defects": int((n == k).sum()) for k in range(4)},
                      **{f"copies_{d}": int(tr[d].gt(0).sum()) for d in D},
                      test_images=len(te), test_per_defect=int(te.defect.value_counts().iloc[0]),
                      test_per_strength=int(te.strength.value_counts().iloc[0])))
    pd.DataFrame(g).to_csv(os.path.join(OUT, "grading_data.csv"), index=False)

    # photograph accounting (report table "Photograph accounting"): release -> usable -> quality pool -> clean -> split
    LAB = os.path.join(ROOT, "labels")
    masters = {"SLID": "SLID_master", "SLID-E": "SLID-E_master", "Infectious Keratitis": "Infectious_Keratitis_master",
               "Nuclear Cataract": "nuclear_cataract_master", "SUSTech-SYSU": "sustech_master"}
    pool = pd.read_csv(os.path.join(LAB, "quality_clean_pool.csv"))
    acc = []
    for ds, f in masters.items():
        m = pd.read_csv(os.path.join(LAB, f + ".csv"))
        st = m.status if "status" in m else pd.Series("usable", index=m.index)
        p = pool[pool.dataset == ds]
        c = L[(L.dataset == ds) & (L.kind == "clean")]
        acc.append(dict(dataset=ds, release=len(m), **{f"status_{k}": int(v) for k, v in st.value_counts().items()},
                        usable=int(st.isin(["usable", "graded"]).sum()), quality_pool=len(p),
                        screened_out=int((~p.clean).sum()), clean=len(c),
                        **{f"clean_{s}": int((c.split == s).sum()) for s in ["train", "val", "test"]}))
    a = pd.DataFrame(acc).fillna(0)
    a.loc["total"] = a.sum(numeric_only=True)
    a.loc["total", "dataset"] = "total"
    num = a.columns.drop("dataset")
    a[num] = a[num].astype(int)
    a.to_csv(os.path.join(OUT, "photo_accounting.csv"), index=False)

    # independence of grading and quality data: split of every clean grading photograph in the quality data
    ind = []
    qc = L[L.kind == "clean"][["dataset", "source_image_id", "split"]]
    for task, ds in [("nuclear_cataract", "Nuclear Cataract"), ("sustech_sysu", "SUSTech-SYSU")]:
        G = pd.read_csv(os.path.join(GD, task, "labels.csv"))
        gc = G[G.kind == "clean"][["image_id", "split"]]
        m = gc.merge(qc[qc.dataset == ds].rename(columns={"source_image_id": "image_id"}), on="image_id", how="left",
                     suffixes=("_grading", "_quality"))
        t = pd.crosstab(m.split_grading, m.split_quality.fillna("not in quality data"))
        t.insert(0, "task", task)
        ind.append(t.reset_index())
    pd.concat(ind).to_csv(os.path.join(OUT, "grading_quality_independence.csv"), index=False)

    print(pd.crosstab(deg.split, deg.n_defects, margins=True), "\n", co, "\n", freq, "\ntriples", len(triples),
          triples.most_common(3), "\n", pd.DataFrame(rows).iloc[:, :4], "\n", pd.DataFrame(g).T)


if __name__ == "__main__":
    main()

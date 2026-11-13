"""E3 leakage check: an IMAGE-level split of a grading task's clean photos, to measure how much
ignoring patients / eyes inflates test scores (published baselines split by image; ours split by group).
Same procedure as the real split except for the groups: StratifiedKFold(shuffle=True, random_state=14) on the photos,
stratified by label_grade; 10 folds -> first = test; remaining 9 folds -> first = val; rest = train.
Only the clean rows of the existing grading data are used (condition A trains on clean photos only), so no images are
rebuilt: the job reads the same zip with this labels file (train_grading.py --labels).
Also reports the leakage this split creates: test photos whose group (Nuclear Cataract volunteer, SUSTech-SYSU eye group)
also has photos in train.
Output: labels/<task>_imgsplit_grading_labels.csv
Usage: python grading/make_image_split.py --task nuclear_cataract | sustech_sysu
"""
import argparse
import hashlib
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTERS = {"sustech_sysu": "sustech_master.csv", "nuclear_cataract": "nuclear_cataract_master.csv"}
SEED = 14


def first_fold(y, n_splits):
    """Positions of the first fold of a shuffled stratified k-fold (the procedure without groups)."""
    return next(iter(StratifiedKFold(n_splits, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y)))[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(MASTERS))
    a = ap.parse_args()
    lab = pd.read_csv(os.path.join(ROOT, "Data", "Grading Data", a.task, "labels.csv"))
    d = lab[lab.kind == "clean"].sort_values("image_id").reset_index(drop=True)
    m = pd.read_csv(os.path.join(ROOT, "labels", MASTERS[a.task])).set_index("image_id")
    y = m.loc[d.image_id, "label_grade"].astype(int).to_numpy()   # the label the real split was stratified on
    d["group_split"] = d.split
    split = np.array(["train"] * len(d), dtype=object)
    te = first_fold(y, 10)
    split[te] = "test"
    rest = np.setdiff1d(np.arange(len(d)), te)
    split[rest[first_fold(y[rest], 9)]] = "val"
    d["split"] = split
    out = os.path.join(ROOT, "labels", f"{a.task}_imgsplit_grading_labels.csv")
    d.drop(columns="group_split").to_csv(out, index=False)
    md5 = hashlib.md5(open(out, "rb").read()).hexdigest()
    print(d.split.value_counts().to_string())
    print(pd.crosstab(d.split, d.grade).to_string())
    train_groups = set(d.group_id[d.split == "train"])
    t = d[d.split == "test"]
    shared = t.group_id.isin(train_groups)
    print(f"test photos whose group has photos in train: {shared.sum} / {len(t)} ({shared.mean:.1%}); "
          f"test groups also in train: {t.group_id[shared].nunique} / {t.group_id.nunique}")
    print(f"test photos that were test in the group split: {(t.group_split == 'test').sum} / {len(t)}")
    print(f"{out}: {len(d)} rows, MD5 {md5}")


if __name__ == "__main__":
    main()

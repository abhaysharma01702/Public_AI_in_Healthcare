"""Stricter same-eye grouping and re-split for the datasets WITHOUT patient IDs.

Why: the leakage check for Quality Data  found the same eye in two splits in SLID (SLID_0076 val /
SLID_0079 train): linked only CONSECUTIVE files, so one eye fell into two groups when a chain broke.
SUSTech-SYSU  has the same risk. No image is dropped.

Rule (applied to SLID and SUSTech-SYSU, all images, on the cached ResNet50 embeddings, Data/derived):
  keep every old group, and additionally link two images if
      (file numbers at most 10 apart and cosine >= 0.85)  or  cosine >= 0.92 anywhere
  -> connected components = new groups.
Split (80/10/10, seed 14, same stratification as before: SLID resolution x healthy/lesion, SUSTech-SYSU type):
  healthy look-alike eyes chain into very large groups under this rule; groups larger than 5% of the
  usable images are assigned to TRAIN, and the rest is split with StratifiedGroupKFold so that val and
  test each get about 10% of all usable images (two stages).

Writes (only with --apply): master CSVs with new group_id / split / split_path (old values kept in
group_id_v1 / split_v1), moves the Data/Split files to their new split folders.
Usage: python -I regroup_no_id_datasets.py            # dry run: statistics only
       python -I regroup_no_id_datasets.py --apply
"""
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cross_dataset_duplicates as CDD  # noqa: E402
from run_log import RunLog  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED, WINDOW, COS_NEAR, COS_FAR, BIG = 14, 10, 0.85, 0.92, 0.05
DATASETS = {"SLID": ("SLID_master.csv", "SLID"), "SUSTech-SYSU": ("sustech_master.csv", "SUSTech-SYSU")}


def num(s):
    return int(re.search(r"(\d+)$", s).group(1))


def new_groups(m, F):
    n = len(m)
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first = {}
    for i, g in enumerate(m.group_id):
        if g in first:
            parent[root(i)] = root(first[g])
        else:
            first[g] = i
    nums = np.array([num(i) for i in m.image_id])
    C = F @ F.T
    iu = np.triu_indices(n, 1)
    gap, c = np.abs(nums[iu[0]] - nums[iu[1]]), C[iu]
    link = ((gap <= WINDOW) & (c >= COS_NEAR)) | (c >= COS_FAR)
    for a, b in zip(iu[0][link], iu[1][link]):
        parent[root(a)] = root(b)
    comp = np.array([root(i) for i in range(n)])
    # name each group after its lowest-numbered image, like the old ids
    prefix = "SLID_g" if m.image_id.iloc[0].startswith("SLID") else "sus_g"
    width = 4 if prefix == "SLID_g" else 3
    name = {r: f"{prefix}{nums[comp == r].min:0{width}d}" for r in set(comp)}
    return np.array([name[r] for r in comp]), C, nums, int(link.sum())


def make_split(groups, y, usable):
    sp = np.array(["train"] * len(groups), dtype=object)
    u = np.where(usable)[0]
    sizes = pd.Series(groups[u]).value_counts()
    big = set(sizes[sizes > BIG * len(u)].index)
    rest = u[~np.isin(groups[u], list(big))]
    target = round(0.10 * len(u))
    k = max(2, round(len(rest) / target))
    r2, te = next(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED).split(rest, y[rest], groups[rest]))
    k2 = max(2, round(len(r2) / target))
    tr, va = next(StratifiedGroupKFold(n_splits=k2, shuffle=True, random_state=SEED).split(r2, y[rest][r2], groups[rest][r2]))
    sp[rest[te]] = "test"
    sp[rest[r2[va]]] = "val"
    # groups made only of excluded images stay in train (as before); every group in exactly one split
    g2s = pd.Series(sp[u], index=groups[u]).groupby(level=0).agg(lambda s: s.iloc[0])
    sp = np.array([g2s.get(g, "train") for g in groups], dtype=object)
    return sp, sorted(int(sizes[b]) for b in big), (k, k2)


def main():
    apply = "--apply" in sys.argv
    with RunLog("regroup_no_id_datasets.py", notes="apply" if apply else "dry run") as log:
        for ds, (csv, key) in DATASETS.items():
            path = os.path.join(ROOT, "labels", csv)
            m = pd.read_csv(path, keep_default_na=False)
            if "group_id_v1" in m:
                raise SystemExit(f"{csv} already regrouped (group_id_v1 exists)")
            c = CDD.load(key)
            pos = {i: k for k, i in enumerate(c["ids"])}
            F = c["F"][[pos[i] for i in m.image_id]].astype(np.float32)
            groups, C, nums, n_links = new_groups(m, F)
            usable = (m.status == "usable").to_numpy()
            if ds == "SLID":
                y = (m.resolution + "_" + np.where(m.lesions == "", "healthy", "lesion")).to_numpy()
                strat = np.where(m.lesions == "", "healthy", "lesion")
            else:
                y = m["type"].astype(str).to_numpy()
                strat = y
            sp, big, ks = make_split(groups, y, usable)
            iu = np.triu_indices(len(m), 1)
            cross = sp[iu[0]] != sp[iu[1]]
            gap, cc = np.abs(nums[iu[0]] - nums[iu[1]]), C[iu]
            left = int((cross & (((gap <= WINDOW) & (cc >= COS_NEAR)) | (cc >= COS_FAR))).sum())
            old_cross = m.split.to_numpy()
            old_left = int(((old_cross[iu[0]] != old_cross[iu[1]]) & (((gap <= WINDOW) & (cc >= COS_NEAR)) | (cc >= COS_FAR))).sum())
            moved = int((sp[usable] != old_cross[usable]).sum())
            print(f"\n{ds}: links added {n_links}; groups {m.group_id.nunique} -> {len(set(groups))}; "
                  f"groups > {BIG:.0%} forced into train: {big}; folds {ks}")
            print(f"  usable per split: {pd.Series(sp[usable]).value_counts.to_dict}  (before: {pd.Series(old_cross[usable]).value_counts.to_dict})")
            print(f"  risky cross-split pairs (rule): before {old_left}, after {left}; usable images changing split: {moved}")
            print("  stratum share % per split:")
            print((pd.crosstab(sp[usable], strat[usable], normalize="index") * 100).round(1).to_string())
            log.add(**{f"{ds}_groups": len(set(groups)), f"{ds}_forced": big, f"{ds}_risky_after": left,
                       f"{ds}_moved": moved, f"{ds}_split": pd.Series(sp[usable]).value_counts().to_dict()})
            if not apply:
                continue
            assert left == 0
            m.insert(m.columns.get_loc("group_id") + 1, "group_id_v1", m.group_id)
            m.insert(m.columns.get_loc("split") + 1, "split_v1", m.split)
            m["group_id"] = groups
            m["split"] = sp
            m["group_source"] = m.group_source.astype(str) + "+window_link_043"
            if "split_path" in m:  # SLID: flat folders Data/Split/SLID/<split>/<id>.png
                new_paths = np.where(m.split_path != "", "Data/Split/SLID/" + m.split + "/" + m.image_id + ".png", "")
                for old, new in zip(m.split_path, new_paths):
                    if old and old != new:
                        os.makedirs(os.path.dirname(os.path.join(ROOT, new)), exist_ok=True)
                        shutil.move(os.path.join(ROOT, old), os.path.join(ROOT, new))
                m["split_path"] = new_paths
            m.to_csv(path, index=False)
            print(f"  written {csv}")


if __name__ == "__main__":
    main()

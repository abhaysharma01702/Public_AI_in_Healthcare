"""Data loading for the quality module (section 2).

- Images: `Data/Quality Data` PNGs (512 px high, 633-771 px wide), resized to height x width
  (512 x 672) with Lanczos (the filter ); random horizontal / vertical flips in training
  (fix 5), drawn from default_rng([seed, epoch, index]) so they do not depend on workers and a
  resumed run sees the same flips. Returned as uint8; normalisation happens on the GPU (engine.py).
- Labels: the 9 measured values of labels.csv; a defect counts as present when its label is > 0
  (2 overexposure images were applied but measured 0 newly blown pixels; they count as absent).
- Batches: SourceGroupedBatchSampler puts all 4 versions of a source photo (clean + d1-d3) in the
  same batch, source order reshuffled every epoch from default_rng([seed, epoch]).
"""
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image

from . import LIGHTING, VALUES


def _with_lighting(df):
    df = df.copy()
    df["lighting"] = df["dataset"].map(LIGHTING)
    return df


def load_split(root, split, extension=None):
    """Rows of labels.csv for one split (train / val / test), in file order. `extension`: an extra labels CSV in
    `root` appended after labels.csv (labels_ext_v3.csv, the d4 versions; paths relative to `root`)."""
    q = pd.read_csv(os.path.join(root, "labels.csv"))
    if extension:
        q = pd.concat([q, pd.read_csv(os.path.join(root, extension))[q.columns]], ignore_index=True)
    return _with_lighting(q[q.split == split].reset_index(drop=True))


def load_stress(root):
    """4-defect stress-test set; test split only."""
    return _with_lighting(pd.read_csv(os.path.join(root, "stress_4defects.csv")))


def load_generalisation(root, impl):
    """Generalisation test: single-defect test images from our generator (impl "ours") or the
    alternative implementation ("alt"), plus the 831 clean test copies."""
    return _with_lighting(pd.read_csv(os.path.join(root, f"gen_{impl}.csv")))


def load_real_check(root):
    """The 30 real val / test photos with visible defects (no measured labels) plus, as negatives,
    the clean val / test sources (labels 0). Column real_defect: defect type noted by eye, '' for the clean photos."""
    q = pd.read_csv(os.path.join(root, "labels.csv"))
    neg = q[(q.kind == "clean") & q.split.isin(["val", "test"])].copy()
    neg["real_defect"] = ""
    rc = pd.read_csv(os.path.join(root, "real_check.csv"))
    pos = pd.DataFrame({
        "sample_id": rc.image_id, "split": rc.split, "dataset": rc.dataset, "source_image_id": rc.image_id,
        "kind": "real_defect", "n_defects": np.nan, "defects": "", "path": rc.path, "real_defect": rc.defect_type,
    })
    for v in VALUES:
        pos[v] = np.nan  # unknown for real photos
    cols = ["sample_id", "split", "dataset", "source_image_id", "kind", "n_defects", "defects", "path",
            "real_defect"] + VALUES
    return _with_lighting(pd.concat([pos[cols], neg[cols]], ignore_index=True))


def limit_sources(df, n, seed):
    """Keep the rows of n randomly chosen source photos (smoke tests only)."""
    if not n:
        return df
    sids = np.sort(df.source_image_id.unique())
    keep = np.random.default_rng(seed).choice(sids, size=min(n, len(sids)), replace=False)
    return df[df.source_image_id.isin(keep)].reset_index(drop=True)


class QualityDataset(torch.utils.data.Dataset):
    def __init__(self, df, root, height, width, train=False, seed=14):
        self.paths = [os.path.join(root, p) for p in df.path]
        self.labels = df[VALUES].to_numpy(np.float32)
        self.size = (int(width), int(height))  # PIL order
        self.train, self.seed, self.epoch = train, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB")
            if im.size != self.size:
                im = im.resize(self.size, Image.LANCZOS)
            x = np.array(im, dtype=np.uint8)  # writable copy
        if self.train:
            h_flip, v_flip = np.random.default_rng([self.seed, self.epoch, i]).random(2) < 0.5
            if h_flip:
                x = x[:, ::-1]
            if v_flip:
                x = x[::-1]
        x = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)  # 3 x H x W, uint8
        return x, torch.from_numpy(self.labels[i]), i


class SourceGroupedBatchSampler(torch.utils.data.Sampler):
    """Batches of `sources_per_batch` source photos x all their versions (4 in Quality Data).
    shuffle=False gives a fixed order (validation loss comparable across epochs). The last batch
    holds the leftover sources, so every image is used exactly once per epoch."""

    def __init__(self, df, sources_per_batch, shuffle, seed=14):
        groups = df.groupby("source_image_id", sort=True).indices  # source id -> row positions
        self.groups = [np.sort(groups[s]) for s in sorted(groups)]
        self.k, self.shuffle, self.seed, self.epoch = int(sources_per_batch), shuffle, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return -(-len(self.groups) // self.k)

    def __iter__(self):
        order = np.arange(len(self.groups))
        if self.shuffle:
            order = np.random.default_rng([self.seed, self.epoch]).permutation(order)
        for b in range(0, len(order), self.k):
            yield np.concatenate([self.groups[g] for g in order[b:b + self.k]]).tolist()


def make_loader(dataset, batch_sampler, num_workers, pin_memory):
    return torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers,
                                       pin_memory=pin_memory, persistent_workers=False)


def sequential_batches(n, batch_size):
    """Fixed-order batches for prediction only (stress / real check have no source groups)."""
    return [list(range(i, min(i + batch_size, n))) for i in range(0, n, batch_size)]

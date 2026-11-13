"""Grading experiments A / B / C / C-oracle. One run = one condition x one seed.

Network: DenseNet121 (ImageNet) -> global average pool (1024) -> concat quality vector (9) ->
dropout -> Linear(1033, n_classes). A and B feed zeros as the quality vector, so all conditions share one network.
Conditions:
  A         train on clean train images only; quality vector 0
  B         train on clean + 3 degraded copies per train image; quality vector 0
  C         as B; quality vector = the quality module's outputs (quality_d3.csv, computed once)
  C_oracle  as B; quality vector = the true measured defect values (labels.csv; 0 for clean)
Equal compute: A sees each clean image 4 times per epoch (different flips), so A, B and C take the same number
of steps per epoch. Model selection: best validation quadratic weighted kappa on the validation images of the run's own
training distribution (A: clean val; B / C: clean + degraded val), early stop after `patience` epochs.
Test (after training, best checkpoint): every test image (clean + 9 fresh defects x 5 strengths); C is also run with
the quality vectors shuffled across test images (C-shuffled, permutation seed 14). Predictions -> predictions_test.csv;
metrics are computed by grading/evaluate_grading.py.
Round 2 options:
  --label category      grade = the dataset's general pattern (0 point-like, 1 mixed, 2 flaky) from the master CSV
  --select loss         checkpoint / early stop on the lowest validation loss instead of the best validation QWK
  --label_smoothing     cross-entropy label smoothing
  --crop_scale s        training crop: a random window of s..1 of the width and height (same fraction, keeps the
                        aspect), then resized to the input size
  --q_standardise       quality vector standardised per value with the training mean / SD (C / C_oracle)
Round 3 option:
  --head token          quality-token transformer head instead of the joined vector (see GradingNet)
--arch visionfm       VisionFM slit-lamp ViT-B/16 (grading/visionfm.py), fully fine-tuned, instead of DenseNet121:
                        features = CLS + mean patch token (1536), same concat head; VisionFM SlitLamp normalisation;
                        input --height / --width divisible by 16 (384 x 512 used); --grad_ckpt saves GPU memory
--rotate d            random rotation in -d..d degrees with a random shift of up to --translate of the size
                        (black fill), before the crop; no colour / brightness / blur / noise (those are our defects,
                        and colour carries the NC grade)
  --mixup a / --cutmix a  per training batch, with probability --mix_prob, Mixup (Beta(a, a)) or CutMix (Beta(a, a),
                        one box) with a shuffled copy of the batch; if both are set each batch picks one at random;
                        loss = lam CE(y) + (1 - lam) CE(y_shuffled)
  --balance p           class-balanced sampling: each training image drawn with weight 1 / count(class)^p (p = 1
                        fully balanced, 0.5 = square-root), same number of draws per epoch
  every run also writes TTA test predictions (tta_pred, tta_p*): softmax averaged over the 4 flips (none / h / v / hv)
: ordinal target and class weights
  --label nc_raw        Nuclear Cataract: regression on the raw LOCS III NC value (1, 1.5, ... 6, from --master NC column)
                        shifted by -1; one output, MSE loss; classes = cut points on the output, tuned on the validation
                        set for QWK (default cuts k + 0.5); test gets pred (tuned cuts), pred_raw and tta_pred / tta_raw
  --class_weight p      cross-entropy class weights 1 / count^p, normalised to mean 1 (classification only)
Usage: python grading/train_grading.py --data_root "Data/Grading Data/sustech_sysu" --cond C --seed 14 --out runs/grading/sustech_sysu
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "data_prep"))
sys.path.insert(0, HERE)
from run_log import RunLog  # noqa: E402
import visionfm  # noqa: E402

VALUES = ["defocus_blur", "motion_blur", "overexposure", "underexposure", "uneven_illumination",
          "haze", "glare", "occlusion", "noise"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255
CONDS = ["A", "B", "C", "C_oracle"]


class GradingNet(nn.Module):
    """head "concat" (round 1 / 2): pooled features + quality vector -> linear.
    head "token" (round 3): the final feature map as tokens (16 x 21 at 512 x 672) projected to d,
    plus a learned position embedding, a quality token (linear map of the 9 values + a learned embedding) and a class
    token -> pre-norm transformer encoder -> class token -> linear. A / B feed zeros, so their quality token is constant."""

    def __init__(self, n_classes, n_q=9, dropout=0.2, pretrained=True, memory_efficient=True, head="concat",
                 d=256, layers=2, heads=4, grid=(16, 21), arch="densenet121", size=(512, 672), grad_ckpt=False):
        super().__init__()
        self.arch = arch
        if arch == "visionfm":   # input normalisation of the encoder (not saved in the state dict)
            assert head == "concat", "visionfm: concat head only"
            self.register_buffer("mean", visionfm.MEAN * 255, persistent=False)
            self.register_buffer("std", visionfm.STD * 255, persistent=False)
            self.features = (visionfm.load_encoder(size=size, grad_ckpt=grad_ckpt) if pretrained
                             else visionfm.ViTBase(size=size, grad_ckpt=grad_ckpt))
            self.kind = head
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(visionfm.FEAT_DIM + n_q, n_classes))
            return
        self.register_buffer("mean", MEAN.clone(), persistent=False)
        self.register_buffer("std", STD.clone(), persistent=False)
        w = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        # memory_efficient: recompute in backward (standard DenseNet121 at 512 x 672 ran out of memory on 11 GB)
        self.features = torchvision.models.densenet121(weights=w, memory_efficient=memory_efficient).features
        self.kind = head
        if head == "concat":
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(1024 + n_q, n_classes))
            return
        self.proj = nn.Linear(1024, d)
        self.q_proj = nn.Linear(n_q, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.q_emb = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.zeros(1, grid[0] * grid[1], d))
        for p in (self.cls, self.q_emb, self.pos):
            nn.init.trunc_normal_(p, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d, dropout=0.1, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, n_classes))

    def forward(self, x, q):
        if self.arch == "visionfm":
            f = self.features(x)
            return self.head(torch.cat([f, q.to(f.dtype)], 1))
        f = torch.relu(self.features(x))
        if self.kind == "concat":
            f = torch.flatten(nn.functional.adaptive_avg_pool2d(f, 1), 1)
            return self.head(torch.cat([f, q], 1))
        t = self.proj(f.flatten(2).transpose(1, 2))
        assert t.shape[1] == self.pos.shape[1], f"feature grid {f.shape[2:]} does not match the position embedding"
        qt = self.q_proj(q.to(t.dtype)).unsqueeze(1) + self.q_emb
        z = self.encoder(torch.cat([self.cls.expand(len(t), -1, -1), qt, t + self.pos], 1))
        return self.head(self.norm(z[:, 0]))


class GradingDataset(torch.utils.data.Dataset):
    def __init__(self, df, root, q, height, width, train, seed, crop_scale=1.0, rotate=0.0, translate=0.0, flip=None):
        self.paths = [os.path.join(root, p) for p in df.path]
        self.y = df.grade.to_numpy(np.int64)
        self.q = q.astype(np.float32)
        self.size, self.train, self.seed, self.epoch = (width, height), train, seed, 0
        self.crop_scale, self.rotate, self.translate = crop_scale, rotate, translate
        self.flip = flip   # test-time augmentation: fixed (h, v) flips for every image

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        # flips (no rotation) and crop drawn per (seed, epoch, index) so workers do not matter
        rng = np.random.default_rng([self.seed, self.epoch, i]) if self.train else None
        h, v = rng.random(2) < 0.5 if self.train else (self.flip or (False, False))
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB")
            if self.train and self.rotate > 0:   # drawn after the flips / crop so their draws stay unchanged
                r = np.random.default_rng([self.seed, self.epoch, i, 80]).random(3)
                w, hh = im.size
                im = im.rotate((2 * r[0] - 1) * self.rotate, Image.BILINEAR,
                               translate=(int((2 * r[1] - 1) * self.translate * w),
                                          int((2 * r[2] - 1) * self.translate * hh)))
            if self.train and self.crop_scale < 1:
                s, fx, fy = rng.random(3)
                s = self.crop_scale + (1 - self.crop_scale) * s
                w, hh = im.size
                cw, ch = int(round(s * w)), int(round(s * hh))
                x0, y0 = int(fx * (w - cw)), int(fy * (hh - ch))
                im = im.crop((x0, y0, x0 + cw, y0 + ch))
            if im.size != self.size:
                im = im.resize(self.size, Image.LANCZOS)
            x = np.array(im, dtype=np.uint8)
        if self.train or self.flip:
            x = x[:, ::-1] if h else x
            x = x[::-1] if v else x
        return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1), torch.from_numpy(self.q[i]), self.y[i], i


def quality_vectors(df, cond, data_root, q_file):
    if cond in ("A", "B"):
        return np.zeros((len(df), len(VALUES)))
    if cond == "C_oracle":
        return df[VALUES].to_numpy(float)
    q = pd.read_csv(os.path.join(data_root, q_file)).set_index("sample_id")
    return q.loc[df.sample_id, [f"q_{v}" for v in VALUES]].to_numpy(float)


def mix_batch(x, q, y, mix, rng):
    """Mixup or CutMix with a shuffled copy of the batch. mix = dict(mixup=a, cutmix=a, prob=p); returns x, q,
    the shuffled labels and lam (lam = 1: batch unchanged). The quality vectors are mixed with the same lam."""
    kinds = [k for k in ("mixup", "cutmix") if mix[k] > 0]
    if not kinds or rng.random() >= mix["prob"]:
        return x, q, y, 1.0
    kind = kinds[rng.integers(len(kinds))]
    lam = float(rng.beta(mix[kind], mix[kind]))
    perm = torch.from_numpy(rng.permutation(len(y))).to(x.device)
    if kind == "mixup":
        x = lam * x + (1 - lam) * x[perm]
    else:   # one box of area (1 - lam), lam corrected to the clipped box
        H, W = x.shape[2:]
        ch, cw = int(H * np.sqrt(1 - lam)), int(W * np.sqrt(1 - lam))
        cy, cx = rng.integers(H), rng.integers(W)
        y0, y1, x0, x1 = max(cy - ch // 2, 0), min(cy + ch // 2, H), max(cx - cw // 2, 0), min(cx + cw // 2, W)
        x = x.clone()
        x[:, :, y0:y1, x0:x1] = x[perm][:, :, y0:y1, x0:x1]
        lam = 1 - (y1 - y0) * (x1 - x0) / (H * W)
    return x, lam * q + (1 - lam) * q[perm], y[perm], lam


def to_class(raw, cuts):
    """continuous output -> class index by cut points (ascending)."""
    return np.searchsorted(np.asarray(cuts), np.asarray(raw), side="right")


def tune_cuts(y, raw, n_classes, passes=3):
    """coordinate descent on the n - 1 cut points for the best QWK on the validation set (start k + 0.5)."""
    cuts = [k + 0.5 for k in range(n_classes - 1)]
    best = qwk(y, to_class(raw, cuts), n_classes)
    for _ in range(passes):
        for k in range(n_classes - 1):
            lo = cuts[k - 1] + 0.05 if k else -0.5
            hi = cuts[k + 1] - 0.05 if k < n_classes - 2 else n_classes - 0.5
            for c in np.arange(lo, hi, 0.05):
                trial = cuts[:k] + [float(c)] + cuts[k + 1:]
                s = qwk(y, to_class(raw, trial), n_classes)
                if s > best:
                    best, cuts = s, trial
    return cuts, best


def run_epoch(model, loader, device, amp, opt=None, scaler=None, label_smoothing=0.0, mix=None, rng=None,
              weight=None, regression=False):
    """One pass; label smoothing and Mixup / CutMix  only in training (the validation loss stays plain
    cross-entropy). regression: y is the raw target, loss = MSE, "logits" = the single output."""
    train = opt is not None
    model.train(train)
    logits, ys, idx, tot = [], [], [], 0.0
    for x, q, y, i in loader:
        x = (x.to(device, non_blocking=True).float() - model.mean) / model.std
        q, y = q.to(device), y.to(device)
        y2, lam = y, 1.0
        if train and mix:
            x, q, y2, lam = mix_batch(x, q, y, mix, rng)
        with torch.set_grad_enabled(train), torch.autocast(device.type, enabled=amp):
            out = model(x, q)
            if regression:
                o = out.float().squeeze(1)
                loss = nn.functional.mse_loss(o, y.float())
                if lam < 1:
                    loss = lam * loss + (1 - lam) * nn.functional.mse_loss(o, y2.float())
            else:
                ls = label_smoothing if train else 0.0
                w = weight if train else None
                loss = nn.functional.cross_entropy(out.float(), y, weight=w, label_smoothing=ls)
                if lam < 1:
                    loss = lam * loss + (1 - lam) * nn.functional.cross_entropy(out.float(), y2, weight=w,
                                                                                label_smoothing=ls)
        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        tot += loss.item() * len(y)
        logits.append(out.detach().float().cpu())
        ys.append(y.cpu())
        idx.append(i)
    logits, ys, idx = torch.cat(logits), torch.cat(ys).numpy(), torch.cat(idx).numpy()
    return tot / len(ys), logits, ys, idx


def qwk(y, p, n_classes):
    """Quadratic weighted kappa, = sklearn cohen_kappa_score(y, p, weights="quadratic", labels=range(n_classes))
    (sklearn is not in the Ada environment). NaN if undefined (a single true and predicted class)."""
    O = np.zeros((n_classes, n_classes))
    np.add.at(O, (y, p), 1)
    i = np.arange(n_classes)
    W = (i[:, None] - i[None, :]) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    den = (W * E).sum()
    return float(1 - (W * O).sum() / den) if den > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--cond", required=True, choices=CONDS)
    ap.add_argument("--seed", type=int, default=14)
    ap.add_argument("--out", required=True, help="parent folder; the run goes to <out>/<cond>_seed<seed>")
    ap.add_argument("--q_file", default="quality_d3.csv")
    ap.add_argument("--labels", default="labels.csv",
                    help="labels file in --data_root, or a path (E3 image-level split)")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=672)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr_backbone", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--max_epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: N images per split")
    ap.add_argument("--label", choices=["grade", "category", "nc_raw"], default="grade",
                    help="grade = labels.csv grade (round 1: type); category = general pattern from --master; "
                         "nc_raw = regression on the raw NC value from --master ")
    ap.add_argument("--class_weight", type=float, default=0.0, help="CE class weights 1 / count^p (0 = off)")
    ap.add_argument("--master", default=os.path.join(ROOT, "labels", "sustech_master.csv"))
    ap.add_argument("--select", choices=["qwk", "loss"], default="qwk", help="checkpoint / early stop criterion")
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--crop_scale", type=float, default=1.0)
    ap.add_argument("--q_standardise", action="store_true")
    ap.add_argument("--head", choices=["concat", "token"], default="concat", help="token = quality-token transformer")
    ap.add_argument("--arch", choices=["densenet121", "visionfm"], default="densenet121")
    ap.add_argument("--grad_ckpt", action="store_true", help="visionfm: recompute blocks in backward (memory)")
    ap.add_argument("--layer_decay", type=float, default=1.0, help="visionfm: layer-wise lr decay (1 = off)")
    ap.add_argument("--warmup_epochs", type=int, default=0, help="linear lr warmup epochs before the cosine")
    ap.add_argument("--rotate", type=float, default=0.0, help="max rotation in degrees (0 = off)")
    ap.add_argument("--translate", type=float, default=0.0, help="max shift as a fraction of the size")
    ap.add_argument("--mixup", type=float, default=0.0, help="Mixup Beta alpha (0 = off)")
    ap.add_argument("--cutmix", type=float, default=0.0, help="CutMix Beta alpha (0 = off)")
    ap.add_argument("--mix_prob", type=float, default=0.5, help="share of batches mixed")
    ap.add_argument("--balance", type=float, default=0.0, help="sampling weight 1 / count^p (0 = off)")
    a = ap.parse_args()
    run_dir = os.path.join(a.out, f"{a.cond}_seed{a.seed}")
    os.makedirs(run_dir, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    with RunLog("grading/train_grading.py", notes=f"{a.cond} seed {a.seed} {a.data_root}") as log:
        lab = pd.read_csv(os.path.join(a.data_root, a.labels))   # an absolute --labels path is used as it is
        if a.label == "category":   # same images, the dataset's general-pattern label
            cat = pd.read_csv(a.master).set_index("image_id").category
            assert lab.image_id.isin(cat.index).all()
            lab["grade"] = lab.image_id.map(cat).astype(int)
        reg = a.label == "nc_raw"
        if reg:   # raw NC per image (half grades kept), on the 0-based class scale; grade stays the rounded class
            nc = pd.read_csv(a.master).set_index("image_id").NC
            assert lab.image_id.isin(nc.index).all()
            lab["y_raw"] = lab.image_id.map(nc).astype(float) - 1
            assert (np.ceil(lab.y_raw).astype(int) == lab.grade).all(), "raw NC does not round up to grade"
        n_classes = int(lab.grade.max()) + 1
        degraded = a.cond != "A"
        tr = lab[(lab.split == "train") & (degraded | (lab.kind == "clean"))].reset_index(drop=True)
        va = lab[(lab.split == "val") & (degraded | (lab.kind == "clean"))].reset_index(drop=True)
        te = lab[lab.split == "test"].reset_index(drop=True)
        if a.limit:
            tr, va, te = (d.groupby("kind").head(a.limit).reset_index(drop=True) for d in (tr, va, te))
        if a.cond == "A":   # equal steps per epoch: each clean image 4x (= clean + 3 degraded in B / C)
            tr = pd.concat([tr] * 4, ignore_index=True)
        q_tr = quality_vectors(tr, a.cond, a.data_root, a.q_file)
        q_mu, q_sd = np.zeros(q_tr.shape[1]), np.ones(q_tr.shape[1])
        if a.q_standardise and a.cond in ("C", "C_oracle"):   # A / B keep their zeros
            q_mu, q_sd = q_tr.mean(0), q_tr.std(0) + 1e-6
        qvec = lambda d: (quality_vectors(d, a.cond, a.data_root, a.q_file) - q_mu) / q_sd
        ds_tr = GradingDataset(tr, a.data_root, (q_tr - q_mu) / q_sd, a.height, a.width, True, a.seed, a.crop_scale,
                               a.rotate, a.translate)
        ds_va = GradingDataset(va, a.data_root, qvec(va), a.height, a.width, False, a.seed)
        if reg:   # the datasets return the raw target; the rounded class is kept for QWK / cut tuning
            ds_tr.y, ds_va.y = tr.y_raw.to_numpy(np.float32), va.y_raw.to_numpy(np.float32)
        weight = None
        if a.class_weight > 0 and not reg:   #
            cnt = tr.grade.value_counts().reindex(range(n_classes)).fillna(1).to_numpy(float)
            w = 1.0 / cnt ** a.class_weight
            weight = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
        g = torch.Generator().manual_seed(a.seed)
        sampler = None
        if a.balance > 0:   # class-balanced draws with replacement, as many per epoch as training images
            cnt = tr.grade.value_counts()
            wts = 1.0 / tr.grade.map(cnt).to_numpy(float) ** a.balance
            sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(wts), len(tr), generator=g)
        dl = lambda ds, shuffle: torch.utils.data.DataLoader(
            ds, batch_size=a.batch, shuffle=shuffle and sampler is None, sampler=sampler if shuffle else None,
            generator=g, num_workers=a.num_workers, pin_memory=amp)
        mix = dict(mixup=a.mixup, cutmix=a.cutmix, prob=a.mix_prob) if a.mixup > 0 or a.cutmix > 0 else None
        mix_rng = np.random.default_rng([a.seed, 80])
        model = GradingNet(1 if reg else n_classes, head=a.head, grid=(a.height // 32, a.width // 32), arch=a.arch,
                           size=(a.height, a.width), grad_ckpt=a.grad_ckpt).to(device)
        rest = [p for n, p in model.named_parameters() if not n.startswith("features.")]   # head (+ token parts)
        if a.layer_decay < 1:   # visionfm: lr x decay^(13 - layer); embeddings = layer 0, blocks 1-12, norm 13
            assert a.arch == "visionfm"
            groups = {}
            for n, p in model.features.named_parameters():
                l = 13 if n.startswith("norm.") else int(n.split(".")[1]) + 1 if n.startswith("blocks.") else 0
                groups.setdefault(l, []).append(p)
            bb = [{"params": ps, "lr": a.lr_backbone * a.layer_decay ** (13 - l)} for l, ps in sorted(groups.items())]
        else:
            bb = [{"params": model.features.parameters(), "lr": a.lr_backbone}]
        opt = torch.optim.AdamW(bb + [{"params": rest, "lr": a.lr_head}], weight_decay=a.weight_decay)
        if a.warmup_epochs:   # linear warmup per epoch, then cosine
            w, T = a.warmup_epochs, a.max_epochs
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: (e + 1) / (w + 1) if e < w
                                                      else 0.5 * (1 + np.cos(np.pi * (e - w) / (T - w))))
        else:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.max_epochs)
        scaler = torch.amp.GradScaler(enabled=amp)
        best, bad, hist = -np.inf, 0, []
        for ep in range(1, a.max_epochs + 1):
            t0 = time.time()
            ds_tr.epoch = ep
            tl, _, _, _ = run_epoch(model, dl(ds_tr, True), device, amp, opt, scaler, a.label_smoothing, mix, mix_rng,
                                    weight, reg)
            vl, lo, y, _ = run_epoch(model, dl(ds_va, False), device, amp, regression=reg)
            sched.step()
            if reg:   # default cuts here; the tuned cuts come from the best checkpoint below
                y = va.grade.to_numpy()
                p = to_class(lo.squeeze(1).numpy(), [k + 0.5 for k in range(n_classes - 1)])
            else:
                p = lo.argmax(1).numpy()
            k, acc = qwk(y, p, n_classes), float((p == y).mean())
            # qwk: an undefined kappa counts as -1 (first epoch still saved); loss: lower is better
            score = -vl if a.select == "loss" else (k if np.isfinite(k) else -1.0)
            star = score > best
            if star:
                best, bad = score, 0
                torch.save({"model": model.state_dict(), "epoch": ep, "val_qwk": k, "val_loss": vl},
                           os.path.join(run_dir, "best.pt"))
            else:
                bad += 1
            hist.append(dict(epoch=ep, train_loss=tl, val_loss=vl, val_qwk=k, val_acc=acc, sec=time.time() - t0))
            mem = f"  peak GPU {torch.cuda.max_memory_allocated / 2**30:.1f} GiB" if amp else ""
            print(f"epoch {ep}/{a.max_epochs}  train {tl:.4f}  val {vl:.4f}  QWK {k:.3f}  acc {acc:.3f}"
                  f"{'  *' if star else ''}  {time.time - t0:.0f} s{mem}", flush=True)
            if bad >= a.patience:
                break
        pd.DataFrame(hist).to_csv(os.path.join(run_dir, "epochs.csv"), index=False)
        # test with the best checkpoint
        ck = torch.load(os.path.join(run_dir, "best.pt"), map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        out = te[["sample_id", "image_id", "group_id", "kind", "defect", "strength", "grade"]].copy()
        cuts = None
        if reg:   # cut points tuned on the validation set with the best checkpoint
            _, lo, _, _ = run_epoch(model, dl(ds_va, False), device, amp, regression=True)
            cuts, val_k = tune_cuts(va.grade.to_numpy(), lo.squeeze(1).numpy(), n_classes)
            print(f"tuned cuts {np.round(cuts, 2).tolist} val QWK {val_k:.3f}")
        variants = [("", qvec(te))]
        if a.cond in ("C", "C_oracle"):   # C-shuffled: the same test images with another image's quality vector
            qv = variants[0][1]
            variants.append(("shuffled_", qv[np.random.default_rng(14).permutation(len(qv))]))
        for prefix, qv in variants:
            ds = GradingDataset(te, a.data_root, qv, a.height, a.width, False, a.seed)
            _, lo, _, _ = run_epoch(model, dl(ds, False), device, amp, regression=reg)
            if reg:
                raw = lo.squeeze(1).numpy()
                out[f"{prefix}pred"], out[f"{prefix}pred_raw"] = to_class(raw, cuts), raw
                continue
            pr = torch.softmax(lo, 1).numpy()
            out[f"{prefix}pred"] = pr.argmax(1)
            for c in range(n_classes):
                out[f"{prefix}p{c}"] = pr[:, c]
        # test-time augmentation, softmax (or the raw output) averaged over the 4 flips
        pr = 0
        for fl in [(False, False), (True, False), (False, True), (True, True)]:
            ds = GradingDataset(te, a.data_root, variants[0][1], a.height, a.width, False, a.seed, flip=fl)
            _, lo, _, _ = run_epoch(model, dl(ds, False), device, amp, regression=reg)
            pr = pr + (lo.squeeze(1).numpy() if reg else torch.softmax(lo, 1).numpy()) / 4
        if reg:
            out["tta_pred"], out["tta_raw"] = to_class(pr, cuts), pr
        else:
            out["tta_pred"] = pr.argmax(1)
            for c in range(n_classes):
                out[f"tta_p{c}"] = pr[:, c]
        out.to_csv(os.path.join(run_dir, "predictions_test.csv"), index=False, float_format="%.6g")
        info = dict(vars(a), cuts=cuts, best_epoch=int(ck["epoch"]), best_val_qwk=float(ck["val_qwk"]),
                    best_val_loss=float(ck.get("val_loss", np.nan)), n_classes=n_classes,
                    q_mean=[float(v) for v in q_mu], q_sd=[float(v) for v in q_sd], n_train=len(tr),
                    n_val=len(va), n_test=len(te), device=str(device),
                    gpu=torch.cuda.get_device_name(0) if amp else None, torch=torch.__version__)
        json.dump(info, open(os.path.join(run_dir, "run_info.json"), "w"), indent=1)
        print(f"best epoch {ck['epoch']} val QWK {ck['val_qwk']:.3f}; test predictions: {len(out)} rows")
        log.add(**{k: v for k, v in info.items() if k != "data_root"})


if __name__ == "__main__":
    main()

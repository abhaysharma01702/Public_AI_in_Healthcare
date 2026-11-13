"""Training / evaluation passes, EMA and checkpoints (section 2)."""
import os
import time

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255


class Batch:
    """Moves a uint8 batch to the device and applies ImageNet normalisation there."""

    def __init__(self, device, channels_last):
        self.device, self.channels_last = device, channels_last
        self.mean, self.std = MEAN.to(device), STD.to(device)

    def __call__(self, x, y):
        x = x.to(self.device, non_blocking=True).float().sub_(self.mean).div_(self.std)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        return x, y.to(self.device, non_blocking=True)


def make_ema(model, decay):
    # use_buffers=True: BatchNorm running statistics are averaged too (EMA includes BN statistics)
    return AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay), use_buffers=True)


def set_warmup_lr(opt, step, warmup_steps):
    if step < warmup_steps:
        f = (step + 1) / warmup_steps
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * f


def autocast(device, amp):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp)


def train_one_epoch(model, ema, loader, loss_fn, opt, scaler, to_batch, device, amp, grad_clip,
                    warmup_steps, global_step, log_every=50, logger=print, max_steps=None):
    model.train()
    sums, n_img, t0, bad = {"loss": 0.0, "mse": 0.0, "rank": 0.0}, 0, time.perf_counter(), 0
    for it, (x, y, _) in enumerate(loader):
        set_warmup_lr(opt, global_step, warmup_steps)
        x, y = to_batch(x, y)
        with autocast(device, amp):
            scores = model(x)
        loss, parts = loss_fn(scores, y)
        if not torch.isfinite(loss):  # skip a rare bad batch instead of losing a long run; stop if it persists
            bad += 1
            logger(f"  warning: non-finite loss at step {global_step} ({parts}); batch skipped")
            if bad > 20:
                raise FloatingPointError("more than 20 consecutive non-finite losses")
            continue
        bad = 0
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt)
        scaler.update()
        ema.update_parameters(model)
        global_step += 1
        b = len(y)
        sums["loss"] += loss.item() * b
        sums["mse"] += parts["mse"].item() * b
        sums["rank"] += parts["rank"].item() * b
        n_img += b
        if log_every and (it + 1) % log_every == 0:
            logger(f"  step {it + 1}/{len(loader)}  loss {sums['loss'] / n_img:.4f}  "
                   f"{n_img / (time.perf_counter - t0):.1f} img/s")
        if max_steps and it + 1 >= max_steps:  # benchmark mode
            break
    secs = time.perf_counter() - t0
    out = {f"train_{k}": v / n_img for k, v in sums.items()}
    out.update(train_seconds=secs, train_images=n_img, train_images_per_s=n_img / secs)
    return out, global_step


@torch.no_grad()
def predict(model, loader, to_batch, device, amp, loss_fn=None):
    """Predictions (sigmoid outputs) in dataset order; mean loss over batches if loss_fn is given."""
    model.eval()
    preds, idxs, sums, nb = [], [], {"loss": 0.0, "mse": 0.0, "rank": 0.0}, 0
    for x, y, i in loader:
        x, y = to_batch(x, y)
        with autocast(device, amp):
            scores = model(x)
        scores = scores.float()
        if loss_fn is not None:
            loss, parts = loss_fn(scores, y)
            sums["loss"] += loss.item()
            sums["mse"] += parts["mse"].item()
            sums["rank"] += parts["rank"].item()
            nb += 1
        preds.append(torch.sigmoid(scores).cpu().numpy())
        idxs.append(i.numpy())
    order = np.argsort(np.concatenate(idxs))
    p = np.concatenate(preds)[order]
    losses = {k: v / nb for k, v in sums.items()} if nb else {}
    return p, losses


def save_atomic(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def rng_state():
    s = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def set_rng_state(s):
    # .cpu: torch.load(map_location="cuda") moves these ByteTensors to the GPU, but RNG states must be on the CPU
    torch.set_rng_state(s["torch"].cpu())
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in s["cuda"]])

"""Train the quality module.

Usage:
  python quality/train.py --config quality/configs/densenet121.yaml \
      --data_root "Data/Quality Data" --run_dir runs/quality/densenet121_seed14 [--set key=value ...]
  Restarting with the same --run_dir continues from last.pt (Kaggle sessions, SLURM requeue).
  --time_budget_h 11.5   stop cleanly after the epoch that would cross the budget (Kaggle ~12 h sessions)
  --benchmark_steps 50   only measure speed / peak memory of the batch size, then exit (no checkpoint)
On SLURM, `#SBATCH --signal=USR1@900` makes the run finish the current epoch, save and exit.
Outputs in run_dir: config.yaml, label_stats.json, last.pt, best.pt (EMA weights), epochs.jsonl,
train.log, status.json; then evaluate.py's predictions / metrics (unless --no_eval).
"""
import argparse
import json
import math
import os
import signal
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "data_prep"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

import run_log  # noqa: E402
from qm import VALUES  # noqa: E402
from qm import data as QD  # noqa: E402
from qm import engine as E  # noqa: E402
from qm import losses as QL  # noqa: E402
from qm import metrics as QX  # noqa: E402
from qm import model as QM  # noqa: E402

# settings that may change when a run is resumed (everything else must match the saved config)
RESUMABLE_CHANGES = {"train.max_epochs", "data.num_workers", "train.log_every", "eval.bootstrap", "eval.sets",
                     "eval.limit_sources"}


def load_config(path, overrides):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for o in overrides or []:
        key, val = o.split("=", 1)
        d = cfg
        *parents, last = key.split(".")
        for p in parents:
            d = d[p]
        if last not in d:
            raise KeyError(f"unknown setting {key}")
        d[last] = yaml.safe_load(val)
    return cfg


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


class Logger:
    def __init__(self, path):
        self.path = path

    def __call__(self, msg):
        print(msg, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. batch.sources_per_batch=6")
    ap.add_argument("--time_budget_h", type=float, default=None)
    ap.add_argument("--benchmark_steps", type=int, default=0)
    ap.add_argument("--no_eval", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    t_start = time.time()
    sys.stdout.reconfigure(encoding="utf-8")

    cfg = load_config(args.config, args.set)
    root, run_dir = os.path.abspath(args.data_root), os.path.abspath(args.run_dir)
    os.makedirs(run_dir, exist_ok=True)
    log = Logger(os.path.join(run_dir, "train.log"))
    last_path, best_path = os.path.join(run_dir, "last.pt"), os.path.join(run_dir, "best.pt")
    resume = os.path.exists(last_path) and not args.benchmark_steps

    cfg_path = os.path.join(run_dir, "config.yaml")
    if resume:
        with open(cfg_path, encoding="utf-8") as f:
            saved = flat(yaml.safe_load(f))
        diff = {k for k, v in flat(cfg).items() if saved.get(k) != v} - RESUMABLE_CHANGES
        if diff:
            raise SystemExit(f"config differs from the saved run in {sorted(diff)}; use a new --run_dir")
    if not args.benchmark_steps:
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = bool(cfg["optim"]["amp"]) and device.type == "cuda"
    torch.backends.cudnn.benchmark = bool(cfg["train"]["cudnn_benchmark"])
    dc, H, W = cfg["data"], cfg["data"]["height"], cfg["data"]["width"]

    # ---- data
    train_full = QD.load_split(root, "train", dc.get("extension"))
    stats = QL.label_stats(train_full, VALUES, cfg["loss"]["rank_min_frac"])  # always from the full training split
    with open(os.path.join(run_dir, "label_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    train_df = QD.limit_sources(train_full, dc["limit_sources_train"], seed)
    if args.benchmark_steps:  # worst case for memory: only sources with the most versions (5 with the extension)
        nv = train_df.groupby("source_image_id").sample_id.transform("size")
        train_df = train_df[nv == nv.max()].reset_index(drop=True)
    val_df = QD.limit_sources(QD.load_split(root, "val", dc.get("extension")), dc["limit_sources_val"], seed)
    train_ds = QD.QualityDataset(train_df, root, H, W, train=True, seed=seed)
    val_ds = QD.QualityDataset(val_df, root, H, W, train=False)
    k = cfg["batch"]["sources_per_batch"]
    train_sampler = QD.SourceGroupedBatchSampler(train_df, k, shuffle=True, seed=seed)
    val_sampler = QD.SourceGroupedBatchSampler(val_df, k, shuffle=False)
    pin = device.type == "cuda"
    train_loader = QD.make_loader(train_ds, train_sampler, dc["num_workers"], pin)
    val_loader = QD.make_loader(val_ds, val_sampler, dc["num_workers"], pin)
    val_y = val_df[VALUES].to_numpy(float)

    # ---- model, loss, optimiser
    mc, oc, tc = cfg["model"], cfg["optim"], cfg["train"]
    mean = [min(max(m, 1e-4), 1 - 1e-4) for m in stats["mean"]]
    model = QM.build_model(mc["name"], len(VALUES), mc["pretrained"], mc["dropout"], mc["memory_efficient"],
                           label_mean=mean).to(device)
    if tc["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    counts = QM.parameter_counts(model)
    loss_fn = QL.QualityLoss(stats["var"], stats["min_diff"], cfg["loss"]["rank_weight"]).to(device)
    if (mc["name"] == "mcf_densenet121") != bool(cfg["loss"].get("output_weights")):
        raise SystemExit("loss.output_weights must be set for mcf_densenet121 and only for it (C1)")
    if cfg["loss"].get("output_weights"):  # MCF-style: weighted loss over its 5 outputs (C1)
        loss_fn = QL.MultiOutputLoss(loss_fn, cfg["loss"]["output_weights"]).to(device)
    opt = torch.optim.AdamW(QM.param_groups(model, oc["lr_backbone"], oc["lr_head"], oc["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=oc["plateau_factor"],
                                                         patience=oc["plateau_patience"])
    ema = E.make_ema(model, oc["ema_decay"])
    to_batch = E.Batch(device, tc["channels_last"])
    warmup_steps = oc["warmup_epochs"] * len(train_sampler)
    st = {"epoch": 0, "global_step": 0, "best_val": math.inf, "best_epoch": None, "bad_epochs": 0}

    if args.benchmark_steps:
        n = args.benchmark_steps
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        out, _ = E.train_one_epoch(model, ema, train_loader, loss_fn, opt, scaler, to_batch, device, amp,
                                   oc["grad_clip"], warmup_steps, 0, log_every=10, logger=log, max_steps=n)
        res = {"sources_per_batch": k, "batch_images": int(len(train_df) / train_df.source_image_id.nunique() * k), "input": [H, W], "steps": n, "amp": amp,
               "images_per_s": round(out["train_images_per_s"], 2), "device": str(device),
               "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
               "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else None,
               "est_train_epoch_min": round(len(train_df) / out["train_images_per_s"] / 60, 1), **counts}
        log(json.dumps(res))
        with open(os.path.join(run_dir, f"benchmark_{k}x4.json"), "w") as f:
            json.dump(res, f, indent=1)
        return

    if resume:
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        plateau.load_state_dict(ck["plateau"])
        st = ck["state"]
        E.set_rng_state(ck["rng"])
        log(f"resumed from {last_path}: {st['epoch']} epochs done, best val loss {st['best_val']:.4f}")

    stop = {"signal": False}
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, lambda *_: stop.update(signal=True))

    log(f"device {device}, amp {amp}, train {len(train_df)} images / {len(train_sampler)} batches, "
        f"val {len(val_df)}, input {H}x{W}, batch {k}x4, params {counts['params_total']:,}")
    run_log.LOG = _writable_log(run_dir)
    stop_reason, epoch_secs = None, []
    with run_log.RunLog("quality/train.py", notes=f"quality module {mc['name']}, run {os.path.basename(run_dir)}") as rl:
        while st["epoch"] < tc["max_epochs"] and st["bad_epochs"] < tc["early_stop_patience"]:
            ep = st["epoch"]
            t0 = time.time()
            train_sampler.set_epoch(ep)
            train_ds.set_epoch(ep)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            tr, st["global_step"] = E.train_one_epoch(
                model, ema, train_loader, loss_fn, opt, scaler, to_batch, device, amp, oc["grad_clip"],
                warmup_steps, st["global_step"], log_every=tc["log_every"], logger=log)
            p, vl = E.predict(ema.module, val_loader, to_batch, device, amp, loss_fn)
            vm = QX.summary(val_y, p, VALUES)
            if ep + 1 >= oc["warmup_epochs"]:
                plateau.step(vl["loss"])
            improved = vl["loss"] < st["best_val"]
            if improved:
                st.update(best_val=vl["loss"], best_epoch=ep + 1, bad_epochs=0)
                E.save_atomic({"model": ema.module.state_dict(), "epoch": ep + 1, "val_loss": vl["loss"],
                               "config": cfg, "label_stats": stats}, best_path)
            else:
                st["bad_epochs"] += 1
            st["epoch"] = ep + 1
            E.save_atomic({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                           "scaler": scaler.state_dict(), "plateau": plateau.state_dict(), "state": st,
                           "rng": E.rng_state(), "config": cfg}, last_path)
            secs = time.time() - t0
            epoch_secs.append(secs)
            rec = {"epoch": ep + 1, **tr, **{f"val_{k2}": v for k2, v in vl.items()}, **{f"val_{k2}": v for k2, v in vm.items()},
                   "epoch_seconds": secs, "improved": improved, "bad_epochs": st["bad_epochs"],
                   "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None}
            lr_names = {g["name"]: g["lr"] for g in opt.param_groups}
            rec["lr_backbone"], rec["lr_head"] = lr_names.get("backbone"), lr_names.get("head")
            with open(os.path.join(run_dir, "epochs.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            log(f"epoch {ep + 1}/{tc['max_epochs']}  train {tr['train_loss']:.4f} (mse {tr['train_mse']:.4f}, "
                f"rank {tr['train_rank']:.4f})  val {vl['loss']:.4f}  AUROC {vm['mean_auroc']:.3f}  "
                f"Spearman {vm['mean_spearman_present']:.3f}  MAE {vm['mean_mae_present']:.4f}  "
                f"{'*' if improved else ''}  {secs / 60:.1f} min")
            if stop["signal"]:
                stop_reason = "signal"
                break
            if args.time_budget_h and (time.time() - t_start + np.mean(epoch_secs) * 1.1) > args.time_budget_h * 3600:
                stop_reason = "time_budget"
                break
        if stop_reason is None:
            stop_reason = "early_stop" if st["bad_epochs"] >= tc["early_stop_patience"] else "max_epochs"
        finished = stop_reason in ("early_stop", "max_epochs")
        status = {"finished": finished, "stop_reason": stop_reason, **st}
        with open(os.path.join(run_dir, "status.json"), "w") as f:
            json.dump(status, f, indent=1)
        log(f"stopped: {stop_reason}; best epoch {st['best_epoch']}, val loss {st['best_val']:.4f}"
            + ("" if finished else "  -> run the same command again to continue"))
        rl.add(run_dir=run_dir, model=mc["name"], input=[H, W], sources_per_batch=k, batch_images_mean=round(len(train_df) / train_df.source_image_id.nunique() * k, 1), train_images=len(train_df),
               val_images=len(val_df), epochs_this_session=len(epoch_secs), epochs_total=st["epoch"],
               mean_epoch_seconds=round(float(np.mean(epoch_secs)), 1) if epoch_secs else None,
               best_epoch=st["best_epoch"], best_val_loss=st["best_val"], stop_reason=stop_reason, device=str(device),
               amp=amp, **counts)

    if finished and not args.no_eval:
        import evaluate
        evaluate.run(run_dir, root, cfg["eval"]["sets"], cfg["eval"]["bootstrap"], device=args.device,
                     num_workers=dc["num_workers"], limit_sources=cfg["eval"].get("limit_sources"), logger=log)


def _writable_log(run_dir):
    """Project logs/run_log.jsonl if writable (laptop, HPC); otherwise the run folder (Kaggle code is read-only)."""
    try:
        os.makedirs(os.path.dirname(run_log.LOG), exist_ok=True)
        with open(run_log.LOG, "a"):
            pass
        return run_log.LOG
    except OSError:
        return os.path.join(run_dir, "run_log.jsonl")


if __name__ == "__main__":
    main()

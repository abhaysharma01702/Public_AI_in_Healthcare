"""Kaggle runner for the quality module. Paste into one Kaggle notebook cell.

Notebook setup: Accelerator = GPU (T4 x2 or P100; the code uses one GPU); Add Input = the two private
datasets slitlamp-quality-data and slitlamp-qm-code. Internet is not needed (ImageNet weights are bundled).
MODE:
  "check"  (about 5-10 min, interactive) on Kaggle's own GPU:
           1. every image listed in the CSVs exists;
           2. tiny training run at full size (16 / 8 photos): 1 epoch, stop, resume to epoch 2, final evaluation
              of all four sets (8 photos each, 50 bootstrap samples), all output files present;
           3. speed test: 50 steps at 8 x 4 (falls back to 6 x 4 if 8 x 4 runs out of memory).
           Ends with "ALL CHECKS PASSED" and the batch size to use.
  "train"  full run; use "Save Version -> Save & Run All" so it runs in the background (<= 12 h).
           If it stops on the time budget, attach this version's output as input in the next version and set
           PREVIOUS_RUN to it: training continues from last.pt.
"""
import glob
import json
import os
import shutil
import subprocess
import sys

MODE = os.environ.get("QM_MODE", "check")   # "check" or "train"
EXTRA_SETTINGS = []                         # e.g. ["batch.sources_per_batch=6"] if the check says so
PREVIOUS_RUN = None                         # e.g. "/kaggle/input/<previous-version-output>/densenet121_seed14"
TIME_BUDGET_H = 11.3                        # leaves time to save the output before the 12 h limit

INPUT = os.environ.get("KAGGLE_INPUT", "/kaggle/input")      # overridable only to test this script locally
WORK = os.environ.get("KAGGLE_WORKING", "/kaggle/working")
RUN = os.path.join(WORK, "densenet121_v2_seed14")       # trained on Quality Data v2


def run(cmd, check=True):
    """Run a command and stream its output into the notebook (plain subprocess output would not show)."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait() and check:
        raise SystemExit(f"command failed with exit code {p.returncode}")
    return p.returncode


def find_one(pattern, what):
    hits = sorted(glob.glob(os.path.join(INPUT, "**", pattern), recursive=True))
    if not hits:
        raise SystemExit(f"{what} not found under {INPUT}; is the dataset attached?")
    return os.path.dirname(hits[0])


try:                                                                        # prefer Quality Data v2
    DATA = find_one("gen_holdout.csv", "Quality Data v2")
except SystemExit:
    DATA = find_one("stress_4defects.csv", "Quality Data")                  # folder with labels.csv
CODE = os.path.dirname(find_one(os.path.join("quality", "train.py"), "code"))  # folder with quality/, data_prep/
os.environ["TORCH_HOME"] = os.path.join(CODE, "torch_home")                  # bundled DenseNet121 weights
print("data:", DATA, "\ncode:", CODE, flush=True)
run([sys.executable, "-c", "import torch, torchvision; print('torch', torch.__version__, 'torchvision', "
     "torchvision.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(0) "
     "if torch.cuda.is_available else 'NONE')"])

import pandas as pd  # noqa: E402  every image listed in the three CSVs must exist
need = pd.concat([pd.read_csv(os.path.join(DATA, f)).path for f in ("labels.csv", "stress_4defects.csv", "real_check.csv")])
missing = [p for p in need if not os.path.exists(os.path.join(DATA, p))]
print(f"images listed: {len(need)}, missing: {len(missing)}", flush=True)
if missing:
    raise SystemExit(f"missing images, e.g. {missing[:3]}")

BASE = [sys.executable, os.path.join(CODE, "quality", "train.py"),
        "--config", os.path.join(CODE, "quality", "configs", "densenet121.yaml"), "--data_root", DATA]


def train(run_dir, settings, *extra):
    sets = EXTRA_SETTINGS + list(settings)
    return BASE + ["--run_dir", run_dir] + list(extra) + (["--set", *sets] if sets else [])


if MODE == "check":
    hw = os.environ.get("QM_CHECK_HW", "512x672").split("x")  # local test of this script uses a smaller size
    tiny = [f"data.height={hw[0]}", f"data.width={hw[1]}", "data.limit_sources_train=16",
            "data.limit_sources_val=8", "batch.sources_per_batch=4", "optim.ema_decay=0.5", "train.log_every=1",
            "eval.bootstrap=50", "eval.limit_sources=8"]
    chk = os.path.join(WORK, "check_run")
    shutil.rmtree(chk, ignore_errors=True)
    print("\n=== tiny run: epoch 1, then stop", flush=True)
    run(train(chk, tiny + ["train.max_epochs=1"], "--no_eval"))
    print("\n=== resume to epoch 2 + final evaluation", flush=True)
    run(train(chk, tiny + ["train.max_epochs=2"]))
    with open(os.path.join(chk, "epochs.jsonl")) as f:
        n_epochs = len(f.readlines())
    files = ["last.pt", "best.pt", "status.json", "metrics_summary.json", "metrics_real_check.csv"] +\
            [f"{k}_{s}.csv" for s in ("val", "test", "stress") for k in ("predictions", "metrics")]
    absent = [f for f in files if not os.path.exists(os.path.join(chk, f))]
    if n_epochs != 2 or absent:
        raise SystemExit(f"CHECK FAILED: epochs logged {n_epochs} (want 2), missing outputs {absent}")
    print(f"tiny run OK: 2 epochs (with resume), all {len(files)} output files present", flush=True)

    steps = os.environ.get("QM_BENCH_STEPS", "50")
    use = None
    for k in (8, 6):
        print(f"\n=== speed test {k} x 4, {steps} steps", flush=True)
        bench = os.path.join(WORK, "benchmark")
        if run(train(bench, [f"batch.sources_per_batch={k}"], "--benchmark_steps", steps), check=False) == 0:
            with open(os.path.join(bench, f"benchmark_{k}x4.json")) as f:
                b = json.load(f)
            use = k
            break
        print(f"{k} x 4 failed (probably out of GPU memory)", flush=True)
    if use is None:
        raise SystemExit("CHECK FAILED: neither 8 x 4 nor 6 x 4 runs")
    print(f"\nALL CHECKS PASSED. Use batch {use} x 4"
          + ("" if use == 8 else ' (set EXTRA_SETTINGS = ["batch.sources_per_batch=6"])')
          + f": {b['images_per_s']} images/s, peak GPU memory {b['peak_gpu_mem_gb']} GB, "
          f"about {b['est_train_epoch_min']} min per training epoch", flush=True)
else:
    if PREVIOUS_RUN and not os.path.exists(os.path.join(RUN, "last.pt")):
        shutil.copytree(PREVIOUS_RUN, RUN, dirs_exist_ok=True)
    final_sets = "eval.sets=[val,test,stress,real_check,gen_train,gen_holdout]" if os.path.exists(
        os.path.join(DATA, "gen_holdout.csv")) else "eval.sets=[val,test,stress,real_check]"
    run(train(RUN, [final_sets], "--time_budget_h", str(TIME_BUDGET_H)))

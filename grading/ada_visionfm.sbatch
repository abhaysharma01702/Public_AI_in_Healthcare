#!/bin/bash
# Reti-Pioneer-style grader  on Ada, one job:
#   1. VisionFM slit-lamp features for every image of the 4 grading zips (one zip at a time on the node SSD, deleted
#      after use) -> $HOME/project/features/<folder>/{features_visionfm.npz, labels.csv, quality_d3.csv}
#   2. bilinear-fusion heads, A / B / C / C_oracle x seeds 14 15 16, both tasks
#      -> $HOME/project/runs/grading/{sustech_sysu_category,nuclear_cataract_grade}_visionfm
# A finished feature folder or run is skipped, so a resubmission continues where it stopped.
# Usage: sbatch --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ada_visionfm.sbatch
#SBATCH --job-name=visionfm
#SBATCH --account=<your_account>
#SBATCH --qos=<your_qos>
#SBATCH --partition=u22
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode004,gnode066,gnode061,gnode044,gnode068,gnode054,gnode079
#SBATCH --mem=28G
#SBATCH --time=06:00:00
#SBATCH --output=$HOME/project/logs/slurm_%x_%j.out
set -euo pipefail
P="$HOME/project"; S="/ssd_scratch/$USER/$SLURM_JOB_ID"; mkdir -p "$S"; trap 'rm -rf "$S"' EXIT
FEAT="$P/features"
echo "node $(hostname) job $SLURM_JOB_ID start $(date -Is)"
source "$HOME/envs/qm/bin/activate"; export TORCH_HOME="$P/torch_home" PYTHONUNBUFFERED=1
# stop at once if the GPU is unusable or the check hangs (as ada_grading.sbatch)
echo "gpu check $(date -Is)"; timeout 60 nvidia-smi -L || echo "nvidia-smi failed or hung"
timeout 120 python -c "import time; t = time.time(); import torch, sys; print('torch import', round(time.time() - t, 1), 's', flush=True); ok = torch.cuda.is_available(); print('cuda', ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 1)" || exit 1
echo "f81ad1ecb6dc4aeb61e20762ff3e027f  $P/weights/visionfm/VFM_SlitLamp_teacher_backbone.pth" | md5sum -c - || exit 1
# zip: MD5: folders inside it
for z in "grading_sustech_sysu.zip:6fdc1f278baa2bf78520197e48aac32b:sustech_sysu" \
         "grading_nuclear_cataract.zip:7e6728927c6c7e955925effb1ebca6c7:nuclear_cataract" \
         "grading_sustech_sysu_extra.zip:6fa9a1c4dff1340ea1bc7b30c62bbe97:sustech_sysu_test_seen sustech_sysu_test_holdout" \
         "grading_nuclear_cataract_extra.zip:794f03a57981a95b5f0882e30f96ca9d:nuclear_cataract_test_seen nuclear_cataract_test_holdout"; do
  zip=${z%%:*}; rest=${z#*:}; md5=${rest%%:*}; folders=${rest#*:}
  todo=""; for f in $folders; do [ -f "$FEAT/$f/features_visionfm.npz" ] || todo="$todo $f"; done
  if [ -z "$todo" ]; then echo "skip $zip (features done)"; continue; fi
  echo "== $zip $(date -Is)"
  echo "$md5  $HOME/data/$zip" | md5sum -c - || exit 1
  python -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$HOME/data/$zip" "$S"
  for f in $todo; do
    md5sum "$S/$f/labels.csv" "$S/$f/quality_d3.csv"
    srun python "$P/grading/extract_features.py" --data_root "$S/$f" --out_dir "$FEAT/$f" --batch 128 --num_workers 8
  done
  rm -rf "${S:?}"/*
done
for t in "sustech_sysu:category" "nuclear_cataract:grade"; do
  task=${t%%:*}; out="$P/runs/grading/${task}_${t#*:}_visionfm"
  conds=""; for c in A B C C_oracle; do for s in 14 15 16; do
    [ -f "$out/${c}_seed${s}/predictions_test_holdout.csv" ] || { conds="$conds $c"; break; }; done; done
  if [ -z "$conds" ]; then echo "skip heads $task (done)"; continue; fi
  echo "== heads $task:$conds $(date -Is)"
  # shellcheck disable=SC2086
  srun python "$P/grading/train_frozen_head.py" --task "$task" --data "$FEAT" --out "$out" --conds $conds
done
echo "end $(date -Is)"

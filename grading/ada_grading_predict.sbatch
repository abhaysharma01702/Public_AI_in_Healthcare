#!/bin/bash
# Extra grading test sets: predict with already-trained runs, no training. One job, one GPU.
# Data zip (the extra test-set folders) unzipped to the node-local SSD, deleted at the end. Each run folder gets
# predictions_<name>.csv next to its predictions_test.csv.
# Usage: sbatch --export=ALL,ZIP=grading_sustech_sysu_extra.zip,ZIP_MD5=<md5>,SETS="sustech_sysu_test_seen:test_seen sustech_sysu_test_holdout:test_holdout",RUNS="sustech_sysu_category sustech_sysu_category_token" ada_grading_predict.sbatch
#SBATCH --job-name=grading_pred
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
echo "node $(hostname) job $SLURM_JOB_ID zip $ZIP sets $SETS runs $RUNS start $(date -Is)"
source "$HOME/envs/qm/bin/activate"; export TORCH_HOME="$P/torch_home" PYTHONUNBUFFERED=1
# stop at once if the GPU is unusable or the check hangs (53015 on gnode066 fell back to CPU; 54152 on gnode061 and 54289 on gnode044 and 54570 on gnode068 hung)
echo "gpu check $(date -Is)"; timeout 60 nvidia-smi -L || echo "nvidia-smi failed or hung"
timeout 120 python -c "import time; t = time.time(); import torch, sys; print('torch import', round(time.time() - t, 1), 's', flush=True); ok = torch.cuda.is_available(); print('cuda', ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 1)" || exit 1
md5sum "$HOME/data/$ZIP"   # checksum on the node, not the gateway
if [ -n "${ZIP_MD5:-}" ]; then echo "$ZIP_MD5  $HOME/data/$ZIP" | md5sum -c - || exit 1; fi
python -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$HOME/data/$ZIP" "$S"
R=""; for r in $RUNS; do R="$R $P/runs/grading/$r"; done
for sn in $SETS; do
  set_dir=${sn%%:*}; name=${sn##*:}
  md5sum "$S/$set_dir/labels.csv" "$S/$set_dir/quality_d3.csv"
  echo "== $set_dir -> predictions_$name.csv $(date -Is)"
  # shellcheck disable=SC2086
  srun python "$P/grading/predict_grading.py" --runs $R --data_root "$S/$set_dir" --name "$name" \
    --master "$P/labels/sustech_master.csv" --num_workers 8
done
echo "end $(date -Is)"

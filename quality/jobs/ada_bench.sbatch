#!/bin/bash
# Batch-size benchmark on Ada's RTX 2080 Ti (11 GB): unzip once to the node-local SSD, check that
# PyTorch sees the GPU, then 50 training steps each for 6x4, 6x4 memory-efficient, 8x4 memory-efficient
# (an out-of-memory failure only skips that configuration). Scratch is deleted at the end.
#SBATCH --job-name=qm_bench
#SBATCH --account=<your_account>
#SBATCH --qos=<your_qos>
#SBATCH --partition=u22
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode004
#SBATCH --mem=28G            # Ada caps about 2.9 GB per CPU: 28 GB fits in our 10 CPUs (48G asked for 17 CPUs)
#SBATCH --time=00:40:00
#SBATCH --output=$HOME/project/logs/slurm_%x_%j.out

set -uo pipefail
P="$HOME/project"
S="/ssd_scratch/$USER/$SLURM_JOB_ID"
mkdir -p "$S"
trap 'rm -rf "$S"' EXIT
echo "node $(hostname)  job $SLURM_JOB_ID  start $(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
source "$HOME/envs/qm/bin/activate"
export TORCH_HOME="$P/torch_home" PYTHONUNBUFFERED=1
python -c "import torch; print('cuda available', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" || exit 1
t0=$(date +%s)
python -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$HOME/data/quality_data_v2.zip" "$S" || exit 1
echo "unzipped in $(( $(date +%s) - t0 )) s; $(du -sh "$S" | cut -f1)"
for cfg in "6 false" "6 true" "8 true"; do
  set -- $cfg
  echo "=== ${1}x4 memory_efficient=$2"
  srun python "$P/quality/train.py" --config "$P/quality/configs/densenet121.yaml" --data_root "$S/Quality Data" \
    --run_dir "$P/runs/benchmark_${1}x4_me$2" --benchmark_steps 50 \
    --set data.num_workers=8 batch.sources_per_batch=$1 model.memory_efficient=$2 2>&1 | grep -v -E "^  step|Warning" | tail -3
done
echo "end $(date -Is)"

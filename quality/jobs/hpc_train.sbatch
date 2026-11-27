#!/bin/bash
# Quality module training on a SLURM cluster. TEMPLATE: the IIIT HPC details (partition, GPU name,
# modules, scratch path, time limit) are not known yet; fill in the lines marked TODO once access is granted.
#SBATCH --job-name=qm_densenet121
#SBATCH --partition=TODO_partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00               # TODO: cluster limit; resubmit the same job to continue from last.pt
#SBATCH --signal=USR1@900             # 15 min before the limit: finish the epoch, save, exit (sent to srun steps)
#SBATCH --requeue
#SBATCH --output=logs/slurm_%x_%j.out

set -euo pipefail
PROJECT=${PROJECT:-$HOME/project}                    # TODO: where the code is
DATA=${DATA:-"$PROJECT/Data/Quality Data"}          # TODO: ideally node-local scratch (14.6 GB of PNGs)
RUN=${RUN:-$PROJECT/runs/quality/densenet121_seed14}
EXTRA=${EXTRA:-}                                      # e.g. "batch.sources_per_batch=6" after the benchmark

# TODO: module load / conda activate with Python 3.10+ and torch / torchvision (CUDA), pandas, scipy, pyyaml, pillow
nvidia-smi

cd "$PROJECT"
if [ "${BENCHMARK:-0}" = "1" ]; then
  srun python quality/train.py --config quality/configs/densenet121.yaml --data_root "$DATA" \
    --run_dir "$RUN/../benchmark" --benchmark_steps 50 ${EXTRA:+--set $EXTRA}
else
  srun python quality/train.py --config quality/configs/densenet121.yaml --data_root "$DATA" \
    --run_dir "$RUN" ${EXTRA:+--set $EXTRA}
fi

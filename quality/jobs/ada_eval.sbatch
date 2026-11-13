#!/bin/bash
# Evaluate an existing run folder on Quality Data v2 sets. RUN = run folder (best.pt + config.yaml),
# SETS = evaluate.py sets. Data unzipped to the node-local SSD, deleted at the end.
#SBATCH --job-name=qm_eval
#SBATCH --account=<your_account>
#SBATCH --qos=<your_qos>
#SBATCH --partition=u22
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode004
#SBATCH --mem=28G
#SBATCH --time=00:45:00
#SBATCH --output=$HOME/project/logs/slurm_%x_%j.out
set -euo pipefail
P="$HOME/project"; S="/ssd_scratch/$USER/$SLURM_JOB_ID"; mkdir -p "$S"; trap 'rm -rf "$S"' EXIT
echo "node $(hostname) job $SLURM_JOB_ID run $RUN sets $SETS start $(date -Is)"
source "$HOME/envs/qm/bin/activate"; export TORCH_HOME="$P/torch_home" PYTHONUNBUFFERED=1
python -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$HOME/data/quality_data_v2.zip" "$S"
srun python "$P/quality/evaluate.py" --run_dir "$RUN" --data_root "$S/Quality Data" --sets $SETS --num_workers 8 --bootstrap 1000
echo "end $(date -Is)"

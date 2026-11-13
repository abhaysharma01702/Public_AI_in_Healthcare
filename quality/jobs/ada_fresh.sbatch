#!/bin/bash
# Phase E (addendum A): the ONE sealed gen_fresh pass for all final models, same evaluate.py for all.
# Each model gets its own output folder runs/fresh_pass/<name> (config.yaml + label_stats.json copied, best.pt linked),
# so the original run folders' metrics_summary.json is not overwritten. Sets: gen_train + gen_fresh
# (gives generalisation_comparison_fresh.csv). Data unzipped to the node-local SSD, deleted at the end.
#SBATCH --job-name=qm_fresh
#SBATCH --account=<your_account>
#SBATCH --qos=<your_qos>
#SBATCH --partition=u22
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode004
#SBATCH --mem=28G
#SBATCH --time=01:30:00
#SBATCH --output=$HOME/project/logs/slurm_%x_%j.out
set -euo pipefail
P="$HOME/project"; S="/ssd_scratch/$USER/$SLURM_JOB_ID"; mkdir -p "$S"; trap 'rm -rf "$S"' EXIT
echo "node $(hostname) job $SLURM_JOB_ID start $(date -Is)"
source "$HOME/envs/qm/bin/activate"; export TORCH_HOME="$P/torch_home" PYTHONUNBUFFERED=1
for z in "$HOME/data/quality_data_v2.zip" "$HOME/data/quality_data_fresh.zip"; do
  python -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$z" "$S"
done
md5sum "$S/Quality Data/gen_fresh.csv"
# name = source run folder
for pair in v1:densenet121_v1_kaggle v2:densenet121_v2_ada D1:densenet121_v3_ada D2:mcf_densenet121_ada D3:swin_t_ada; do
  name=${pair%%:*}; src="$P/runs/${pair#*:}"; out="$P/runs/fresh_pass/$name"
  mkdir -p "$out"; cp "$src/config.yaml" "$out/"; [ -f "$src/label_stats.json" ] && cp "$src/label_stats.json" "$out/"
  ln -sf "$src/best.pt" "$out/best.pt"; md5sum "$src/best.pt"
  echo "== $name ($src) $(date -Is)"
  srun python "$P/quality/evaluate.py" --run_dir "$out" --data_root "$S/Quality Data" --sets gen_train gen_fresh --num_workers 8 --bootstrap 1000
done
echo "end $(date -Is)"

#!/bin/bash
# Evaluation of grading runs on Ada: evaluate_grading.py (numpy metrics, no sklearn) for every
# (run folder, test set) pair, all in parallel, one CPU each. CPU only: no GPU requested.
# RUNS = run folders under runs/grading; folders starting with nuclear_cataract get the Nuclear Cataract master.
# Usage: sbatch --export=ALL,RUNS="sustech_sysu_category_visionfm nuclear_cataract_grade_visionfm" ada_evaluate.sbatch
#SBATCH --job-name=evaluate
#SBATCH --account=<your_account>
#SBATCH --qos=<your_qos>
#SBATCH --partition=u22
#SBATCH --cpus-per-task=10
#SBATCH --exclude=gnode004,gnode066,gnode061,gnode044,gnode068,gnode054,gnode079
#SBATCH --mem=28G
#SBATCH --time=03:00:00
#SBATCH --output=$HOME/project/logs/slurm_%x_%j.out
set -euo pipefail
P="$HOME/project"; SETS=${SETS:-"test test_seen test_holdout"}
echo "node $(hostname) job $SLURM_JOB_ID runs $RUNS sets $SETS start $(date -Is)"
source "$HOME/envs/qm/bin/activate"; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for r in $RUNS; do
  m=""; case "$r" in nuclear_cataract*) m="--master $P/labels/nuclear_cataract_master.csv";; esac
  for s in $SETS; do
    # shellcheck disable=SC2086
    ( t=$(date +%s); python "$P/grading/evaluate_grading.py" --runs "$P/runs/grading/$r" --pred "$s" $m \
        > "$P/logs/evaluate_${r}_${s}.log" 2>&1; echo "done $r $s exit $? $(( $(date +%s) - t )) s" ) &
  done
done
wait
echo "end $(date -Is)"

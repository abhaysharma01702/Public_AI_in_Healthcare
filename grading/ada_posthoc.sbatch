#!/bin/bash
# Post-hoc quality experiments: quality_posthoc.py for 2 tasks x 3 test sets, in parallel, one CPU each.
# CPU only: no GPU requested. Quality files uploaded to ~/project/labels/posthoc/<task>_<set>_quality_d3.csv.
#SBATCH --job-name=posthoc
#SBATCH --account=<your_account>
#SBATCH --qos=<your_qos>
#SBATCH --partition=u22
#SBATCH --cpus-per-task=10
#SBATCH --exclude=gnode004,gnode066,gnode061,gnode044,gnode068,gnode054,gnode079,gnode091
#SBATCH --mem=28G
#SBATCH --time=02:00:00
#SBATCH --output=$HOME/project/logs/slurm_%x_%j.out
set -euo pipefail
P="$HOME/project"; R="$P/runs/grading"
echo "node $(hostname) job $SLURM_JOB_ID start $(date -Is)"
source "$HOME/envs/qm/bin/activate"; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$P/grading"
for t in "nuclear_cataract nuclear_cataract_grade" "sustech_sysu sustech_sysu_category"; do
  set -- $t
  for s in test test_seen test_holdout; do
    ( t0=$(date +%s); python quality_posthoc.py --dense "$R/$2" --vfm "$R/$2_visionfm_ft" \
        --quality "$P/labels/posthoc/$1_${s}_quality_d3.csv" --pred $s --out "$P/results/posthoc/$1" \
        > "$P/logs/posthoc_$1_$s.log" 2>&1 && rc=0 || rc=$?; echo "done $1 $s exit $rc $(( $(date +%s) - t0 )) s" ) &
  done
done
wait
echo "end $(date -Is)"

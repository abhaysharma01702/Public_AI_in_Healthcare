#!/bin/bash
# Quality module on IIIT Ada: RTX 2080 Ti 11 GB, 1 job at a time, 6 h walltime (admin), 10 cores.
# The data zip in $HOME is unpacked onto the node-local SSD (ssd_scratch) at the start and deleted at the end (trap),
# so the shared home filesystem only ever sees one large file read.
# MODE=benchmark: 50-step speed / memory test (SETS = extra --set overrides, e.g. "batch.sources_per_batch=6")
# MODE=train: training with resume; stops cleanly at 5.2 h (budget) so the final evaluation fits in 6 h;
#                  resubmit the SAME command to continue from last.pt (only 1 job may exist, so no queued chain).
# Usage (from the gateway): sbatch --export=ALL,MODE=benchmark,SETS="model.memory_efficient=true batch.sources_per_batch=6" ada_train.sbatch
# With the extension: add EXT_ZIP=$HOME/data/quality_data_ext_v3.zip to --export (unpacked into the same folder).
# Other models: add CONFIG=mcf_densenet121 or CONFIG=swin_t (default densenet121; benchmark dir runs/benchmark_<config>).
#SBATCH --job-name=qm_train
#SBATCH --account=<your_account>
#SBATCH --qos=<your_qos>
#SBATCH --partition=u22
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode004
#SBATCH --mem=28G            # Ada caps about 2.9 GB per CPU: 28 GB fits in our 10 CPUs (48G asked for 17 CPUs)
#SBATCH --time=06:00:00
#SBATCH --signal=USR1@900
#SBATCH --output=$HOME/project/logs/slurm_%x_%j.out

set -euo pipefail
P="$HOME/project"
ZIP="${ZIP:-$HOME/data/quality_data_v2.zip}"
EXT_ZIP="${EXT_ZIP:-}"   # extension (d4 versions), unpacked into the same "Quality Data" folder; sets data.extension
MODE="${MODE:-train}"
RUN="${RUN:-$P/runs/densenet121_v2_ada}"
SETS="${SETS:-}"
CONFIG="${CONFIG:-densenet121}"
S="/ssd_scratch/$USER/$SLURM_JOB_ID"
mkdir -p "$S"
trap 'rm -rf "$S"' EXIT
echo "node $(hostname)  job $SLURM_JOB_ID  mode $MODE  start $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
source "$HOME/envs/qm/bin/activate"
export TORCH_HOME="$P/torch_home" PYTHONUNBUFFERED=1
t0=$(date +%s)
python -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$ZIP" "$S"
[ -n "$EXT_ZIP" ] && python -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$EXT_ZIP" "$S"
echo "unzipped in $(( $(date +%s) - t0 )) s"
ARGS=(--config "$P/quality/configs/$CONFIG.yaml" --data_root "$S/Quality Data" --set data.num_workers=8)
[ -n "$EXT_ZIP" ] && ARGS+=(data.extension=labels_ext_v3.csv)
[ -n "$SETS" ] && ARGS+=($SETS)
if [ "$MODE" = "benchmark" ]; then
  srun python "$P/quality/train.py" "${ARGS[@]}" --run_dir "$P/runs/benchmark$([ "$CONFIG" = densenet121 ] || echo "_$CONFIG")" --benchmark_steps 50
else
  ARGS+=("eval.sets=[val,test,stress,real_check,gen_train,gen_holdout]")
  srun python "$P/quality/train.py" "${ARGS[@]}" --run_dir "$RUN" --time_budget_h 5.2
fi
echo "end $(date -Is)"

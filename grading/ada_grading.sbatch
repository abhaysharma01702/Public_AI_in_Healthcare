#!/bin/bash
# Grading runs  on Ada: one job runs every condition x seed in CONDS x SEEDS one after another.
# Data zip unzipped to the node-local SSD, deleted at the end. Runs write to $HOME/project/runs/grading/<task>.
# Usage: sbatch --export=ALL,TASK=sustech_sysu,CONDS="A B C C_oracle",SEEDS="14 15 16",RECIPE=r2 ada_grading.sbatch
#SBATCH --job-name=grading
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
TASK=${TASK:-sustech_sysu}; CONDS=${CONDS:-"A B C C_oracle"}; SEEDS=${SEEDS:-"14 15 16"}; RECIPE=${RECIPE:-r1}
# r1 = round 1 (type grade); r2 = round 2: category label + fixes, separate output folder
case "$RECIPE" in
  r1) OUT="$P/runs/grading/$TASK"; ARGS="";;
  r2) OUT="$P/runs/grading/${TASK}_category"
      ARGS="--label category --select loss --label_smoothing 0.1 --crop_scale 0.8 --q_standardise --lr_backbone 3e-5";;
  # r3 = round 3: r2 + quality-token transformer head (head learning rate 3e-4)
  r3) OUT="$P/runs/grading/${TASK}_category_token"
      ARGS="--label category --select loss --label_smoothing 0.1 --crop_scale 0.8 --q_standardise --lr_backbone 3e-5 --lr_head 3e-4 --head token";;
  # r2g = r2 options with the task's own grade label (Nuclear Cataract NC grade)
  r2g) OUT="$P/runs/grading/${TASK}_grade"
      ARGS="--select loss --label_smoothing 0.1 --crop_scale 0.8 --q_standardise --lr_backbone 3e-5";;
  # E3 leakage check: condition A on an image-level split of the same clean photos (labels file from
  # grading/make_image_split.py, uploaded to ~/project/labels); r2_img for SUSTech-SYSU, r2g_img for Nuclear Cataract
  r2_img) OUT="$P/runs/grading/${TASK}_category_imgsplit"
      ARGS="--label category --select loss --label_smoothing 0.1 --crop_scale 0.8 --q_standardise --lr_backbone 3e-5 --labels $P/labels/${TASK}_imgsplit_grading_labels.csv";;
  r2g_img) OUT="$P/runs/grading/${TASK}_grade_imgsplit"
      ARGS="--select loss --label_smoothing 0.1 --crop_scale 0.8 --q_standardise --lr_backbone 3e-5 --labels $P/labels/${TASK}_imgsplit_grading_labels.csv";;
  # fine-tuned VisionFM slit-lamp ViT-B/16 at 384 x 512, otherwise r2 / r2g;
  # vfm_bench = 2 epochs of r2g_vfm to measure time per epoch and GPU memory (separate output folder)
  r2_vfm) OUT="$P/runs/grading/${TASK}_category_visionfm_ft"
      ARGS="--label category --select loss --label_smoothing 0.1 --crop_scale 0.8 --q_standardise --lr_backbone 3e-5 --arch visionfm --height 384 --width 512 ${VFM_ARGS:-}";;
  r2g_vfm) OUT="$P/runs/grading/${TASK}_grade_visionfm_ft"
      ARGS="--select loss --label_smoothing 0.1 --crop_scale 0.8 --q_standardise --lr_backbone 3e-5 --arch visionfm --height 384 --width 512 ${VFM_ARGS:-}";;
  vfm_bench) OUT="$P/runs/grading/${TASK}_visionfm_bench"
      ARGS="--select loss --label_smoothing 0.1 --crop_scale 0.8 --q_standardise --lr_backbone 3e-5 --arch visionfm --height 384 --width 512 --max_epochs 2 ${VFM_ARGS:-}";;
  # r2_vfm / r2g_vfm + layer-wise lr decay 0.75 + 2 warmup epochs
  r2_vfmld) OUT="$P/runs/grading/${TASK}_category_visionfm_ftld"
      ARGS="--label category --select loss --label_smoothing 0.1 --crop_scale 0.8 --q_standardise --lr_backbone 3e-5 --arch visionfm --height 384 --width 512 --layer_decay 0.75 --warmup_epochs 2 ${VFM_ARGS:-}";;
  r2g_vfmld) OUT="$P/runs/grading/${TASK}_grade_visionfm_ftld"
      ARGS="--select loss --label_smoothing 0.1 --crop_scale 0.8 --q_standardise --lr_backbone 3e-5 --arch visionfm --height 384 --width 512 --layer_decay 0.75 --warmup_epochs 2 ${VFM_ARGS:-}";;
  # augmentation without defects on the option-1 grader (fine-tuned VisionFM), condition A;
  # AUG = geo | mix | bal | all; output ${TASK}_aug_<AUG>; SUSTech-SYSU keeps the category label
  aug) G="--rotate 15 --translate 0.05"; M="--mixup 0.4 --cutmix 1.0 --mix_prob 0.5"; B="--balance 0.5"
      case "${AUG:-geo}" in geo) X="$G";; mix) X="$G $M";; bal) X="$G $B";; all) X="$G $M $B";;
        *) echo "unknown AUG $AUG"; exit 1;; esac
      L=""; [ "$TASK" = sustech_sysu ] && L="--label category"
      OUT="$P/runs/grading/${TASK}_aug_${AUG:-geo}${TAG:-}"
      ARGS="$L --select loss --label_smoothing 0.1 --crop_scale 0.8 --q_standardise --lr_backbone 3e-5 --arch visionfm --height 384 --width 512 $X ${VFM_ARGS:-}";;
  # ordinal target / class weights on the fine-tuned VisionFM, condition A.
  # reg  = Nuclear Cataract only: regression on the raw NC value + tuned cut points (output ${TASK}_reg)
  # cw   = class-weighted cross-entropy 1 / count^0.5 (output ${TASK}_cw); EXTRA = further train_grading options
  reg) [ "$TASK" = nuclear_cataract ] || { echo "reg: nuclear_cataract only"; exit 1; }
      OUT="$P/runs/grading/${TASK}_reg${TAG:-}"
      ARGS="--label nc_raw --master $P/labels/nuclear_cataract_master.csv --select loss --crop_scale 0.8 --q_standardise --lr_backbone 3e-5 --arch visionfm --height 384 --width 512 ${EXTRA:-} ${VFM_ARGS:-}";;
  cw) L=""; [ "$TASK" = sustech_sysu ] && L="--label category"
      OUT="$P/runs/grading/${TASK}_cw${TAG:-}"
      ARGS="$L --select loss --label_smoothing 0.1 --crop_scale 0.8 --q_standardise --lr_backbone 3e-5 --arch visionfm --height 384 --width 512 --class_weight 0.5 ${EXTRA:-} ${VFM_ARGS:-}";;
  *) echo "unknown RECIPE $RECIPE"; exit 1;;
esac
echo "node $(hostname) job $SLURM_JOB_ID task $TASK recipe $RECIPE conds $CONDS seeds $SEEDS start $(date -Is)"
source "$HOME/envs/qm/bin/activate"; export TORCH_HOME="$P/torch_home" PYTHONUNBUFFERED=1
# stop at once if the GPU is unusable or the check hangs (53015 on gnode066 fell back to CPU; 54152 on gnode061 and 54289 on gnode044 and 54570 on gnode068 hung)
echo "gpu check $(date -Is)"; timeout 60 nvidia-smi -L || echo "nvidia-smi failed or hung"
timeout 120 python -c "import time; t = time.time(); import torch, sys; print('torch import', round(time.time() - t, 1), 's', flush=True); ok = torch.cuda.is_available(); print('cuda', ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 1)" || exit 1
md5sum "$HOME/data/grading_${TASK}.zip"   # compare with the local zip's MD5 (checksum on the node, not the gateway)
if [ -n "${ZIP_MD5:-}" ]; then echo "$ZIP_MD5  $HOME/data/grading_${TASK}.zip" | md5sum -c - || exit 1; fi
python -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$HOME/data/grading_${TASK}.zip" "$S"
md5sum "$S/$TASK/labels.csv" "$S/$TASK/quality_d3.csv"
# optional: also predict the seen / holdout test sets right after each run (EXTRA_MD5 = MD5 of
# grading_${TASK}_extra.zip), so the checkpoints can leave Ada (home quota) without a separate prediction job
if [ -n "${EXTRA_MD5:-}" ]; then
  echo "$EXTRA_MD5  $HOME/data/grading_${TASK}_extra.zip" | md5sum -c - || exit 1
  python -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$HOME/data/grading_${TASK}_extra.zip" "$S"
fi
for c in $CONDS; do
  for s in $SEEDS; do
    if [ -f "$OUT/${c}_seed${s}/predictions_test.csv" ]; then echo "skip $c seed $s (done)"; continue; fi
    echo "== $c seed $s $(date -Is)"
    # shellcheck disable=SC2086
    srun python "$P/grading/train_grading.py" --data_root "$S/$TASK" --cond "$c" --seed "$s" \
      --out "$OUT" --num_workers 8 $ARGS
    if [ -n "${EXTRA_MD5:-}" ]; then
      for x in test_seen test_holdout; do
        srun python "$P/grading/predict_grading.py" --runs "$OUT" --data_root "$S/${TASK}_$x" --name "$x" --num_workers 8
      done
    fi
  done
done
if [ -n "${EXTRA_MD5:-}" ]; then   # catch-up: runs trained by an earlier, interrupted job (finished ones are skipped)
  for x in test_seen test_holdout; do
    srun python "$P/grading/predict_grading.py" --runs "$OUT" --data_root "$S/${TASK}_$x" --name "$x" --num_workers 8
  done
fi
echo "end $(date -Is)"

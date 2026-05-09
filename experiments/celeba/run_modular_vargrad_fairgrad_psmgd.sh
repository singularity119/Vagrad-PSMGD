#!/bin/bash
set -euo pipefail

REPO_ROOT=/root/Vagrad-PSMGD-modular
DATA_ROOT=/root/autodl-tmp/dataset/celeba
EXP_ROOT=/root/autodl-tmp/experiment/celeba_experiment
SAVE_DIR=/root/autodl-tmp/exp_logs_save/modular/celeba/save
LOG_ROOT=/root/autodl-tmp/exp_logs_save/modular/celeba/log

method="${METHOD:-modular}"
preprocessing="${PREPROCESSING:-vargrad}"
solver="${SOLVER:-fairgrad}"
scheduler="${SCHEDULER:-psmgd_periodic}"

beta="${BETA:-0.85}"
psmgd_R="${PSMGD_R:-10}"
psmgd_alpha="${PSMGD_ALPHA:-0.5}"
alpha="${ALPHA:-2.0}"
seed="${SEED:-0}"
batch_size="${BATCH_SIZE:-256}"
epochs="${EPOCHS:-15}"
lr="${LR:-3e-4}"

mkdir -p "$SAVE_DIR"
mkdir -p "$LOG_ROOT"
cd "$REPO_ROOT/experiments/celeba"
export PYTHONPATH="$REPO_ROOT"
export OMP_NUM_THREADS=8

if [[ "$scheduler" == "psmgd_periodic" ]]; then
  run_name="${method}_${preprocessing}_${solver}_alpha${alpha}_beta${beta}_psmgd_R${psmgd_R}_a${psmgd_alpha}_sd${seed}"
else
  run_name="${method}_${preprocessing}_${solver}_alpha${alpha}_beta${beta}_${scheduler}_sd${seed}"
fi

log_file="$LOG_ROOT/${run_name}.log"

nohup python -u trainer.py \
  --method "$method" \
  --preprocessing "$preprocessing" \
  --solver "$solver" \
  --scheduler "$scheduler" \
  --beta "$beta" \
  --psmgd-R "$psmgd_R" \
  --psmgd-alpha "$psmgd_alpha" \
  --alpha "$alpha" \
  --seed "$seed" \
  --batch-size "$batch_size" \
  --n-epochs "$epochs" \
  --lr "$lr" \
  --data-path "$DATA_ROOT" \
  --save-dir "$SAVE_DIR" \
  > "$log_file" 2>&1 < /dev/null &

echo "Started CelebA run: $log_file"

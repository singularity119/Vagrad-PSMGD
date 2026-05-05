#!/bin/bash
set -euo pipefail

REPO_ROOT=/root/Vagrad-PSMGD-modular
DATA_ROOT=/root/autodl-tmp/dataset/nyuv2
EXP_ROOT=/root/autodl-tmp/experiment/nyuv2_experiment
SAVE_DIR=/root/autodl-tmp/exp_logs_save/modular/nyuv2/save
LOG_ROOT=/root/autodl-tmp/exp_logs_save/modular/nyuv2/log

method="${METHOD:-modular}"
preprocessing="${PREPROCESSING:-vargrad}"
solver="${SOLVER:-fairgrad}"
scheduler="${SCHEDULER:-psmgd_periodic}"
use_momentum="${USE_MOMENTUM:-true}"

beta_v="${BETA_V:-0.9}"
beta_m="${BETA_M:-0.9}"
psmgd_R="${PSMGD_R:-10}"
psmgd_alpha="${PSMGD_ALPHA:-0.5}"
alpha="${ALPHA:-1.0}"
seed="${SEED:-0}"
batch_size="${BATCH_SIZE:-2}"
epochs="${EPOCHS:-200}"
lr="${LR:-1e-4}"
model="${MODEL:-mtan}"

mkdir -p "$SAVE_DIR"
mkdir -p "$LOG_ROOT"
cd "$REPO_ROOT/experiments/nyuv2"
export PYTHONPATH="$REPO_ROOT"
export OMP_NUM_THREADS=8

if [[ "$scheduler" == "psmgd_periodic" ]]; then
  run_name="${method}_${preprocessing}_${solver}_psmgd_beta${beta_v}_R${psmgd_R}_a${psmgd_alpha}_alpha${alpha}_sd${seed}"
else
  if [[ "$use_momentum" == "true" ]]; then
    use_momentum_flag=1
  else
    use_momentum_flag=0
  fi
  run_name="${method}_${preprocessing}_${solver}_${scheduler}_mom${use_momentum_flag}_sd${seed}"
fi

log_file="$LOG_ROOT/${run_name}.log"

nohup python -u trainer.py \
  --method "$method" \
  --preprocessing "$preprocessing" \
  --solver "$solver" \
  --scheduler "$scheduler" \
  --use-momentum "$use_momentum" \
  --beta-v "$beta_v" \
  --beta-m "$beta_m" \
  --psmgd-R "$psmgd_R" \
  --psmgd-alpha "$psmgd_alpha" \
  --alpha "$alpha" \
  --seed "$seed" \
  --batch-size "$batch_size" \
  --n-epochs "$epochs" \
  --lr "$lr" \
  --model "$model" \
  --data-path "$DATA_ROOT" \
  --save-dir "$SAVE_DIR" \
  > "$log_file" 2>&1 < /dev/null &

echo "Started NYUv2 run: $log_file"

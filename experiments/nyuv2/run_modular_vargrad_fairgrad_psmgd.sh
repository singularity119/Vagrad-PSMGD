#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/Vargrad_PSMGD_modular}"
DATA_ROOT=/root/autodl-tmp/dataset/nyuv2
EXP_ROOT=/root/autodl-tmp/experiment/nyuv2_experiment
SAVE_DIR=/root/autodl-tmp/exp_logs_save/modular/nyuv2/save
LOG_ROOT=/root/autodl-tmp/exp_logs_save/modular/nyuv2/log

method="${METHOD:-modular}"
preprocessing="${PREPROCESSING:-vargrad}"
solver="${SOLVER:-fairgrad}"
scheduler="${SCHEDULER:-psmgd_periodic}"

beta="${BETA:-0.85}"
psmgd_R="${PSMGD_R:-10}"
psmgd_alpha="${PSMGD_ALPHA:-0.5}"
alpha="${ALPHA:-2.0}"
seed="${SEED:-0}"
batch_size="${BATCH_SIZE:-2}"
epochs="${EPOCHS:-200}"
lr="${LR:-1e-4}"
model="${MODEL:-mtan}"
save_u_telemetry="${SAVE_U_TELEMETRY:-false}"
python_bin="${PYTHON_BIN:-/root/miniconda3/bin/python}"

mkdir -p "$SAVE_DIR"
mkdir -p "$LOG_ROOT"
cd "$REPO_ROOT/experiments/nyuv2"
export PYTHONPATH="$REPO_ROOT"
export OMP_NUM_THREADS=8

if [[ "$scheduler" == "psmgd_periodic" ]]; then
  run_name="${method}_${preprocessing}_beta${beta}_${solver}_alpha${alpha}_psmgd_R${psmgd_R}_a${psmgd_alpha}_sd${seed}"
else
  run_name="${method}_${preprocessing}_beta${beta}_${solver}_alpha${alpha}_${scheduler}_sd${seed}"
fi

log_file="$LOG_ROOT/${run_name}.log"

nohup "$python_bin" -u trainer.py \
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
  --model "$model" \
  --data-path "$DATA_ROOT" \
  --save-dir "$SAVE_DIR" \
  --save-u-telemetry "$save_u_telemetry" \
  > "$log_file" 2>&1 < /dev/null &

echo "Started NYUv2 run: $log_file"

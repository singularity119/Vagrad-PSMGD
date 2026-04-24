#!/bin/bash
set -euo pipefail

REPO_ROOT=/root/Vagrad-PSMGD-modular
DATA_ROOT=/root/autodl-tmp/dataset/qm9
EXP_ROOT=/root/autodl-tmp/experiment/quantum_chemistry_experiment
SAVE_DIR=/root/autodl-tmp/exp_logs_save/modular/quantum_chemistry/save
LOG_ROOT=/root/autodl-tmp/exp_logs_save/modular/quantum_chemistry/log

method=modular
preprocessing=vargrad
solver=fairgrad
scheduler=psmgd_periodic
use_momentum=true

beta_v=0.9
beta_m=0.9
psmgd_R=10
psmgd_alpha=0.5
alpha=1.0
seed=0
batch_size=120
epochs=300
lr=1e-3
scale_y=true

mkdir -p "$SAVE_DIR"
mkdir -p "$LOG_ROOT"
cd "$REPO_ROOT/experiments/quantum_chemistry"
export PYTHONPATH="$REPO_ROOT"
export OMP_NUM_THREADS=8

log_file="$LOG_ROOT/${method}_${preprocessing}_${solver}_psmgd_beta${beta_v}_R${psmgd_R}_a${psmgd_alpha}_alpha${alpha}_sd${seed}.log"

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
  --scale-y "$scale_y" \
  --data-path "$DATA_ROOT" \
  --save-dir "$SAVE_DIR" \
  > "$log_file" 2>&1 < /dev/null &

echo "Started QM9 run: $log_file"

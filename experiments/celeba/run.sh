#!/bin/bash
set -euo pipefail

REPO_ROOT=/root/Vagrad-PSMGD-modular
DATA_ROOT=/root/autodl-tmp/dataset/celeba
EXP_ROOT=/root/autodl-tmp/experiment/celeba_experiment
SAVE_DIR=/root/autodl-tmp/exp_logs_save/modular/celeba/save
LOG_ROOT=/root/autodl-tmp/exp_logs_save/modular/celeba/log

mkdir -p "$SAVE_DIR"
mkdir -p "$LOG_ROOT"
cd "$REPO_ROOT/experiments/celeba"
export PYTHONPATH="$REPO_ROOT"

method=fairgrad
alpha=2.0
seed=0

nohup python -u trainer.py \
  --method="$method" \
  --seed="$seed" \
  --alpha="$alpha" \
  --data-path "$DATA_ROOT" \
  --save-dir "$SAVE_DIR" \
  > "$LOG_ROOT/$method-alpha$alpha-sd$seed.log" 2>&1 &

echo "Training started. Logs: $LOG_ROOT/$method-alpha$alpha-sd$seed.log"

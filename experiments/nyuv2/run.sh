#!/bin/bash
set -euo pipefail

REPO_ROOT=/root/Vagrad-PSMGD-modular
DATA_ROOT=/root/autodl-tmp/dataset/nyuv2
EXP_ROOT=/root/autodl-tmp/experiment/nyuv2_experiment
SAVE_DIR=/root/autodl-tmp/exp_logs_save/modular/nyuv2/save
LOG_ROOT=/root/autodl-tmp/exp_logs_save/modular/nyuv2/log

mkdir -p "$SAVE_DIR"
mkdir -p "$LOG_ROOT"
cd "$REPO_ROOT/experiments/nyuv2"
export PYTHONPATH="$REPO_ROOT"

method=fairgrad
alpha=2.0
seed=0
init=xaiver

nohup python -u trainer.py \
  --method="$method" \
  --seed="$seed" \
  --alpha="$alpha" \
  --data-path "$DATA_ROOT" \
  --save-dir "$SAVE_DIR" \
  > "$LOG_ROOT/$method-alpha$alpha-sd$seed-$init.log" 2>&1 &

echo "Training started. Logs: $LOG_ROOT/$method-alpha$alpha-sd$seed-$init.log"

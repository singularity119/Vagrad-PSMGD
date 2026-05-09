#!/bin/bash
set -euo pipefail

REPO_ROOT=/root/Vagrad-PSMGD-modular
RUN_DIR="$REPO_ROOT/experiments/nyuv2"
LOG_ROOT=/root/autodl-tmp/exp_logs_save/modular/nyuv2/log
SAVE_ROOT=/root/autodl-tmp/exp_logs_save/modular/nyuv2/save

SOURCE_SEED="${SOURCE_SEED:-1}"
TARGET_SEED="${TARGET_SEED:-0}"
MODEL="${MODEL:-mtan}"
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"

SOURCE_RUN_NAME="modular_vargrad_fairgrad_alpha2.0_beta0.85_every_step_sd${SOURCE_SEED}"
TARGET_RUN_NAME="modular_vargrad_fairgrad_alpha2.0_beta0.85_every_step_sd${TARGET_SEED}"
SOURCE_LOG="$LOG_ROOT/$SOURCE_RUN_NAME.log"
TARGET_LOG="$LOG_ROOT/$TARGET_RUN_NAME.log"
TARGET_STATS="$SAVE_ROOT/$TARGET_RUN_NAME.stats"
WATCH_LOG="$LOG_ROOT/$TARGET_RUN_NAME.after_sd${SOURCE_SEED}.watchdog.log"

SOURCE_PATTERN="python -u trainer.py --method modular --preprocessing vargrad --solver fairgrad --scheduler every_step --beta 0.85 --psmgd-R 10 --psmgd-alpha 0.5 --alpha 2.0 --seed ${SOURCE_SEED}"
TARGET_PATTERN="python -u trainer.py --method modular --preprocessing vargrad --solver fairgrad --scheduler every_step --beta 0.85 --psmgd-R 10 --psmgd-alpha 0.5 --alpha 2.0 --seed ${TARGET_SEED}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  mkdir -p "$LOG_ROOT"
  echo "$(timestamp) $*" >> "$WATCH_LOG"
}

matching_pid() {
  local pattern="$1"
  pgrep -f "$pattern" | head -n 1 || true
}

source_complete() {
  [[ -f "$SOURCE_LOG" ]] && grep -q "Final Performance:" "$SOURCE_LOG"
}

target_complete() {
  [[ -f "$TARGET_LOG" ]] && grep -q "Final Performance:" "$TARGET_LOG"
}

backup_target_artifacts() {
  local ts
  ts=$(date '+%Y%m%d_%H%M%S')
  if [[ -s "$TARGET_LOG" ]]; then
    cp -p "$TARGET_LOG" "$TARGET_LOG.before_chain_start_$ts"
    log "backed up existing target log to $TARGET_LOG.before_chain_start_$ts"
  fi
  if [[ -s "$TARGET_STATS" ]]; then
    cp -p "$TARGET_STATS" "$TARGET_STATS.before_chain_start_$ts"
    log "backed up existing target stats to $TARGET_STATS.before_chain_start_$ts"
  fi
}

start_target() {
  log "starting target run: seed=${TARGET_SEED} run_name=${TARGET_RUN_NAME}"
	(
	  cd "$RUN_DIR"
	  export SCHEDULER=every_step
	  export SEED="$TARGET_SEED"
	  export MODEL
	  exec bash ./run_modular_vargrad_fairgrad_psmgd.sh
	) >> "$WATCH_LOG" 2>&1
}

log "chain watcher started: source=${SOURCE_RUN_NAME} target=${TARGET_RUN_NAME} check_interval=${CHECK_INTERVAL}s"

while true; do
  target_pid=$(matching_pid "$TARGET_PATTERN")
  if [[ -n "$target_pid" ]]; then
    log "target already running with pid=$target_pid; exiting"
    exit 0
  fi

  if target_complete; then
    log "target already complete; exiting"
    exit 0
  fi

  if source_complete; then
    log "source complete; launching target"
    backup_target_artifacts
    start_target
    sleep 10
    target_pid=$(matching_pid "$TARGET_PATTERN")
    if [[ -n "$target_pid" ]]; then
      log "target started with pid=$target_pid; exiting"
      exit 0
    fi
    log "launch command returned but no target process was found"
    exit 1
  fi

  source_pid=$(matching_pid "$SOURCE_PATTERN")
  if [[ -n "$source_pid" ]]; then
    log "waiting: source still running with pid=$source_pid"
  else
    log "waiting: source is not complete yet and no matching source process was found"
  fi
  sleep "$CHECK_INTERVAL"
done

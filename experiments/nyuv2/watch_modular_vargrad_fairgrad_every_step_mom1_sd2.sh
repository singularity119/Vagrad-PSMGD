#!/bin/bash
set -euo pipefail

REPO_ROOT=/root/Vagrad-PSMGD-modular
RUN_DIR="$REPO_ROOT/experiments/nyuv2"
LOG_ROOT=/root/autodl-tmp/exp_logs_save/modular/nyuv2/log
SAVE_ROOT=/root/autodl-tmp/exp_logs_save/modular/nyuv2/save
SEED="${SEED:-2}"
MODEL="${MODEL:-mtan}"
RUN_NAME="${RUN_NAME:-modular_vargrad_fairgrad_every_step_mom1_sd${SEED}}"
TRAIN_LOG="$LOG_ROOT/$RUN_NAME.log"
STATS_FILE="$SAVE_ROOT/$RUN_NAME.stats"
WATCH_LOG="$LOG_ROOT/$RUN_NAME.watchdog.log"
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"
STALE_SECONDS="${STALE_SECONDS:-1200}"

MATCH_PATTERN="python -u trainer.py --method modular --preprocessing vargrad --solver fairgrad --scheduler every_step --use-momentum true --beta-v 0.9 --beta-m 0.9 --psmgd-R 10 --psmgd-alpha 0.5 --alpha 1.0 --seed ${SEED}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  echo "$(timestamp) $*" >> "$WATCH_LOG"
}

current_pid() {
  pgrep -f "$MATCH_PATTERN" | head -n 1 || true
}

is_complete() {
  [[ -f "$TRAIN_LOG" ]] && grep -q "Final Performance:" "$TRAIN_LOG"
}

backup_artifacts() {
  local ts
  ts=$(date '+%Y%m%d_%H%M%S')
  if [[ -s "$TRAIN_LOG" ]]; then
    cp -p "$TRAIN_LOG" "$TRAIN_LOG.watchdog_restart_$ts"
    log "backed up train log to $TRAIN_LOG.watchdog_restart_$ts"
  fi
  if [[ -s "$STATS_FILE" ]]; then
    cp -p "$STATS_FILE" "$STATS_FILE.watchdog_restart_$ts"
    log "backed up stats to $STATS_FILE.watchdog_restart_$ts"
  fi
}

start_run() {
  log "starting run via run_modular_vargrad_fairgrad_psmgd.sh"
  (
    cd "$RUN_DIR"
    export SCHEDULER=every_step
    export USE_MOMENTUM=true
    export SEED
    export MODEL
    exec bash ./run_modular_vargrad_fairgrad_psmgd.sh
  ) >> "$WATCH_LOG" 2>&1
}

log "watchdog started: run_name=${RUN_NAME} seed=${SEED} check_interval=${CHECK_INTERVAL}s stale_seconds=${STALE_SECONDS}s"

while true; do
  pid=$(current_pid)
  if [[ -z "$pid" ]]; then
    if is_complete; then
      log "training completed normally; Final Performance found, watchdog exiting"
      exit 0
    fi
    log "training process missing"
    backup_artifacts
    start_run
    sleep 10
    pid=$(current_pid)
    if [[ -n "$pid" ]]; then
      log "training restarted with pid=$pid"
    else
      log "restart command returned but no matching training process was found"
    fi
  else
    if [[ -f "$TRAIN_LOG" ]]; then
      now=$(date +%s)
      mtime=$(stat -c %Y "$TRAIN_LOG")
      age=$((now - mtime))
      if (( age > STALE_SECONDS )); then
        log "warning: pid=$pid is alive but train log has not changed for ${age}s"
      else
        gpu_line=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | awk -F, -v pid="$pid" '$1 == pid {print $0}' || true)
        if [[ -n "$gpu_line" ]]; then
          log "ok: pid=$pid log_age=${age}s gpu=[$gpu_line]"
        else
          log "ok: pid=$pid log_age=${age}s gpu=[not listed]"
        fi
      fi
    else
      log "warning: pid=$pid is alive but train log does not exist"
    fi
  fi
  sleep "$CHECK_INTERVAL"
done

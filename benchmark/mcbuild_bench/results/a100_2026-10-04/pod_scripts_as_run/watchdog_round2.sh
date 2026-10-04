#!/usr/bin/env bash
# Unattended supervision of the second round. Every POLL seconds:
#   * all twelve cells have a meta.json      -> WATCHDOG_COMPLETE, exit
#   * no runner alive and not complete       -> start round2_run.sh again (it skips finished cells
#                                               and resumes a partial one from its checkpoint)
#   * runner alive but nothing written for   -> STALL: kill the reader and the runner so the next
#     STALL_S seconds (no log line, no answer)   poll restarts them (a model load prints progress
#                                               every second, so a healthy load never trips this)
# Restarts are capped at MAX_RESTARTS so a cell that dies instantly cannot loop on a rented GPU.
set -uo pipefail
MCB=/workspace/mcb; OUT=$MCB/runs/round2; LOG=$MCB/runs/watchdog_round2.log
MAX_RESTARTS="${MAX_RESTARTS:-5}"; POLL="${POLL:-300}"; STALL_S="${STALL_S:-1500}"
CELLS="truncB_W8000 truncB_W16000 truncB_W32000
       proposed_W8000_w0.1 proposed_W8000_w0.3 proposed_W8000_w1.0
       proposed_W16000_w0.1 proposed_W16000_w0.3 proposed_W16000_w1.0
       proposed_W32000_w0.1 proposed_W32000_w0.3 proposed_W32000_w1.0"
log() { printf '[%s] %s
' "$(date -u +%FT%T)" "$*" >> "$LOG"; }
complete() { for c in $CELLS; do [ -f "$OUT/$c/meta.json" ] || return 1; done; return 0; }
running() { pgrep -f "round2_run.sh|mcbuild_bench.run_arms" >/dev/null; }
newest_write_age() {  # seconds since the last byte written by the runner (logs or checkpoints)
  local t; t=$(stat -c %Y "$OUT"/*.log "$OUT"/*/answers.jsonl "$MCB/runs/round2_run.log" 2>/dev/null | sort -n | tail -1)
  [ -n "$t" ] && echo $(( $(date +%s) - t )) || echo 0
}

restarts=0
log "watchdog start (poll ${POLL}s, stall after ${STALL_S}s, max ${MAX_RESTARTS} restarts)"
while true; do
  if complete; then log "ALL CELLS COMPLETE"; echo WATCHDOG_COMPLETE >> "$LOG"; break; fi
  if running; then
    age=$(newest_write_age)
    if [ "$age" -gt "$STALL_S" ]; then
      log "STALL: runner alive but nothing written for ${age}s; killing reader and runner"
      pkill -f "mcbuild_bench.run_arms"; sleep 5; pkill -f "round2_run.sh"; sleep 5
      pkill -9 -f "mcbuild_bench.run_arms" 2>/dev/null
    fi
  fi
  if ! running; then
    left=$(for c in $CELLS; do [ -f "$OUT/$c/meta.json" ] || echo "$c"; done | tr '
' ' ')
    if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
      log "runner gone and $MAX_RESTARTS restarts used; giving up. missing: $left"
      echo WATCHDOG_GAVE_UP >> "$LOG"; break
    fi
    restarts=$((restarts + 1))
    log "runner gone, restart #$restarts; missing: $left"
    nohup bash $MCB/round2_run.sh >> $MCB/runs/round2_run.log 2>&1 &
    sleep 90
  fi
  sleep "$POLL"
done

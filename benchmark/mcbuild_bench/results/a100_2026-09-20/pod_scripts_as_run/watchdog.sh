#!/usr/bin/env bash
# Unattended supervision of the main run (Ryosuke: "同じようにプロセスを常に監視するように").
# Every 5 minutes: if every expected cell has a meta.json the run is complete and this exits; if no
# runner process is alive and the run is NOT complete, main_resume.sh is started again (it skips the
# finished cells and continues a partial one from its checkpoint). Capped at MAX_RESTARTS so a cell
# that fails instantly cannot loop forever on a rented GPU. Nothing here deletes or overwrites data.
set -uo pipefail
MCB=/workspace/mcb; OUT=$MCB/runs/main; LOG=$MCB/runs/watchdog.log
MAX_RESTARTS="${MAX_RESTARTS:-5}"; POLL="${POLL:-300}"
CELLS="proposed_W8000_w0.1 proposed_W8000_w0.3 proposed_W8000_w1.0
       proposed_W16000_w0.1 proposed_W16000_w0.3 proposed_W16000_w1.0
       proposed_W32000_w0.1 proposed_W32000_w0.3 proposed_W32000_w1.0 A_full"
log() { printf '[%s] %s\n' "$(date -u +%FT%T)" "$*" >> "$LOG"; }

complete() { for c in $CELLS; do [ -f "$OUT/$c/meta.json" ] || return 1; done; return 0; }
running() { pgrep -f "main_run.sh|main_resume.sh|mcbuild_bench.run_arms" >/dev/null; }

restarts=0
log "watchdog start (poll ${POLL}s, max ${MAX_RESTARTS} restarts)"
while true; do
  if complete; then log "ALL CELLS COMPLETE"; echo WATCHDOG_COMPLETE >> "$LOG"; break; fi
  if ! running; then
    left=$(for c in $CELLS; do [ -f "$OUT/$c/meta.json" ] || echo "$c"; done | tr '\n' ' ')
    if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
      log "runner gone and $MAX_RESTARTS restarts used; giving up. missing: $left"
      echo WATCHDOG_GAVE_UP >> "$LOG"; break
    fi
    restarts=$((restarts + 1))
    log "runner gone, restart #$restarts; missing: $left"
    nohup bash $MCB/main_resume.sh >> $MCB/runs/main_resume.log 2>&1 &
    sleep 90   # let the model load before the next liveness check
  fi
  sleep "$POLL"
done

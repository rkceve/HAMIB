#!/usr/bin/env bash
# Unattended supervision of the second round. Every 5 minutes: if all twelve cells have a
# meta.json the run is complete and this exits; if no runner is alive and the run is not complete,
# round2_run.sh is started again (it skips finished cells and resumes a partial one from its
# checkpoint). Capped at MAX_RESTARTS so a cell that fails instantly cannot loop on a rented GPU.
set -uo pipefail
MCB=/workspace/mcb; OUT=$MCB/runs/round2; LOG=$MCB/runs/watchdog_round2.log
MAX_RESTARTS="${MAX_RESTARTS:-5}"; POLL="${POLL:-300}"
CELLS="truncB_W8000 truncB_W16000 truncB_W32000
       proposed_W8000_w0.1 proposed_W8000_w0.3 proposed_W8000_w1.0
       proposed_W16000_w0.1 proposed_W16000_w0.3 proposed_W16000_w1.0
       proposed_W32000_w0.1 proposed_W32000_w0.3 proposed_W32000_w1.0"
log() { printf '[%s] %s\n' "$(date -u +%FT%T)" "$*" >> "$LOG"; }
complete() { for c in $CELLS; do [ -f "$OUT/$c/meta.json" ] || return 1; done; return 0; }
running() { pgrep -f "round2_run.sh|mcbuild_bench.run_arms" >/dev/null; }

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
    nohup bash $MCB/round2_run.sh >> $MCB/runs/round2_run.log 2>&1 &
    sleep 90
  fi
  sleep "$POLL"
done

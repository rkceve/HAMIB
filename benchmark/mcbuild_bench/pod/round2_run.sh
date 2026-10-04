#!/usr/bin/env bash
# Second measurement round (DECISIONS H34 + H35, Ryosuke 2026-10-01/03). Twelve reader-only cells
# on the UNCHANGED corpus, questions, cd.json, prompt and windows of the 2026-09-21 run:
#
#   truncB_W<W>            --arm B: the raw transcript cut to W, newest round trips first,
#                          no diagram, no bias.  The truncation control.
#   proposed_W<W>_w<w>     --arm proposed with the bias inherited by satellites and the
#                          effective bias w*mass capped at 3.0.
#
# W in {8000, 16000, 32000}; w in {0.1, 0.3, 1.0}.  The full-transcript baseline (94/96) and the
# w = 0 cells of the first run are reused, so nothing here needs the judge or the manager.
# Outputs go to /workspace/mcb/runs/round2/; A_full is linked in from the first run so the scorer
# can be pointed at this directory alone.
#
# Usage on the pod:  bash round2_run.sh [cd.json]        (resumable; rerun to continue)
# A cell that exits non-zero is reported as CELL_FAIL and skipped; the pass ends with exit 1.
set -uo pipefail
MCB=/workspace/mcb; CD="${1:-$MCB/runs/v4_36rt/cd.json}"; OUT=$MCB/runs/round2; mkdir -p "$OUT"
export HF_HOME=$MCB/hf PATH="$HOME/.local/bin:$PATH"
source /root/venvs/venv_reader/bin/activate
cd $MCB/cms-prototype
log() { printf '[%s] %s\n' "$(date -u +%T)" "$*"; }
COMMON=(--model-id Qwen/Qwen3.8-27B --questions benchmark/mcbuild_bench/data/questions.json
        --session benchmark/mcbuild_bench/data/session_redacted.json --prefill-chunk 8192
        --prefill-scale 0.0 --max-new-tokens 48 --quantization none --resume)

[ -f "$CD" ] || { log "no CD at $CD"; echo ROUND2_DONE; exit 1; }
[ -d "$MCB/runs/main/A_full" ] || { log "first-run baseline missing at runs/main/A_full"; echo ROUND2_DONE; exit 1; }
[ -e "$OUT/A_full" ] || ln -s "$MCB/runs/main/A_full" "$OUT/A_full"

run_cell() {  # name, then run_arms arguments
  local cell="$1"; shift
  if [ -f "$OUT/$cell/meta.json" ]; then log "cell $cell already complete"; return 0; fi
  log "cell $cell"
  python -m benchmark.mcbuild_bench.run_arms "${COMMON[@]}" "$@" \
    --out "$OUT/$cell" --gpu-csv "$OUT/$cell/gpu.csv" >> "$OUT/$cell.log" 2>&1
  local rc=$?; log "cell $cell exit=$rc"
  # a failed cell is reported and the queue moves on; the watchdog's restart retries it later
  [ "$rc" = 0 ] || { echo "CELL_FAIL $cell"; FAILED="$FAILED $cell"; }
}
FAILED=""

for W in 8000 16000 32000; do
  run_cell "truncB_W${W}" --arm B --W "$W" --w 0 --inject none
done
for W in 8000 16000 32000; do for w in 0.1 0.3 1.0; do
  run_cell "proposed_W${W}_w${w}" --arm proposed --W "$W" --w "$w" \
    --inject planet+satellites --bias-cap 3.0 --cd "$CD"
done; done
[ -z "$FAILED" ] || { log "cells that failed this pass:$FAILED"; echo ROUND2_DONE; exit 1; }
echo ROUND2_DONE

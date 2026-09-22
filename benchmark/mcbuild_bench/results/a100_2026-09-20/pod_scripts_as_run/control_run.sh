#!/usr/bin/env bash
# w = 0 control (H32, Ryosuke: "下駄なしの実験を回してから止めよう" — approved 2026-09-21).
# Same windows as the main run, same diagram, same prompt, mass vector scaled by zero: the reader
# sees exactly the correlation-diagram window the proposed cells saw, with no attention bias at all.
# It is the control the 2026-09-21 grid lacks: every cell there was monotonically better at lower w,
# so "the window helps" and "the bias hurts" cannot be told apart without this column.
# Usage on the pod:  bash control_run.sh [cd.json]
set -uo pipefail
MCB=/workspace/mcb; CD="${1:-$MCB/runs/v4_36rt/cd.json}"; OUT=$MCB/runs/main; mkdir -p "$OUT"
export HF_HOME=$MCB/hf PATH="$HOME/.local/bin:$PATH"
source /root/venvs/venv_reader/bin/activate
cd $MCB/cms-prototype
log() { printf '[%s] %s\n' "$(date -u +%T)" "$*"; }
COMMON=(--model-id Qwen/Qwen3.8-27B --questions benchmark/mcbuild_bench/data/questions.json
        --session benchmark/mcbuild_bench/data/session_redacted.json --prefill-chunk 8192
        --prefill-scale 0.0 --max-new-tokens 48 --quantization none --resume)

[ -f "$CD" ] || { log "no CD at $CD"; echo CONTROL_DONE; exit 1; }
for W in 8000 16000 32000; do
  cell="proposed_W${W}_w0.0"
  if [ -f "$OUT/$cell/meta.json" ]; then log "cell $cell already complete"; continue; fi
  log "cell $cell (no bias)"
  python -m benchmark.mcbuild_bench.run_arms "${COMMON[@]}" --arm proposed --W "$W" --w 0.0 --inject planet \
    --cd "$CD" --out "$OUT/$cell" --gpu-csv "$OUT/$cell/gpu.csv" >> "$OUT/$cell.log" 2>&1
  rc=$?; log "cell $cell exit=$rc"
  [ "$rc" = 0 ] || { echo "CELL_FAIL $cell"; echo CONTROL_DONE; exit 1; }
done
echo CONTROL_DONE

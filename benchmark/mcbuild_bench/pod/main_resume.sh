#!/usr/bin/env bash
# Resume the main run: the same cells as main_run.sh, minus the V5 gate, every one with --resume.
# A cell that already has meta.json (written only on success) is skipped; a cell with a partial
# answers.jsonl continues from its last answered question (the checkpoint identity is re-checked,
# so a cell run under different settings refuses rather than splicing).
# Usage on the pod:  bash main_resume.sh [cd.json]
set -uo pipefail
MCB=/workspace/mcb; CD="${1:-$MCB/runs/v4_36rt/cd.json}"; OUT=$MCB/runs/main; mkdir -p "$OUT"
export HF_HOME=$MCB/hf PATH="$HOME/.local/bin:$PATH"
source /root/venvs/venv_reader/bin/activate
cd $MCB/cms-prototype
log() { printf '[%s] %s\n' "$(date -u +%T)" "$*"; }
COMMON=(--model-id Qwen/Qwen3.8-27B --questions benchmark/mcbuild_bench/data/questions.json
        --session benchmark/mcbuild_bench/data/session_redacted.json --prefill-chunk 8192
        --prefill-scale 0.0 --max-new-tokens 48 --quantization none --resume)

[ -f "$CD" ] || { log "no CD at $CD"; echo MAIN_DONE; exit 1; }

for W in 8000 16000 32000; do for w in 0.1 0.3 1.0; do
  cell="proposed_W${W}_w${w}"
  if [ -f "$OUT/$cell/meta.json" ]; then log "cell $cell already complete"; continue; fi
  log "cell $cell"
  python -m benchmark.mcbuild_bench.run_arms "${COMMON[@]}" --arm proposed --W "$W" --w "$w" --inject planet \
    --cd "$CD" --out "$OUT/$cell" --gpu-csv "$OUT/$cell/gpu.csv" >> "$OUT/$cell.log" 2>&1
  rc=$?; log "cell $cell exit=$rc"
  [ "$rc" = 0 ] || { echo "CELL_FAIL $cell"; echo MAIN_DONE; exit 1; }
done; done

if [ -f "$OUT/A_full/meta.json" ]; then
  log "arm A already complete"
else
  log "arm A on all 96 questions"
  python -m benchmark.mcbuild_bench.run_arms "${COMMON[@]}" --arm A --W full --w 0 --inject none \
    --out "$OUT/A_full" --gpu-csv "$OUT/A_full/gpu.csv" >> "$OUT/A_full.log" 2>&1
  rc=$?; log "arm A exit=$rc"
  [ "$rc" = 0 ] || { echo "CELL_FAIL A_full"; echo MAIN_DONE; exit 1; }
fi
echo MAIN_DONE

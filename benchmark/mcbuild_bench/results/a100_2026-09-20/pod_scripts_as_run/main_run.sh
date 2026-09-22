#!/usr/bin/env bash
# V5 + main run (DESIGN.md §10, DECISIONS H26 w grid): reader = raw Qwen3.8-27B bf16 on the A100.
#   V5   : arm A (full transcript) on the first 5 questions — OOM / seconds-per-question check.
#   cells: proposed × W {8000,16000,32000} × w {0.1,0.3,1.0}; each writes questions_subset.json (H22 d).
#   A    : arm A once on all 96 questions; the per-W subsets are applied at scoring time (questions
#          are answered independently, greedy, so this equals running A per subset — see H29).
# Usage on the pod:  bash main_run.sh [cd.json]   (default: the 36-round-trip manager output)
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
python - "$CD" <<'PY' || { echo MAIN_DONE; exit 1; }
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print("cd summary:", d.get("summary"), "| stopped:", d.get("stopped"))
assert not d.get("stopped"), "CD is partial (stopped) — not used for the main run"
PY

log "V5: arm A on 5 questions"
python - <<'PY'
import json
q = json.load(open("benchmark/mcbuild_bench/data/questions.json", encoding="utf-8"))
json.dump(q[:5], open("/workspace/mcb/runs/questions_5.json", "w", encoding="utf-8"), ensure_ascii=False)
PY
python -m benchmark.mcbuild_bench.run_arms --arm A --W full --w 0 --inject none \
  --questions /workspace/mcb/runs/questions_5.json --session benchmark/mcbuild_bench/data/session_redacted.json \
  --model-id Qwen/Qwen3.8-27B --prefill-chunk 8192 --prefill-scale 0.0 --max-new-tokens 48 --quantization none \
  --out "$OUT/v5_A5" --gpu-csv "$OUT/v5_A5/gpu.csv" > "$OUT/v5_A5.log" 2>&1
rc=$?; log "V5 exit=$rc"; tail -3 "$OUT/v5_A5.log"
[ "$rc" = 0 ] || { echo V5_FAIL; echo MAIN_DONE; exit 1; }

for W in 8000 16000 32000; do for w in 0.1 0.3 1.0; do
  cell="proposed_W${W}_w${w}"; log "cell $cell"
  python -m benchmark.mcbuild_bench.run_arms "${COMMON[@]}" --arm proposed --W "$W" --w "$w" --inject planet \
    --cd "$CD" --out "$OUT/$cell" --gpu-csv "$OUT/$cell/gpu.csv" > "$OUT/$cell.log" 2>&1
  rc=$?; log "cell $cell exit=$rc"; tail -2 "$OUT/$cell.log"
  [ "$rc" = 0 ] || { echo "CELL_FAIL $cell"; echo MAIN_DONE; exit 1; }
done; done

log "arm A on all 96 questions"
python -m benchmark.mcbuild_bench.run_arms "${COMMON[@]}" --arm A --W full --w 0 --inject none \
  --out "$OUT/A_full" --gpu-csv "$OUT/A_full/gpu.csv" > "$OUT/A_full.log" 2>&1
rc=$?; log "arm A exit=$rc"; tail -2 "$OUT/A_full.log"
[ "$rc" = 0 ] || echo "CELL_FAIL A_full"
echo MAIN_DONE

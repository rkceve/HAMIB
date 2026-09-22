#!/usr/bin/env bash
# V4 (DESIGN.md §10): manager phase on the first N round trips — vLLM summarizer + Jev, budget-guarded.
# Usage on the pod:  bash v4_manager.sh <n_round_trips> [max_jev_input_tokens] [max_jev_requests]
# Key: /root/.typesafe_key (mode 600, never printed). Outputs under /workspace/mcb/runs/v4_<n>rt/.
set -uo pipefail
N="${1:?round trips}"; MAX_TOK="${2:-100000000}"; MAX_REQ="${3:-20000}"
MCB=/workspace/mcb; R=$MCB/runs/v4_${N}rt; mkdir -p "$R"
# 8001 is taken by RunPod's nginx proxy on this template (it answers 502/405 pages); use a free port.
PORT="${MCB_VLLM_PORT:-8123}"
export HF_HOME=$MCB/hf PATH="$HOME/.local/bin:$PATH"
export TYPESAFE_API_KEY="$(tr -d '\r\n ' < /root/.typesafe_key)"
log() { printf '[%s] %s\n' "$(date -u +%T)" "$*"; }

log "1/4 summarizer (transformers server, Qwen3.5-4B bf16, reader venv; vLLM cannot run on this driver — H27) in tmux"
tmux kill-session -t summarizer 2>/dev/null || true
tmux new -d -s summarizer "source /root/venvs/venv_reader/bin/activate && cd $MCB/cms-prototype && HF_HOME=$HF_HOME python -m benchmark.mcbuild_bench.pod.summarizer_server --model-id Qwen/Qwen3.5-4B --port $PORT --served-model-name summarizer 2>&1 | tee $R/summarizer_server.log"
up=0
for i in $(seq 1 60); do
  if curl -s "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q '"summarizer"'; then log "summarizer up after $((i*10)) s"; up=1; break; fi
  if ! tmux has-session -t summarizer 2>/dev/null; then break; fi
  sleep 10
done
if [ "$up" != 1 ]; then log "SUMMARIZER DID NOT START (nothing sent to Jev)"; tail -30 "$R/summarizer_server.log"; echo V4_DONE; exit 1; fi
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{"model":"summarizer","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' | head -c 400; echo

log "2/4 build_cd on the first $N round trips (budget: $MAX_REQ requests / $MAX_TOK input tokens)"
source /root/venvs/venv_reader/bin/activate
cd $MCB/cms-prototype
python -m benchmark.mcbuild_bench.build_cd \
  --session benchmark/mcbuild_bench/data/session_redacted.json \
  --out "$R/cd.json" --jev-accounting "$R/jev_calls.jsonl" --summarizer-accounting "$R/summarizer_calls.jsonl" \
  --summarizer-url "http://127.0.0.1:$PORT/v1" --gpu-csv "$R/gpu_manager.csv" \
  --max-round-trips "$N" --max-jev-requests "$MAX_REQ" --max-jev-input-tokens "$MAX_TOK" --run-id "v4_${N}rt" \
  > "$R/build_cd.log" 2>&1
log "build_cd exit=$?"
tail -5 "$R/build_cd.log"

log "3/4 stop the summarizer"
tmux kill-session -t summarizer 2>/dev/null || true

log "4/4 summary"
python - "$R" <<'PY'
import json, sys, glob, os
r = sys.argv[1]
cd = glob.glob(os.path.join(r, "cd*.json"))
for p in cd:
    d = json.load(open(p, encoding="utf-8"))
    print(os.path.basename(p), "| summary:", d.get("summary"), "| stopped:", d.get("stopped"))
    m = d.get("manifest", {})
    print("  jev:", {k: m.get(k) for k in ("jev_requests", "jev_input_tokens_total", "jev_output_tokens_total", "jev_cost_usd", "jev_budget")})
    print("  summarizer:", {k: m.get(k) for k in ("summarizer_calls", "summarizer_prompt_tokens", "summarizer_completion_tokens")}, "| wall_s:", m.get("wall_s"))
PY
echo V4_DONE

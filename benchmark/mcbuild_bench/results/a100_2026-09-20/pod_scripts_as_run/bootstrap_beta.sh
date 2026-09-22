#!/usr/bin/env bash
# mcbuild-bench pod bootstrap — variant β (one 80 GB pod, BF16 reader, transformers 5.17.0, vLLM 0.29.0).
# Run ONCE on a fresh RunPod PyTorch/CUDA 12.x pod as root:  bash bootstrap_beta.sh <git-ref>
# Idempotent: re-running skips finished steps. Nothing here spends Jev tokens or starts a GPU job.
set -euo pipefail
REF="${1:-mcbuild-bench}"
WORK=${MCB_WORK:-/workspace/mcb}
VENV=${MCB_VENV:-/root/venvs/venv_beta}   # venv on the root overlay disk, weights on the volume
REPO_URL="${MCB_REPO_URL:-git@github.com:rkceve/cms-prototype.git}"
export HF_HOME=$WORK/hf HF_HUB_ENABLE_HF_TRANSFER=1 PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$WORK" "$HF_HOME" "$WORK/runs" "$(dirname "$VENV")"
cd "$WORK"

log() { printf '\n[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

log "0/6 GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nvidia-smi --query-gpu=power.draw --format=csv,noheader | grep -qv 'N/A' || { echo "power.draw not readable (E3 needs it)"; exit 2; }

log "1/6 tools"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux)

log "2/6 repository @ $REF"
# private repo + proxy ssh without scp: a snapshot typed in with pod/transfer.sh (no .git) is used as is
if [ -d cms-prototype ]; then
  cd cms-prototype; git rev-parse --short HEAD 2>/dev/null || cat SNAPSHOT_REF 2>/dev/null || echo "snapshot (no git)"
else
  git clone -q "$REPO_URL" cms-prototype && cd cms-prototype && git checkout -q "$REF" && git rev-parse --short HEAD
fi

log "3/6 venv (one environment: vllm 0.29.0 pins torch 2.13.0; transformers 5.17.0)"
[ -d "$VENV" ] || uv venv -q --python 3.11 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
uv pip install -q "vllm==0.29.0" "transformers==5.17.0" "flash-linear-attention==0.5.2" accelerate tiktoken pytest jsonschema numpy safetensors huggingface_hub
uv pip install -q causal-conv1d || echo "WARN: causal-conv1d wheel/build failed (conv falls back to torch; fla kernels still used)"
python - <<'PY'
import torch, transformers, vllm
print("torch", torch.__version__, "cuda", torch.version.cuda, "| transformers", transformers.__version__, "| vllm", vllm.__version__)
from transformers.utils.import_utils import is_flash_linear_attention_available, is_causal_conv1d_available
print("fla_importable", is_flash_linear_attention_available(), "| causal_conv1d", is_causal_conv1d_available())
PY

log "4/6 weights (reader BF16 55.6 GB, summarizer 9.3 GB) — resumable"
python - <<'PY'
from huggingface_hub import snapshot_download
for rid in ("Qwen/Qwen3.8-27B", "Qwen/Qwen3.5-4B"):
    p = snapshot_download(rid, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.py"], ignore_patterns=["model_mtp*"])
    print(rid, "->", p)
PY

log "5/6 CPU-side self-tests on the pod (no GPU job yet)"
python -m pytest tests/mcbuild -q -p no:cacheprovider -x 2>&1 | tail -2
python verify_attention_math.py | tail -1

df -h / "$WORK" | tail -2
log "6/6 ready. Next (Fable over ssh, in tmux):"
cat <<'TXT'
  # summarizer (only during the manager phase):
  tmux new -d -s vllm 'source $VENV/bin/activate && vllm serve Qwen/Qwen3.5-4B --language-model-only --max-model-len 8192 --gpu-memory-utilization 0.25 --served-model-name summarizer --port 8001 --dtype bfloat16 2>&1 | tee /workspace/runs/vllm.log'
  # V2 load check + V3 injection probe, then V4 (manager, 3 round trips), V5 (baseline A, 5 questions) — see DESIGN.md §10.
TXT

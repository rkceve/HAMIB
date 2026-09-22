#!/usr/bin/env bash
# mcbuild-bench pod bootstrap — variant α (48 GB card, INT4 reader RedHatAI/Qwen3.8-27B-INT4).
# Two virtual environments because vLLM 0.29.0 needs transformers >= 5.10.4 while the INT4 checkpoint
# only stays packed in memory under transformers 5.8.0 (DESIGN.md §1; DECISIONS H4).
# Usage (as root on the pod):  MCB_REPO_URL=<clone url> bash bootstrap_alpha.sh [git-ref]
# Idempotent; logs to $MCB/bootstrap.log when run through run_bootstrap below. Spends no Jev tokens.
set -euo pipefail
REF="${1:-mcbuild-bench}"
MCB=/workspace/mcb
REPO_URL="${MCB_REPO_URL:-git@github.com:rkceve/cms-prototype.git}"
export HF_HOME=$MCB/hf HF_HUB_ENABLE_HF_TRANSFER=1 PIP_DISABLE_PIP_VERSION_CHECK=1 DEBIAN_FRONTEND=noninteractive
mkdir -p "$MCB" "$HF_HOME" $MCB/runs
cd "$MCB"
log() { printf '\n[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

log "0/7 GPU + disk"
nvidia-smi --query-gpu=name,memory.total,driver_version,power.draw --format=csv,noheader
df -h / /workspace | tail -2

log "1/7 tools (uv, tmux)"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux >/dev/null)
uv --version; tmux -V

log "2/7 repository @ $REF"
# The repo is private and the proxy ssh has no scp: a snapshot may have been typed in as a tarball
# (pod/transfer.sh) — then cms-prototype/ exists without .git and we use it as is.
if [ -d cms-prototype ]; then
  cd cms-prototype; git rev-parse --short HEAD 2>/dev/null || cat SNAPSHOT_REF 2>/dev/null || echo "snapshot (no git)"
else
  git clone -q "$REPO_URL" cms-prototype && cd cms-prototype && git checkout -q "$REF" && git rev-parse --short HEAD
fi

log "3/7 venv_reader: transformers 5.8.0 (pinned) + compressed-tensors + torch"
[ -d $MCB/venv_reader ] || uv venv -q --python 3.11 $MCB/venv_reader
# shellcheck disable=SC1091
source $MCB/venv_reader/bin/activate
uv pip install -q "torch>=2.10" --index-url https://download.pytorch.org/whl/cu128 || uv pip install -q "torch>=2.10"
uv pip install -q "transformers==5.8.0" "compressed-tensors>=0.18" "flash-linear-attention==0.5.2" accelerate tiktoken hf_transfer pytest jsonschema numpy safetensors huggingface_hub sentence-transformers
uv pip install -q causal-conv1d || echo "WARN: causal-conv1d wheel/build failed (conv falls back to torch; fla kernels still used)"
python - <<'PY'
import torch, transformers, compressed_tensors
print("reader venv: torch", torch.__version__, "cuda", torch.version.cuda, "| transformers", transformers.__version__, "| compressed-tensors", compressed_tensors.__version__)
from transformers.utils.import_utils import is_flash_linear_attention_available, is_causal_conv1d_available
print("fla_importable", is_flash_linear_attention_available(), "| causal_conv1d", is_causal_conv1d_available())
assert transformers.__version__ == "5.8.0"
PY
deactivate

log "4/7 venv_vllm: vllm 0.29.0 (summarizer server only)"
[ -d $MCB/venv_vllm ] || uv venv -q --python 3.11 $MCB/venv_vllm
# shellcheck disable=SC1091
source $MCB/venv_vllm/bin/activate
uv pip install -q "vllm==0.29.0" hf_transfer
python -c "import vllm, transformers; print('vllm venv: vllm', vllm.__version__, '| transformers', transformers.__version__)"
deactivate

log "5/7 weights (INT4 reader 18.6 GB, summarizer 9.3 GB) — resumable"
# shellcheck disable=SC1091
source $MCB/venv_reader/bin/activate
python - <<'PY'
from huggingface_hub import snapshot_download
for rid in ("RedHatAI/Qwen3.8-27B-INT4", "Qwen/Qwen3.5-4B"):
    p = snapshot_download(rid, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.py", "*.yaml"], ignore_patterns=["model_mtp*"])
    print(rid, "->", p)
PY
du -sh "$HF_HOME" | tail -1

log "6/7 CPU-side self-tests on the pod (reader venv, transformers 5.8.0)"
python -m pytest tests/mcbuild -q -p no:cacheprovider -x 2>&1 | tail -2
python verify_attention_math.py | tail -1

log "7/7 bootstrap done"
df -h / /workspace | tail -2

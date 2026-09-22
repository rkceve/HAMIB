#!/usr/bin/env bash
# mcbuild-bench pod bootstrap — variant β (one 80 GB pod, BF16 reader, transformers 5.17.0).
# Two venvs (2026-09-20, A100 pod, driver 570 = CUDA 12.8): vllm 0.29.0 pins torch 2.13+cu130, which cannot see
# the GPU on a 12.8 driver (vllm 0.27/0.28 pin torch 2.13 too); the reader venv uses torch 2.11.0+cu128 and the
# summarizer venv vllm 0.26.0 (pins torch 2.11.0). No cu128 wheel of torch 2.13 exists (checked 2026-09-20).
# Run ONCE on a fresh RunPod PyTorch/CUDA 12.x pod as root:  bash bootstrap_beta.sh <git-ref>
# Idempotent: re-running skips finished steps. Nothing here spends Jev tokens or starts a GPU job.
set -euo pipefail
REF="${1:-mcbuild-bench}"
WORK=${MCB_WORK:-/workspace/mcb}
VENV=${MCB_VENV:-/root/venvs/venv_reader}   # reader venv on the root overlay disk, weights on the volume
VENV_VLLM=${MCB_VENV_VLLM:-/root/venvs/venv_vllm}
REPO_URL="${MCB_REPO_URL:-git@github.com:<owner>/<repo>.git}"
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

log "3/6 venvs: reader (torch 2.11.0+cu128, transformers 5.17.0) and summarizer (vllm 0.26.0, torch 2.11.0+cu128)"
[ -d "$VENV" ] || uv venv -q --python 3.11 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
uv pip install -q "torch==2.11.0" "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cu128
uv pip install -q "transformers==5.17.0" "flash-linear-attention==0.5.2" accelerate tiktoken pytest jsonschema numpy safetensors huggingface_hub
# causal_conv1d (H31): without it the 48 linear-attention layers run the reference PyTorch
# convolution and prefill is ~809 tok/s on an A100.  No wheel exists for torch 2.11, so it is built
# from the GitHub source tree; `uv pip` (NOT `pip`, which in a uv venv is /usr/local/bin/pip and
# builds against the SYSTEM torch) with --no-build-isolation, sm_80 only to keep the build short.
uv pip install -q ninja packaging setuptools wheel
( set -e
  CCD=$WORK/ccd_src; mkdir -p "$CCD"; cd "$CCD"
  [ -d causal-conv1d-1.5.0.post8 ] || { curl -sL -o v.tar.gz       https://github.com/Dao-AILab/causal-conv1d/archive/refs/tags/v1.5.0.post8.tar.gz && tar xzf v.tar.gz; }
  cd causal-conv1d-1.5.0.post8
  CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=64 TORCH_CUDA_ARCH_LIST="8.0" CUDA_HOME=/usr/local/cuda     PATH="/usr/local/cuda/bin:$PATH" uv pip install . --no-build-isolation --no-cache
) || echo "WARN: causal-conv1d build failed (conv falls back to torch; fla kernels still used)"
python - <<'PY'
import torch, transformers
assert torch.cuda.is_available(), "torch cannot see the GPU: wrong CUDA build for this driver"
print("reader venv: torch", torch.__version__, "cuda", torch.version.cuda, "| transformers", transformers.__version__)
from transformers.utils.import_utils import is_flash_linear_attention_available, is_causal_conv1d_available
print("fla_importable", is_flash_linear_attention_available(), "| causal_conv1d", is_causal_conv1d_available())
PY
deactivate
[ -d "$VENV_VLLM" ] || uv venv -q --python 3.11 "$VENV_VLLM"
# shellcheck disable=SC1091
source "$VENV_VLLM/bin/activate"
uv pip install -q "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0" --index-url https://download.pytorch.org/whl/cu128
uv pip install -q "vllm==0.26.0"
python -c "import torch, vllm; assert torch.cuda.is_available(); print('vllm venv:', vllm.__version__, 'torch', torch.__version__, torch.version.cuda)"
deactivate
# shellcheck disable=SC1091
source "$VENV/bin/activate"

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
  tmux new -d -s vllm 'source $VENV_VLLM/bin/activate && vllm serve Qwen/Qwen3.5-4B --language-model-only --max-model-len 8192 --gpu-memory-utilization 0.25 --served-model-name summarizer --port 8123 --dtype bfloat16 2>&1 | tee $WORK/runs/vllm.log'   # 8001 is RunPod's nginx proxy
  # V2 load check + V3 injection probe, then V4 (manager, 3 round trips), V5 (baseline A, 5 questions) — see DESIGN.md §10.
TXT

#!/usr/bin/env bash
# Pod bootstrap for the second measurement round (reader only: no summarizer, no vLLM, no judge).
#
# Same pins as the first run so the reader is the same program: torch 2.11.0+cu128,
# transformers 5.17.0, flash-linear-attention 0.5.2, causal_conv1d 1.5.0.post8 built for sm_80.
# The cu128 build also runs on a newer driver (the 2026-10-03 pod has driver 580 / CUDA 13.0).
#
# Order is chosen for wall time: the venv first, then the 55.6 GB weight download in the
# background while the convolution kernel compiles, then the inputs, then the CPU self-test.
# Run ONCE as root on a fresh pod:  bash bootstrap_round2.sh      (idempotent; safe to rerun)
# Expects, if already uploaded: /workspace/mcb/round2_inputs.tgz (cd.json + first-run A_full)
#                               /workspace/mcb/cms-prototype      (repo snapshot, for the self-test)
set -uo pipefail
WORK=/workspace/mcb; VENV=/root/venvs/venv_reader
export HF_HOME=$WORK/hf HF_HUB_ENABLE_HF_TRANSFER=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
mkdir -p "$WORK/runs" "$HF_HOME" "$(dirname "$VENV")"
log() { printf '\n[%s] %s\n' "$(date -u +%T)" "$*"; }

log "0/5 GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version,power.draw --format=csv,noheader

log "1/5 reader venv"
[ -d "$VENV" ] || uv venv -q --python 3.11 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
uv pip install -q "torch==2.11.0" "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cu128
uv pip install -q "transformers==5.17.0" "flash-linear-attention==0.5.2" accelerate tiktoken pytest \
  jsonschema numpy scipy safetensors huggingface_hub hf_transfer ninja packaging setuptools wheel
python - <<'PY'
import torch, transformers
assert torch.cuda.is_available(), "torch cannot see the GPU: wrong CUDA build for this driver"
print("torch", torch.__version__, "cuda", torch.version.cuda, "| transformers", transformers.__version__,
      "| gpu", torch.cuda.get_device_name(0))
PY

log "2/5 weights: Qwen/Qwen3.8-27B (55.6 GB), downloading in the background -> $WORK/runs/weights.log"
( python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen3.8-27B",
                      allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.py"],
                      ignore_patterns=["model_mtp*"])
print("WEIGHTS_READY", p, flush=True)
PY
) > "$WORK/runs/weights.log" 2>&1 &
WPID=$!

log "3/5 causal_conv1d from the GitHub source tree (sm_80 only) -> $WORK/runs/ccd_build.log"
( set -e
  CCD=$WORK/ccd_src; mkdir -p "$CCD"; cd "$CCD"
  [ -d causal-conv1d-1.5.0.post8 ] || { curl -sL -o v.tar.gz \
      https://github.com/Dao-AILab/causal-conv1d/archive/refs/tags/v1.5.0.post8.tar.gz && tar xzf v.tar.gz; }
  cd causal-conv1d-1.5.0.post8
  CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=64 TORCH_CUDA_ARCH_LIST="8.0" CUDA_HOME=/usr/local/cuda \
    uv pip install -q . --no-build-isolation --no-cache
) > "$WORK/runs/ccd_build.log" 2>&1 && echo "causal_conv1d: built" \
  || echo "WARN: causal_conv1d build failed (reference conv is used; it made no speed difference last time)"
python -c "from transformers.utils.import_utils import is_causal_conv1d_available as f; print('causal_conv1d visible:', f())"

log "4/5 inputs from the first run"
if [ -f "$WORK/round2_inputs.tgz" ]; then
  tar xzf "$WORK/round2_inputs.tgz" -C "$WORK" && ls "$WORK/runs/v4_36rt" "$WORK/runs/main/A_full"
else
  echo "round2_inputs.tgz not uploaded yet"
fi

log "   waiting for the weights"
wait "$WPID"; tail -1 "$WORK/runs/weights.log"

log "5/5 CPU self-test"
if [ -d "$WORK/cms-prototype" ]; then
  cd "$WORK/cms-prototype"
  python -m pytest tests/mcbuild -q -p no:cacheprovider -x 2>&1 | tail -1
  python verify_attention_math.py 2>&1 | tail -1
else
  echo "repo snapshot not in place yet; run the self-test after the transfer"
fi
df -h / "$WORK" | tail -2
echo BOOTSTRAP_DONE

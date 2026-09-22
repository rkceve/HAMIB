source /root/venvs/venv_reader/bin/activate
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=64 TORCH_CUDA_ARCH_LIST="8.0" CUDA_HOME=/usr/local/cuda
cd /workspace/mcb/ccd_src/causal-conv1d-1.5.0.post8
echo "=== start $(date -u +%T)"
uv pip install . --no-build-isolation --no-cache
echo "=== exit=$? $(date -u +%T)"
python - <<"PY"
import torch
from transformers.utils.import_utils import is_causal_conv1d_available
from causal_conv1d import causal_conv1d_fn
x = torch.randn(1, 8, 32, device="cuda", dtype=torch.bfloat16)
w = torch.randn(8, 4, device="cuda", dtype=torch.bfloat16)
print("forward ok, out", tuple(causal_conv1d_fn(x, w).shape))
print("transformers sees it:", is_causal_conv1d_available())
PY
echo CCD_DONE

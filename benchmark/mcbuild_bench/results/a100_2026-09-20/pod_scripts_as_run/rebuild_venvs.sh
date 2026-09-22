set -uo pipefail
export HF_HOME=/workspace/mcb/hf PATH="$HOME/.local/bin:$PATH" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/mcb; R=/workspace/mcb/runs
echo "[$(date -u +%T)] driver 570 = CUDA 12.8 -> torch must be a cu128 build. venv_reader: torch 2.11.0+cu128 + transformers 5.17.0"
rm -rf /root/venvs/venv_reader; uv venv -q --python 3.11 /root/venvs/venv_reader
source /root/venvs/venv_reader/bin/activate
uv pip install -q "torch==2.11.0" "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2
uv pip install -q "transformers==5.17.0" "flash-linear-attention==0.5.2" accelerate tiktoken pytest jsonschema numpy safetensors huggingface_hub 2>&1 | tail -2
uv pip install -q causal-conv1d 2>&1 | tail -1 || echo "WARN causal-conv1d"
python -c "import torch,transformers;print(\"reader:\",torch.__version__,\"cuda\",torch.version.cuda,\"available\",torch.cuda.is_available(),\"|\",transformers.__version__)"
python -c "from transformers.utils.import_utils import is_flash_linear_attention_available as f, is_causal_conv1d_available as c; print(\"fla\",f(),\"conv1d\",c())"
cd /workspace/mcb/cms-prototype
python -m pytest tests/mcbuild -q -p no:cacheprovider 2>&1 | tail -2
python verify_attention_math.py | tail -1
deactivate
echo "[$(date -u +%T)] venv_vllm: vllm 0.28.0 (torch 2.11 era, cu128)"
rm -rf /root/venvs/venv_vllm /root/venvs/venv_beta; uv venv -q --python 3.11 /root/venvs/venv_vllm
source /root/venvs/venv_vllm/bin/activate
uv pip install -q "vllm==0.28.0" 2>&1 | tail -2
python -c "import torch,vllm,transformers;print(\"vllm venv:\",vllm.__version__,\"torch\",torch.__version__,\"cuda\",torch.version.cuda,\"available\",torch.cuda.is_available(),\"| transformers\",transformers.__version__)"
deactivate
df -h / /workspace | tail -2
echo "[$(date -u +%T)] REBUILD_DONE"

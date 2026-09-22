set -uo pipefail
export PATH="$HOME/.local/bin:$PATH" HF_HOME=/workspace/mcb/hf
echo "[$(date -u +%T)] venv_vllm: vllm 0.26.0 (pins torch 2.11.0 -> cu128 wheels usable on the CUDA 12.8 driver)"
rm -rf /root/venvs/venv_vllm; uv venv -q --python 3.11 /root/venvs/venv_vllm
source /root/venvs/venv_vllm/bin/activate
uv pip install -q "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0" --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2
uv pip install -q "vllm==0.26.0" 2>&1 | tail -2
python -c "import torch,vllm,transformers;print(\"vllm venv:\",vllm.__version__,\"torch\",torch.__version__,\"cuda\",torch.version.cuda,\"available\",torch.cuda.is_available(),\"| transformers\",transformers.__version__); torch.cuda.set_device(0); print(\"cuda init ok\")" 2>&1 | tail -2
echo "[$(date -u +%T)] smoke: serve Qwen3.5-4B on 8123 for 8 minutes max"
(timeout 480 vllm serve Qwen/Qwen3.5-4B --language-model-only --max-model-len 8192 --gpu-memory-utilization 0.25 --served-model-name summarizer --port 8123 --dtype bfloat16 > /workspace/mcb/runs/vllm_smoke.log 2>&1 &)
for i in $(seq 1 45); do curl -s http://127.0.0.1:8123/v1/models 2>/dev/null | grep -q "\"summarizer\"" && { echo "server up after $((i*10)) s"; break; }; sleep 10; done
curl -s http://127.0.0.1:8123/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"summarizer\",\"messages\":[{\"role\":\"user\",\"content\":\"Rewrite as one sentence: The demo server runs Paper 1.21.8 on Java 21.\"}],\"max_tokens\":40,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" | head -c 600; echo
pkill -f "vllm serve" 2>/dev/null; sleep 3
grep -i -m3 "error\|exception" /workspace/mcb/runs/vllm_smoke.log | cut -c1-200
echo "[$(date -u +%T)] REBUILD_VLLM_DONE"

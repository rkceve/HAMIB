#!/usr/bin/env bash
set -uo pipefail
MCB=/workspace/mcb; R=$MCB/runs; mkdir -p "$R"
exec > >(tee -a $R/queue.log) 2>&1
echo "[$(date -u +%FT%TZ)] queue started (waits for the venv rebuild)"
while ! grep -q "REBUILD_DONE" $R/rebuild.log 2>/dev/null; do sleep 30; done
if grep -q "available False\|failed\|Error" $R/rebuild.log; then echo "REBUILD PROBLEM:"; grep -E "available|failed|Error|passed" $R/rebuild.log | tail -6; fi
export HF_HOME=$MCB/hf PATH="$HOME/.local/bin:$PATH" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /root/venvs/venv_reader/bin/activate
cd $MCB/cms-prototype
echo "[$(date -u +%FT%TZ)] V2 start (bf16)"
python -m benchmark.mcbuild_bench.pod.v2_load_check --model-id Qwen/Qwen3.8-27B --quantization none --max-packed-gb 70 --out $R/v2.json > $R/v2.log 2>&1; echo "V2 exit=$?"
grep -E "V2 |\"pass\"|peak_mem|forward_512_s|gen_2000|\"error\"|Error:|loaded_class|loading_info|answer" $R/v2.log | grep -v "Loading weights" | tail -16
if grep -q "V2 PASS" $R/v2.log; then
  echo "[$(date -u +%FT%TZ)] V3 start"
  python -m benchmark.mcbuild_bench.pod.v3_probe --model-id Qwen/Qwen3.8-27B --quantization none --w-grid 0,0.1,0.3,1,3,10 --out $R/v3.json > $R/v3.log 2>&1; echo "V3 exit=$?"
  grep -E "^\{|V3 |Error|Traceback|prompt tokens" $R/v3.log | grep -v "Loading weights" | tail -12
else
  echo "V2 failed; V3 skipped"; grep -E "Error|Traceback" $R/v2.log | tail -5
fi
echo "[$(date -u +%FT%TZ)] queue finished"; echo QUEUE_END

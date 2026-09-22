"""V2 (DESIGN.md §10): load the reader on the pod and check the facts the experiment relies on.

Run inside venv_reader from the repository root:
    python -m benchmark.mcbuild_bench.pod.v2_load_check --model-id RedHatAI/Qwen3.8-27B-INT4 --out /workspace/mcb/runs/v2.json

Pass rules (variant α, 46 GB card): attn_implementation == "sdpa"; 16 full-attention layers reachable;
flash-linear-attention importable; loading_info missing == mismatched == 0; peak memory after one
512-token forward < 24 GB (INT4 weights stay packed under transformers 5.8.0); seconds per generated
token measured on a 2,000-token prompt. Nothing here touches Jev.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from benchmark.bineval.run_reader import (
    build_prompt,
    check_model_supported,
    linear_attention_kernel_report,
    load_reader,
)

GB = 1024 ** 3


def gb(x: int) -> float:
    return round(x / GB, 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quantization", choices=("none", "nf4", "fp4"), default="none")
    ap.add_argument("--max-packed-gb", type=float, default=24.0)
    args = ap.parse_args()

    report: dict = {"model_id": args.model_id, "torch": torch.__version__}
    import transformers

    report["transformers"] = transformers.__version__
    report["kernels"] = linear_attention_kernel_report()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    llm = load_reader(args.model_id, max_new_tokens=8, w=0.0, quantization=args.quantization)
    report["load_s"] = round(time.perf_counter() - t0, 1)
    report["loaded_class_name"] = llm.loaded_class_name
    report["loading_info"] = llm.loading_info
    report["attn_implementation"] = llm.attn_implementation
    layer_info = check_model_supported(llm._model.config, allow_linear_layers=True)
    report["n_sdpa_layers"] = layer_info["n_sdpa_layers"]
    report["n_linear_layers"] = layer_info["n_linear_layers"]
    linear_classes = sorted({type(m).__name__ for m in llm._model.modules() if isinstance(m, torch.nn.Linear)})
    report["linear_module_classes"] = linear_classes  # CompressedLinear expected for INT4 under 5.8.0
    report["mem_after_load_gb"] = gb(torch.cuda.memory_allocated())

    filler = "the quick brown fox jumps over the lazy dog. " * 55  # ≈ 512 tokens
    prompt = build_prompt(filler, "What animal jumps?")
    t0 = time.perf_counter()
    out = llm.generate(prompt)
    report["forward_512_s"] = round(time.perf_counter() - t0, 2)
    report["answer_512"] = out[:80]
    report["peak_mem_after_512_gb"] = gb(torch.cuda.max_memory_allocated())
    report["stats_after_512"] = llm.mass_injection_stats()
    report["last_generated_tokens"] = llm.last_generated_tokens
    report["last_prefill_ms"] = llm.last_prefill_ms
    report["last_decode_ms"] = llm.last_decode_ms

    long_filler = "the quick brown fox jumps over the lazy dog. " * 220  # ≈ 2,000 tokens
    llm._max_new_tokens = 32
    t0 = time.perf_counter()
    out = llm.generate(build_prompt(long_filler, "What animal jumps? Answer in one word."))
    total = time.perf_counter() - t0
    n = llm.last_generated_tokens or 1
    report["gen_2000_total_s"] = round(total, 2)
    report["gen_2000_prefill_ms"] = llm.last_prefill_ms
    report["gen_2000_s_per_token"] = round((llm.last_decode_ms or 0) / 1000.0 / max(1, n - 1), 3)
    report["peak_mem_gb"] = gb(torch.cuda.max_memory_allocated())

    checks = {
        "sdpa": report["attn_implementation"] == "sdpa",
        "n_sdpa_layers_16": report["n_sdpa_layers"] == 16,
        "fla_importable": bool(report["kernels"].get("fla_importable")),
        "no_missing_weights": (report["loading_info"] or {}).get("missing", 1) == 0
        and (report["loading_info"] or {}).get("mismatched", 1) == 0,
        "packed_memory": report["peak_mem_after_512_gb"] < args.max_packed_gb,
    }
    report["checks"] = checks
    report["pass"] = all(checks.values())
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False)
    print(json.dumps(report, indent=1, ensure_ascii=False))
    print("V2 PASS" if report["pass"] else "V2 FAIL")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

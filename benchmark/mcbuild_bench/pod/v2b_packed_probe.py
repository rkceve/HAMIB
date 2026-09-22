"""V2b: can the INT4 reader run WITHOUT the compressed-tensors first-forward decompression?

Facts from the pod (2026-09-20): transformers 5.8.0 + compressed-tensors 0.18 loaded
RedHatAI/Qwen3.8-27B-INT4 with 0 missing weights, then the first generate() ran
"Decompressing model" (the pre-forward hook registered by ModelCompressor.compress_model)
and went out of memory at 44 GB on the 46 GB A40: the hook rebuilds the whole model in BF16.

This probe loads the model, reports the module classes compressed-tensors installed and
whether the decompression hook is present, REMOVES the hook, and runs one 512-token forward.
If the linear modules are CompressedLinear (per-forward dequantization) the forward succeeds
with the weights still packed (< 24 GB). Nothing here touches Jev.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from benchmark.bineval.run_reader import build_prompt, load_reader

GB = 1024 ** 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quantization", choices=("none", "nf4", "fp4"), default="none")
    args = ap.parse_args()
    rep: dict = {"model_id": args.model_id}
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    llm = load_reader(args.model_id, max_new_tokens=8, w=0.0, quantization=args.quantization)
    rep["load_s"] = round(time.perf_counter() - t0, 1)
    model = llm._model
    rep["loaded_class_name"] = llm.loaded_class_name
    rep["loading_info"] = llm.loading_info
    rep["mem_after_load_gb"] = round(torch.cuda.memory_allocated() / GB, 2)
    classes: dict[str, int] = {}
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            classes[type(m).__name__] = classes.get(type(m).__name__, 0) + 1
    rep["linear_module_classes"] = classes
    hooks = []
    for name, mod in model.named_modules():
        if hasattr(mod, "ct_decompress_hook"):
            hooks.append(name or "<root>")
    rep["decompress_hook_on"] = hooks
    # the hook is a forward PRE-hook on the root module: remove it so the first forward
    # does not rebuild the model in bf16
    removed = 0
    for mod in model.modules():
        h = getattr(mod, "ct_decompress_hook", None)
        if h is not None:
            h.remove()
            delattr(mod, "ct_decompress_hook")
            removed += 1
    rep["hooks_removed"] = removed
    rep["forward_pre_hooks_root"] = len(getattr(model, "_forward_pre_hooks", {}))
    # one Linear of a full-attention layer: what parameters does it hold?
    for name, mod in model.named_modules():
        if name.endswith("layers.3.self_attn.q_proj"):
            rep["q_proj_params"] = {n: [list(p.shape), str(p.dtype)] for n, p in mod.named_parameters(recurse=False)}
            rep["q_proj_class"] = type(mod).__name__
            break
    filler = "the quick brown fox jumps over the lazy dog. " * 55
    t0 = time.perf_counter()
    try:
        out = llm.generate(build_prompt(filler, "What animal jumps?"))
        rep["forward_512_s"] = round(time.perf_counter() - t0, 2)
        rep["answer"] = out[:80]
        rep["ok"] = True
    except torch.OutOfMemoryError as exc:  # noqa: F841
        rep["ok"] = False
        rep["error"] = "OOM: " + str(exc)[:300]
    except Exception as exc:  # noqa: BLE001
        rep["ok"] = False
        rep["error"] = type(exc).__name__ + ": " + str(exc)[:300]
    rep["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / GB, 2)
    rep["mem_now_gb"] = round(torch.cuda.memory_allocated() / GB, 2)
    for name, mod in model.named_modules():
        if name.endswith("layers.3.self_attn.q_proj"):
            rep["q_proj_params_after"] = {n: [list(p.shape), str(p.dtype)] for n, p in mod.named_parameters(recurse=False)}
            break
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, ensure_ascii=False)
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    print("V2b PACKED-OK" if rep.get("ok") and rep["peak_mem_gb"] < 24 else "V2b FAIL")
    return 0 if rep.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Answer the questions with a Hugging Face reader, optionally with mass injection.

Per question: ``build_prompt`` puts the arm's context and the question into
``READER_PROMPT`` (only the context differs between arms) ->
``build_reader_mass_vector`` finds the planet markers (``[PN ...]``) in the
prompt tokens and turns them into a per-token mass vector -> the model
generates greedily while the patched ``F.scaled_dot_product_attention`` adds
``w * mass`` to the attention scores -> ``first_line_answer`` extracts the
answer.  ``w = 0`` runs the same code path with a zero bias, so the text-only
and injected arms differ in exactly one number.  ``write_answers`` saves
``{qid: answer}`` plus a ``<out>.meta.json`` sidecar for auditing the run.

``check_model_supported`` and ``check_context_fits`` refuse an unusable model
or an impossible cell from the config alone, before any weights are
downloaded.  Everything except ``load_reader`` / ``run_reader`` runs on a CPU
machine; importing this module loads no model.

CLI (GPU host only)::

    python -m benchmark.bineval.run_reader \
        --model-id Qwen/Qwen3.8-27B \
        --questions benchmark/bineval/questions_restaurant.json \
        --context-file ctx.txt --out answers/cd_mass_6x_w0.25.json \
        --w 0.25 --inject planet

Scoring is local and separate::

    python -m benchmark.bineval.score_binary --answers <answers.json> \
        --questions benchmark/bineval/questions_restaurant.json \
        --out <report.json> --subset generated --max-words 32

``--subset generated`` is required there.  The reader answers only the
generated subset (173 of the 189 non-excluded questions); the scorer's own
default, ``--subset all``, would count the other 16 as failures and report a
pass rate about 8 points too low.
"""

from __future__ import annotations

import argparse
import json
import re
import math
import struct
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from server.cd_parser import find_marker_spans, marker_positions

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
THINK_BLOCK_RE = re.compile(r"^\s*<think>.*?</think>", re.DOTALL)

# Identical for every arm.  Do not edit: published results used this exact text.
READER_PROMPT = (
    "{context_block}\n\n"
    "Answer the question using only the information above. Reply with the "
    "answer only, in a few words. If the information is not present, reply: "
    "unknown.\n"
    "Question: {question}\n"
    "Answer:"
)

INJECT_MODES = ("planet", "planet+satellites", "none")


def chat_wrap(tokenizer: Any, prompt: str) -> str:
    """``prompt`` as one user turn of the model's chat template, thinking off.

    Given the raw prompt, Qwen3.8-27B opens a ``<think>`` block and spends the
    whole 48-token budget inside it, so no answer appears.  With
    ``enable_thinking=False`` the template closes the thinking block itself and
    generation starts on the answer.  A tokenizer without a chat template gets
    the raw prompt back (which is still the same for every arm).
    """
    apply = getattr(tokenizer, "apply_chat_template", None)
    if apply is None or not getattr(tokenizer, "chat_template", None):
        return prompt
    messages = [{"role": "user", "content": prompt}]
    try:
        return apply(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:  # template without the enable_thinking switch
        return apply(messages, tokenize=False, add_generation_prompt=True)


def build_prompt(context_block: str, question: str, *, tokenizer: Any = None) -> str:
    """The reader prompt; only ``context_block`` differs between arms.

    With ``tokenizer`` the prompt is wrapped in its chat template
    (``chat_wrap``).  Token budgets must be measured on this same final string,
    so every caller that counts tokens passes the tokenizer too.
    """
    prompt = READER_PROMPT.format(context_block=context_block, question=question)
    return prompt if tokenizer is None else chat_wrap(tokenizer, prompt)


# --------------------------------------------------------------------------
# mass vector (pure, CPU-testable with a fake tokenizer)
# --------------------------------------------------------------------------

@dataclass
class MassVectorInfo:
    """Marker-scan counts for one prompt, recorded per question in the meta."""

    positions_found: int = 0
    spans: int = 0
    planet_spans: int = 0
    satellite_spans: int = 0
    inject: str = "none"
    cap: float | None = None


def build_reader_mass_vector(
    prompt_ids: Sequence[int],
    tokenizer: Any,
    inject: str,
    cap: float | None = None,
    *,
    w: float = 1.0,
    prompt_text: str | None = None,
    device: Any = None,
) -> tuple[Any, MassVectorInfo]:
    """Find the CD markers in the prompt and build the mass vector.

    Returns ``(vector or None, info)``.  ``inject`` is one of INJECT_MODES:
    "planet" puts each planet's mass on the tokens of its text,
    "planet+satellites" also gives each satellite its planet's mass, and "none"
    skips the scan (baselines).  ``cap`` optionally caps each element (None =
    no cap: the specification fixes no cap, and ``w`` is the only knob).

    Raises RuntimeError when ``w > 0`` and the prompt contains ``[PN`` but no
    injectable position was found: the injected arm would silently become the
    text-only baseline.
    """
    if inject not in INJECT_MODES:
        raise ValueError("inject must be one of %r, got %r" % (INJECT_MODES, inject))
    if inject == "none":
        return None, MassVectorInfo(inject="none", cap=cap)

    ids = list(prompt_ids)
    spans = find_marker_spans(ids, tokenizer)
    info = MassVectorInfo(
        spans=len(spans),
        planet_spans=sum(1 for lvl, _m, _p in spans if lvl == "planet"),
        satellite_spans=sum(1 for lvl, _m, _p in spans if lvl == "satellite"),
        inject=inject,
        cap=cap,
    )

    positions = marker_positions(
        spans,
        inject_levels={"planet"},
        satellite_inherit=(inject == "planet+satellites"),
    )
    info.positions_found = len(positions)

    if w > 0 and not positions:
        if prompt_text is None:
            prompt_text = tokenizer.decode(ids, skip_special_tokens=False)
        if "[PN" in prompt_text:
            raise RuntimeError(
                "mass injection requested (w=%g, inject=%s) but the marker scan "
                "found 0 positions while the prompt contains '[PN' "
                "(spans=%d planets=%d): refusing to run a silent baseline"
                % (w, inject, len(spans), info.planet_spans)
            )

    if not positions:
        return None, info

    from server.mass_vector import positions_to_mass_vector  # imports torch

    vec = positions_to_mass_vector(
        positions,
        len(ids),
        cap=float("inf") if cap is None else float(cap),
        scale=1.0,
        device=device,
    )
    return vec, info


# --------------------------------------------------------------------------
# pre-flight model check (pure, CPU-testable with fake configs)
# --------------------------------------------------------------------------

FULL_ATTENTION = "full_attention"
LINEAR_ATTENTION = "linear_attention"
# Running the injection on only some layers must be an explicit choice, so a
# model with linear-attention layers is refused by default.  The chosen hybrid
# reader is the exception (see default_allow_linear_layers).
DEFAULT_ALLOW_LINEAR_LAYERS = False

# The chosen reader (also used as the judge).  Its text_config.layer_types
# repeats 3 x linear_attention + 1 x full_attention over 64 layers (checked
# against transformers 5.8), so the sdpa patch reaches 16 layers and skips 48.
HYBRID_MODEL_ID = "Qwen/Qwen3.8-27B"
HYBRID_N_SDPA_LAYERS = 16
HYBRID_N_LINEAR_LAYERS = 48
HYBRID_NUM_HIDDEN_LAYERS = 64


def is_hybrid_model_id(model_id: str | None) -> bool:
    """Is this the hybrid reader?

    Compared case-insensitively and without a revision suffix, so
    ``qwen/qwen3.8-27b`` and ``Qwen/Qwen3.8-27B@main`` both match.
    """
    if not model_id:
        return False
    head = str(model_id).split("@", 1)[0].rstrip("/")
    return head.lower() == HYBRID_MODEL_ID.lower()


def default_allow_linear_layers(model_id: str | None) -> bool:
    """The ``--allow-linear-layers`` default for ``model_id``.

    True only for the hybrid reader: it is the default reader, and refusing it
    would force a flag onto every run.  The 16/64 split stays visible
    (``n_sdpa_layers`` in every meta sidecar, ``n_sdpa_layers_expected`` in
    run_info.json), so every result must be read as a 16-of-64-layer
    intervention.  Any other hybrid model still fails loudly.
    """
    return is_hybrid_model_id(model_id)


def expected_n_sdpa_layers(model_id: str | None) -> int | None:
    """How many layers the patch must reach, from the model id alone.

    16 for the hybrid reader; None ("nothing to assert") for any other model.
    """
    return HYBRID_N_SDPA_LAYERS if is_hybrid_model_id(model_id) else None


def linear_attention_kernel_report() -> dict:
    """Can this host use the fast kernels for Qwen's linear-attention layers?

    transformers uses them only when flash-linear-attention (fla >= 0.2.2) and
    causal-conv1d are importable and CUDA is available; otherwise it falls back,
    with a warning, to a slower pure-torch implementation that gives the same
    results.

    Reported, never asserted: the fallback only changes the speed (and,
    marginally, the numerics) of the 48 linear layers, which the mass patch
    never touches.
    """
    report: dict[str, Any] = {
        "fla_importable": False,
        "fla_version": None,
        "causal_conv1d_importable": False,
        "torch_cuda_available": None,
        "fast_path_available": None,
        "would_use_torch_fallback": None,
        "error": None,
    }
    try:
        from transformers.utils.import_utils import (
            is_causal_conv1d_available,
            is_flash_linear_attention_available,
        )
    except Exception as exc:  # transformers absent / renamed the helpers
        report["error"] = "%s: %s" % (type(exc).__name__, exc)
        return report

    try:
        import torch  # noqa: PLC0415

        report["torch_cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        report["torch_cuda_available"] = False

    report["fla_importable"] = bool(is_flash_linear_attention_available())
    report["causal_conv1d_importable"] = bool(is_causal_conv1d_available())
    if report["fla_importable"]:
        try:
            from importlib.metadata import version as _pkg_version  # noqa: PLC0415

            report["fla_version"] = _pkg_version("flash-linear-attention")
        except Exception:
            report["fla_version"] = None
    fast = report["fla_importable"] and report["causal_conv1d_importable"]
    report["fast_path_available"] = fast
    report["would_use_torch_fallback"] = not fast
    return report


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def resolve_text_config(model_config: Any) -> Any:
    """The sub-config holding the language model's attention settings.

    A vision-language wrapper (e.g. transformers' ``Qwen3VLConfig``) has no
    ``num_hidden_layers`` / ``max_position_embeddings`` at its top level; they
    live in ``text_config``.  Tries ``get_text_config()`` first, then a
    ``text_config`` attribute or key (plain dicts and test fakes).  A text-only
    config resolves to itself.
    """
    def has_layers(cfg: Any) -> bool:
        return cfg is not None and _cfg_get(cfg, "num_hidden_layers") is not None

    getter = getattr(model_config, "get_text_config", None)
    if callable(getter):
        try:
            sub_cfg = getter()
        except Exception:  # a fake / partial config must not break the check
            sub_cfg = None
        if has_layers(sub_cfg):
            return sub_cfg
    sub_cfg = _cfg_get(model_config, "text_config")
    return sub_cfg if has_layers(sub_cfg) else model_config


def layer_type_summary(layer_types: Any) -> dict[str, int]:
    """``["full_attention", "linear_attention", ...]`` -> ``{type: count}``."""
    return dict(Counter(str(t) for t in layer_types or ()))


def check_model_supported(
    model_config: Any, *, allow_linear_layers: bool = DEFAULT_ALLOW_LINEAR_LAYERS
) -> dict:
    """Check which attention layers the mass patch can reach; raise if unusable.

    Sliding-window attention is always refused: the mass vector is indexed by
    absolute position, and a sliding-window layer truncates the KV cache, so
    ``mass_vector[j]`` would land on an unrelated token.

    ``linear_attention`` layers never call ``F.scaled_dot_product_attention``,
    so the bias cannot reach them.  A model that has them is accepted only with
    ``allow_linear_layers=True``, and the number of layers the bias does reach
    is reported as ``n_sdpa_layers``.  Anything that compares
    ``bias_applied_calls`` with a layer count must use ``n_sdpa_layers``, never
    ``num_hidden_layers``.

    Returns the inspected values for the meta sidecar.
    """
    # A vision-language wrapper has none of these keys at its top level and
    # would pass vacuously.
    model_config = resolve_text_config(model_config)
    layer_types = _cfg_get(model_config, "layer_types")
    use_sliding = _cfg_get(model_config, "use_sliding_window")
    sliding_window = _cfg_get(model_config, "sliding_window")
    sliding_pattern = _cfg_get(model_config, "sliding_window_pattern")
    n_hidden = _cfg_get(model_config, "num_hidden_layers")
    summary = layer_type_summary(layer_types)
    if layer_types:
        n_sdpa: int | None = summary.get(FULL_ATTENTION, 0)
        n_linear = summary.get(LINEAR_ATTENTION, 0)
    else:
        n_sdpa = int(n_hidden) if n_hidden is not None else None
        n_linear = 0
    info = {
        "model_type": _cfg_get(model_config, "model_type"),
        "layer_types": list(layer_types) if layer_types else None,
        "layer_types_summary": summary or None,
        "use_sliding_window": use_sliding,
        "sliding_window": sliding_window,
        "sliding_window_pattern": sliding_pattern,
        # Lets a later reader see whether a context came close to the limit.
        "max_position_embeddings": _cfg_get(model_config, "max_position_embeddings"),
        "num_hidden_layers": n_hidden,
        "n_sdpa_layers": n_sdpa,  # the layers the patch can actually bias
        "n_linear_layers": n_linear,
        "allow_linear_layers": bool(allow_linear_layers),
    }

    if layer_types:
        allowed = {FULL_ATTENTION}
        if allow_linear_layers:
            allowed.add(LINEAR_ATTENTION)
        offenders = sorted({str(t) for t in layer_types if str(t) not in allowed})
        if offenders:
            hint = ""
            if offenders == [LINEAR_ATTENTION]:
                n_layers = len(list(layer_types))
                hint = (
                    " This is a HYBRID-attention model: %d of %d layers are "
                    "linear and the sdpa mass patch cannot reach them. Pass "
                    "allow_linear_layers=True (--allow-linear-layers) to run the "
                    "injection on the %s full-attention layers only, and read "
                    "every result as an %s/%d-layer intervention."
                    % (n_linear, n_layers, n_sdpa, n_sdpa, n_layers)
                )
            raise ValueError(
                "reader model uses attention layers the mass patch cannot bias: "
                "%r (layer_types_summary=%r)."
                "%s" % (offenders, summary, hint)
            )
        if not n_sdpa:
            raise ValueError(
                "reader model has no full_attention layer at all "
                "(layer_types_summary=%r): there is nothing the sdpa patch "
                "could bias" % (summary,)
            )
        return info

    if use_sliding is True:
        raise ValueError(
            "reader model has use_sliding_window=True (sliding_window=%r): "
            "mass injection needs every layer to be full attention" % (sliding_window,)
        )
    if use_sliding is None and (sliding_window or sliding_pattern):
        raise ValueError(
            "reader model declares a sliding window (sliding_window=%r, "
            "sliding_window_pattern=%r) and no use_sliding_window=False switch; "
            "mass injection needs every layer to be full attention"
            % (sliding_window, sliding_pattern)
        )
    return info


# --------------------------------------------------------------------------
# pre-flight context length / memory check (pure, CPU-testable)
# --------------------------------------------------------------------------

# bf16: 2 bytes per weight and per KV element.
BYTES_PER_ELEMENT = 2
# Leave 10% of the card for activations, fragmentation and the CUDA context.
GPU_MEM_HEADROOM = 0.9
# Arm budgets are tiktoken cl100k counts, but the reader tokenizes with the
# model's own tokenizer (Qwen3.8: 248k vocabulary).  Budget for the difference.
DEFAULT_TOKEN_MARGIN = 0.15


def full_attention_layers(model_config: Any) -> int | None:
    """Number of layers whose KV cache grows with the context.

    With ``layer_types``: the ``full_attention`` entries only.  A
    ``linear_attention`` layer keeps a fixed-size state, so counting it would
    overstate a hybrid model's KV cache about 4x.  Without: ``num_hidden_layers``
    (a dense model is all full attention).
    """
    cfg = resolve_text_config(model_config)
    layer_types = _cfg_get(cfg, "layer_types")
    if layer_types:
        return sum(1 for t in layer_types if str(t) == FULL_ATTENTION)
    layers = _cfg_get(cfg, "num_hidden_layers")
    return int(layers) if layers else None


def kv_cache_bytes(model_config: Any, tokens: int) -> int | None:
    """bf16 KV-cache size for ``tokens`` positions, or None when unknowable.

    2 (K and V) x full-attention layers x kv_heads x head_dim x 2 bytes x
    tokens.  ``head_dim`` comes from the config when present, else
    hidden_size / num_attention_heads.
    """
    cfg = resolve_text_config(model_config)
    layers = full_attention_layers(cfg)
    kv_heads = _cfg_get(cfg, "num_key_value_heads")
    if kv_heads is None:
        kv_heads = _cfg_get(cfg, "num_attention_heads")
    head_dim = _cfg_get(cfg, "head_dim")
    if head_dim is None:
        hidden = _cfg_get(cfg, "hidden_size")
        heads = _cfg_get(cfg, "num_attention_heads")
        if hidden and heads:
            head_dim = int(hidden) // int(heads)
    if not (layers and kv_heads and head_dim):
        return None
    return (
        2 * int(layers) * int(kv_heads) * int(head_dim) * BYTES_PER_ELEMENT * int(tokens)
    )


def check_context_fits(
    model_config: Any,
    prompt_tokens: int,
    max_new_tokens: int,
    gpu_mem_gb: float,
    *,
    headroom: float = GPU_MEM_HEADROOM,
    token_margin: float = DEFAULT_TOKEN_MARGIN,
    weight_bytes: int | None = None,
) -> dict:
    """Refuse a cell that cannot run, before any weights are downloaded.

    Raises ValueError, with every number in the message, when
    1. ``prompt_tokens * (1 + token_margin) + max_new_tokens`` exceeds
       ``max_position_embeddings``: an overflow does not crash, it silently
       produces garbage answers that would be scored as the arm's result; or
    2. weights + KV cache exceed ``gpu_mem_gb * headroom``: running out of
       memory after a 56 GB download wastes an hour of GPU time per cell.

    ``token_margin`` covers the difference between the tiktoken counts used
    for arm budgets and the model's own tokenizer.  The weight size is
    ``weight_bytes`` (see ``safetensors_total_bytes``), else the config's
    ``num_parameters``.  With neither, or without ``max_position_embeddings`` /
    ``num_hidden_layers``, it raises rather than pass vacuously (0 weight
    bytes would make any card look big enough).

    Returns the inspected numbers for the meta sidecar.
    """
    cfg = resolve_text_config(model_config)
    max_pos = _cfg_get(cfg, "max_position_embeddings")
    n_layers = _cfg_get(cfg, "num_hidden_layers")
    if max_pos is None or n_layers is None:
        raise ValueError(
            "cannot pre-flight the context: the resolved text config states "
            "max_position_embeddings=%r and num_hidden_layers=%r (model_type=%r). "
            "A vision-language wrapper keeps both under text_config; passing the "
            "wrapper here used to make this check pass with None values, which is "
            "exactly the silent failure it exists to prevent"
            % (max_pos, n_layers, _cfg_get(cfg, "model_type"))
        )

    margin = float(token_margin)
    effective_prompt = int(math.ceil(int(prompt_tokens) * (1.0 + margin)))
    total_tokens = effective_prompt + int(max_new_tokens)
    kv_bytes = kv_cache_bytes(cfg, total_tokens)
    n_params = _cfg_get(cfg, "num_parameters")
    if weight_bytes is not None:
        weight_bytes = int(weight_bytes)
    elif n_params:
        weight_bytes = int(n_params) * BYTES_PER_ELEMENT
    else:
        raise ValueError(
            "weight bytes unknown; pass weight_bytes (the config states no "
            "num_parameters, and 0 weight bytes would make any card look big "
            "enough; see safetensors_total_bytes)"
        )
    limit_bytes = float(gpu_mem_gb) * (1024 ** 3) * headroom
    need_bytes = (kv_bytes or 0) + weight_bytes

    info = {
        "prompt_tokens": int(prompt_tokens),
        "token_margin": margin,
        "effective_prompt_tokens": effective_prompt,
        "max_new_tokens": int(max_new_tokens),
        "total_tokens": total_tokens,
        "max_position_embeddings": max_pos,
        "num_hidden_layers": int(n_layers),
        "full_attention_layers": full_attention_layers(cfg),
        "kv_cache_bytes": kv_bytes,
        "weight_bytes": weight_bytes,
        "gpu_mem_gb": float(gpu_mem_gb),
        "headroom": headroom,
        "budget_bytes": limit_bytes,
        "needed_bytes": need_bytes,
    }

    if total_tokens > int(max_pos):
        raise ValueError(
            "context does not fit: prompt_tokens=%d x (1 + margin %.2f) = %d + "
            "max_new_tokens=%d = %d > max_position_embeddings=%d"
            % (prompt_tokens, margin, effective_prompt, max_new_tokens,
               total_tokens, int(max_pos))
        )
    if kv_bytes is not None and need_bytes > limit_bytes:
        raise ValueError(
            "context does not fit in memory: KV=%.2f GB + weights=%s GB = "
            "%.2f GB > %.2f GB (%.0f%% of %.1f GB) for %d tokens"
            % (
                kv_bytes / (1024 ** 3),
                "%.2f" % (weight_bytes / (1024 ** 3)),
                need_bytes / (1024 ** 3),
                limit_bytes / (1024 ** 3),
                100 * headroom,
                gpu_mem_gb,
                total_tokens,
            )
        )
    return info


SAFETENSORS_INDEX = "model.safetensors.index.json"
SAFETENSORS_SINGLE = "model.safetensors"


def safetensors_header_bytes(path: str | Path) -> int:
    """Weight bytes of ONE safetensors file from its header alone: the file
    starts with a little-endian u64 header length, then the JSON header whose
    ``data_offsets`` end at the last byte of tensor data."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return max(
        (int(v["data_offsets"][1]) for k, v in header.items() if k != "__metadata__"),
        default=0,
    )


def _index_total_size(index_path: str | Path) -> int | None:
    """``metadata.total_size`` of a ``model.safetensors.index.json``, or None."""
    meta = json.loads(Path(index_path).read_text(encoding="utf-8")).get("metadata") or {}
    size = meta.get("total_size")
    return int(size) if size else None


def safetensors_total_bytes(model_id: str) -> int | None:
    """Total weight bytes of a checkpoint, without downloading the weights.

    Local directory: ``metadata.total_size`` of the index file, else the header
    of ``model.safetensors``.  Hub id: download only the index file; for a
    single-file checkpoint read the header with a range request
    (``parse_safetensors_file_metadata``).  None when the size cannot be
    obtained (offline, no auth, ...); callers must then refuse to pre-flight,
    never assume 0.
    """
    local = Path(model_id)
    if local.is_dir():
        index = local / SAFETENSORS_INDEX
        if index.is_file():
            return _index_total_size(index)
        single = local / SAFETENSORS_SINGLE
        if single.is_file():
            return safetensors_header_bytes(single)
        return None
    try:
        import huggingface_hub
        from huggingface_hub.utils import EntryNotFoundError
    except ImportError:
        return None
    try:
        return _index_total_size(huggingface_hub.hf_hub_download(model_id, SAFETENSORS_INDEX))
    except EntryNotFoundError:
        pass  # no index: a single-file checkpoint
    except Exception:  # noqa: BLE001 -- offline / auth / network: unobtainable, not 0
        return None
    try:
        file_meta = huggingface_hub.parse_safetensors_file_metadata(model_id, SAFETENSORS_SINGLE)
    except Exception:  # noqa: BLE001 -- same: unobtainable
        return None
    return max((int(t.data_offsets[1]) for t in file_meta.tensors.values()), default=None)


# --------------------------------------------------------------------------
# questions
# --------------------------------------------------------------------------

def load_questions(
    path: str | Path,
    *,
    subset: str = "generated",
    include_excluded: bool = False,
    max_questions: int | None = None,
) -> list[dict]:
    """The questions to answer: ``subset`` ("generated", "legacy", or anything
    else for all), without excluded ones unless ``include_excluded``, at most
    ``max_questions``."""
    with Path(path).open(encoding="utf-8") as f:
        items = json.load(f)
    out = []
    for q in items:
        if not include_excluded and q.get("excluded"):
            continue
        if subset == "generated" and q.get("legacy"):
            continue
        if subset == "legacy" and not q.get("legacy"):
            continue
        out.append(q)
    if max_questions is not None:
        out = out[:max_questions]
    return out


# --------------------------------------------------------------------------
# reader (GPU only)
# --------------------------------------------------------------------------

@dataclass
class ReaderRun:
    answers: dict[str, str] = field(default_factory=dict)
    per_question: dict[str, dict] = field(default_factory=dict)
    run_meta: dict = field(default_factory=dict)


def load_reader(
    model_id: str,
    *,
    max_new_tokens: int = 48,
    prefill_scale: float = 0.0,
    bias_cap: float | None = None,
    w: float = 0.0,
    prefill_last_row: bool = False,
    quantization: str = "none",
) -> Any:
    """Load the HF reader with the sdpa mass patch active.  GPU only.

    ``quantization``: "none" loads bf16 (or the checkpoint's own
    quantization); "nf4"/"fp4" load an unquantized checkpoint in bitsandbytes
    4-bit, dequantized per matmul, which is what fits on a 46 GB card.

    ``bias_cap`` is the only cap: it clamps the effective bias ``w * mass``
    inside ``build_mass_bias``; the mass vector itself is built uncapped.

    ``prefill_last_row`` (off by default) also applies the bias during prefill,
    to the last prompt row only.  That row produces the first answer token,
    which decode-only injection can never influence.
    """
    from server.mass_weighted_gemma import MassWeightedLLM

    llm = MassWeightedLLM(
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        quantization=str(quantization),
        prefill_last_row=bool(prefill_last_row),
    )
    # MassWeightedLLM reads these knobs from config.yaml in __init__ and has no
    # constructor parameters for them; each cell needs its own values.
    llm._mass_weight = float(w)
    llm._prefill_mass_scale = float(prefill_scale)
    llm._bias_cap = None if bias_cap is None else float(bias_cap)
    llm.load()
    # The mass patch replaces torch's scaled_dot_product_attention.  A model
    # running "eager" or flash_attention_2 never calls it: the bias would be
    # built and counted as applied but never added, and the injected arm would
    # silently be the text-only arm.  load() asks for sdpa; check it got it.
    impl = llm.attn_implementation
    if impl != "sdpa":
        raise RuntimeError(
            "reader model loaded with attn_implementation=%r, expected 'sdpa': "
            "the mass patch only intervenes on the sdpa path, so any other "
            "implementation would run an un-injected baseline" % (impl,)
        )
    return llm


def first_line_answer(text: str) -> str:
    """The answer is the FIRST non-empty line of the generation.

    A leading ``<think>...</think>`` block is removed first, and a bare
    ``<think>`` / ``</think>`` line is skipped, so an unterminated thinking
    block cannot become the answer.

    The prompt asks for "the answer only, in a few words", but a chat model
    often adds a sentence of justification, which would either trip the
    scorer's word cap or smuggle in a second candidate answer.  The full text is
    kept in the meta sidecar as ``raw``.
    """
    body = THINK_BLOCK_RE.sub("", text, count=1) if "<think>" in text else text
    for line in body.splitlines():
        stripped = line.strip()
        if stripped and stripped != THINK_OPEN and stripped != THINK_CLOSE:
            return stripped
    return body.strip()


def run_reader(
    llm: Any,
    context_block: str,
    questions: list[dict],
    *,
    w: float,
    inject: str,
    bias_cap: float | None = None,
    arm: str | None = None,
    progress: Callable[[str], None] | None = None,
    chat_template: bool = False,
) -> ReaderRun:
    """Answer every question one at a time (greedy), with its own mass vector.

    GPU only: ``llm`` is a loaded MassWeightedLLM.

    When ``arm`` is a ``cd_*`` arm and ``w > 0``, the context must contain at
    least one planet span.  ``build_reader_mass_vector`` only refuses when the
    text ``[PN`` is present, so a CD serialized without any planet (every node
    promoted to a sun) would otherwise run as a silent baseline under an
    injected cell name.

    ``bias_cap`` is not applied here and only keeps the signature: the single
    cap lives in the loaded model (``load_reader(bias_cap=...)``).  Capping the
    mass vector as well would clamp ``mass`` before ``w * mass`` (mass 10,
    w 0.5, cap 3 would give 1.5 instead of 3).

    ``chat_template=True`` wraps each prompt in the tokenizer's chat template
    (see ``chat_wrap``).
    """
    tokenizer = llm.tokenizer
    result = ReaderRun()
    for q in questions:
        prompt = build_prompt(context_block, q["question"],
                              tokenizer=tokenizer if chat_template else None)
        ids = tokenizer(prompt, return_tensors=None)["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        # Put the mass vector on the model's device (None = torch's default
        # device, for a model with no parameters or not loaded yet).
        try:
            device = next(llm._model.parameters()).device
        except (StopIteration, AttributeError):
            device = None
        vec, info = build_reader_mass_vector(
            ids, tokenizer, inject, None,  # uncapped: the cap is applied in the model
            w=w, prompt_text=prompt, device=device,
        )
        if vec is not None:
            llm.set_mass_vector(vec)
        else:
            llm.clear_mass_vector()
        try:
            text = llm.generate(prompt)
        finally:
            stats = llm.mass_injection_stats()
            # Read before clear_mass_vector() resets it; absent (0) on a stub.
            last_row_calls = int(getattr(llm, "bias_applied_prefill_last_row_calls", 0))
            llm.clear_mass_vector()
        if arm and arm.startswith("cd_") and w > 0 and info.planet_spans == 0:
            raise RuntimeError(
                "arm %r with w=%g found 0 planet spans in its context "
                "(spans=%d, satellites=%d): the CD serialized without a single "
                "[PN line, so this cell would be the text-only baseline under an "
                "injected name" % (arm, w, info.spans, info.satellite_spans)
            )
        result.answers[q["qid"]] = first_line_answer(text)
        result.per_question[q["qid"]] = {
            "raw": text,
            "positions_found": info.positions_found,
            "spans": info.spans,
            "planet_spans": info.planet_spans,
            "satellite_spans": info.satellite_spans,
            "prompt_tokens": len(ids),
            "bias_applied_calls": stats["bias_applied_calls"],
            "bias_skipped_prefill_calls": stats["bias_skipped_prefill_calls"],
            "bias_skipped_sliding_calls": stats["bias_skipped_sliding_calls"],
            "bias_applied_prefill_last_row_calls": last_row_calls,
        }
        if progress:
            progress(q["qid"])
    return result


def run_meta(
    *,
    model_id: str,
    layer_info: dict,
    w: float,
    inject: str,
    prefill_scale: float,
    bias_cap: float | None,
    context_tokens: int,
    arm: str | None = None,
    context_check: dict | None = None,
    extra: dict | None = None,
) -> dict:
    import torch  # local: keeps `import run_reader` free of torch on CPU hosts
    import transformers

    meta = {
        "model_id": model_id,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "layer_types": layer_info.get("layer_types"),
        # The layers the injection actually reaches, next to the raw depth: a
        # hybrid model biases only its full-attention layers.
        "layer_types_summary": layer_info.get("layer_types_summary"),
        "n_sdpa_layers": layer_info.get("n_sdpa_layers"),
        "n_linear_layers": layer_info.get("n_linear_layers"),
        "allow_linear_layers": layer_info.get("allow_linear_layers"),
        "model_config_checked": layer_info,
        "max_position_embeddings": layer_info.get("max_position_embeddings"),
        "context_check": context_check,
        "arm": arm,
        "w": w,
        "inject": inject,
        "prefill_scale": prefill_scale,
        "bias_cap": bias_cap,
        "context_tokens": context_tokens,
    }
    if extra:
        meta.update(extra)
    return meta


def write_answers(out_path: str | Path, run: ReaderRun) -> None:
    """``{qid: answer}`` to ``out_path``; run meta + per-question data to ``<out>.meta.json``."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(run.answers, f, ensure_ascii=False, indent=2)
    payload = {**run.run_meta, "per_question": run.per_question}
    with out.with_suffix(".meta.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def resolve_allow_linear_layers(args: argparse.Namespace) -> bool:
    """``--allow-linear-layers`` / ``--no-allow-linear-layers`` / the default.

    With neither flag (``None``) the model id decides (True only for the
    hybrid reader).  An explicit flag always wins.
    """
    explicit = getattr(args, "allow_linear_layers", None)
    if explicit is not None:
        return bool(explicit)
    return default_allow_linear_layers(getattr(args, "model_id", None))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="bineval reader with mass injection")
    p.add_argument("--model-id", required=True)
    p.add_argument("--questions", required=True)
    p.add_argument("--context-file")
    p.add_argument("--arm")
    p.add_argument("--chat", default=None)
    p.add_argument("--cd-json", default=None)
    p.add_argument("--ratio", type=float, default=None)
    p.add_argument("--policy", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-level-markers", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--w", type=float, default=0.0)
    p.add_argument("--inject", choices=list(INJECT_MODES), default="planet")
    p.add_argument("--prefill-scale", type=float, default=0.0)
    p.add_argument("--bias-cap", type=float, default=None,
                   help="optional cap on the effective bias (default: none, per spec)")
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--max-questions", type=int, default=None)
    p.add_argument("--subset", choices=("all", "legacy", "generated"), default="generated")
    p.add_argument("--dtype", choices=("bf16",), default="bf16")
    p.add_argument(
        "--allow-linear-layers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "accept a HYBRID model (linear + full attention, e.g. "
            "Qwen/Qwen3.8-27B): the injection then reaches the full-attention "
            "layers ONLY, and n_sdpa_layers in the meta says how many. "
            "DEFAULT: on for %s (the decided reader), off for every other "
            "model id. --no-allow-linear-layers forces the dense-only rule."
            % HYBRID_MODEL_ID
        ),
    )
    p.add_argument(
        "--gpu-mem-gb",
        type=float,
        default=None,
        help=(
            "GPU memory of the target card; enables the B4 pre-flight "
            "(context length + KV-cache size). Omit to skip the memory term"
        ),
    )
    return p


def resolve_context(args: argparse.Namespace) -> tuple[str, int, str | None]:
    """(context_text, tokens, arm_name) from --context-file or --arm."""
    from benchmark.bineval import arms as arms_mod

    if args.context_file:
        text = Path(args.context_file).read_text(encoding="utf-8")
        return text, arms_mod.make_token_counter()(text), args.arm
    if not args.arm:
        raise SystemExit("one of --context-file / --arm is required")
    chat = arms_mod.load_chat(args.chat) if args.chat else arms_mod.load_chat()
    ctx = arms_mod.build_arm_context(
        args.arm,
        chat=chat,
        cd_json=args.cd_json,
        ratio=args.ratio,
        policy=args.policy,
        seed=args.seed,
        level_markers=not args.no_level_markers,
    )
    return ctx.text, ctx.tokens, ctx.name


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    context, tokens, arm = resolve_context(args)
    questions = load_questions(
        args.questions, subset=args.subset, max_questions=args.max_questions
    )

    # Pre-flight from the config alone, so an unusable model or an impossible
    # cell fails before 56 GB of weights are downloaded.
    from transformers import AutoConfig

    allow_linear = resolve_allow_linear_layers(args)
    config = AutoConfig.from_pretrained(args.model_id)
    layer_info = check_model_supported(config, allow_linear_layers=allow_linear)
    context_check = None
    if args.gpu_mem_gb is not None:
        context_check = check_context_fits(
            config, tokens, args.max_new_tokens, args.gpu_mem_gb,
            weight_bytes=safetensors_total_bytes(args.model_id),
        )

    llm = load_reader(
        args.model_id,
        max_new_tokens=args.max_new_tokens,
        prefill_scale=args.prefill_scale,
        bias_cap=args.bias_cap,
        w=args.w,
    )

    run = run_reader(
        llm, context, questions,
        w=args.w, inject=args.inject, bias_cap=args.bias_cap, arm=arm,
    )
    run.run_meta = run_meta(
        model_id=args.model_id,
        layer_info=layer_info,
        w=args.w,
        inject=args.inject,
        prefill_scale=args.prefill_scale,
        bias_cap=args.bias_cap,
        context_tokens=tokens,
        arm=arm,
        context_check=context_check,
    )
    write_answers(args.out, run)
    print("wrote %s (%d answers, %d context tokens)" % (args.out, len(run.answers), tokens))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

"""run_reader.py — HF reader with mass injection (S2.2).

ONE question per generation, greedy, ``max_new_tokens=48``. The prompt is
IDENTICAL across arms (only the context block changes), so an arm difference can
never be a prompt difference.

The mass patch always runs on the same code path: ``w = 0`` makes the injected
bias numerically inert (``w * mass == 0``) but the sdpa patch, the marker scan
and the mass vector are all still built, so the text-only arm and the injected
arm differ in exactly one scalar.

Answers are written as ``{qid: text}`` (the schema ``score_binary.py`` maps by
qid) plus a sidecar ``<out>.meta.json`` carrying the provenance the run needs to
be auditable: model id, transformers/torch versions, layer_types, per-question
positions_found, and ``mass_injection_stats()``.

CPU-testable pieces (no model, no torch models imported at module import time):
``build_prompt``, ``build_reader_mass_vector``, ``check_model_supported``.
Model loading happens only inside ``load_reader`` / ``run_reader``.

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

``--subset generated`` is NOT optional (F4).  The reader answers the questions
``load_questions`` selects, and ITS default subset is ``generated``: 173 of the
189 non-excluded questions.  Scoring those answers with the scorer's own
default (``--subset all``) puts 189 in the denominator and reports a pass rate
~8 points low, with the 16 legacy questions counted as unanswered failures.

MODEL SUPPORT (A3)
------------------
The mass patch only intervenes on ``F.scaled_dot_product_attention``.  A HYBRID
model (Qwen/Qwen3.8-27B: ``text_config.layer_types`` is 3 x
``linear_attention`` + 1 x ``full_attention``, repeating -- 48 linear and 16
full of 64 layers) can only be injected on its full-attention layers.
``check_model_supported`` therefore refuses such a config unless
``allow_linear_layers`` is given, and always reports ``n_sdpa_layers``: the
number of layers the bias actually reaches.  Sliding attention is refused
unconditionally -- the 1D mass vector is indexed by ABSOLUTE position and a
sliding layer truncates the KV cache.

Since the 2026-09-07 owner decision Qwen/Qwen3.8-27B is the DECIDED reader, so
``--allow-linear-layers`` DEFAULTS to on for that model id and to off for every
other one (``default_allow_linear_layers``).  The 16/64 arithmetic is not
hidden by that default: ``n_sdpa_layers`` = 16 is in every meta sidecar and
every result must be read as a 16-of-64-layer intervention.
``--no-allow-linear-layers`` forces the dense-only rule back on.
"""

from __future__ import annotations

import argparse
import json
import re
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from server.cd_parser import find_marker_spans, marker_positions

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
THINK_BLOCK_RE = re.compile(r"^\s*<think>.*?</think>", re.DOTALL)

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
    """``prompt`` as a single user turn of the model's chat template, thinking OFF.

    mcbuild-bench H6 alternative, taken because smoke criterion F4 failed on the
    raw completion prompt: Qwen3.8-27B opens a ``<think>`` block and spends the
    48-token budget inside it, so the answer never appears.  With
    ``enable_thinking=False`` the template itself closes the thinking block, and
    generation starts on the answer.  Falls back to the raw prompt when the
    tokenizer has no chat template (the string is then arm-invariant either way).
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
    """The reader prompt. Arm-invariant by construction.

    ``tokenizer`` (mcbuild-bench, H6 alternative) wraps the prompt in that
    tokenizer's chat template with thinking disabled; the token budget W must be
    measured on the SAME string, so every caller that counts tokens passes it too.
    """
    prompt = READER_PROMPT.format(context_block=context_block, question=question)
    return prompt if tokenizer is None else chat_wrap(tokenizer, prompt)


# --------------------------------------------------------------------------
# mass vector (pure, CPU-testable with a fake tokenizer)
# --------------------------------------------------------------------------

@dataclass
class MassVectorInfo:
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
    """Marker scan -> (mass vector | None, info).

    ``inject`` is one of INJECT_MODES. ``cap`` is the optional per-element mass
    cap (None = no cap, which is the spec default: "no fixed cap", the only knob
    is w).

    Raises RuntimeError when ``w > 0`` and the prompt visibly contains ``[PN``
    markers but the scan found no injectable position: that combination would
    silently degrade the injected arm into the text-only baseline.
    """
    if inject not in INJECT_MODES:
        raise ValueError("inject must be one of %r, got %r" % (INJECT_MODES, inject))
    if inject == "none":
        # DECISIONS H11: baselines run without any marker scan.
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
        text = prompt_text
        if text is None:
            text = tokenizer.decode(ids, skip_special_tokens=False)
        if "[PN" in text:
            raise RuntimeError(
                "mass injection requested (w=%g, inject=%s) but the marker scan "
                "found 0 positions while the prompt contains '[PN' "
                "(spans=%d planets=%d): refusing to run a silent baseline"
                % (w, inject, len(spans), info.planet_spans)
            )

    if not positions:
        return None, info

    from server.mass_vector import positions_to_mass_vector

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
# A3: a hybrid model runs the injection on its full-attention layers only, so
# the choice has to be made explicitly rather than crashed into.  It stays the
# default for an UNKNOWN model id; for the hybrid the owner has made the choice
# once and for all (see ``default_allow_linear_layers``).
DEFAULT_ALLOW_LINEAR_LAYERS = False

# A3 (2026-09-07 owner decision): judge AND reader are Qwen/Qwen3.8-27B.  These
# three numbers are the model's config.json as verified against transformers
# 5.8: text_config.layer_types repeats 3 x linear_attention + 1 x
# full_attention over 64 layers (configuration_qwen3_5.py:106-111 builds
# exactly that from full_attention_interval=4), so the sdpa mass patch reaches
# 16 layers and skips 48.
HYBRID_MODEL_ID = "Qwen/Qwen3.8-27B"
HYBRID_N_SDPA_LAYERS = 16
HYBRID_N_LINEAR_LAYERS = 48
HYBRID_NUM_HIDDEN_LAYERS = 64


def is_hybrid_model_id(model_id: str | None) -> bool:
    """Is this the hybrid reader the 16/64 injection was decided for?

    Compared case-insensitively and without a revision suffix, so
    ``qwen/qwen3.8-27b`` and ``Qwen/Qwen3.8-27B@main`` both resolve.
    """
    if not model_id:
        return False
    head = str(model_id).split("@", 1)[0].rstrip("/")
    return head.lower() == HYBRID_MODEL_ID.lower()


def default_allow_linear_layers(model_id: str | None) -> bool:
    """The ``--allow-linear-layers`` default for ``model_id``.

    True for the hybrid: it is now the DEFAULT reader, and refusing the default
    model would make the tool unusable without a flag on every invocation.  The
    16/64 arithmetic is not hidden by this -- ``n_sdpa_layers`` is still in
    every meta sidecar and ``n_sdpa_layers_expected`` in run_info.json.  False
    for everything else, so an unvetted hybrid still fails loudly.
    """
    return is_hybrid_model_id(model_id)


def expected_n_sdpa_layers(model_id: str | None) -> int | None:
    """How many layers the patch MUST reach, from the model id alone.

    Known only for the hybrid (16).  ``None`` means "no expectation to assert",
    which is what a dense or unknown model gets.
    """
    return HYBRID_N_SDPA_LAYERS if is_hybrid_model_id(model_id) else None


def linear_attention_kernel_report() -> dict:
    """Are the qwen3_5 linear-attention fast-path kernels importable HERE?

    transformers 5.8 gates them at ``modeling_qwen3_5.py:49-64`` on
    ``is_flash_linear_attention_available()`` (fla >= 0.2.2 AND torch CUDA,
    ``utils/import_utils.py:810-812``) and ``is_causal_conv1d_available()``
    (causal_conv1d AND torch CUDA, ``import_utils.py:815-817``).  When either
    is missing ``is_fast_path_available`` (``modeling_qwen3_5.py:205-207``) is
    False and the module falls back to its pure-torch implementations
    (``modeling_qwen3_5.py:407-418``) with a warning -- correct, slower.

    Reported, never asserted: the fallback changes speed and (marginally)
    numerics of the 48 LINEAR layers, which the mass patch never touches.
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
    """The sub-config carrying the LANGUAGE model's attention settings.

    F1: a vision-language wrapper (transformers 5.8 ``Qwen3VLConfig``) has NO
    ``num_hidden_layers`` / ``max_position_embeddings`` at the top level -- they
    live under ``text_config``.  transformers exposes ``get_text_config()`` for
    exactly this; the raw ``text_config`` attribute is the fallback for plain
    dicts and hand-built fakes.  A text-only config resolves to itself.
    """
    getter = getattr(model_config, "get_text_config", None)
    if callable(getter):
        try:
            sub_cfg = getter()
        except Exception:  # a fake / partial config must not break the check
            sub_cfg = None
        if sub_cfg is not None and _cfg_get(sub_cfg, "num_hidden_layers", None) is not None:
            return sub_cfg
    sub_cfg = _cfg_get(model_config, "text_config", None)
    if sub_cfg is not None and _cfg_get(sub_cfg, "num_hidden_layers", None) is not None:
        return sub_cfg
    return model_config


def layer_type_summary(layer_types: Any) -> dict[str, int]:
    """``["full_attention", "linear_attention", ...]`` -> ``{type: count}``."""
    out: dict[str, int] = {}
    for t in layer_types or ():
        out[str(t)] = out.get(str(t), 0) + 1
    return out


def check_model_supported(
    model_config: Any, *, allow_linear_layers: bool = DEFAULT_ALLOW_LINEAR_LAYERS
) -> dict:
    """Which attention layers the mass patch can actually reach, else raise.

    The 1D mass vector is indexed by ABSOLUTE position; a sliding-window layer
    truncates the KV cache, so mass_vector[j] would land on an unrelated token
    (see build_mass_bias / F2).  Sliding attention is therefore ALWAYS refused.

    A3: ``linear_attention`` layers (Qwen3.8-27B is 48 linear + 16 full of 64)
    never call ``F.scaled_dot_product_attention``, so the bias cannot reach
    them.  That is a measured trade-off, not a bug, so it is accepted only with
    ``allow_linear_layers=True`` and the reachable layer count is reported as
    ``n_sdpa_layers``.  Every caller that compares ``bias_applied_calls``
    against a layer count MUST use ``n_sdpa_layers``, never
    ``num_hidden_layers``.

    Returns the inspected values for the meta sidecar.
    """
    # F1: vision-language wrappers keep the language-model attention settings
    # under ``text_config``; the wrapper's top level has none of the keys and
    # would pass vacuously.
    model_config = resolve_text_config(model_config)
    layer_types = _cfg_get(model_config, "layer_types", None)
    use_sliding = _cfg_get(model_config, "use_sliding_window", None)
    sliding_window = _cfg_get(model_config, "sliding_window", None)
    sliding_pattern = _cfg_get(model_config, "sliding_window_pattern", None)
    model_type = _cfg_get(model_config, "model_type", None)
    n_hidden = _cfg_get(model_config, "num_hidden_layers", None)
    summary = layer_type_summary(layer_types)
    if layer_types:
        n_sdpa: int | None = summary.get(FULL_ATTENTION, 0)
        n_linear = summary.get(LINEAR_ATTENTION, 0)
    else:
        n_sdpa = int(n_hidden) if n_hidden is not None else None
        n_linear = 0
    info = {
        "model_type": model_type,
        "layer_types": list(layer_types) if layer_types else None,
        "layer_types_summary": summary or None,
        "use_sliding_window": use_sliding,
        "sliding_window": sliding_window,
        "sliding_window_pattern": sliding_pattern,
        # B4: recorded in every cell's meta so a later reader can tell whether a
        # context was close to the model's position limit.
        "max_position_embeddings": _cfg_get(
            model_config, "max_position_embeddings", None
        ),
        "num_hidden_layers": n_hidden,
        # A3: the number of layers the sdpa patch can actually bias.
        "n_sdpa_layers": n_sdpa,
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
                hint = (
                    " This is a HYBRID-attention model: %d of %d layers are "
                    "linear and the sdpa mass patch cannot reach them. Pass "
                    "allow_linear_layers=True (--allow-linear-layers) to run the "
                    "injection on the %s full-attention layers only, and read "
                    "every result as an %s/%d-layer intervention."
                    % (n_linear, len(list(layer_types)), n_sdpa, n_sdpa,
                       len(list(layer_types)))
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
# A4: arm budgets are tiktoken cl100k counts; the reader tokenizes with the
# MODEL's tokenizer (Qwen3.8: 248k vocabulary). Budget for the difference.
DEFAULT_TOKEN_MARGIN = 0.15


def full_attention_layers(model_config: Any) -> int | None:
    """Number of layers that hold a GROWING KV cache (A3d).

    ``layer_types`` present -> the ``full_attention`` entries only: a
    ``linear_attention`` layer keeps a constant-size recurrent state, so
    counting it would inflate a hybrid model's KV estimate by ~4x.  Absent ->
    ``num_hidden_layers`` (a dense model is all full attention).
    """
    cfg = resolve_text_config(model_config)
    layer_types = _cfg_get(cfg, "layer_types", None)
    if layer_types:
        return sum(1 for t in layer_types if str(t) == FULL_ATTENTION)
    layers = _cfg_get(cfg, "num_hidden_layers", None)
    return int(layers) if layers else None


def kv_cache_bytes(model_config: Any, tokens: int) -> int | None:
    """bf16 KV-cache size for ``tokens`` positions, or None when unknowable.

    2 (K and V) x FULL-attention layers x kv_heads x head_dim x 2 bytes x
    tokens.  ``head_dim`` is taken from the config when present and otherwise
    derived as hidden_size / num_attention_heads (the usual convention; Qwen3
    states it explicitly).
    """
    cfg = resolve_text_config(model_config)
    layers = full_attention_layers(cfg)
    kv_heads = _cfg_get(cfg, "num_key_value_heads", None)
    if kv_heads is None:
        kv_heads = _cfg_get(cfg, "num_attention_heads", None)
    head_dim = _cfg_get(cfg, "head_dim", None)
    if head_dim is None:
        hidden = _cfg_get(cfg, "hidden_size", None)
        heads = _cfg_get(cfg, "num_attention_heads", None)
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
    """Refuse a cell that cannot physically run, BEFORE any weight is fetched.

    ``weight_bytes`` (item K, 2026-09-18): the checkpoint's weight size in
    bytes, from ``safetensors_total_bytes``.  It wins over the config's
    ``num_parameters``; when it is None AND the config states no
    ``num_parameters`` the check RAISES instead of silently budgeting 0 bytes
    of weights (which made a 55.6 GB reader "fit" on any card).

    Two independent refusals:

    1. ``prompt_tokens * (1 + token_margin) + max_new_tokens >
       max_position_embeddings``: the arm's context is longer than the model's
       position table.  A silent overflow here does not crash -- it produces
       garbage answers that would then be scored as an arm result.
    2. weights + KV cache > ``gpu_mem_gb * headroom``: an OOM after a 56 GB
       download costs an hour of GPU time per cell.

    A4 (``token_margin``): every arm budget in this repo is counted with
    tiktoken ``cl100k_base``, but the reader tokenizes with the MODEL's
    tokenizer (Qwen3.8 has a 248k vocabulary).  The two counts differ by a few
    percent in either direction, so the pre-flight budgets a 15% safety margin
    on the prompt rather than pretending the two agree.  ``token_margin=0.0``
    restores the raw comparison.

    F1: the LANGUAGE config is resolved here (``get_text_config()`` /
    ``text_config``), and a config that states neither
    ``max_position_embeddings`` nor ``num_hidden_layers`` RAISES: silently
    returning ``None`` for both made this pre-flight pass vacuously on exactly
    the vision-language wrapper it exists to protect.

    The weight term comes from ``weight_bytes`` or, failing that, from the
    config's ``num_parameters``; it is never guessed and never skipped.

    Returns the inspected numbers for the meta sidecar; raises ValueError with
    every number in the message when a cell is refused.
    """
    cfg = resolve_text_config(model_config)
    max_pos = _cfg_get(cfg, "max_position_embeddings", None)
    n_layers = _cfg_get(cfg, "num_hidden_layers", None)
    if max_pos is None or n_layers is None:
        raise ValueError(
            "cannot pre-flight the context: the resolved text config states "
            "max_position_embeddings=%r and num_hidden_layers=%r (model_type=%r). "
            "A vision-language wrapper keeps both under text_config; passing the "
            "wrapper here used to make this check pass with None values, which is "
            "exactly the silent failure it exists to prevent"
            % (max_pos, n_layers, _cfg_get(cfg, "model_type", None))
        )

    margin = float(token_margin)
    effective_prompt = int(math.ceil(int(prompt_tokens) * (1.0 + margin)))
    total_tokens = effective_prompt + int(max_new_tokens)
    kv_bytes = kv_cache_bytes(cfg, total_tokens)
    n_params = _cfg_get(cfg, "num_parameters", None)
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
    import json as _json
    import struct

    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = _json.loads(f.read(n))
    return max(
        (int(v["data_offsets"][1]) for k, v in header.items() if k != "__metadata__"),
        default=0,
    )


def safetensors_total_bytes(model_id: str) -> int | None:
    """Total weight bytes of a checkpoint WITHOUT downloading weights (item K).

    Local directory: ``model.safetensors.index.json`` ``metadata.total_size``,
    else the header of ``model.safetensors``.  Hub id: ``hf_hub_download`` of
    the index file only; for a single-file checkpoint
    ``parse_safetensors_file_metadata`` (header via range request).  None when
    unobtainable -- callers must then refuse to pre-flight, never assume 0.
    """
    import json as _json

    local = Path(model_id)
    if local.is_dir():
        index = local / SAFETENSORS_INDEX
        if index.is_file():
            meta = _json.loads(index.read_text(encoding="utf-8")).get("metadata") or {}
            size = meta.get("total_size")
            return int(size) if size else None
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
        path = huggingface_hub.hf_hub_download(model_id, SAFETENSORS_INDEX)
        meta = _json.loads(Path(path).read_text(encoding="utf-8")).get("metadata") or {}
        size = meta.get("total_size")
        return int(size) if size else None
    except EntryNotFoundError:
        pass
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
    """Load the HF reader with the sdpa mass patch active.

    ``quantization``: "none" = bf16 (or the checkpoint's own pre-quantization);
    "nf4"/"fp4" = bitsandbytes 4-bit on an unquantized checkpoint (weights
    stay 4-bit in memory, dequantized per matmul) — the 46 GB-card fallback
    of DECISIONS H24.

    GPU ONLY -- never called on the CPU dev machine (D-11).

    ``bias_cap`` is the ONLY cap (Astra round 2, item 4): it clamps the
    effective bias ``w * mass`` inside ``build_mass_bias``; the mass vector
    itself is built uncapped.  ``prefill_last_row`` is the H15 (b) switch
    (DECISIONS H15/H18), off by default.
    """
    from server.mass_weighted_gemma import MassWeightedLLM

    llm = MassWeightedLLM(
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        quantization=str(quantization),
        prefill_last_row=bool(prefill_last_row),
    )
    # The knobs below have no constructor parameters in MassWeightedGemma; they
    # are read from config.yaml at __init__ time. The experiment must set them
    # per cell, so they are assigned here explicitly (documented deviation).
    llm._mass_weight = float(w)
    llm._prefill_mass_scale = float(prefill_scale)
    llm._bias_cap = None if bias_cap is None else float(bias_cap)
    llm.load()
    # M1: the mass patch replaces torch's scaled_dot_product_attention. A model
    # that runs "eager" (or flash_attention_2) never calls it, so the bias would
    # be built, counted as applied and silently never added -- the injected arm
    # would be the text-only arm with extra bookkeeping. load() now asks for
    # sdpa explicitly; this asserts the model actually took it.
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

    A leading ``<think>...</think>`` block (a thinking model answers inside it
    first) is removed before the scan, and a bare ``<think>`` line is skipped, so
    an unterminated block cannot become the answer.

    The reader prompt asks for "the answer only, in a few words", but a chat
    model routinely adds a second sentence of justification. score_binary
    matches on the whole string with --max-words, so the extra prose either
    trips the word cap or smuggles in a second candidate answer. The full text
    is kept in the meta sidecar as ``raw`` so nothing is lost.
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
    """Answer every question one at a time; greedy; mass vector per question.

    GPU ONLY (``llm`` is a loaded MassWeightedLLM).

    ``arm`` enables the M2 guard: a ``cd_*`` arm run with ``w > 0`` MUST have at
    least one planet span in its context. ``build_reader_mass_vector`` only
    refuses when the literal text ``[PN`` is present, so a CD that serialized to
    zero planets (every node promoted to a sun) would pass that check and run as
    a silent baseline under an injected cell name.

    ``bias_cap`` is NOT applied here (Astra round 2, item 4): the mass vector is
    built uncapped and the single effective-bias cap lives in the loaded model
    (``load_reader(bias_cap=...)`` -> ``build_mass_bias``).  Passing it to the
    mass vector as well clamped ``mass`` first and then ``w * mass`` (mass 10,
    w 0.5, cap 3 gave 1.5 instead of 3).  The parameter is kept for the call
    signature and recorded by the caller's meta only.
    """
    tokenizer = llm.tokenizer
    result = ReaderRun()
    for q in questions:
        prompt = build_prompt(context_block, q["question"],
                              tokenizer=tokenizer if chat_template else None)
        ids = tokenizer(prompt, return_tensors=None)["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        # Place the mass vector on the model's device. A parameter-less or
        # not-yet-loaded model falls back to the torch default device; the
        # exception list is narrow on purpose (no bare except).
        try:
            device = next(llm._model.parameters()).device
        except (StopIteration, AttributeError):
            device = None
        vec, info = build_reader_mass_vector(
            ids, tokenizer, inject, None,  # item 4: uncapped mass; the cap is in the model
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
            # H15 (b) counter: an attribute (0 / absent on a stub), reset by clear.
            last_row_calls = int(getattr(llm, "bias_applied_prefill_last_row_calls", 0))
            llm.clear_mass_vector()
        if arm and arm.startswith("cd_") and w > 0 and info.planet_spans == 0:
            raise RuntimeError(
                "arm %r with w=%g found 0 planet spans in its context "
                "(spans=%d, satellites=%d): the CD serialized without a single "
                "[PN line, so this cell would be the text-only baseline under an "
                "injected name" % (arm, w, info.spans, info.satellite_spans)
            )
        answer = first_line_answer(text)
        result.answers[q["qid"]] = answer
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
        # A3: the layer count the injection actually reaches, next to the raw
        # depth -- a hybrid model biases only its full-attention layers.
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
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(run.answers, f, ensure_ascii=False, indent=2)
    meta_path = out.with_suffix(".meta.json")
    payload = dict(run.run_meta)
    payload["per_question"] = run.per_question
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def resolve_allow_linear_layers(args: argparse.Namespace) -> bool:
    """``--allow-linear-layers`` / ``--no-allow-linear-layers`` / the default.

    ``None`` (neither flag given) means "decide from the model id", which is
    True only for the hybrid reader.  An explicit flag always wins, so
    ``--no-allow-linear-layers`` restores the dense-only rule for the hybrid.
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

    # Pre-flight BEFORE the weights are pulled: a sliding-window model must
    # fail fast, not after 56GB of bf16 has been downloaded.
    from transformers import AutoConfig

    allow_linear = resolve_allow_linear_layers(args)
    config = AutoConfig.from_pretrained(args.model_id)
    layer_info = check_model_supported(config, allow_linear_layers=allow_linear)
    context_check = None
    if args.gpu_mem_gb is not None:
        # B4: refuse an impossible cell before 56 GB of weights are downloaded.
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

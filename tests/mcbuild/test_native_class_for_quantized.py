"""A1 loading-class decision for pre-quantized checkpoints (2026-09-20).

Reproduced on the pod: RedHatAI/Qwen3.8-27B-INT4's quantization_config.ignore
list names modules under the full multimodal tree
(``model.language_model.layers.N.linear_attn.in_proj_a``). Loading through
the text-only AutoModelForCausalLM class (Qwen3_5ForCausalLM) drops the
``language_model.`` segment, so the ignore rules never match and
compressed-tensors quantizes modules the checkpoint left unquantized -> 288
missing weights, refused by _record_loading_info. See
server/mass_weighted_gemma.py prefers_native_multimodal_class() and load().
"""
from __future__ import annotations

from server.mass_weighted_gemma import prefers_native_multimodal_class


def test_quantized_conditional_generation_checkpoint_prefers_native_class() -> None:
    assert prefers_native_multimodal_class(["Qwen3_5ForConditionalGeneration"], True) is True


def test_unquantized_conditional_generation_checkpoint_uses_causal_lm_first() -> None:
    assert prefers_native_multimodal_class(["Qwen3_5ForConditionalGeneration"], False) is False


def test_quantized_causal_lm_only_checkpoint_uses_causal_lm() -> None:
    # a checkpoint that was never a *ForConditionalGeneration has no
    # language_model. prefix problem regardless of quantization
    assert prefers_native_multimodal_class(["Qwen3_5ForCausalLM"], True) is False


def test_no_architectures_declared_defaults_to_causal_lm() -> None:
    assert prefers_native_multimodal_class([], True) is False
    assert prefers_native_multimodal_class([], False) is False

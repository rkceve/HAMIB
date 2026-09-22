"""CPU end-to-end test of the injection path on a tiny Qwen3_5 checkpoint (item L).

A ``Qwen3_5ForConditionalGeneration`` with 4 layers (3 linear + 1 full
attention) is built from a config, saved with ``save_pretrained`` and loaded
through ``MassWeightedGemma`` exactly as ``run_reader.load_reader`` does.  The
tokenizer is a char-level fake (no network).  Assertions:

  (a) the checkpoint loads through the text-only class with 0 missing keys
      (``loaded_class_name`` / ``loading_info``, item A1);
  (b) chunked prefill (chunk 5, prompt 13 tokens) yields the same greedy tokens
      as the unchunked path when no mass vector is set;
  (c) with a mass vector and w > 0: ``bias_applied_calls == S * (n - 1)``,
      ``bias_skipped_prefill_calls == S * n_prefill_forwards`` and
      ``bias_skipped_sliding_calls == 0`` (S = 1 sdpa layer here), and the
      wall-ms split is recorded (items A2 / A3);
  (d) a 1-token FINAL prefill chunk (prompt 11, chunk 5 -> 5, 5, 1) gets the
      prefill rule: no bias is applied during prefill (item A2);
  (e) with prefill_scale > 0 a chunked prefill slices the mass vector to the
      growing cache instead of raising the sliding-window error (item A2).
"""

from __future__ import annotations

import math
import time

import pytest
import torch

pytest.importorskip("transformers")

from tests.mcbuild._tiny_qwen import N_SDPA_LAYERS, VOCAB  # noqa: E402


def _mass(n: int) -> torch.Tensor:
    v = torch.zeros(n)
    v[3:6] = 2.0
    return v


def _set_chunk(llm, chunk: int | None) -> None:
    llm._model.generation_config.prefill_chunk_size = chunk


def _generated_ids(llm, prompt: str) -> list[int]:
    return [ord(c) % VOCAB for c in llm.generate(prompt)]


def test_tiny_qwen35_end_to_end(llm) -> None:
    t0 = time.perf_counter()
    prompt13 = "abcdefghijklm"
    prompt11 = "abcdefghijk"
    assert len(prompt13) == 13 and len(prompt11) == 11

    # (a) explicit, verified model class
    assert llm.loaded_class_name == "Qwen3_5ForCausalLM"
    assert llm.checkpoint_architectures == ["Qwen3_5ForConditionalGeneration"]
    assert llm.loading_info["missing"] == 0 and llm.loading_info["mismatched"] == 0
    assert llm.attn_implementation == "sdpa"

    # (b) chunked prefill == unchunked greedy, no mass vector
    llm.clear_mass_vector()
    _set_chunk(llm, None)
    plain = _generated_ids(llm, prompt13)
    assert len(plain) == 4 and llm.last_generated_tokens == 4
    _set_chunk(llm, 5)
    chunked = _generated_ids(llm, prompt13)
    assert chunked == plain
    assert llm.mass_injection_stats() == {
        "bias_applied_calls": 0, "bias_skipped_prefill_calls": 0, "bias_skipped_sliding_calls": 0,
    }

    # (c) decode-only injection counters, unchunked then chunked (13 = 5 + 5 + 3)
    n = 4
    for chunk in (None, 5):
        _set_chunk(llm, chunk)
        llm.set_mass_vector(_mass(13))
        llm.generate(prompt13)
        n_prefill = 1 if chunk is None else math.ceil(13 / chunk)
        stats = llm.mass_injection_stats()
        assert llm.last_generated_tokens == n
        assert llm.last_decode_forwards == n - 1
        assert stats["bias_applied_calls"] == N_SDPA_LAYERS * (n - 1), (chunk, stats)
        assert stats["bias_skipped_prefill_calls"] == N_SDPA_LAYERS * n_prefill, (chunk, stats)
        assert stats["bias_skipped_sliding_calls"] == 0, (chunk, stats)
        assert isinstance(llm.last_prefill_ms, float) and isinstance(llm.last_decode_ms, float)
        assert llm._prefill_done is True

    # (d) 1-token final chunk: 11 = 5 + 5 + 1 -> the last chunk is PREFILL, no bias
    _set_chunk(llm, 5)
    llm.set_mass_vector(_mass(11))
    llm.generate(prompt11)
    stats = llm.mass_injection_stats()
    assert stats["bias_applied_calls"] == N_SDPA_LAYERS * (n - 1), stats
    assert stats["bias_skipped_prefill_calls"] == N_SDPA_LAYERS * 3, stats
    assert stats["bias_skipped_sliding_calls"] == 0, stats

    # (e) prefill_scale > 0 under chunked prefill: sliced, never "sliding"
    llm._prefill_mass_scale = 0.5
    try:
        _set_chunk(llm, 5)
        llm.set_mass_vector(_mass(13))
        llm.generate(prompt13)
        stats = llm.mass_injection_stats()
        assert stats["bias_applied_calls"] == N_SDPA_LAYERS * (3 + n - 1), stats
        assert stats["bias_skipped_prefill_calls"] == 0, stats
        assert stats["bias_skipped_sliding_calls"] == 0, stats
    finally:
        llm._prefill_mass_scale = 0.0
        _set_chunk(llm, None)
        llm.clear_mass_vector()

    print("tiny e2e body: %.1f s" % (time.perf_counter() - t0))

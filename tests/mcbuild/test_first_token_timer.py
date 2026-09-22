"""Item 6: the prefill/decode split of ``MassWeightedGemma.generate`` is measured
by a ``transformers.LogitsProcessor`` whose FIRST call marks the end of prefill.

No model is loaded here: the processor is unit-tested in isolation.
"""

from __future__ import annotations

import time

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from server.mass_weighted_gemma import MassWeightedGemma, make_first_token_timer


def test_timer_is_a_logits_processor_and_keeps_the_first_timestamp() -> None:
    timer = make_first_token_timer()
    assert isinstance(timer, LogitsProcessor)
    assert timer.first_call_t is None
    ids = torch.zeros((1, 3), dtype=torch.long)
    scores = torch.randn(1, 7)
    before = time.perf_counter()
    out = timer(ids, scores)
    first = timer.first_call_t
    assert out is scores  # scores are returned untouched
    assert first is not None and first >= before
    time.sleep(0.01)
    timer(ids, scores)
    assert timer.first_call_t == first  # the second call does not move it


def test_timer_runs_inside_a_logits_processor_list() -> None:
    timer = make_first_token_timer()
    lst = LogitsProcessorList([timer])
    scores = torch.randn(1, 5)
    out = lst(torch.zeros((1, 2), dtype=torch.long), scores)
    assert torch.equal(out, scores)
    assert timer.first_call_t is not None


def test_generate_timing_attributes_default_to_none() -> None:
    llm = MassWeightedGemma(model_id="dummy/model", quantization="none")
    assert llm.last_generated_tokens is None
    assert llm.last_prefill_ms is None
    assert llm.last_decode_ms is None

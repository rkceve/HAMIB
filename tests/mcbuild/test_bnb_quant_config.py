"""bitsandbytes 4-bit fallback (DECISIONS H24, 2026-09-20): config fields the reader relies on."""
from __future__ import annotations

import pytest
import torch

from server.mass_weighted_gemma import BNB_SKIP_MODULES, bnb_quant_config


def test_nf4_config_fields() -> None:
    cfg = bnb_quant_config("nf4")
    assert cfg["load_in_4bit"] is True
    assert cfg["bnb_4bit_quant_type"] == "nf4"
    assert cfg["bnb_4bit_use_double_quant"] is True
    assert cfg["bnb_4bit_compute_dtype"] is torch.bfloat16  # SDPA patch adds the bias in the query dtype
    assert cfg["llm_int8_skip_modules"] == ["in_proj_a", "in_proj_b", "lm_head"] == BNB_SKIP_MODULES


def test_unknown_quant_type_is_refused() -> None:
    with pytest.raises(ValueError):
        bnb_quant_config("int4")


def test_run_reader_forwards_quantization(monkeypatch) -> None:
    from benchmark.bineval import run_reader

    captured: dict = {}

    class FakeLLM:
        def __init__(self, **kw):
            captured.update(kw)
            self.attn_implementation = "sdpa"

        def load(self):
            pass

    monkeypatch.setattr("server.mass_weighted_gemma.MassWeightedLLM", FakeLLM)
    run_reader.load_reader("x", quantization="nf4")
    assert captured["quantization"] == "nf4"
    run_reader.load_reader("x")
    assert captured["quantization"] == "none"

"""Shared tiny Qwen3_5 CPU fixture for the mcbuild end-to-end tests.

A ``Qwen3_5ForConditionalGeneration`` with 4 layers (3 linear + 1 full
attention, 4 query heads : 2 kv heads) is built from a config, saved with
``save_pretrained`` and loaded through ``MassWeightedGemma`` exactly as
``run_reader.load_reader`` does.  The tokenizer is a char-level fake (no
network).  Import the fixtures into a test module::

    from tests.mcbuild._tiny_qwen import tiny_checkpoint, llm  # noqa: F401
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")

from transformers import BatchEncoding  # noqa: E402

TEXT_CONFIG = dict(
    hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
    num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
    linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
    full_attention_interval=4, vocab_size=512, max_position_embeddings=256,
    rope_parameters={
        "rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
        "mrope_section": [2, 1, 1], "mrope_interleaved": True,
    },
)
VISION_CONFIG = dict(
    depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
    patch_size=14, spatial_merge_size=2, temporal_patch_size=2, in_channels=3,
    hidden_act="gelu", num_position_embeddings=16, deepstack_visual_indexes=[],
)
N_SDPA_LAYERS = 1  # layer_types = 3 x linear_attention + 1 x full_attention
VOCAB = 512


class CharTokenizer:
    """Char-level fake: one token per character, id = ord(c) % VOCAB.

    ``return_tensors="pt"`` (MassWeightedGemma.generate) gives a BatchEncoding
    of tensors; ``return_tensors=None`` (run_reader.run_reader) gives plain
    lists, as a HF tokenizer does.
    """

    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text: str, return_tensors=None, **_kw):
        ids = [ord(c) % VOCAB for c in text]
        if return_tensors is None:
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        t = torch.tensor([ids], dtype=torch.long)
        return BatchEncoding(
            {"input_ids": t, "attention_mask": torch.ones_like(t)}, tensor_type=None
        )

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(chr(int(i)) for i in ids)


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory) -> Path:
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    torch.manual_seed(0)
    cfg = Qwen3_5Config(text_config=dict(TEXT_CONFIG), vision_config=dict(VISION_CONFIG))
    assert cfg.text_config.layer_types == ["linear_attention"] * 3 + ["full_attention"]
    model = Qwen3_5ForConditionalGeneration(cfg)
    out = tmp_path_factory.mktemp("tiny_qwen35")
    model.save_pretrained(out)
    return out


def make_llm(tiny_checkpoint: Path, monkeypatch, *, w: float = 0.5, max_new_tokens: int = 4,
             **kwargs):
    """A loaded ``MassWeightedGemma`` on the tiny checkpoint (caller restores)."""
    import transformers

    from server.mass_weighted_gemma import MassWeightedGemma

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda *a, **k: CharTokenizer())
    llm = MassWeightedGemma(model_id=str(tiny_checkpoint), quantization="none",
                            max_new_tokens=max_new_tokens, do_sample=False, **kwargs)
    llm._mass_weight = w
    llm._prefill_mass_scale = 0.0
    llm._bias_cap = None
    llm.load()
    return llm


@pytest.fixture
def llm(tiny_checkpoint: Path, monkeypatch):
    obj = make_llm(tiny_checkpoint, monkeypatch)
    try:
        yield obj
    finally:
        obj.restore_sdpa()

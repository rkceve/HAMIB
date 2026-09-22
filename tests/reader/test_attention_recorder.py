"""B2: the opt-in attention recorder of MassWeightedGemma, and the ratio logic.

CPU only, NO model: the recorder is driven with random q/k/v tensors through the
patched ``F.scaled_dot_product_attention``, and its output is compared against a
hand-written softmax.  The instrument's verdict function is exercised on
synthetic share vectors.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from experiments.modal_spec_run import (
    INSTRUMENT_MIN_RATIO,
    build_instrument_prompt,
    check_ratio_grows,
    mass_share,
)
from server.mass_weighted_gemma import MassWeightedLLM


@pytest.fixture()
def llm():
    """A MassWeightedLLM with the sdpa patch active and no model behind it."""
    obj = MassWeightedLLM(model_id="fake/none", max_new_tokens=1, do_sample=False)
    obj._mass_weight = 1.0
    obj._prefill_mass_scale = 0.0
    obj._bias_cap = None
    obj._patch_sdpa()
    try:
        yield obj
    finally:
        obj.restore_sdpa()


def _manual_probs(q, k, mask=None, scale=None):
    factor = (q.shape[-1] ** -0.5) if scale is None else scale
    logits = (q.float() @ k.float().transpose(-2, -1)) * factor
    if mask is not None:
        logits = logits + mask.float()
    return torch.softmax(logits, dim=-1)[..., -1, :].mean(dim=(0, 1))


# --------------------------------------------------------------------------
# recorder math
# --------------------------------------------------------------------------

def test_recorder_matches_a_manual_softmax(llm) -> None:
    torch.manual_seed(0)
    b, h, sk, d = 2, 3, 7, 8
    q = torch.randn(b, h, 1, d)
    k = torch.randn(b, h, sk, d)
    v = torch.randn(b, h, sk, d)

    llm.start_attention_recording()
    F.scaled_dot_product_attention(q, k, v)
    recorded = llm.stop_attention_recording()

    assert len(recorded) == 1
    expected = _manual_probs(q, k)
    assert recorded[0].shape == (sk,)
    assert torch.allclose(recorded[0], expected, atol=1e-6)
    # a probability distribution, on cpu, in float32
    assert recorded[0].dtype is torch.float32
    assert float(recorded[0].sum()) == pytest.approx(1.0, abs=1e-5)


def test_recorder_includes_the_float_mask(llm) -> None:
    torch.manual_seed(1)
    q = torch.randn(1, 2, 1, 4)
    k = torch.randn(1, 2, 5, 4)
    v = torch.randn(1, 2, 5, 4)
    mask = torch.zeros(1, 1, 1, 5)
    mask[..., 2] = 4.0  # the "mass" on position 2

    llm.start_attention_recording()
    F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    recorded = llm.stop_attention_recording()

    assert torch.allclose(recorded[0], _manual_probs(q, k, mask), atol=1e-6)
    # the biased position dominates
    assert int(recorded[0].argmax()) == 2


def test_recorder_folds_a_bool_mask_the_same_way_as_combine_attn_mask(llm) -> None:
    torch.manual_seed(2)
    q = torch.randn(1, 1, 1, 4)
    k = torch.randn(1, 1, 4, 4)
    v = torch.randn(1, 1, 4, 4)
    allowed = torch.tensor([[[[True, True, False, True]]]])

    llm.start_attention_recording()
    F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
    recorded = llm.stop_attention_recording()

    # a forbidden key gets (essentially) zero probability, not 0.0 added bias
    assert float(recorded[0][2]) == pytest.approx(0.0, abs=1e-6)
    assert float(recorded[0].sum()) == pytest.approx(1.0, abs=1e-5)


def test_recorder_honours_an_explicit_scale(llm) -> None:
    torch.manual_seed(3)
    q = torch.randn(1, 1, 1, 4)
    k = torch.randn(1, 1, 6, 4)
    v = torch.randn(1, 1, 6, 4)

    llm.start_attention_recording()
    F.scaled_dot_product_attention(q, k, v, scale=0.05)
    recorded = llm.stop_attention_recording()

    assert torch.allclose(recorded[0], _manual_probs(q, k, scale=0.05), atol=1e-6)


def test_recorder_repeats_kv_heads_for_gqa(llm) -> None:
    torch.manual_seed(4)
    q = torch.randn(1, 4, 1, 8)
    k = torch.randn(1, 2, 5, 8)

    llm.start_attention_recording()
    llm._record_attention_row(q, k, None, None, True)
    recorded = llm.stop_attention_recording()

    expected = _manual_probs(q, k.repeat_interleave(2, dim=1))
    assert torch.allclose(recorded[0], expected, atol=1e-6)


def test_recorder_ignores_prefill_calls(llm) -> None:
    """Only seq_q == 1 (decode) calls are recorded: the 1D mass bias is not
    applied at prefill, so a prefill row would measure the un-injected model."""
    torch.manual_seed(5)
    q = torch.randn(1, 2, 9, 4)   # seq_q = 9 -> prefill
    k = torch.randn(1, 2, 9, 4)
    v = torch.randn(1, 2, 9, 4)

    llm.start_attention_recording()
    F.scaled_dot_product_attention(q, k, v)
    assert llm.stop_attention_recording() == []


def test_recording_is_off_by_default(llm) -> None:
    q = torch.randn(1, 1, 1, 4)
    k = torch.randn(1, 1, 3, 4)
    v = torch.randn(1, 1, 3, 4)
    F.scaled_dot_product_attention(q, k, v)
    assert llm.recorded_attention == []


def test_one_recorded_vector_per_call(llm) -> None:
    q = torch.randn(1, 1, 1, 4)
    k = torch.randn(1, 1, 3, 4)
    v = torch.randn(1, 1, 3, 4)
    llm.start_attention_recording()
    for _ in range(5):
        F.scaled_dot_product_attention(q, k, v)
    assert len(llm.stop_attention_recording()) == 5


# --------------------------------------------------------------------------
# mass_share / check_ratio_grows on synthetic vectors
# --------------------------------------------------------------------------

def test_mass_share_on_a_synthetic_vector() -> None:
    probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
    assert mass_share(probs, [1, 2]) == pytest.approx(0.5)
    assert mass_share(probs, []) == 0.0
    # out-of-range positions are dropped, not counted
    assert mass_share(probs, [99]) == 0.0


def test_ratio_logic_accepts_a_growing_share() -> None:
    """The share a mass-4.0 planet gets under w = 0 / 0.25 / 1.0, as the softmax
    would produce it: s -> s*e^(w*4) / (s*e^(w*4) + (1-s))."""
    import math

    base = 0.02
    shares = {}
    for w in (0.0, 0.25, 1.0):
        boost = math.exp(w * 4.0)
        shares[w] = base * boost / (base * boost + (1 - base))
    verdict = check_ratio_grows(shares)
    assert verdict["ratio_monotone"]
    assert verdict["ratio_at_least_min"]
    assert verdict["ratio"] >= INSTRUMENT_MIN_RATIO
    # ...and renormalisation keeps it well BELOW the naive exp(4) = 54.6
    assert verdict["ratio"] < math.exp(4.0)


def test_ratio_logic_rejects_a_flat_share() -> None:
    verdict = check_ratio_grows({0.0: 0.05, 0.25: 0.05, 1.0: 0.05})
    assert not verdict["ratio_monotone"]
    assert not verdict["ratio_at_least_min"]


def test_ratio_logic_rejects_a_shrinking_share() -> None:
    verdict = check_ratio_grows({0.0: 0.4, 0.25: 0.3, 1.0: 0.2})
    assert not verdict["ratio_monotone"]
    assert not verdict["ratio_at_least_min"]


def test_ratio_logic_rejects_growth_below_the_bound() -> None:
    verdict = check_ratio_grows({0.0: 0.10, 0.25: 0.12, 1.0: 0.15})
    assert verdict["ratio_monotone"]
    assert not verdict["ratio_at_least_min"]


# --------------------------------------------------------------------------
# the instrument prompt
# --------------------------------------------------------------------------

def test_instrument_prompt_has_exactly_one_planet_line() -> None:
    prompt = build_instrument_prompt()
    assert sum(1 for ln in prompt.splitlines() if "[PN" in ln) == 1
    assert "[PN4.0]" in prompt
    assert any(ln.startswith("[SN] ") for ln in prompt.splitlines())
    assert any(ln.startswith("    [RN] ") for ln in prompt.splitlines())


def test_instrument_prompt_is_roughly_300_tokens() -> None:
    from benchmark.bineval.arms import make_token_counter

    tokens = make_token_counter()(build_instrument_prompt())
    assert 200 <= tokens <= 450, tokens

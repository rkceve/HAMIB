"""Item A2: ``build_mass_bias(phase_is_decode=...)`` and the prefill/decode
phase flag of ``MassWeightedGemma`` (CPU, no model)."""

from __future__ import annotations

import pytest
import torch

from server.mass_weighted_gemma import MassWeightedGemma, build_mass_bias, make_first_token_timer

DT, DEV = torch.float32, torch.device("cpu")


def _mass(n: int) -> torch.Tensor:
    v = torch.zeros(n)
    v[2] = 1.0
    return v


def _bias(seq_q, seq_k, mass, **kw):
    return build_mass_bias(
        seq_q, seq_k, m_matrix=None, mass_vector=mass, mass_weight=0.7,
        prefill_mass_scale=kw.pop("prefill_mass_scale", 0.0), dtype=DT, device=DEV, **kw,
    )


def test_phase_none_keeps_the_seq_q_heuristic() -> None:
    mass = _mass(8)
    assert _bias(1, 8, mass) is not None        # seq_q == 1 -> decode
    assert _bias(4, 8, mass) is None            # seq_q > 1, scale 0 -> prefill skip
    with pytest.raises(RuntimeError, match="sliding-window"):
        _bias(1, 6, mass)                       # cropped cache on decode -> raise


def test_explicit_prefill_with_seq_q_one_applies_no_bias() -> None:
    """A 1-token final prefill chunk must get the PREFILL rule."""
    mass = _mass(11)
    assert _bias(1, 11, mass, phase_is_decode=False) is None
    assert _bias(1, 11, mass, phase_is_decode=True) is not None


def test_explicit_prefill_slices_a_growing_cache() -> None:
    mass = _mass(13)
    out = _bias(5, 10, mass, phase_is_decode=False, prefill_mass_scale=0.5)
    assert out is not None and tuple(out.shape) == (1, 1, 1, 10)
    assert out[0, 0, 0, 2].item() == pytest.approx(0.7 * 0.5)
    # decode with a cropped cache still raises
    with pytest.raises(RuntimeError, match="sliding-window"):
        _bias(1, 10, mass, phase_is_decode=True)


def test_prefill_done_defaults_true_and_timer_callback_flips_it() -> None:
    inst = MassWeightedGemma.__new__(MassWeightedGemma)  # verify_attention_math style
    assert inst._prefill_done is True
    hits: list[int] = []
    timer = make_first_token_timer(on_first_call=lambda: hits.append(1))
    ids = torch.zeros((1, 3), dtype=torch.long)
    timer(ids, torch.randn(1, 5))
    timer(ids, torch.randn(1, 5))
    assert hits == [1]

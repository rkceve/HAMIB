"""Astra round 2 (2026-09-18), items 1-4, CPU only, no model.

1. GQA + explicit mask: when the bias is applied and ``enable_gqa`` is set the
   patch expands K/V itself and calls the original with ``enable_gqa=False``
   (so torch never falls back to the float32 math kernel); the no-bias path is
   untouched.
2. Patch idempotence / ownership: ``_ORIGINAL_SDPA`` captured once; double
   patch on one instance -> single wrapper; a second instance -> RuntimeError;
   restore -> original identity.
3. Recorder: records by PHASE (``is_decode``), not ``seq_q == 1``; refuses
   ``seq_k > RECORD_MAX_KEYS``.
4. Single bias cap: run_reader passes ``cap=None`` to the mass vector and the
   only cap is the effective-bias cap in ``build_mass_bias``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from benchmark.bineval import run_reader as rr
from server import mass_weighted_gemma as mwg
from server.mass_weighted_gemma import MassWeightedGemma, build_mass_bias

D = 8


def _bare(mass_vector=None, *, mass_weight: float = 0.7, **attrs) -> MassWeightedGemma:
    """verify_attention_math style: no __init__, only what patched_sdpa reads."""
    inst = MassWeightedGemma.__new__(MassWeightedGemma)
    inst._mass_weight = mass_weight
    inst._prefill_mass_scale = 0.0
    inst._qk_norm_mode = "off"
    inst._m_matrix = None
    inst._mass_vector = mass_vector
    inst._allow_sliding_layers = False
    inst._bias_cap = None
    inst.bias_applied_calls = 0
    inst.bias_skipped_prefill_calls = 0
    inst.bias_skipped_sliding_calls = 0
    for k, v in attrs.items():
        setattr(inst, k, v)
    return inst


def _mass(n: int) -> torch.Tensor:
    v = torch.zeros(n)
    v[2] = 1.5
    v[5] = 0.5
    return v


@pytest.fixture(autouse=True)
def _sdpa_is_pristine():
    assert F.scaled_dot_product_attention is mwg._ORIGINAL_SDPA
    yield
    assert F.scaled_dot_product_attention is mwg._ORIGINAL_SDPA


class _Spy:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, query, key, value, **kw):
        self.calls.append({"key": key, "value": value, **kw})
        return mwg._ORIGINAL_SDPA(query, key, value, **kw)


# ---------------------------------------------------------------- item 1


def test_gqa_with_bias_expands_kv_and_disables_enable_gqa() -> None:
    g = torch.Generator().manual_seed(1)
    h_q, h_kv, seq_k = 4, 2, 10
    q = torch.randn(1, h_q, 1, D, generator=g)
    k = torch.randn(1, h_kv, seq_k, D, generator=g)
    v = torch.randn(1, h_kv, seq_k, D, generator=g)
    mass = _mass(seq_k)
    inst = _bare(mass)
    inst._patch_sdpa()
    spy = _Spy()
    inst._original_sdpa = spy
    try:
        got = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    finally:
        inst.restore_sdpa()
    assert len(spy.calls) == 1
    call = spy.calls[0]
    assert call["enable_gqa"] is False
    assert tuple(call["key"].shape) == (1, h_q, seq_k, D)
    assert tuple(call["value"].shape) == (1, h_q, seq_k, D)
    assert call["key"].dtype is k.dtype and call["value"].dtype is v.dtype
    k_rep = k.repeat_interleave(h_q // h_kv, dim=1)
    v_rep = v.repeat_interleave(h_q // h_kv, dim=1)
    ref = mwg._ORIGINAL_SDPA(q, k_rep, v_rep, attn_mask=(0.7 * mass).view(1, 1, 1, seq_k))
    assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max()
    assert inst.bias_applied_calls == 1


def test_gqa_expansion_keeps_bf16() -> None:
    g = torch.Generator().manual_seed(2)
    q = torch.randn(1, 4, 1, D, generator=g).to(torch.bfloat16)
    k = torch.randn(1, 2, 6, D, generator=g).to(torch.bfloat16)
    v = torch.randn(1, 2, 6, D, generator=g).to(torch.bfloat16)
    inst = _bare(_mass(6))
    inst._patch_sdpa()
    spy = _Spy()
    inst._original_sdpa = spy
    try:
        F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    finally:
        inst.restore_sdpa()
    assert spy.calls[0]["key"].dtype is torch.bfloat16
    assert spy.calls[0]["value"].dtype is torch.bfloat16
    assert spy.calls[0]["enable_gqa"] is False


def test_no_bias_path_forwards_enable_gqa_untouched() -> None:
    g = torch.Generator().manual_seed(3)
    q = torch.randn(1, 4, 1, D, generator=g)
    k = torch.randn(1, 2, 6, D, generator=g)
    v = torch.randn(1, 2, 6, D, generator=g)
    inst = _bare(None)  # no mass vector: nothing to add
    inst._patch_sdpa()
    spy = _Spy()
    inst._original_sdpa = spy
    try:
        got = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    finally:
        inst.restore_sdpa()
    assert spy.calls[0]["enable_gqa"] is True
    assert tuple(spy.calls[0]["key"].shape) == (1, 2, 6, D)
    assert torch.equal(got, mwg._ORIGINAL_SDPA(q, k, v, enable_gqa=True))


# ---------------------------------------------------------------- item 2


def test_original_sdpa_is_the_torch_kernel() -> None:
    assert mwg._ORIGINAL_SDPA is torch._C._nn.scaled_dot_product_attention


def test_double_patch_on_one_instance_is_a_single_wrapper() -> None:
    inst = _bare(_mass(6))
    inst._patch_sdpa()
    try:
        first = F.scaled_dot_product_attention
        inst._patch_sdpa()  # idempotent
        assert F.scaled_dot_product_attention is first
        assert inst._original_sdpa is mwg._ORIGINAL_SDPA
        q, k, v = (torch.randn(1, 1, 1, D), torch.randn(1, 1, 6, D), torch.randn(1, 1, 6, D))
        got = F.scaled_dot_product_attention(q, k, v)
        ref = mwg._ORIGINAL_SDPA(q, k, v, attn_mask=(0.7 * _mass(6)).view(1, 1, 1, 6))
        assert torch.allclose(got, ref, atol=1e-6)
        assert inst.bias_applied_calls == 1  # one wrapper, counted once
    finally:
        inst.restore_sdpa()
    assert F.scaled_dot_product_attention is mwg._ORIGINAL_SDPA
    assert torch.nn.functional.scaled_dot_product_attention is mwg._ORIGINAL_SDPA


def test_second_instance_cannot_patch_over_the_first() -> None:
    a, b = _bare(_mass(6)), _bare(_mass(6))
    a._patch_sdpa()
    try:
        with pytest.raises(RuntimeError, match="already patched"):
            b._patch_sdpa()
        assert F.scaled_dot_product_attention is not mwg._ORIGINAL_SDPA
    finally:
        a.restore_sdpa()
    assert F.scaled_dot_product_attention is mwg._ORIGINAL_SDPA
    # ownership cleared: b may patch now, and restore hands the kernel back
    b._patch_sdpa()
    b.restore_sdpa()
    assert F.scaled_dot_product_attention is mwg._ORIGINAL_SDPA


def test_restore_by_a_non_owner_leaves_the_patch_alone() -> None:
    a, b = _bare(_mass(6)), _bare(_mass(6))
    a._patch_sdpa()
    try:
        patched = F.scaled_dot_product_attention
        b.restore_sdpa()  # b never patched
        assert F.scaled_dot_product_attention is patched
    finally:
        a.restore_sdpa()


# ---------------------------------------------------------------- item 3


def test_recorder_uses_the_phase_not_seq_q() -> None:
    q, k, v = torch.randn(1, 2, 1, D), torch.randn(1, 2, 9, D), torch.randn(1, 2, 9, D)
    inst = _bare(None)
    inst._patch_sdpa()
    try:
        inst.start_attention_recording()
        inst._prefill_done = False  # a 1-token FINAL prefill chunk
        F.scaled_dot_product_attention(q, k, v)
        assert inst.recorded_attention == []
        inst._prefill_done = True  # decode
        F.scaled_dot_product_attention(q, k, v)
        assert len(inst.stop_attention_recording()) == 1
    finally:
        inst.restore_sdpa()


def test_recorder_refuses_more_than_record_max_keys() -> None:
    assert mwg.RECORD_MAX_KEYS == 4096
    n = mwg.RECORD_MAX_KEYS + 1
    q, k, v = torch.randn(1, 1, 1, D), torch.randn(1, 1, n, D), torch.randn(1, 1, n, D)
    inst = _bare(None)
    inst._patch_sdpa()
    try:
        inst.start_attention_recording()
        with pytest.raises(RuntimeError, match="4097"):
            F.scaled_dot_product_attention(q, k, v)
        # at the limit it records
        F.scaled_dot_product_attention(q, k[:, :, :4096], v[:, :, :4096])
        assert len(inst.stop_attention_recording()) == 1
    finally:
        inst.restore_sdpa()


# ---------------------------------------------------------------- item 4

BLOCK = "<CONTEXT>\n[SN] Plan\n  [PN10.0] Budget\n</CONTEXT>"


class _CharTok:
    def __init__(self, text: str) -> None:
        self.text = text

    def __call__(self, prompt, return_tensors=None):
        self.text = prompt
        return {"input_ids": list(range(len(prompt)))}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.text[i] for i in ids)


def test_single_cap_mass_10_w_half_cap_3_gives_bias_3() -> None:
    tok = _CharTok(BLOCK)
    vec, info = rr.build_reader_mass_vector(list(range(len(BLOCK))), tok, "planet", None, w=0.5)
    assert info.planet_spans == 1 and float(vec.max()) == 10.0  # uncapped mass
    bias = build_mass_bias(
        1, len(BLOCK), m_matrix=None, mass_vector=vec, mass_weight=0.5,
        prefill_mass_scale=0.0, dtype=torch.float32, device=torch.device("cpu"),
        bias_cap=3.0, phase_is_decode=True,
    )
    assert float(bias.max()) == pytest.approx(3.0)  # was 1.5 under the double cap


def test_run_reader_passes_no_cap_to_the_mass_vector(monkeypatch) -> None:
    seen: list = []
    real = rr.build_reader_mass_vector

    def spy(prompt_ids, tokenizer, inject, cap=None, **kw):
        seen.append(cap)
        return real(prompt_ids, tokenizer, inject, cap, **kw)

    monkeypatch.setattr(rr, "build_reader_mass_vector", spy)

    class _Stub:
        tokenizer = _CharTok("")
        _model = None

        def generate(self, prompt):
            return "x"

        def set_mass_vector(self, v):
            pass

        def clear_mass_vector(self):
            pass

        def mass_injection_stats(self):
            return {"bias_applied_calls": 0, "bias_skipped_prefill_calls": 0,
                    "bias_skipped_sliding_calls": 0}

    rr.run_reader(_Stub(), BLOCK, [{"qid": "q", "question": "?"}], w=0.5, inject="planet",
                  bias_cap=3.0)
    assert seen == [None]

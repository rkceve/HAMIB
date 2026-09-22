"""H15 option (b): the ``prefill_last_row`` switch (Astra round 2, item 6).

Off by default.  When on, during the LAST prefill forward (``seq_k ==
prompt_len``) the bias ``w * mass`` is added to the final query row only:
the row that produces the first answer token.  Every other row keeps the
plain prefill output.  Counted in ``bias_applied_prefill_last_row_calls``.

(a) switch off: counters as before, first token == no-bias run;
(b) switch on, large w, tiny model: counter == n_sdpa_layers, last-position
    logits differ from the switch-off run, earlier positions unchanged;
(c) closure numerics on CPU (2 q heads : 1 kv head, GQA + causal): the patched
    final row equals a float32 softmax((q_last k^T) * scale + bias) v within
    1e-4 and the other rows equal the original kernel.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from server import mass_weighted_gemma as mwg
from server.mass_weighted_gemma import MassWeightedGemma
from tests.mcbuild._tiny_qwen import N_SDPA_LAYERS, VOCAB, make_llm

PROMPT = "abcdefghijklm"


def _mass(n: int) -> torch.Tensor:
    v = torch.zeros(n)
    v[3:6] = 2.0
    return v


def _prefill_logits(llm, prompt: str) -> torch.Tensor:
    """One forward of the prompt in the state generate() gives its prefill."""
    ids = llm.tokenizer(prompt, return_tensors="pt")["input_ids"].to(next(llm._model.parameters()).device)
    llm._prompt_len = ids.shape[1]
    llm._prefill_done = False
    try:
        with torch.no_grad():
            out = llm._model(input_ids=ids, use_cache=False)
    finally:
        llm._prefill_done = True
    return out.logits[0].float()


# ---------------------------------------------------------------- (a) / (b) tiny model


def test_constructor_default_is_off_and_kwarg_sets_it(tiny_checkpoint, monkeypatch) -> None:
    on = make_llm(tiny_checkpoint, monkeypatch, prefill_last_row=True)
    try:
        assert on.prefill_last_row is True
    finally:
        on.restore_sdpa()
    off = make_llm(tiny_checkpoint, monkeypatch)
    try:
        assert off.prefill_last_row is False
        assert MassWeightedGemma.prefill_last_row is False  # __new__ instances too
    finally:
        off.restore_sdpa()


def test_switch_off_keeps_counters_and_first_token(llm) -> None:
    llm.clear_mass_vector()
    plain = llm.generate(PROMPT)
    llm.set_mass_vector(_mass(len(PROMPT)))
    injected = llm.generate(PROMPT)
    n = llm.last_generated_tokens
    assert injected[0] == plain[0]  # first token comes out of the unbiased prefill
    assert llm.mass_injection_stats() == {
        "bias_applied_calls": N_SDPA_LAYERS * (n - 1),
        "bias_skipped_prefill_calls": N_SDPA_LAYERS,
        "bias_skipped_sliding_calls": 0,
    }
    assert llm.bias_applied_prefill_last_row_calls == 0


def test_switch_on_biases_only_the_last_prompt_row(tiny_checkpoint, monkeypatch) -> None:
    llm = make_llm(tiny_checkpoint, monkeypatch, w=50.0, prefill_last_row=True)
    try:
        llm.set_mass_vector(_mass(len(PROMPT)))
        llm.generate(PROMPT)
        n = llm.last_generated_tokens
        assert llm.bias_applied_prefill_last_row_calls == N_SDPA_LAYERS
        assert llm.mass_injection_stats() == {
            "bias_applied_calls": N_SDPA_LAYERS * (n - 1),
            "bias_skipped_prefill_calls": N_SDPA_LAYERS,
            "bias_skipped_sliding_calls": 0,
        }
        assert llm._prompt_len == len(PROMPT)

        on = _prefill_logits(llm, PROMPT)
        # the manual forward is another last-chunk forward; the counter is only
        # reset by set_mass_vector / clear_mass_vector
        assert llm.bias_applied_prefill_last_row_calls == 2 * N_SDPA_LAYERS
        llm.prefill_last_row = False
        off = _prefill_logits(llm, PROMPT)
        llm.prefill_last_row = True
        assert not torch.allclose(on[-1], off[-1], atol=1e-4), "last-row bias had no effect"
        assert torch.allclose(on[:-1], off[:-1], atol=1e-5), "earlier rows were touched"
        # a forward of the prompt WITHOUT its last token is not the last chunk:
        # seq_k != prompt_len, so nothing is biased and the output is the plain one
        llm._prompt_len = len(PROMPT)
        llm._prefill_done = False
        before = llm.bias_applied_prefill_last_row_calls
        ids = llm.tokenizer(PROMPT[:-1], return_tensors="pt")["input_ids"].to(next(llm._model.parameters()).device)
        try:
            with torch.no_grad():
                short_on = llm._model(input_ids=ids, use_cache=False).logits[0].float()
        finally:
            llm._prefill_done = True
        assert llm.bias_applied_prefill_last_row_calls == before
        assert torch.allclose(short_on, off[:-1], atol=1e-5)
    finally:
        llm.restore_sdpa()


def test_switch_on_refuses_prefill_scale(tiny_checkpoint, monkeypatch) -> None:
    llm = make_llm(tiny_checkpoint, monkeypatch, prefill_last_row=True)
    try:
        llm._prefill_mass_scale = 0.5
        llm.set_mass_vector(_mass(len(PROMPT)))
        with pytest.raises(ValueError, match="prefill_last_row"):
            llm.generate(PROMPT)
    finally:
        llm.restore_sdpa()


# ---------------------------------------------------------------- (c) closure numerics

D = 8


def _bare(mass, *, prompt_len: int, w: float = 0.7, on: bool = True) -> MassWeightedGemma:
    inst = MassWeightedGemma.__new__(MassWeightedGemma)
    inst._mass_weight = w
    inst._prefill_mass_scale = 0.0
    inst._qk_norm_mode = "off"
    inst._m_matrix = None
    inst._mass_vector = mass
    inst._allow_sliding_layers = False
    inst._bias_cap = None
    inst.bias_applied_calls = 0
    inst.bias_skipped_prefill_calls = 0
    inst.bias_skipped_sliding_calls = 0
    inst.bias_applied_prefill_last_row_calls = 0
    inst.prefill_last_row = on
    inst._prefill_done = False
    inst._prompt_len = prompt_len
    return inst


def _manual_last_row(q, k, v, bias_row, scale):
    """float32 softmax((q_last k^T) * scale + bias_row) v, kv heads repeated."""
    groups = q.shape[1] // k.shape[1]
    k32 = k.float().repeat_interleave(groups, dim=1)
    v32 = v.float().repeat_interleave(groups, dim=1)
    q_last = q[:, :, -1:, :].float()
    logits = (q_last @ k32.transpose(-2, -1)) * scale + bias_row.float()
    return torch.softmax(logits, dim=-1) @ v32


@pytest.mark.parametrize("use_bool_mask", [False, True])
def test_closure_last_row_matches_manual_softmax(use_bool_mask: bool) -> None:
    g = torch.Generator().manual_seed(7)
    h_q, h_kv, seq = 2, 1, 6
    q = torch.randn(1, h_q, seq, D, generator=g)
    k = torch.randn(1, h_kv, seq, D, generator=g)
    v = torch.randn(1, h_kv, seq, D, generator=g)
    mass = _mass(seq)
    w, scale = 0.7, D ** -0.5
    if use_bool_mask:
        mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool)).view(1, 1, seq, seq)
        mask[..., 1] = False  # one padded key, forbidden in every row
        kw = dict(attn_mask=mask, is_causal=False)
    else:
        kw = dict(attn_mask=None, is_causal=True)

    inst = _bare(mass, prompt_len=seq, w=w)
    inst._patch_sdpa()
    try:
        got = F.scaled_dot_product_attention(q, k, v, enable_gqa=True, scale=scale, **kw)
    finally:
        inst.restore_sdpa()
    ref_plain = mwg._ORIGINAL_SDPA(q, k, v, enable_gqa=True, scale=scale, **kw)
    assert torch.allclose(got[:, :, :-1], ref_plain[:, :, :-1], atol=1e-6), "earlier rows changed"

    bias_row = (w * mass).view(1, 1, 1, seq)
    if use_bool_mask:
        bias_row = bias_row.masked_fill(~mask[..., -1:, :], float("-inf"))
    ref_last = _manual_last_row(q, k, v, bias_row, scale)
    err = (got[:, :, -1:] - ref_last).abs().max().item()
    assert err < 1e-4, err
    assert not torch.allclose(got[:, :, -1:], ref_plain[:, :, -1:], atol=1e-3)
    assert inst.bias_applied_prefill_last_row_calls == 1
    assert inst.bias_skipped_prefill_calls == 1  # the other rows are still a skipped prefill
    assert inst.bias_applied_calls == 0


def test_closure_ignores_non_final_chunks_and_the_off_switch() -> None:
    g = torch.Generator().manual_seed(8)
    q = torch.randn(1, 2, 4, D, generator=g)
    k = torch.randn(1, 1, 4, D, generator=g)
    v = torch.randn(1, 1, 4, D, generator=g)
    plain = mwg._ORIGINAL_SDPA(q, k, v, enable_gqa=True, is_causal=True)
    # seq_k (4) != prompt_len (9): an earlier chunk of a chunked prefill
    inst = _bare(_mass(9), prompt_len=9)
    inst._patch_sdpa()
    try:
        got = F.scaled_dot_product_attention(q, k, v, enable_gqa=True, is_causal=True)
    finally:
        inst.restore_sdpa()
    assert torch.equal(got, plain) and inst.bias_applied_prefill_last_row_calls == 0
    # switch off: the last chunk is a plain prefill
    inst = _bare(_mass(4), prompt_len=4, on=False)
    inst._patch_sdpa()
    try:
        got = F.scaled_dot_product_attention(q, k, v, enable_gqa=True, is_causal=True)
    finally:
        inst.restore_sdpa()
    assert torch.equal(got, plain) and inst.bias_applied_prefill_last_row_calls == 0
    assert inst.bias_skipped_prefill_calls == 1


def test_vocab_constant_is_shared() -> None:
    assert VOCAB == 512

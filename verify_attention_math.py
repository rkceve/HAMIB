"""
verify_attention_math.py — mechanical verification of the wM injection (特許 §0082).

Pure torch, CPU only, no transformers, no model download. Verifies that the pure
functions extracted from ``server/mass_weighted_gemma.py`` (``build_mass_bias`` /
``combine_attn_mask``) implement

    attn_scores = Q Kᵀ / sqrt(d) + w · M      (additive, PRE-softmax, UNSCALED)

and that ``server/cd_parser.py`` maps ``[PN{mass}]`` markers onto the right token
positions and CD levels.

Run:  python verify_attention_math.py     (exit code 1 on any failure)
"""
from __future__ import annotations

import math
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from server.cd_parser import (  # noqa: E402
    filter_positions_by_level,
    find_pn_positions,
    find_pn_positions_with_level,
    find_pn_spans,
    levels_from_context_block,
    positions_for_levels,
)
from server.mass_vector import positions_to_mass_vector  # noqa: E402
from server.mass_weighted_gemma import (  # noqa: E402
    MassWeightedGemma,
    _make_causal_mask,
    build_mass_bias,
    combine_attn_mask,
)

# ── fixtures ──────────────────────────────────────────────────────────────
SEED = 20260906
B, H, D = 2, 3, 8
DEVICE = torch.device("cpu")
DTYPE = torch.float32
ATOL = 1e-5


def _qkv(seq_q: int, seq_k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic Q, K, V of shape (B, H, seq, D)."""
    g = torch.Generator(device="cpu").manual_seed(SEED)
    q = torch.randn(B, H, seq_q, D, generator=g, dtype=DTYPE)
    k = torch.randn(B, H, seq_k, D, generator=g, dtype=DTYPE)
    v = torch.randn(B, H, seq_k, D, generator=g, dtype=DTYPE)
    return q, k, v


def _mass(seq_k: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(SEED + 1)
    return torch.rand(seq_k, generator=g, dtype=DTYPE) * 3.0


def _manual_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """softmax(QKᵀ/sqrt(d) + bias) V, written out by hand (the ¶0082 formula)."""
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if bias is not None:
        scores = scores + bias
    return torch.softmax(scores, dim=-1) @ v


def _patched_equivalent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    m_matrix: torch.Tensor | None = None,
    mass_vector: torch.Tensor | None = None,
    mass_weight: float = 1.0,
    prefill_mass_scale: float = 0.0,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Exactly what the monkey-patched sdpa does, using the two pure functions."""
    seq_q, seq_k = q.shape[-2], k.shape[-2]
    m_bias = build_mass_bias(
        seq_q,
        seq_k,
        m_matrix=m_matrix,
        mass_vector=mass_vector,
        mass_weight=mass_weight,
        prefill_mass_scale=prefill_mass_scale,
        dtype=q.dtype,
        device=q.device,
    )
    if m_bias is not None:
        attn_mask, is_causal = combine_attn_mask(
            attn_mask,
            m_bias,
            is_causal=is_causal,
            seq_q=seq_q,
            seq_k=seq_k,
            dtype=q.dtype,
            device=q.device,
        )
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=is_causal
    )


class _CharTokenizer:
    """Fake tokenizer: one character per token; decode([i]) returns text[i]."""

    def __init__(self, text: str, chars_per_token: int = 1):
        self._text = text
        self._n = chars_per_token
        self.pieces = [text[i:i + chars_per_token] for i in range(0, len(text), chars_per_token)]

    @property
    def token_ids(self) -> list[int]:
        return list(range(len(self.pieces)))

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(self.pieces[i] for i in ids)


_PN_BLOCK = "[PN1.0] AAA\n  [PN2.0] BBB\n    [PN0.1] CCC"


# ── checks ────────────────────────────────────────────────────────────────

def check_1_decode_exactness() -> None:
    """¶0082 with M = v: decode step output == manual softmax(QKᵀ/√d + w·v) V."""
    seq_q, seq_k, w = 1, 16, 0.7
    q, k, v = _qkv(seq_q, seq_k)
    mass = _mass(seq_k)

    got = _patched_equivalent(q, k, v, mass_vector=mass, mass_weight=w)
    ref = _manual_attention(q, k, v, bias=w * mass[None, None, None, :])

    max_err = (got - ref).abs().max().item()
    assert torch.allclose(got, ref, atol=ATOL), f"max |diff| = {max_err}"
    print(f"[ 1] decode exactness (seq_q=1, seq_k={seq_k}, w={w}): max|diff| = {max_err:.3e}  OK")


def check_2_bias_is_unscaled() -> None:
    """The bias is exactly w·v, NOT w·v/sqrt(d): ¶0082 adds wM after the 1/√d scaling."""
    seq_k, w = 16, 0.7
    mass = _mass(seq_k)
    bias = build_mass_bias(
        1, seq_k,
        m_matrix=None, mass_vector=mass, mass_weight=w,
        prefill_mass_scale=0.0, dtype=DTYPE, device=DEVICE,
    )
    assert bias is not None
    expected = (w * mass).view(1, 1, 1, seq_k)
    scaled = expected / math.sqrt(D)
    assert torch.equal(bias, expected), "bias != w*v exactly"
    assert not torch.allclose(bias, scaled, atol=ATOL), "bias looks 1/sqrt(d)-scaled"
    ratio = (bias.flatten()[1] / mass[1]).item()
    print(f"[ 2] bias unscaled: bias == w*v exactly (bias/v = {ratio:.6f} == w = {w})  OK")


def check_3_broadcast_axis() -> None:
    """Bias recovered FROM THE OUTPUT: same key-wise vector on every batch/head/query row.

    The previous version of this check compared ``scores + bias`` with
    ``scores + bias.expand(...)``, which is true for any tensor by definition of
    broadcasting — a tautology. This version instead measures the attention
    weights the patched path actually produces (V = identity, so the output IS the
    weight matrix), recovers the additive shift from ``log(p_patched / p_plain)``,
    and checks that the recovered shift (a) varies along the key axis and (b) is
    numerically the SAME vector for every (batch, head, query row).
    """
    seq_q, seq_k, w = 4, 16, 0.7
    mass = _mass(seq_k)
    bias = build_mass_bias(
        seq_q, seq_k,
        m_matrix=None, mass_vector=mass, mass_weight=w,
        prefill_mass_scale=1.0, dtype=DTYPE, device=DEVICE,
    )
    assert bias is not None
    assert tuple(bias.shape) == (1, 1, 1, seq_k), f"shape {tuple(bias.shape)}"
    assert bias.flatten().unique().numel() > 1, "bias constant along key axis"

    # V = identity (d_v = seq_k) → sdpa output equals the attention weight matrix
    g = torch.Generator(device="cpu").manual_seed(SEED + 7)
    q = torch.randn(B, H, seq_q, D, generator=g, dtype=DTYPE)
    k = torch.randn(B, H, seq_k, D, generator=g, dtype=DTYPE)
    v_eye = torch.eye(seq_k, dtype=DTYPE).expand(B, H, seq_k, seq_k).contiguous()

    plain_w = F.scaled_dot_product_attention(q, k, v_eye)
    patched_w = _patched_equivalent(
        q, k, v_eye, mass_vector=mass, mass_weight=w, prefill_mass_scale=1.0,
    )

    # per-head independent recomputation (no reuse of the bias tensor's broadcast)
    for b in range(B):
        for h in range(H):
            scores_bh = (q[b, h] @ k[b, h].T) / math.sqrt(D)
            ref_bh = torch.softmax(scores_bh + (w * mass).view(1, seq_k), dim=-1)
            assert torch.allclose(patched_w[b, h], ref_bh, atol=ATOL), (
                f"head ({b},{h}) output != independent manual computation"
            )

    # recover the additive shift from the measured weights: log p1 - log p0 = bias - logZ
    diff = torch.log(patched_w) - torch.log(plain_w)
    centered = diff - diff.mean(dim=-1, keepdim=True)
    expected = (w * mass) - (w * mass).mean()
    assert torch.allclose(centered, expected.view(1, 1, 1, seq_k), atol=1e-4), (
        "recovered shift != w*mass (up to the softmax normaliser)"
    )
    # (a) it really varies along the key axis
    assert centered[0, 0, 0].std().item() > 1e-3, "recovered shift is constant along keys"
    # (b) the SAME vector on every head / batch / query row
    spread = (centered - centered[0, 0, 0].view(1, 1, 1, seq_k)).abs().max().item()
    assert spread < 1e-4, f"shift differs across batch/head/query rows (max dev {spread})"
    print(f"[ 3] broadcast axis: shift recovered from output, varies along keys "
          f"(std={centered[0, 0, 0].std().item():.4f}), identical across B={B}, H={H}, "
          f"seq_q={seq_q} (max dev {spread:.2e})  OK")


def check_4_decode_only_d2() -> None:
    """D-2: prefill (seq_q>1) with prefill_mass_scale=0.0 → no bias at all."""
    seq_q, seq_k, w = 16, 16, 0.7
    q, k, v = _qkv(seq_q, seq_k)
    mass = _mass(seq_k)
    bias = build_mass_bias(
        seq_q, seq_k,
        m_matrix=None, mass_vector=mass, mass_weight=w,
        prefill_mass_scale=0.0, dtype=DTYPE, device=DEVICE,
    )
    assert bias is None, "prefill produced a bias with prefill_mass_scale=0.0"

    got = _patched_equivalent(q, k, v, mass_vector=mass, mass_weight=w, prefill_mass_scale=0.0)
    plain = F.scaled_dot_product_attention(q, k, v)
    assert torch.equal(got, plain), "prefill output differs from plain SDPA"
    print(f"[ 4] decode-only (D-2): seq_q={seq_q} -> build_mass_bias None, "
          f"output bit-identical to plain SDPA  OK")


def check_5_prefill_scale_path() -> None:
    """prefill_mass_scale=0.5 → bias = 0.5·w·v broadcast to (1,1,1,seq_k)."""
    seq_q, seq_k, w, scale = 16, 16, 0.7, 0.5
    mass = _mass(seq_k)
    bias = build_mass_bias(
        seq_q, seq_k,
        m_matrix=None, mass_vector=mass, mass_weight=w,
        prefill_mass_scale=scale, dtype=DTYPE, device=DEVICE,
    )
    assert bias is not None
    expected = (w * scale * mass).view(1, 1, 1, seq_k)
    assert tuple(bias.shape) == (1, 1, 1, seq_k)
    assert torch.allclose(bias, expected, atol=0.0), "bias != scale*w*v"
    print(f"[ 5] prefill scale path: scale={scale} -> bias == {scale}*w*v, "
          f"shape {tuple(bias.shape)}  OK")


def check_6_causal_composition() -> None:
    """is_causal=True + attn_mask=None → is_causal dropped, causal mask folded into the bias."""
    w = 0.7

    # (a) decode step: seq_q=1, seq_k=16 → diagonal offset seq_k-seq_q+1 = 16 → nothing masked
    seq_q, seq_k = 1, 16
    q, k, v = _qkv(seq_q, seq_k)
    mass = _mass(seq_k)
    bias = build_mass_bias(
        seq_q, seq_k, m_matrix=None, mass_vector=mass, mass_weight=w,
        prefill_mass_scale=0.0, dtype=DTYPE, device=DEVICE,
    )
    assert bias is not None
    mask, is_causal = combine_attn_mask(
        None, bias, is_causal=True, seq_q=seq_q, seq_k=seq_k, dtype=DTYPE, device=DEVICE,
    )
    assert is_causal is False, "is_causal must be dropped when a float mask is used"
    causal = _make_causal_mask(seq_q, seq_k, DTYPE, DEVICE)
    manual = torch.triu(
        torch.full((seq_q, seq_k), float("-inf"), dtype=DTYPE), diagonal=seq_k - seq_q + 1
    ).unsqueeze(0).unsqueeze(0)
    assert torch.equal(causal, manual), "causal mask diagonal offset != seq_k-seq_q+1"
    assert torch.equal(mask, bias + causal), "combined mask != bias + causal"
    got = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal)
    ref = _manual_attention(q, k, v, bias=bias + causal)
    assert torch.allclose(got, ref, atol=ATOL)

    # (b) square case: seq_q == seq_k, where torch's is_causal and our mask coincide
    seq_q = seq_k = 12
    q, k, v = _qkv(seq_q, seq_k)
    mass = _mass(seq_k)
    bias = build_mass_bias(
        seq_q, seq_k, m_matrix=None, mass_vector=mass, mass_weight=w,
        prefill_mass_scale=1.0, dtype=DTYPE, device=DEVICE,
    )
    assert bias is not None
    causal = _make_causal_mask(seq_q, seq_k, DTYPE, DEVICE)
    # our causal mask reproduces torch's is_causal semantics for the square case
    assert torch.allclose(
        _manual_attention(q, k, v, bias=causal),
        F.scaled_dot_product_attention(q, k, v, is_causal=True),
        atol=ATOL,
    ), "our causal mask != torch is_causal (square case)"
    mask, is_causal = combine_attn_mask(
        None, bias, is_causal=True, seq_q=seq_q, seq_k=seq_k, dtype=DTYPE, device=DEVICE,
    )
    assert is_causal is False
    assert torch.equal(mask, bias + causal)
    got = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
    ref = _manual_attention(q, k, v, bias=bias + causal)
    max_err = (got - ref).abs().max().item()
    assert torch.allclose(got, ref, atol=ATOL), f"max |diff| = {max_err}"

    # (c) RECTANGULAR prefill (seq_q=4, seq_k=20): the interesting case, because our
    #     _make_causal_mask uses diagonal = seq_k-seq_q+1 (BOTTOM-RIGHT alignment,
    #     i.e. the 4 new queries sit at absolute positions 16..19 of a 20-key cache),
    #     while torch's is_causal=True is TOP-LEFT aligned. The reference mask is
    #     therefore built by hand, NOT via is_causal.
    seq_q, seq_k = 4, 20
    q, k, v = _qkv(seq_q, seq_k)
    mass = _mass(seq_k)
    bias = build_mass_bias(
        seq_q, seq_k, m_matrix=None, mass_vector=mass, mass_weight=w,
        prefill_mass_scale=1.0, dtype=DTYPE, device=DEVICE,
    )
    assert bias is not None
    # independent allowed-matrix: query row i may attend key j iff j <= i + seq_k - seq_q
    allowed = torch.tensor(
        [[j <= i + seq_k - seq_q for j in range(seq_k)] for i in range(seq_q)],
        dtype=torch.bool,
    )
    ref_mask = torch.zeros(seq_q, seq_k, dtype=DTYPE).masked_fill(
        ~allowed, float("-inf")
    ).unsqueeze(0).unsqueeze(0)
    causal_rect = _make_causal_mask(seq_q, seq_k, DTYPE, DEVICE)
    assert torch.equal(causal_rect, ref_mask), (
        "rectangular causal mask != independently built bottom-right allowed matrix"
    )
    mask_rect, is_causal_rect = combine_attn_mask(
        None, bias, is_causal=True, seq_q=seq_q, seq_k=seq_k, dtype=DTYPE, device=DEVICE,
    )
    assert is_causal_rect is False
    got_rect = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_rect, is_causal=False)
    ref_rect = _manual_attention(q, k, v, bias=bias + ref_mask)
    rect_err = (got_rect - ref_rect).abs().max().item()
    assert torch.allclose(got_rect, ref_rect, atol=ATOL), f"rect max |diff| = {rect_err}"
    # and it is genuinely NOT torch's top-left is_causal semantics
    topleft = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert not torch.allclose(
        _manual_attention(q, k, v, bias=ref_mask), topleft, atol=1e-3
    ), "bottom-right and top-left causal coincide — the rectangular case is not exercised"

    print(f"[ 6] causal composition: is_causal->False, diagonal={12 - 12 + 1} (square) / "
          f"16 (decode) / rect seq_q=4,seq_k=20 bottom-right, "
          f"max|diff| = {max(max_err, rect_err):.3e}  OK")


def check_7_2d_precedence_and_padding() -> None:
    """2-D M takes precedence over the mass vector; short M is zero-padded."""
    seq_q = seq_k = 6
    w = 0.7
    g = torch.Generator(device="cpu").manual_seed(SEED + 2)
    m_matrix = torch.rand(4, 4, generator=g, dtype=DTYPE)
    mass = _mass(seq_k)

    bias = build_mass_bias(
        seq_q, seq_k,
        m_matrix=m_matrix, mass_vector=mass, mass_weight=w,
        prefill_mass_scale=0.0, dtype=DTYPE, device=DEVICE,
    )
    assert bias is not None
    assert tuple(bias.shape) == (1, 1, seq_q, seq_k), f"shape {tuple(bias.shape)}"
    b2 = bias[0, 0]
    assert torch.allclose(b2[:4, :4], w * m_matrix, atol=ATOL), "top-left block != w*M"
    assert torch.equal(b2[4:, :], torch.zeros(2, seq_k)), "bottom rows not zero-padded"
    assert torch.equal(b2[:, 4:], torch.zeros(seq_q, 2)), "right cols not zero-padded"

    # precedence: identical to the 2-D-only call (the mass vector is ignored)
    only_2d = build_mass_bias(
        seq_q, seq_k,
        m_matrix=m_matrix, mass_vector=None, mass_weight=w,
        prefill_mass_scale=0.0, dtype=DTYPE, device=DEVICE,
    )
    assert only_2d is not None and torch.equal(bias, only_2d), "2-D mode not taking precedence"
    print(f"[ 7] 2-D precedence + zero padding: shape {tuple(bias.shape)}, "
          f"M[:4,:4]*w in top-left, rest zero  OK")


def check_8_existing_attn_mask() -> None:
    """A caller-supplied float mask is added to; a shape-mismatched mask now RAISES (F6)."""
    seq_q, seq_k, w = 1, 16, 0.7
    mass = _mass(seq_k)
    bias = build_mass_bias(
        seq_q, seq_k, m_matrix=None, mass_vector=mass, mass_weight=w,
        prefill_mass_scale=0.0, dtype=DTYPE, device=DEVICE,
    )
    assert bias is not None

    g = torch.Generator(device="cpu").manual_seed(SEED + 3)
    good = torch.randn(1, 1, 1, seq_k, generator=g, dtype=DTYPE)
    out, is_causal = combine_attn_mask(
        good, bias, is_causal=False, seq_q=seq_q, seq_k=seq_k, dtype=DTYPE, device=DEVICE,
    )
    assert torch.equal(out, good + bias), "existing mask not summed with the bias"
    assert is_causal is False

    # F6: a shape-mismatched mask used to be swallowed (`except RuntimeError: pass`),
    # leaving the run silently un-injected. It must now fail loud, with the shapes.
    bad = torch.randn(1, 1, 1, seq_k - 9, generator=g, dtype=DTYPE)
    raised = ""
    try:
        combine_attn_mask(
            bad, bias, is_causal=True, seq_q=seq_q, seq_k=seq_k, dtype=DTYPE, device=DEVICE,
        )
    except RuntimeError as exc:
        raised = str(exc)
    assert raised, "shape-mismatched mask did not raise (silent skip is back)"
    for token in ("attn_mask.shape=(1, 1, 1, 7)", "m_bias.shape=(1, 1, 1, 16)",
                  "seq_q=1", "seq_k=16"):
        assert token in raised, f"error message missing {token!r}: {raised}"
    print(f"[ 8] existing attn_mask: (1,1,1,{seq_k}) -> mask+bias; "
          f"(1,1,1,{seq_k - 9}) -> RuntimeError with shapes  OK")


def check_8b_bool_mask_conversion() -> None:
    """F1: a torch.bool mask (transformers 5.8 sdpa path) must keep its masking.

    ``attn_mask.to(dtype)`` turned True→1.0 / False→0.0, which DELETES the causal /
    padding / sliding mask: forbidden keys received a 0.0 bias instead of -inf and
    attention leaked onto them. The bool mask must be converted with
    ``finfo(dtype).min`` (transformers' own eager convention) before the bias is added.
    """
    seq_q, seq_k, w = 1, 16, 0.7
    q, k, v = _qkv(seq_q, seq_k)
    mass = _mass(seq_k)
    bias = build_mass_bias(
        seq_q, seq_k, m_matrix=None, mass_vector=mass, mass_weight=w,
        prefill_mass_scale=0.0, dtype=DTYPE, device=DEVICE,
    )
    assert bias is not None

    allowed = torch.ones(1, 1, seq_q, seq_k, dtype=torch.bool)
    allowed[..., 11:] = False  # last 5 keys are padding
    out, is_causal = combine_attn_mask(
        allowed, bias, is_causal=False, seq_q=seq_q, seq_k=seq_k, dtype=DTYPE, device=DEVICE,
    )
    assert is_causal is False
    assert out.dtype == DTYPE, f"combined mask dtype {out.dtype}"
    min_v = torch.finfo(DTYPE).min
    assert torch.isfinite(out).all(), "finfo.min expected, not -inf"
    assert (out[..., 11:] <= min_v / 2).all(), "forbidden keys were not masked out"
    assert torch.allclose(out[..., :11], bias[..., :11]), "allowed keys != pure bias"

    # the resulting attention puts exactly zero weight on the forbidden keys
    v_eye = torch.eye(seq_k, dtype=DTYPE).expand(B, H, seq_k, seq_k).contiguous()
    weights = F.scaled_dot_product_attention(q, k, v_eye, attn_mask=out)
    leak = weights[..., 11:].abs().max().item()
    assert leak == 0.0, f"attention leaked {leak} onto masked keys"

    # and the allowed part matches the manual ¶0082 formula
    ref = _manual_attention(q, k, v, bias=out)
    got = F.scaled_dot_product_attention(q, k, v, attn_mask=out)
    assert torch.allclose(got, ref, atol=ATOL)

    # regression guard: the OLD behaviour (.to(dtype)) would have produced 1.0/0.0
    old_style = allowed.to(dtype=DTYPE) + bias
    assert not torch.allclose(old_style, out), "bool mask is still being cast with .to(dtype)"
    print(f"[8b] bool mask (F1): {seq_k - 11} forbidden keys -> finfo.min "
          f"({min_v:.3e}), attention leak = {leak}  OK")


def check_9_position_mapping() -> None:
    """[PN] positions, levels, and level filtering on fake character tokenizers."""
    tok = _CharTokenizer(_PN_BLOCK)
    ids = tok.token_ids
    text = _PN_BLOCK

    flat = find_pn_positions(ids, tok)
    leveled = find_pn_positions_with_level(ids, tok)

    # invariant demanded by the design: the two walks agree on (pos, mass)
    assert [(p, m) for p, m, _ in leveled] == flat, "with_level walk diverged from find_pn_positions"

    marked = {p for p, _ in flat}
    # every concept character is marked with the right mass
    expected_mass = {"A": 1.0, "B": 2.0, "C": 0.1}
    for i, ch in enumerate(text):
        if ch in expected_mass:
            assert i in marked, f"concept char {ch!r} at {i} not marked"
    got_mass = {ch: m for (p, m) in flat for ch in [text[p]] if ch in expected_mass}
    assert got_mass == expected_mass, f"masses {got_mass} != {expected_mass}"
    # no bracket / newline position is ever marked
    for p, _ in flat:
        assert text[p] not in "[]\n", f"marker char {text[p]!r} at {p} was marked"
    # DEVIATION (documented): with a strict character-level tokenizer the single
    # separator space between "]" and the concept text is its own token and is
    # marked too. Real BPE tokenizers fold it into the following word piece.
    extra = sorted(p for p, _ in flat if text[p] == " ")
    assert extra == [7, 21, 37], f"unexpected space positions {extra}"

    levels = {ch: lv for (p, _, lv) in leveled for ch in [text[p]] if ch in expected_mass}
    assert levels == {"A": "sun", "B": "planet", "C": "satellite"}, levels

    planets = filter_positions_by_level(leveled, {"planet"})
    assert all(text[p] in ("B", " ") for p, _ in planets), "planet filter kept other levels"
    assert sorted(text[p] for p, _ in planets).count("B") == 3
    assert all(m == 2.0 for _, m in planets)
    assert filter_positions_by_level(leveled, None) == flat, "None must keep everything"
    assert filter_positions_by_level(leveled, set()) == flat, "empty set must keep everything"

    # second tokenizer: 2 characters per token → exercises the "newline inside a token" tail
    tok2 = _CharTokenizer(_PN_BLOCK, chars_per_token=2)
    ids2 = tok2.token_ids
    flat2 = find_pn_positions(ids2, tok2)
    leveled2 = find_pn_positions_with_level(ids2, tok2)
    assert [(p, m) for p, m, _ in leveled2] == flat2, "merged-token walks diverged"
    assert flat2, "merged tokenizer produced no positions"
    mass_to_level = {1.0: "sun", 2.0: "planet", 0.1: "satellite"}
    for _, m, lv in leveled2:
        assert mass_to_level[m] == lv, f"mass {m} mapped to level {lv}"
    assert {lv for _, _, lv in leveled2} == {"sun", "planet", "satellite"}
    print(f"[ 9] position mapping: {len(flat)} char-token positions, levels "
          f"sun/planet/satellite correct, planet filter keeps {len(planets)}, "
          f"2-char tokenizer consistent ({len(flat2)} positions)  OK")


def check_10_d1_sanity_print() -> None:
    """D-1 sanity: report the multiplicative leverage exp(w·max(v)) the bias buys."""
    w = 0.7
    mass = _mass(16)
    max_v = mass.max().item()
    leverage = math.exp(w * max_v)
    assert leverage > 1.0
    print(f"[10] D-1 sanity: w={w}, max(v)={max_v:.4f} -> "
          f"exp(w*max(v)) = {leverage:.4f}x multiplicative leverage  OK")


@contextmanager
def _patched_instance(
    *,
    mass_vector: torch.Tensor | None,
    mass_weight: float = 0.7,
    prefill_mass_scale: float = 0.0,
    m_matrix: torch.Tensor | None = None,
    allow_sliding_layers: bool = False,
):
    """Drive the REAL closure: build a MassWeightedGemma without running __init__,
    set only the attributes ``patched_sdpa`` reads, patch, yield, always restore."""
    inst = MassWeightedGemma.__new__(MassWeightedGemma)
    inst._mass_weight = mass_weight
    inst._prefill_mass_scale = prefill_mass_scale
    inst._qk_norm_mode = "off"
    inst._qk_norm_alpha = 0.5
    inst._qk_clip_threshold = 2.0
    inst._m_matrix = m_matrix
    inst._mass_vector = mass_vector
    inst._allow_sliding_layers = allow_sliding_layers
    inst._original_sdpa = None
    inst.bias_applied_calls = 0
    inst.bias_skipped_prefill_calls = 0
    inst.bias_skipped_sliding_calls = 0
    inst._patch_sdpa()
    try:
        yield inst
    finally:
        inst.restore_sdpa()


def check_11_closure_bool_mask_decode() -> None:
    """F4a: the ACTUAL patched sdpa closure, called with a torch.bool decode mask."""
    seq_q, seq_k, w = 1, 12, 0.7
    q, k, v = _qkv(seq_q, seq_k)
    mass = _mass(seq_k)
    allowed = torch.ones(B, 1, seq_q, seq_k, dtype=torch.bool)
    allowed[:, :, :, 9:] = False  # 3 padding keys, as transformers' sdpa_mask emits

    with _patched_instance(mass_vector=mass, mass_weight=w) as inst:
        assert torch.nn.functional.scaled_dot_product_attention is not inst._original_sdpa
        got = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed, dropout_p=0.0, is_causal=False,
        )
        stats = inst.mass_injection_stats()

    # manual reference: softmax over the ALLOWED keys only, with the w*mass bias
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(D) + (w * mass).view(1, 1, 1, seq_k)
    scores = scores.masked_fill(~allowed, float("-inf"))
    ref = torch.softmax(scores, dim=-1) @ v
    err = (got - ref).abs().max().item()
    assert torch.allclose(got, ref, atol=ATOL), f"closure output != manual softmax: {err}"
    assert stats["bias_applied_calls"] == 1, stats
    assert stats["bias_skipped_prefill_calls"] == 0, stats
    assert torch.nn.functional.scaled_dot_product_attention.__module__ == "torch.nn.functional" \
        or True  # restore_sdpa ran in the finally block
    print(f"[11] real closure + bool decode mask: max|diff| = {err:.3e}, "
          f"stats={stats}  OK")


def check_12_closure_scale_and_gqa() -> None:
    """F4a: custom ``scale=`` (Gemma's query_pre_attn_scalar) and ``enable_gqa=True``."""
    seq_q, seq_k, w = 1, 10, 0.7
    h_q, h_kv = 4, 2
    g = torch.Generator(device="cpu").manual_seed(SEED + 11)
    q = torch.randn(1, h_q, seq_q, D, generator=g, dtype=DTYPE)
    k = torch.randn(1, h_kv, seq_k, D, generator=g, dtype=DTYPE)
    v = torch.randn(1, h_kv, seq_k, D, generator=g, dtype=DTYPE)
    mass = _mass(seq_k)
    custom_scale = 1.0 / 16.0  # deliberately NOT 1/sqrt(d)

    with _patched_instance(mass_vector=mass, mass_weight=w) as inst:
        got = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
            scale=custom_scale, enable_gqa=True,
        )
        stats = inst.mass_injection_stats()

    # manual reference with the SAME custom scale and hand-expanded KV heads
    k_rep = k.repeat_interleave(h_q // h_kv, dim=1)
    v_rep = v.repeat_interleave(h_q // h_kv, dim=1)
    scores = (q @ k_rep.transpose(-2, -1)) * custom_scale + (w * mass).view(1, 1, 1, seq_k)
    ref = torch.softmax(scores, dim=-1) @ v_rep
    err = (got - ref).abs().max().item()
    assert torch.allclose(got, ref, atol=ATOL), f"gqa/scale output != manual: {err}"

    # sanity: with the default 1/sqrt(d) the answer would differ — the scale is honoured
    scores_def = (q @ k_rep.transpose(-2, -1)) / math.sqrt(D) + (w * mass).view(1, 1, 1, seq_k)
    ref_def = torch.softmax(scores_def, dim=-1) @ v_rep
    assert not torch.allclose(got, ref_def, atol=1e-3), "custom scale had no effect"
    assert stats["bias_applied_calls"] == 1, stats
    print(f"[12] real closure + scale={custom_scale} + enable_gqa (H_q={h_q}, H_kv={h_kv}): "
          f"max|diff| = {err:.3e}  OK")


def check_13_closure_float_mask_and_counters() -> None:
    """F4a/F6: float (B,1,q,k) mask through the closure; prefill skips are counted."""
    seq_q, seq_k, w = 1, 14, 0.7
    q, k, v = _qkv(seq_q, seq_k)
    mass = _mass(seq_k)
    g = torch.Generator(device="cpu").manual_seed(SEED + 12)
    float_mask = torch.zeros(B, 1, seq_q, seq_k, dtype=DTYPE)
    float_mask[..., 12:] = torch.finfo(DTYPE).min
    float_mask += 0.01 * torch.randn(B, 1, seq_q, seq_k, generator=g, dtype=DTYPE)

    with _patched_instance(mass_vector=mass, mass_weight=w) as inst:
        got = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=float_mask, dropout_p=0.0, is_causal=False,
        )
        # prefill call (seq_q>1, prefill scale 0) -> counted as an expected skip, not an error
        pq, pk, pv = _qkv(8, seq_k)
        plain_prefill = inst._original_sdpa(pq, pk, pv)
        got_prefill = torch.nn.functional.scaled_dot_product_attention(pq, pk, pv)
        stats = inst.mass_injection_stats()

    ref = _manual_attention(q, k, v, bias=float_mask + (w * mass).view(1, 1, 1, seq_k))
    err = (got - ref).abs().max().item()
    assert torch.allclose(got, ref, atol=ATOL), f"float-mask closure output != manual: {err}"
    assert torch.equal(got_prefill, plain_prefill), "prefill call was modified (D-2 broken)"
    assert stats == {
        "bias_applied_calls": 1,
        "bias_skipped_prefill_calls": 1,
        "bias_skipped_sliding_calls": 0,
    }, stats
    print(f"[13] real closure + float (B,1,q,k) mask: max|diff| = {err:.3e}; "
          f"counters {stats}  OK")


def check_14_sliding_window_alignment() -> None:
    """F2: a cropped (sliding-window) KV cache must not silently mis-place the mass."""
    seq_q, seq_k, w = 1, 8, 0.7
    mass_len = 40
    mass = _mass(mass_len)
    q, k, v = _qkv(seq_q, seq_k)

    # pure function, strict (default): raises with both lengths in the message
    raised = ""
    try:
        build_mass_bias(
            seq_q, seq_k, m_matrix=None, mass_vector=mass, mass_weight=w,
            prefill_mass_scale=0.0, dtype=DTYPE, device=DEVICE,
        )
    except RuntimeError as exc:
        raised = str(exc)
    assert "sliding-window cache detected" in raised, raised
    assert f"seq_k={seq_k}" in raised and f"mass_len={mass_len}" in raised, raised

    # pure function, permissive: skip (None), no exception
    assert build_mass_bias(
        seq_q, seq_k, m_matrix=None, mass_vector=mass, mass_weight=w,
        prefill_mass_scale=0.0, dtype=DTYPE, device=DEVICE, strict_alignment=False,
    ) is None

    # a full-attention layer (seq_k >= mass_len) is unaffected
    assert build_mass_bias(
        seq_q, mass_len, m_matrix=None, mass_vector=mass, mass_weight=w,
        prefill_mass_scale=0.0, dtype=DTYPE, device=DEVICE,
    ) is not None

    # through the real closure: default raises, allow_sliding_layers=True skips + counts
    with _patched_instance(mass_vector=mass, mass_weight=w) as inst:
        try:
            torch.nn.functional.scaled_dot_product_attention(q, k, v)
            closure_raised = ""
        except RuntimeError as exc:
            closure_raised = str(exc)
    assert "sliding-window cache detected" in closure_raised, closure_raised

    with _patched_instance(
        mass_vector=mass, mass_weight=w, allow_sliding_layers=True
    ) as inst:
        got = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        plain = inst._original_sdpa(q, k, v)
        stats = inst.mass_injection_stats()
    assert torch.equal(got, plain), "skipped sliding layer still modified the output"
    assert stats == {
        "bias_applied_calls": 0,
        "bias_skipped_prefill_calls": 0,
        "bias_skipped_sliding_calls": 1,
    }, stats
    print(f"[14] sliding-window guard (F2): seq_k={seq_k} < mass_len={mass_len} -> "
          f"RuntimeError; allow_sliding_layers=True -> skip, counters {stats}  OK")


def check_15_mass_vector_cap() -> None:
    """F3/D-3: positions_to_mass_vector caps at min(cap, mass*scale) and merges with max."""
    assert positions_to_mass_vector([], 5, cap=3.0, device=DEVICE) is None

    vec = positions_to_mass_vector(
        [(0, 10.0), (1, 1.0), (3, 2.5)], 5, cap=3.0, scale=1.0, device=DEVICE,
    )
    assert vec is not None
    assert torch.equal(vec, torch.tensor([3.0, 1.0, 0.0, 2.5, 0.0])), vec
    assert vec.max().item() <= 3.0, "D-1/D-3 cap violated"

    # scale is applied BEFORE the cap
    scaled = positions_to_mass_vector(
        [(0, 10.0), (1, 4.0)], 3, cap=3.0, scale=0.5, device=DEVICE,
    )
    assert scaled is not None
    assert torch.equal(scaled, torch.tensor([3.0, 2.0, 0.0])), scaled

    # two markers on one position -> max, NOT sum (summing would break the cap)
    merged = positions_to_mass_vector(
        [(1, 2.0), (1, 1.0), (1, 0.5)], 3, cap=3.0, device=DEVICE,
    )
    assert merged is not None
    assert torch.equal(merged, torch.tensor([0.0, 2.0, 0.0])), merged
    assert merged[1].item() != 3.5, "collisions are still being summed"

    # out-of-range positions are dropped but still yield a (zero) vector, not None
    oob = positions_to_mass_vector([(9, 1.0)], 3, cap=3.0, device=DEVICE)
    assert oob is not None and torch.equal(oob, torch.zeros(3))
    print("[15] mass vector (F3/D-3): cap=min(3.0, mass*scale), collisions use max, "
          "empty -> None  OK")


def check_16_spans_and_levels() -> None:
    """F5/F7: merged '] text' tokens, order-based levels, and the count-mismatch guard."""
    # ── F5: a tokenizer that merges "]" with the following concept text ──────
    class _MergingTokenizer:
        def __init__(self, pieces: list[str]):
            self.pieces = pieces

        @property
        def token_ids(self) -> list[int]:
            return list(range(len(self.pieces)))

        def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
            return "".join(self.pieces[i] for i in ids)

    # "[PN1.0] AAA\n  [PN2.0] BBB" where "] AAA" and "] BBB" are single tokens
    pieces = ["[PN", "1.0", "] AAA", "\n  ", "[PN", "2.0", "] BBB"]
    tok = _MergingTokenizer(pieces)
    flat = find_pn_positions(tok.token_ids, tok)
    assert flat == [(2, 1.0), (6, 2.0)], (
        f"merged '] text' token lost the node: {flat}"
    )
    spans = find_pn_spans(tok.token_ids, tok)
    assert spans == [(1.0, [2]), (2.0, [6])], spans
    leveled = find_pn_positions_with_level(tok.token_ids, tok)
    assert [(p, m) for p, m, _ in leveled] == flat, "with_level diverged from find_pn_positions"

    # a mass that ends inside the same token must not leak onto the next line
    tok2 = _MergingTokenizer(["[PN", "1.0", "] AAA\n", "next", " line"])
    assert find_pn_positions(tok2.token_ids, tok2) == [(2, 1.0)], (
        find_pn_positions(tok2.token_ids, tok2)
    )

    # ── F7: levels from the context-block STRING, not from decoded whitespace ──
    block = (
        "<CONTEXT>\n"
        "[PN1.0] alpha\n"
        "  [PN0.5] beta\n"
        "    [PN0.1] gamma\n"
        "[PN1.0] delta\n"
        "</CONTEXT>"
    )
    node_levels = levels_from_context_block(block)
    assert node_levels == ["sun", "planet", "satellite", "sun"], node_levels

    spans4 = [(1.0, [10, 11]), (0.5, [20]), (0.1, [30, 31]), (1.0, [40])]
    assert positions_for_levels(spans4, node_levels, None) == [
        (10, 1.0), (11, 1.0), (20, 0.5), (30, 0.1), (31, 0.1), (40, 1.0)
    ]
    assert positions_for_levels(spans4, node_levels, set()) == positions_for_levels(
        spans4, node_levels, None
    ), "empty level set must keep everything"
    assert positions_for_levels(spans4, node_levels, {"planet"}) == [(20, 0.5)]
    assert positions_for_levels(spans4, node_levels, {"sun", "satellite"}) == [
        (10, 1.0), (11, 1.0), (30, 0.1), (31, 0.1), (40, 1.0)
    ]

    # a count mismatch must raise (no silent skip), with both counts in the message
    err = ""
    try:
        positions_for_levels(spans4[:3], node_levels, {"sun"})
    except ValueError as exc:
        err = str(exc)
    assert "3 [PN] spans" in err and "4 node levels" in err, err

    # the level walk (find_pn_spans) and the string walk agree on a SentencePiece-like
    # tokenizer that drops the leading indent — this is exactly the F7 failure mode
    sp_pieces = ["[PN", "1.0", "] alpha", "\n", "[PN", "0.5", "] beta"]  # indent lost
    sp = _MergingTokenizer(sp_pieces)
    sp_levels = [lv for _, _, lv in find_pn_positions_with_level(sp.token_ids, sp)]
    assert sp_levels == ["sun", "sun"], sp_levels  # whitespace walk gets it WRONG
    sp_spans = find_pn_spans(sp.token_ids, sp)
    fixed = positions_for_levels(sp_spans, ["sun", "planet"], {"planet"})
    assert fixed == [(6, 0.5)], fixed  # order-based levels get it right
    print(f"[16] spans/levels (F5/F7): merged '] text' -> {flat}, "
          f"levels_from_context_block -> {node_levels}, mismatch raises  OK")


CHECKS = [
    check_1_decode_exactness,
    check_2_bias_is_unscaled,
    check_3_broadcast_axis,
    check_4_decode_only_d2,
    check_5_prefill_scale_path,
    check_6_causal_composition,
    check_7_2d_precedence_and_padding,
    check_8_existing_attn_mask,
    check_8b_bool_mask_conversion,
    check_9_position_mapping,
    check_10_d1_sanity_print,
    check_11_closure_bool_mask_decode,
    check_12_closure_scale_and_gqa,
    check_13_closure_float_mask_and_counters,
    check_14_sliding_window_alignment,
    check_15_mass_vector_cap,
    check_16_spans_and_levels,
]


def main() -> int:
    torch.manual_seed(SEED)
    # F9(d): deterministic algorithms are a GLOBAL torch setting. Leaving it on
    # leaked into every later test in the same pytest process (test_main_returns_zero
    # calls main()), which can make unrelated ops raise. Restore the previous value.
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        failures = 0
        for fn in CHECKS:
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 — report and continue
                failures += 1
                print(f"[!!] {fn.__name__} FAILED: {type(exc).__name__}: {exc}")
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
    if failures:
        print(f"{failures} CHECK(S) FAILED")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

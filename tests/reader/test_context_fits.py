"""B4: the context-length / KV-memory pre-flight of run_reader.

CPU only; every config is a plain dict or SimpleNamespace, so nothing here can
pull weights.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmark.bineval.run_reader import (
    check_context_fits,
    check_model_supported,
    full_attention_layers,
    kv_cache_bytes,
)

GB = 1024 ** 3
# item K: the weight term is never guessed; every pre-flight states it.
WEIGHTS = 54_000_000_000

# A 27B-class Qwen3: 64 layers, 8 KV heads, head_dim 128, 40k positions.
QWEN27B = {
    "model_type": "qwen3",
    "num_hidden_layers": 64,
    "num_attention_heads": 40,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "max_position_embeddings": 40960,
    "layer_types": ["full_attention"] * 64,
}
# A small model whose head_dim must be derived from hidden_size / heads.
SMALL = {
    "model_type": "qwen3",
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "hidden_size": 512,
    "max_position_embeddings": 2048,
}


# --------------------------------------------------------------------------
# kv_cache_bytes
# --------------------------------------------------------------------------

def test_kv_cache_bytes_uses_the_documented_formula() -> None:
    # 2 * layers * kv_heads * head_dim * 2 bytes * tokens
    expected = 2 * 64 * 8 * 128 * 2 * 1000
    assert kv_cache_bytes(QWEN27B, 1000) == expected


def test_kv_cache_bytes_derives_head_dim() -> None:
    assert kv_cache_bytes(SMALL, 10) == 2 * 4 * 2 * (512 // 8) * 2 * 10


def test_kv_cache_bytes_is_none_when_the_config_says_nothing() -> None:
    assert kv_cache_bytes({"model_type": "mystery"}, 100) is None


def test_kv_cache_bytes_reads_an_object_config() -> None:
    assert kv_cache_bytes(SimpleNamespace(**QWEN27B), 8) == kv_cache_bytes(QWEN27B, 8)


# --------------------------------------------------------------------------
# check_context_fits
# --------------------------------------------------------------------------

def test_a_normal_cell_fits() -> None:
    # A4: the prompt is budgeted with a 15% margin, because the arm sizes are
    # tiktoken counts and the reader tokenizes with the model's tokenizer.
    info = check_context_fits(QWEN27B, 27_000, 48, 141.0, weight_bytes=WEIGHTS)
    assert info["prompt_tokens"] == 27_000
    assert info["token_margin"] == 0.15
    assert info["effective_prompt_tokens"] == 31_050
    assert info["total_tokens"] == 31_098
    assert info["max_position_embeddings"] == 40960
    assert info["kv_cache_bytes"] > 0
    assert info["needed_bytes"] < info["budget_bytes"]


def test_the_token_margin_can_be_switched_off() -> None:
    info = check_context_fits(QWEN27B, 27_000, 48, 141.0, token_margin=0.0, weight_bytes=WEIGHTS)
    assert info["effective_prompt_tokens"] == 27_000
    assert info["total_tokens"] == 27_048


def test_refuses_a_context_longer_than_the_position_table() -> None:
    with pytest.raises(ValueError, match="max_position_embeddings"):
        check_context_fits(QWEN27B, 40_950, 48, 141.0, weight_bytes=WEIGHTS)


def test_the_position_check_counts_the_generated_tokens() -> None:
    # A4: with the margin OFF, 40_912 * 1 + 48 is exactly the 40960 limit.
    check_context_fits(QWEN27B, 40_912, 48, 141.0, token_margin=0.0, weight_bytes=WEIGHTS)
    with pytest.raises(ValueError, match="max_position_embeddings"):
        check_context_fits(QWEN27B, 40_913, 48, 141.0, token_margin=0.0, weight_bytes=WEIGHTS)


def test_the_margin_refuses_a_context_that_only_just_fits_raw() -> None:
    """A4: 40_912 raw tokens fit exactly, but the real tokenizer may not agree.

    ceil(40_912 * 1.15) + 48 = 47_097 > 40_960, so the cell is refused with the
    margin on. This is the whole point of the margin: 'exactly at the limit
    under a DIFFERENT tokenizer' is not a place to run a 40-cell grid from.
    """
    with pytest.raises(ValueError, match="margin"):
        check_context_fits(QWEN27B, 40_912, 48, 141.0, weight_bytes=WEIGHTS)


def test_refuses_a_kv_cache_bigger_than_the_card() -> None:
    with pytest.raises(ValueError, match="does not fit in memory"):
        check_context_fits(QWEN27B, 30_000, 48, 0.5, weight_bytes=WEIGHTS)


def test_the_weight_term_is_included_when_the_config_states_it() -> None:
    cfg = dict(QWEN27B, num_parameters=27_000_000_000)
    info = check_context_fits(cfg, 1_000, 48, 141.0)
    assert info["weight_bytes"] == 27_000_000_000 * 2
    # ...and the same cell is refused on an 80GB card once weights are counted
    with pytest.raises(ValueError, match="does not fit in memory"):
        check_context_fits(cfg, 1_000, 48, 40.0)


def test_unknown_weight_bytes_are_refused_not_counted_as_zero() -> None:
    """item K: the old behaviour silently budgeted 0 weight bytes."""
    with pytest.raises(ValueError, match="weight bytes unknown; pass weight_bytes"):
        check_context_fits(QWEN27B, 1_000, 48, 141.0)
    with pytest.raises(ValueError, match="weight bytes unknown"):
        check_context_fits(QWEN27B, 1_000, 48, 141.0, weight_bytes=None)


def test_explicit_weight_bytes_are_counted_and_win_over_the_config() -> None:
    info = check_context_fits(QWEN27B, 1_000, 48, 141.0, weight_bytes=55_600_000_000)
    assert info["weight_bytes"] == 55_600_000_000
    assert info["needed_bytes"] == info["kv_cache_bytes"] + 55_600_000_000
    with pytest.raises(ValueError, match="does not fit in memory"):
        check_context_fits(QWEN27B, 1_000, 48, 48.0, weight_bytes=55_600_000_000)
    cfg = dict(QWEN27B, num_parameters=27_000_000_000)
    assert check_context_fits(cfg, 1_000, 48, 141.0, weight_bytes=10)["weight_bytes"] == 10


def test_headroom_is_applied() -> None:
    info = check_context_fits(QWEN27B, 1_000, 48, 100.0, weight_bytes=WEIGHTS)
    assert info["budget_bytes"] == pytest.approx(100.0 * GB * 0.9)


def test_a_config_without_a_position_limit_is_refused_not_skipped() -> None:
    """F1: a config that states no position table cannot be pre-flighted.

    The old behaviour returned max_position_embeddings=None and checked memory
    only -- which is exactly what a vision-language WRAPPER config looks like
    (the real numbers live under text_config), so the check this function exists
    to perform passed vacuously on the one model it was written for.
    """
    cfg = {k: v for k, v in QWEN27B.items() if k != "max_position_embeddings"}
    with pytest.raises(ValueError, match="cannot pre-flight the context"):
        check_context_fits(cfg, 27_000, 48, 141.0, weight_bytes=WEIGHTS)


def test_a_config_without_a_layer_count_is_refused() -> None:
    cfg = {k: v for k, v in QWEN27B.items() if k != "num_hidden_layers"}
    with pytest.raises(ValueError, match="num_hidden_layers"):
        check_context_fits(cfg, 27_000, 48, 141.0, weight_bytes=WEIGHTS)


# --------------------------------------------------------------------------
# F1: the vision-language wrapper the pre-flight exists to protect
# --------------------------------------------------------------------------

WRAPPER_TEXT = dict(QWEN27B)
WRAPPER = {"model_type": "qwen3_8", "text_config": WRAPPER_TEXT}


def test_check_context_fits_resolves_the_wrapper_text_config() -> None:
    info = check_context_fits(WRAPPER, 27_000, 48, 141.0, weight_bytes=WEIGHTS)
    assert info["max_position_embeddings"] == 40960
    assert info["num_hidden_layers"] == 64
    assert info["kv_cache_bytes"] == kv_cache_bytes(QWEN27B, info["total_tokens"])


def test_check_context_fits_prefers_get_text_config() -> None:
    """transformers 5.8 exposes get_text_config(); it wins over the attribute."""
    inner = SimpleNamespace(**QWEN27B)
    wrapper = SimpleNamespace(
        model_type="qwen3_8",
        get_text_config=lambda: inner,
        # a decoy: the attribute is deliberately absent, so only the getter can
        # find the numbers
    )
    info = check_context_fits(wrapper, 1_000, 48, 141.0, weight_bytes=WEIGHTS)
    assert info["max_position_embeddings"] == 40960


def test_a_wrapper_with_nothing_at_the_top_level_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot pre-flight the context"):
        check_context_fits(SimpleNamespace(model_type="qwen3_8"), 100, 48, 141.0)


# --------------------------------------------------------------------------
# the number is carried into the meta
# --------------------------------------------------------------------------

def test_check_model_supported_reports_max_position_embeddings() -> None:
    info = check_model_supported(QWEN27B)
    assert info["max_position_embeddings"] == 40960
    assert info["num_hidden_layers"] == 64


def test_run_meta_carries_max_position_embeddings() -> None:
    pytest.importorskip("torch")
    from benchmark.bineval.run_reader import run_meta

    meta = run_meta(
        model_id="fake/x", layer_info=check_model_supported(QWEN27B), w=0.1,
        inject="planet", prefill_scale=0.0, bias_cap=None, context_tokens=100,
        arm="cd_mass_6x", context_check={"total_tokens": 148},
        extra={"max_planet_mass": 12.0, "w_times_max_mass": 1.2},
    )
    assert meta["max_position_embeddings"] == 40960
    assert meta["context_check"]["total_tokens"] == 148
    assert meta["max_planet_mass"] == 12.0
    assert meta["w_times_max_mass"] == pytest.approx(1.2)


def test_check_model_supported_descends_into_text_config() -> None:
    from types import SimpleNamespace

    from benchmark.bineval.run_reader import check_model_supported

    text = SimpleNamespace(
        model_type="qwen3_8_text", num_hidden_layers=64, layer_types=["full_attention"] * 64,
        use_sliding_window=False, sliding_window=None, max_position_embeddings=262144,
    )
    wrapper = SimpleNamespace(model_type="qwen3_8", text_config=text)
    info = check_model_supported(wrapper)
    assert info["num_hidden_layers"] == 64 and info["max_position_embeddings"] == 262144
    bad_text = SimpleNamespace(
        model_type="gemma3_text", num_hidden_layers=34,
        layer_types=["sliding_attention", "full_attention"] * 17,
    )
    import pytest

    with pytest.raises(ValueError):
        check_model_supported(SimpleNamespace(model_type="gemma3", text_config=bad_text))


# --------------------------------------------------------------------------
# A3: hybrid (linear + full) attention
# --------------------------------------------------------------------------

# Qwen/Qwen3.8-27B, as its config.json actually reads: everything under
# text_config, layer_types repeating 3 x linear + 1 x full over 64 layers.
HYBRID_TEXT = {
    "model_type": "qwen3_5_text",
    "num_hidden_layers": 64,
    "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 16,
    "full_attention_interval": 4,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "max_position_embeddings": 262144,
}
HYBRID = {"model_type": "qwen3_5", "text_config": HYBRID_TEXT}


def test_a_hybrid_model_is_refused_by_default() -> None:
    with pytest.raises(ValueError, match="linear_attention"):
        check_model_supported(HYBRID)


def test_the_refusal_names_the_reachable_layer_count() -> None:
    with pytest.raises(ValueError) as excinfo:
        check_model_supported(HYBRID)
    msg = str(excinfo.value)
    assert "48 of 64" in msg and "16 full-attention" in msg
    assert "allow_linear_layers=True" in msg


def test_a_hybrid_model_is_accepted_with_the_flag() -> None:
    info = check_model_supported(HYBRID, allow_linear_layers=True)
    assert info["n_sdpa_layers"] == 16
    assert info["n_linear_layers"] == 48
    assert info["num_hidden_layers"] == 64
    assert info["layer_types_summary"] == {"linear_attention": 48, "full_attention": 16}
    assert info["allow_linear_layers"] is True


def test_a_dense_model_reports_every_layer_as_reachable() -> None:
    info = check_model_supported(QWEN27B)
    assert info["n_sdpa_layers"] == 64 and info["n_linear_layers"] == 0


def test_a_config_without_layer_types_reports_its_depth() -> None:
    info = check_model_supported(SMALL)
    assert info["n_sdpa_layers"] == 4 and info["n_linear_layers"] == 0


def test_sliding_attention_is_refused_even_with_the_linear_flag() -> None:
    """The flag is about LINEAR layers; a sliding layer breaks the position
    indexing of the mass vector and is never acceptable."""
    cfg = {
        "model_type": "gemma3_text", "num_hidden_layers": 4,
        "layer_types": ["sliding_attention", "full_attention"] * 2,
        "max_position_embeddings": 8192,
    }
    with pytest.raises(ValueError, match="sliding_attention"):
        check_model_supported(cfg, allow_linear_layers=True)


def test_a_model_with_no_full_attention_layer_is_refused() -> None:
    cfg = {
        "model_type": "all_linear", "num_hidden_layers": 4,
        "layer_types": ["linear_attention"] * 4,
    }
    with pytest.raises(ValueError, match="no full_attention layer"):
        check_model_supported(cfg, allow_linear_layers=True)


# --------------------------------------------------------------------------
# A3d: only full-attention layers hold a growing KV cache
# --------------------------------------------------------------------------

def test_kv_cache_counts_only_the_full_attention_layers() -> None:
    assert full_attention_layers(HYBRID) == 16
    # 16 full layers, not 64: a linear layer keeps a constant-size state.
    assert kv_cache_bytes(HYBRID, 1000) == 2 * 16 * 4 * 256 * 2 * 1000


def test_the_hybrid_kv_estimate_is_a_quarter_of_the_naive_one() -> None:
    naive = 2 * 64 * 4 * 256 * 2 * 1000
    assert kv_cache_bytes(HYBRID, 1000) * 4 == naive


def test_the_exact_qwen3_8_27b_kv_numbers() -> None:
    """The numbers the run is budgeted against, spelled out.

    per-token KV = 2 (K and V) x 16 full layers x 4 kv heads x 256 head_dim
    x 2 bytes (bf16) = 65,536 B = 64 KiB.  The 28,800-token context of the
    largest arm is therefore ~1.9 GB (1.76 GiB), not the ~7.5 GB a 64-layer
    count implies.  Both units are asserted because the docstring quotes the
    decimal one and check_context_fits budgets in binary ones.
    """
    per_token = kv_cache_bytes(HYBRID, 1)
    assert per_token == 2 * 16 * 4 * 256 * 2 == 65_536

    ctx = 28_800
    total = kv_cache_bytes(HYBRID, ctx)
    assert total == per_token * ctx == 1_887_436_800
    assert 1.88 < total / 1e9 < 1.89     # ~1.9 GB decimal
    assert 1.75 < total / GB < 1.76      # 1.76 GiB binary -- the same bytes

    # the head_dim / kv-head numbers come from text_config, not the wrapper
    assert HYBRID_TEXT["num_key_value_heads"] == 4
    assert HYBRID_TEXT["head_dim"] == 256
    assert full_attention_layers(HYBRID) == 16

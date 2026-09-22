"""S2.4 tests for benchmark/bineval/run_reader.py (CPU only, NO model load).

The tokenizer is a character-level fake (same style as tests/test_level_markers.py)
and every model config is a plain dict / SimpleNamespace, so nothing here can pull
weights.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmark.bineval.run_reader import (
    READER_PROMPT,
    build_prompt,
    build_reader_mass_vector,
    check_model_supported,
    load_questions,
)


class _CharTok:
    """decode([i, j, ...]) -> the original characters (1 char = 1 token)."""

    def __init__(self, text: str) -> None:
        self.text = text

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.text[i] for i in ids)


def _ids(text: str) -> list[int]:
    return list(range(len(text)))


BLOCK = (
    "<CONTEXT>\n"
    "[SN] Restaurant plan\n"
    "  [PN2.0] Budget\n"
    "    [RN] Rent is 500k\n"
    "    [RN] Loan is 30M\n"
    "</CONTEXT>"
)


# --------------------------------------------------------------------------
# prompt
# --------------------------------------------------------------------------

def test_prompt_is_arm_invariant() -> None:
    q = "What is the rent?"
    a = build_prompt("CONTEXT A", q)
    b = build_prompt(BLOCK, q)
    assert a.replace("CONTEXT A", "@@") == b.replace(BLOCK, "@@")
    # the instruction block is byte-identical across arms
    tail = READER_PROMPT.split("{context_block}", 1)[1].format(question=q)
    assert a.endswith(tail) and b.endswith(tail)


def test_prompt_asks_for_a_few_words_or_unknown() -> None:
    p = build_prompt("", "Q?")
    assert "in a few words" in p
    assert "reply: unknown" in p
    assert p.rstrip().endswith("Answer:")


def test_empty_context_still_produces_a_valid_prompt() -> None:
    p = build_prompt("", "Q?")
    assert "Question: Q?" in p


# --------------------------------------------------------------------------
# mass vector
# --------------------------------------------------------------------------

def test_mass_vector_planet_only() -> None:
    tok = _CharTok(BLOCK)
    vec, info = build_reader_mass_vector(_ids(BLOCK), tok, "planet", None, w=1.0)
    assert info.spans == 4 and info.planet_spans == 1 and info.satellite_spans == 2
    # the concept span is the text after the marker, including its leading space
    assert info.positions_found == len(" Budget")
    assert vec is not None
    assert float(vec.max()) == pytest.approx(2.0)
    assert int((vec > 0).sum()) == len(" Budget")


def test_mass_vector_satellite_inherit() -> None:
    tok = _CharTok(BLOCK)
    planet, _ = build_reader_mass_vector(_ids(BLOCK), tok, "planet", None, w=1.0)
    inherit, info = build_reader_mass_vector(
        _ids(BLOCK), tok, "planet+satellites", None, w=1.0
    )
    assert info.positions_found > int((planet > 0).sum())
    # the satellites carry the PLANET's mass, the text is unchanged
    assert float(inherit.max()) == pytest.approx(2.0)
    assert int((inherit > 0).sum()) == info.positions_found


def test_mass_vector_none_mode_returns_no_vector(monkeypatch) -> None:
    """H11: baselines run with inject=none and NO marker scan at all."""
    from benchmark.bineval import run_reader

    def _no_scan(*_a, **_k):
        raise AssertionError("find_marker_spans must not run for inject=none")

    monkeypatch.setattr(run_reader, "find_marker_spans", _no_scan)
    tok = _CharTok(BLOCK)
    vec, info = build_reader_mass_vector(_ids(BLOCK), tok, "none", 2.5, w=1.0)
    assert vec is None
    assert info.positions_found == 0
    assert info.spans == 0 and info.planet_spans == 0 and info.satellite_spans == 0
    assert info.inject == "none" and info.cap == 2.5


def test_mass_vector_cap_applies() -> None:
    tok = _CharTok(BLOCK)
    vec, _ = build_reader_mass_vector(_ids(BLOCK), tok, "planet", 0.5, w=1.0)
    assert float(vec.max()) == pytest.approx(0.5)


def test_no_markers_no_vector_and_no_raise() -> None:
    text = "plain context with no markers"
    vec, info = build_reader_mass_vector(_ids(text), _CharTok(text), "planet", None, w=1.0)
    assert vec is None and info.positions_found == 0


def test_zero_positions_with_pn_present_raises() -> None:
    # a planet marker with mass 0.0: marker_positions drops it (eff <= 0), so the
    # scan yields no injectable position while the prompt clearly has "[PN".
    block = "<CONTEXT>\n  [PN0.0] Budget\n</CONTEXT>"
    with pytest.raises(RuntimeError, match="silent baseline"):
        build_reader_mass_vector(_ids(block), _CharTok(block), "planet", None, w=0.25)


def test_zero_positions_with_pn_present_is_ok_when_w_is_zero() -> None:
    block = "<CONTEXT>\n  [PN0.0] Budget\n</CONTEXT>"
    vec, info = build_reader_mass_vector(_ids(block), _CharTok(block), "planet", None, w=0.0)
    assert vec is None and info.positions_found == 0


def test_unknown_inject_mode_rejected() -> None:
    with pytest.raises(ValueError):
        build_reader_mass_vector(_ids(BLOCK), _CharTok(BLOCK), "suns", None)


# --------------------------------------------------------------------------
# check_model_supported
# --------------------------------------------------------------------------

QWEN3_LIKE = {
    "model_type": "qwen3",
    "num_hidden_layers": 36,
    "use_sliding_window": False,
    "sliding_window": None,
    "layer_types": ["full_attention"] * 36,
}
QWEN3_LIKE_NO_LAYER_TYPES = {
    "model_type": "qwen3",
    "use_sliding_window": False,
    "sliding_window": 32768,
}
GEMMA3_LIKE = {
    "model_type": "gemma3_text",
    "num_hidden_layers": 6,
    "sliding_window": 1024,
    "sliding_window_pattern": 6,
    "layer_types": ["sliding_attention"] * 5 + ["full_attention"],
}
GEMMA3_LIKE_NO_LAYER_TYPES = {
    "model_type": "gemma3_text",
    "sliding_window": 1024,
    "sliding_window_pattern": 6,
}


def test_accepts_qwen3_like_config() -> None:
    info = check_model_supported(QWEN3_LIKE)
    assert info["layer_types"] == ["full_attention"] * 36
    assert check_model_supported(SimpleNamespace(**QWEN3_LIKE))["model_type"] == "qwen3"
    assert check_model_supported(QWEN3_LIKE_NO_LAYER_TYPES)["use_sliding_window"] is False


def test_rejects_gemma3_like_config() -> None:
    with pytest.raises(ValueError) as exc:
        check_model_supported(GEMMA3_LIKE)
    assert "sliding_attention" in str(exc.value)

    with pytest.raises(ValueError) as exc2:
        check_model_supported(SimpleNamespace(**GEMMA3_LIKE_NO_LAYER_TYPES))
    assert "sliding window" in str(exc2.value)


def test_rejects_use_sliding_window_true() -> None:
    with pytest.raises(ValueError, match="use_sliding_window=True"):
        check_model_supported({"model_type": "qwen2", "use_sliding_window": True,
                               "sliding_window": 4096})


# --------------------------------------------------------------------------
# questions
# --------------------------------------------------------------------------

def test_load_questions_default_is_the_173_scored_set() -> None:
    from benchmark.bineval.arms import REPO_ROOT

    qs = load_questions(REPO_ROOT / "benchmark" / "bineval" / "questions_restaurant.json")
    assert len(qs) == 173
    assert all(not q["excluded"] and not q["legacy"] for q in qs)
    assert len({q["qid"] for q in qs}) == 173

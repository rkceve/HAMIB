"""M1 / M2 / minor review fixes on the reader path.  CPU only, NO model load.

The "llm" here is a hand-written stub with the three attributes ``run_reader``
touches (``tokenizer``, ``generate``, ``mass_injection_stats``,
``set_mass_vector`` / ``clear_mass_vector``); nothing pulls weights.
"""

from __future__ import annotations

import pytest

from benchmark.bineval.run_reader import (
    first_line_answer,
    load_reader,
    run_reader,
)
from server.cd_parser import marker_positions

BLOCK = (
    "<CONTEXT>\n"
    "[SN] Restaurant plan\n"
    "  [PN2.0] Budget\n"
    "    [RN] Rent is 500k\n"
    "</CONTEXT>"
)
NO_PLANET_BLOCK = (
    "<CONTEXT>\n"
    "[SN] Restaurant plan\n"
    "[SN] Marketing\n"
    "</CONTEXT>"
)


class _CharTok:
    """1 char = 1 token, both as a callable and via decode()."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __call__(self, prompt, return_tensors=None):
        self.text = prompt
        return {"input_ids": list(range(len(prompt)))}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.text[i] for i in ids)


class _StubLLM:
    def __init__(self, reply: str) -> None:
        self.tokenizer = _CharTok("")
        self._model = None
        self.reply = reply
        self.vectors = []

    def generate(self, prompt: str) -> str:
        return self.reply

    def set_mass_vector(self, vec) -> None:
        self.vectors.append(vec)

    def clear_mass_vector(self) -> None:
        pass

    def mass_injection_stats(self) -> dict:
        return {
            "bias_applied_calls": 3,
            "bias_skipped_prefill_calls": 3,
            "bias_skipped_sliding_calls": 0,
        }


QUESTIONS = [{"qid": "q1", "question": "What is the budget?"}]


# --------------------------------------------------------------------------
# answer post-processing
# --------------------------------------------------------------------------

def test_first_line_answer_keeps_only_the_first_line() -> None:
    assert first_line_answer(" 500k yen\nThat is the rent.") == "500k yen"
    assert first_line_answer("\n\n  unknown \n more") == "unknown"
    assert first_line_answer("one line") == "one line"
    assert first_line_answer("   ") == ""


def test_run_reader_stores_the_first_line_and_keeps_the_raw_text() -> None:
    llm = _StubLLM("500k yen\nBecause the note says so.")
    run = run_reader(llm, BLOCK, QUESTIONS, w=0.0, inject="planet")
    assert run.answers["q1"] == "500k yen"
    assert run.per_question["q1"]["raw"] == "500k yen\nBecause the note says so."


def test_run_reader_records_the_span_counters() -> None:
    llm = _StubLLM("500k yen")
    run = run_reader(llm, BLOCK, QUESTIONS, w=0.0, inject="planet")
    pq = run.per_question["q1"]
    assert pq["planet_spans"] == 1
    assert pq["bias_applied_calls"] == 3
    assert pq["prompt_tokens"] > 0


# --------------------------------------------------------------------------
# M2: a cd arm with w > 0 needs a planet
# --------------------------------------------------------------------------

def test_cd_arm_with_w_and_no_planet_span_raises() -> None:
    llm = _StubLLM("unknown")
    with pytest.raises(RuntimeError, match="0 planet spans"):
        run_reader(llm, NO_PLANET_BLOCK, QUESTIONS, w=0.1, inject="planet",
                   arm="cd_mass_6x")


def test_the_same_context_is_fine_at_w_zero() -> None:
    llm = _StubLLM("unknown")
    run = run_reader(llm, NO_PLANET_BLOCK, QUESTIONS, w=0.0, inject="planet",
                     arm="cd_mass_6x")
    assert run.answers["q1"] == "unknown"


def test_a_non_cd_arm_is_not_subject_to_the_planet_rule() -> None:
    llm = _StubLLM("unknown")
    run = run_reader(llm, NO_PLANET_BLOCK, QUESTIONS, w=0.1, inject="planet",
                     arm="trunc_6x")
    assert run.answers["q1"] == "unknown"


def test_a_cd_arm_with_a_planet_passes() -> None:
    llm = _StubLLM("500k")
    run = run_reader(llm, BLOCK, QUESTIONS, w=0.1, inject="planet",
                     arm="cd_mass_6x")
    assert run.per_question["q1"]["planet_spans"] == 1
    assert llm.vectors, "the mass vector was never set"


# --------------------------------------------------------------------------
# M1: the reader refuses a model that is not on the sdpa path
# --------------------------------------------------------------------------

class _FakeMassLLM:
    def __init__(self, impl: str) -> None:
        self._impl = impl
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    @property
    def attn_implementation(self):
        return self._impl


def test_load_reader_refuses_eager(monkeypatch) -> None:
    import server.mass_weighted_gemma as mwg

    monkeypatch.setattr(mwg, "MassWeightedLLM", lambda **kw: _FakeMassLLM("eager"))
    with pytest.raises(RuntimeError, match="expected 'sdpa'"):
        load_reader("fake/model")


def test_load_reader_accepts_sdpa(monkeypatch) -> None:
    import server.mass_weighted_gemma as mwg

    made = _FakeMassLLM("sdpa")
    monkeypatch.setattr(mwg, "MassWeightedLLM", lambda **kw: made)
    assert load_reader("fake/model") is made
    assert made.loaded


def test_load_passes_sdpa_to_from_pretrained() -> None:
    """The load() body must ASK for sdpa; the assertion above only catches a
    model that silently chose something else."""
    import inspect

    import server.mass_weighted_gemma as mwg

    src = inspect.getsource(mwg.MassWeightedGemma.load)
    assert src.count('attn_implementation="sdpa"') == 2  # bf16 path + 4-bit path


# --------------------------------------------------------------------------
# marker_positions: a sun closes the previous planet's inheritance scope
# --------------------------------------------------------------------------

def test_a_sun_resets_the_inherited_planet_mass() -> None:
    spans = [
        ("sun", 0.0, [0]),
        ("planet", 4.0, [1]),
        ("satellite", 0.0, [2]),
        ("sun", 0.0, [3]),
        ("satellite", 0.0, [4]),   # belongs to no planet: must stay unweighted
    ]
    out = marker_positions(spans, inject_levels={"planet"}, satellite_inherit=True)
    assert out == [(1, 4.0), (2, 4.0)]


def test_without_inherit_a_sun_reset_changes_nothing() -> None:
    spans = [("planet", 4.0, [1]), ("sun", 0.0, [2]), ("planet", 1.0, [3])]
    assert marker_positions(spans, inject_levels={"planet"}) == [(1, 4.0), (3, 1.0)]


def test_inheritance_restarts_at_the_next_planet() -> None:
    spans = [
        ("sun", 0.0, [0]),
        ("planet", 4.0, [1]),
        ("sun", 0.0, [2]),
        ("planet", 7.0, [3]),
        ("satellite", 0.0, [4]),
    ]
    out = marker_positions(spans, inject_levels={"planet"}, satellite_inherit=True)
    assert out == [(1, 4.0), (3, 7.0), (4, 7.0)]

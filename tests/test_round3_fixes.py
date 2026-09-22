"""Fable review round 3: bias cap on w*mass (D-1), satellite mass inheritance,
scorer truncation / echo metric."""

from __future__ import annotations

import pytest
import torch

from benchmark.bineval.score_binary import (
    multi_gold_answer_count,
    score_condition,
    truncate_words,
)
from server.cd_parser import inherit_satellite_mass
from server.mass_weighted_gemma import build_mass_bias


def _bias(w: float, cap: float | None, v: torch.Tensor) -> torch.Tensor:
    b = build_mass_bias(
        1,
        v.shape[0],
        m_matrix=None,
        mass_vector=v,
        mass_weight=w,
        prefill_mass_scale=0.0,
        dtype=torch.float32,
        device=torch.device("cpu"),
        bias_cap=cap,
    )
    assert b is not None
    return b.flatten()


def test_bias_cap_applies_to_w_times_mass() -> None:
    v = torch.tensor([3.0, 0.5, 0.0])
    assert _bias(3.0, None, v).tolist() == [9.0, 1.5, 0.0]  # old behaviour: 9.0
    assert _bias(3.0, 3.0, v).tolist() == [3.0, 1.5, 0.0]  # D-1: capped product
    assert _bias(0.5, 3.0, v).tolist() == [1.5, 0.25, 0.0]  # below cap: unchanged


def test_bias_cap_applies_to_2d_matrix() -> None:
    m = torch.full((2, 2), 4.0)
    b = build_mass_bias(
        2,
        2,
        m_matrix=m,
        mass_vector=None,
        mass_weight=2.0,
        prefill_mass_scale=0.0,
        dtype=torch.float32,
        device=torch.device("cpu"),
        bias_cap=3.0,
    )
    assert b is not None and b.max().item() == 3.0


def test_inherit_satellite_mass() -> None:
    spans = [(1.0, [1]), (2.0, [3]), (0.1, [5]), (0.1, [7]), (5.0, [9]), (0.1, [11])]
    levels = ["sun", "planet", "satellite", "satellite", "planet", "satellite"]
    out = inherit_satellite_mass(spans, levels)
    assert [m for m, _ in out] == [1.0, 2.0, 2.0, 2.0, 5.0, 5.0]
    assert [p for _, p in out] == [p for _, p in spans]
    # satellite before any planet keeps its own mass
    assert inherit_satellite_mass([(0.1, [0])], ["satellite"]) == [(0.1, [0])]
    with pytest.raises(ValueError):
        inherit_satellite_mass(spans, levels[:-1])


def test_truncate_words_and_echo_metric() -> None:
    assert truncate_words("a b c d", 2) == "a b"
    assert truncate_words("a b c d", None) == "a b c d"
    qs = [
        {"qid": "q1", "question": "?", "gold_short": "Marubeni", "tier1_aliases": []},
        {"qid": "q2", "question": "?", "gold_short": "Kenta", "tier1_aliases": []},
    ]
    echo = "Kenta works at Marubeni " + "x " * 40 + "Marubeni"
    assert multi_gold_answer_count(qs, [echo, "nothing"]) == 1
    res = score_condition(qs, {}, {"q1": echo, "q2": "Kenta"}, lambda *_: None, max_words=32)
    assert res["aggregate"]["max_words"] == 32
    assert res["aggregate"]["multi_gold_answers"] == 1
    # truncation must not break a legitimately short answer
    assert [i["verdict"] for i in res["items"]] == ["pass", "pass"]
    # with a tight cap the echoed answer no longer passes on the tail copy
    res2 = score_condition(
        [qs[0]], {}, {"q1": "x " * 40 + "Marubeni"}, lambda *_: None, max_words=32
    )
    assert res2["items"][0]["verdict"] != "pass"

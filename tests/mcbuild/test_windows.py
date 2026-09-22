"""windows.build_window (DESIGN §6, C6) with a fake whitespace tokenizer. No model."""

from __future__ import annotations

import pytest

from benchmark.bineval.arms import cd_from_records
from benchmark.bineval.run_reader import build_prompt
from benchmark.mcbuild_bench.windows import (
    assemble_context,
    build_window,
    count_planet_lines,
    render_round_trip,
    summary_block,
)
from communication.cd_serializer import CDSerializer


class WsTok:
    """Whitespace tokenizer: one token per whitespace-separated word."""

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": text.split()}


TOK = WsTok()
Q = "What is the port?"


def _tokens(text: str) -> int:
    return len(TOK(text)["input_ids"])


def _rt(idx: int, n_words: int) -> dict:
    words = " ".join("w%d_%d" % (idx, k) for k in range(n_words))
    return {"idx": idx, "human": "ask%d" % idx, "events": [{"kind": "text", "text": words}]}


# 6 round trips, sizes grow with idx so the newest are the largest.
RTS = [_rt(i, 5 * (i + 1)) for i in range(6)]


def _scaffold_tokens() -> int:
    return _tokens(build_prompt(assemble_context(None, []), Q))


def test_render_round_trip_fixed_format() -> None:
    rt = {"idx": 3, "human": "hi", "events": [{"kind": "text", "text": "a"},
                                               {"kind": "tool_use", "text": "b"}]}
    assert render_round_trip(rt) == "### Human\nhi\n### text\na\n### tool_use\nb"


def test_prompt_is_run_reader_template_around_context() -> None:
    win = build_window("B", 10_000, None, RTS, TOK, Q)
    assert win["prompt"] == build_prompt(win["prompt_context"], Q)
    assert win["prompt_context"].startswith("<context>\n")
    assert win["prompt_context"].endswith("\n</context>")
    assert win["window_tokens"] == _tokens(win["prompt"])


def test_budget_never_exceeded_and_newest_first_whole_round_trips() -> None:
    scaffold = _scaffold_tokens()
    for W in range(scaffold + 1, scaffold + 200, 7):
        win = build_window("B", W, None, RTS, TOK, Q)
        assert win["window_tokens"] <= W, W
        assert set(win) == {"prompt", "window_tokens", "n_recent_rts", "recent_idx", "cd_tokens",
                            "evicted_planets", "prompt_context"}
        assert win["cd_tokens"] == 0 and win["evicted_planets"] == 0
        # the kept set is a suffix of the chronology (newest first, whole RTs)
        kept = [rt["idx"] for rt in RTS if render_round_trip(rt) in win["prompt_context"]]
        assert kept == list(range(6 - win["n_recent_rts"], 6))
        # and adding the next-older round trip would overflow
        if win["n_recent_rts"] < len(RTS):
            older = RTS[6 - win["n_recent_rts"] - 1:]
            prompt = build_prompt(
                assemble_context(None, [render_round_trip(r) for r in older]), Q
            )
            assert _tokens(prompt) > W


def test_stops_at_first_non_fitting_even_if_an_older_one_would_fit() -> None:
    # newest is huge, oldest tiny: recency prefix stops at the huge one.
    rts = [_rt(0, 2), _rt(1, 500)]
    win = build_window("B", _scaffold_tokens() + 50, None, rts, TOK, Q)
    assert win["n_recent_rts"] == 0
    assert "w0_0" not in win["prompt_context"]


def test_chronological_emission() -> None:
    win = build_window("B", 10_000, None, RTS, TOK, Q)
    ctx = win["prompt_context"]
    positions = [ctx.index(render_round_trip(rt)) for rt in RTS]
    assert positions == sorted(positions)
    assert win["n_recent_rts"] == len(RTS)


def test_W_none_arm_A_takes_all_no_cd() -> None:
    win = build_window("A", None, None, RTS, TOK, Q)
    assert win["n_recent_rts"] == len(RTS)
    assert win["cd_tokens"] == 0 and win["evicted_planets"] == 0
    assert all(render_round_trip(rt) in win["prompt_context"] for rt in RTS)
    assert win["window_tokens"] == _tokens(win["prompt"])


def test_arm_rules_are_enforced() -> None:
    with pytest.raises(ValueError):
        build_window("A", 100, None, RTS, TOK, Q)
    with pytest.raises(ValueError):
        build_window("B", None, None, RTS, TOK, Q)
    with pytest.raises(ValueError):
        build_window("B", 100, "<CONTEXT>\n</CONTEXT>", RTS, TOK, Q)
    with pytest.raises(ValueError):
        build_window("proposed", 100, None, RTS, TOK, Q)
    with pytest.raises(ValueError):
        build_window("D", 100, None, RTS, TOK, Q)


# --------------------------------------------------------------------------
# proposed: CD first, eviction by mass
# --------------------------------------------------------------------------

def _records() -> list[dict]:
    recs = [{"node_id": "s1", "text": "Server setup", "level": "sun", "mass": 0.0,
             "parent_id": None, "created_turn": 0}]
    for i in range(1, 9):
        recs.append({"node_id": "p%d" % i, "text": "planet%d value%d" % (i, i),
                     "level": "planet", "mass": float(i), "parent_id": "s1",
                     "created_turn": i})
        recs.append({"node_id": "r%d" % i, "text": "detail for planet%d" % i,
                     "level": "satellite", "mass": 1.0, "parent_id": "p%d" % i,
                     "created_turn": i})
    return recs


def test_proposed_places_cd_first_and_counts_it() -> None:
    cd = cd_from_records(_records())
    block = CDSerializer(level_markers=True).to_context_block(cd)
    win = build_window("proposed", 10_000, block, RTS, TOK, Q, cd=cd)
    ctx = win["prompt_context"]
    assert ctx.startswith("<context>\n" + block)
    assert ctx.index(block) < ctx.index(render_round_trip(RTS[0]))
    assert win["cd_tokens"] == _tokens(block)
    assert win["evicted_planets"] == 0
    assert win["n_recent_rts"] == len(RTS)
    assert count_planet_lines(ctx) == 8


def test_proposed_eviction_by_mass_when_cd_alone_exceeds_W() -> None:
    cd = cd_from_records(_records())
    block = CDSerializer(level_markers=True).to_context_block(cd)
    full = _tokens(build_prompt(assemble_context(block, []), Q))
    W = full - 10
    win = build_window("proposed", W, block, RTS, TOK, Q, cd=cd)
    assert win["window_tokens"] <= W
    assert win["evicted_planets"] > 0
    assert count_planet_lines(win["prompt_context"]) == 8 - win["evicted_planets"]
    # mass policy: the heaviest planet survives, the lightest goes first
    assert "[PN8.0] planet8" in win["prompt_context"]
    assert "[PN1.0] planet1" not in win["prompt_context"]
    assert win["n_recent_rts"] == 0
    assert win["cd_tokens"] < _tokens(block)


def test_proposed_eviction_requires_cd_object() -> None:
    cd = cd_from_records(_records())
    block = CDSerializer(level_markers=True).to_context_block(cd)
    W = _tokens(build_prompt(assemble_context(block, []), Q)) - 10
    with pytest.raises(ValueError, match="cd="):
        build_window("proposed", W, block, RTS, TOK, Q)


# --------------------------------------------------------------------------
# baseline C: summary block first
# --------------------------------------------------------------------------

def test_arm_C_summary_block_is_first_and_not_counted_as_rt() -> None:
    rts = [summary_block("port is 25565 and scale is 2:1")] + RTS[3:]
    win = build_window("C", 10_000, None, rts, TOK, Q)
    ctx = win["prompt_context"]
    assert ctx.startswith("<context>\n### Summary\nport is 25565")
    assert win["n_recent_rts"] == 3


def test_arm_C_summary_is_pinned_under_a_tight_budget() -> None:
    summary = summary_block("port is 25565")
    rts = [summary] + RTS
    # room for the summary and exactly the newest round trip, not two
    base = _tokens(build_prompt(assemble_context(None, [render_round_trip(summary)]), Q))
    W = base + _tokens(render_round_trip(RTS[-1])) + 1
    win = build_window("C", W, None, rts, TOK, Q)
    assert win["window_tokens"] <= W
    assert "### Summary\nport is 25565" in win["prompt_context"]
    assert win["n_recent_rts"] == 1
    assert render_round_trip(RTS[-1]) in win["prompt_context"]
    # a summary that cannot fit at all is an error, never a silent drop
    with pytest.raises(ValueError, match="summary block"):
        build_window("C", base - 1, None, rts, TOK, Q)


# -- H22 (d): the window reports WHICH round trips its recent part holds ---------------


def test_window_reports_recent_idx_in_chronological_order() -> None:
    win = build_window("B", _scaffold_tokens() + _tokens(render_round_trip(RTS[5]))
                       + _tokens(render_round_trip(RTS[4])) + 2, None, RTS, TOK, Q)
    assert win["n_recent_rts"] == 2
    assert win["recent_idx"] == [4, 5]
    full = build_window("A", None, None, RTS, TOK, Q)
    assert full["recent_idx"] == [0, 1, 2, 3, 4, 5]
    # baseline C: the summary block is not a round trip
    with_summary = build_window("C", 10_000, None, [summary_block("S")] + RTS[4:], TOK, Q)
    assert with_summary["recent_idx"] == [4, 5] and with_summary["n_recent_rts"] == 2

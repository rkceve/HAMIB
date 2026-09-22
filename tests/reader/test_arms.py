"""S2.4 tests for benchmark/bineval/arms.py (CPU only, no model)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.bineval.arms import (
    DEFAULT_CHAT,
    EXCLUDED_SESSIONS,
    ArmSpec,
    build_arm_context,
    cd_from_records,
    excluded_sessions_from_markdown,
    load_chat,
    make_token_counter,
    parse_arm,
    raw_tokens,
    truncation_sessions,
)

# RESULTS_v1.md sec.1: truncation 2x = last 23 sessions, 6x = 7, 10x = 4, 20x = 2.
EXPECTED_TRUNC_SESSIONS = {2: 23, 6: 7, 10: 4, 20: 2}
# RESULTS_v1.md sec.1 "Context tokens" column for the same rows.
EXPECTED_TRUNC_TOKENS = {2: 85651, 6: 26621, 10: 14801, 20: 6228}


@pytest.fixture(scope="module")
def chat() -> dict:
    return load_chat(DEFAULT_CHAT)


def test_excluded_session_set_matches_exclusions_md() -> None:
    assert set(EXCLUDED_SESSIONS) == excluded_sessions_from_markdown()
    assert set(EXCLUDED_SESSIONS) == {19, 24, 26, 32, 33, 35, 40}


def test_truncation_session_counts_reproduce_results_v1(chat: dict) -> None:
    counter = make_token_counter()
    counts = {
        r: len(truncation_sessions(chat, r, counter)) for r in EXPECTED_TRUNC_SESSIONS
    }
    assert counts == EXPECTED_TRUNC_SESSIONS


def test_the_note_header_is_paid_out_of_the_truncation_budget(chat: dict) -> None:
    """Review fix: the NOTE header is part of the arm's context, so it is part
    of the arm's budget.  At the published ratios it changes no session count
    and no body token count -- 85,651 / 26,621 / 14,801 / 6,228 still reproduce
    exactly -- but the TOTAL now stays inside raw/r instead of exceeding it."""
    from benchmark.bineval.arms import truncation_header_tokens

    counter = make_token_counter()
    header = truncation_header_tokens(counter)
    assert header > 0
    base = raw_tokens(chat, counter)
    for r, expected in EXPECTED_TRUNC_TOKENS.items():
        ctx = build_arm_context("trunc_%dx" % r, chat=chat, counter=counter)
        assert ctx.meta["header_tokens"] == header
        # the RESULTS_v1 number is the SESSION BODY, unchanged
        assert ctx.meta["body_tokens"] == expected, "ratio %dx" % r
        assert ctx.meta["n_sessions"] == EXPECTED_TRUNC_SESSIONS[r]
        # ...and header + body is now inside the budget
        assert ctx.tokens == expected + header
        assert ctx.tokens <= base / r


def test_header_tokens_shrink_the_budget(chat: dict) -> None:
    counter = make_token_counter()
    huge = 10 ** 9  # a header nobody can afford
    assert truncation_sessions(chat, 6, counter, header_tokens=huge) == []


def test_truncation_arms_reproduce_results_v1_token_counts(chat: dict) -> None:
    """The session bodies (without the truncation NOTE header) match RESULTS_v1."""
    from benchmark.bineval.arms import format_sessions

    counter = make_token_counter()
    for r, expected in EXPECTED_TRUNC_TOKENS.items():
        body = format_sessions(truncation_sessions(chat, r, counter))
        assert counter(body) == expected, "ratio %dx" % r


def test_truncation_is_a_suffix_of_the_chat(chat: dict) -> None:
    kept = truncation_sessions(chat, 6)
    all_n = [s["n"] for s in chat["sessions"]]
    assert [s["n"] for s in kept] == all_n[-len(kept):]


def test_raw_token_base_is_the_frozen_protocol_number(chat: dict) -> None:
    # PROTOCOL.md "Budget definition": restaurant raw = 172,773 tokens.
    assert raw_tokens(chat) == 172773


def test_parse_arm() -> None:
    assert parse_arm("full") == ArmSpec("full")
    assert parse_arm("floor") == ArmSpec("floor")
    assert parse_arm("trunc_6x") == ArmSpec("trunc", ratio=6.0)
    assert parse_arm("cd_mass_10x") == ArmSpec("cd", ratio=10.0, policy="mass")
    assert parse_arm("summary_9x") == ArmSpec("summary", ratio=9.0)
    assert parse_arm("oracle_cd_full") == ArmSpec("oracle")
    with pytest.raises(ValueError):
        parse_arm("cd_semantic_6x")


def test_full_arm_drops_every_excluded_session(chat: dict) -> None:
    ctx = build_arm_context("full", chat=chat)
    assert set(ctx.meta["sessions"]).isdisjoint(EXCLUDED_SESSIONS)
    assert len(ctx.meta["sessions"]) == len(chat["sessions"]) - len(EXCLUDED_SESSIONS)
    for n in EXCLUDED_SESSIONS:
        assert ("=== Session %d " % n) not in ctx.text


@pytest.mark.parametrize(
    "arm", ["full", "trunc_2x", "trunc_6x", "trunc_10x", "trunc_20x",
            "summary_9x", "oracle_cd_full", "floor"],
)
def test_every_arm_reports_tokens(arm: str, chat: dict) -> None:
    ctx = build_arm_context(arm, chat=chat)
    counter = make_token_counter()
    assert ctx.tokens == counter(ctx.text) == ctx.meta["tokens"]
    assert ctx.tokens >= 0
    if arm == "floor":
        assert ctx.text == "" and ctx.tokens == 0
    else:
        assert ctx.tokens > 0


def test_summary_arm_measured_ratio_is_about_9x(chat: dict) -> None:
    ctx = build_arm_context("summary_9x", chat=chat)
    # RESULTS_v1: 19,115 tokens = 9.0x. The file name (summary_6x.txt) is kept.
    assert ctx.tokens == 19115
    assert 8.9 <= ctx.meta["measured_ratio"] <= 9.1
    assert ctx.meta["source_file"] == "results/pilot/summary_6x.txt"


# --------------------------------------------------------------------------
# CD arm
# --------------------------------------------------------------------------

_CD_RECORDS = [
    {"node_id": "s1", "text": "Restaurant plan", "level": "sun", "mass": 0.0,
     "parent_id": None, "created_turn": 1},
    {"node_id": "p1", "text": "Budget", "level": "planet", "mass": 2.0,
     "parent_id": "s1", "created_turn": 1},
    {"node_id": "r1", "text": "Rent is 500k", "level": "satellite", "mass": 0.1,
     "parent_id": "p1", "created_turn": 1},
    {"node_id": "r2", "text": "Loan is 30M", "level": "satellite", "mass": 0.1,
     "parent_id": "p1", "created_turn": 2},
]


@pytest.fixture()
def cd_file(tmp_path: Path) -> Path:
    p = tmp_path / "cd.json"
    p.write_text(json.dumps({"nodes": _CD_RECORDS}), encoding="utf-8")
    return p


def test_cd_from_records_rebuilds_the_hierarchy() -> None:
    cd = cd_from_records(_CD_RECORDS)
    assert len(cd.suns) == 1
    assert len(cd.suns[0].planets) == 1
    assert len(cd.suns[0].planets[0].satellites) == 2


def test_cd_arm_uses_marker_format_when_requested(chat: dict, cd_file: Path) -> None:
    ctx = build_arm_context("cd_mass_6x", chat=chat, cd_json=cd_file, level_markers=True)
    assert "[SN] Restaurant plan" in ctx.text
    assert "[PN2.0] Budget" in ctx.text
    assert "[RN] Rent is 500k" in ctx.text
    assert ctx.meta["policy"] == "mass"
    assert ctx.meta["budget"] == int(172773 / 6)
    assert ctx.meta["nodes_kept"] == 4
    assert ctx.tokens > 0


def test_cd_arm_legacy_format_when_markers_off(chat: dict, cd_file: Path) -> None:
    ctx = build_arm_context("cd_mass_6x", chat=chat, cd_json=cd_file, level_markers=False)
    assert "[SN]" not in ctx.text
    assert ctx.text.count("[PN") == 4


def test_cd_arm_budget_is_enforced(chat: dict, cd_file: Path) -> None:
    counter = make_token_counter()
    ctx = build_arm_context(
        "cd_mass_6x", chat=chat, cd_json=cd_file, level_markers=True, counter=counter
    )
    assert counter(ctx.text) <= ctx.meta["budget"]


def test_cd_arm_requires_cd_json(chat: dict) -> None:
    with pytest.raises(ValueError):
        build_arm_context("cd_mass_6x", chat=chat, cd_json=None)

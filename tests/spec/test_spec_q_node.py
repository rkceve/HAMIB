"""S1.4: Q_NODE prompt hygiene, parsing, retry, fallback, clamping, tie table."""

from __future__ import annotations

import json

import pytest
from _spec_helpers import make_manager, node_reply, scripted_judge

from management.harness.backends import FakeJudge, first_fenced_span
from management.harness.prompts import (
    FALLBACK_NODE_CHARS,
    K_NODE,
    Q_NODE,
    Q_NODE_EXAMPLE_LINE,
)
from management.harness.spec_manager import (
    SCORE_MAX,
    SCORE_MIN,
    clamp_score,
    first_json_object,
    level_from_scores,
    parse_node_object,
)
from models.node import NodeLevel

SUN, PLANET, SATELLITE = NodeLevel.SUN, NodeLevel.PLANET, NodeLevel.SATELLITE


# -- prompt hygiene ---------------------------------------------------------


def test_q_node_fences_the_source_verbatim_and_carries_max_chars() -> None:
    fragment = "Rent is 500k yen.\nThe loan is 30M yen."
    prompt = Q_NODE.format(text=fragment, max_chars=120)
    assert first_fenced_span(prompt) == fragment
    assert "120" in prompt
    # M8: the ONLY braces in the rendered prompt are the example object's, and
    # they survive str.format because the template doubles them.
    assert prompt.count("{") == 1 and prompt.count("}") == 1
    assert Q_NODE_EXAMPLE_LINE in prompt


def test_q_node_states_the_json_keys_then_one_example_last() -> None:
    """Contract change (M8): the format sentence is still stated after the axes,
    and a single concrete example is now the very last line."""
    prompt = Q_NODE.format(text="x", max_chars=120)
    lines = prompt.strip().splitlines()
    assert lines[-1] == Q_NODE_EXAMPLE_LINE
    assert lines[-2] == (
        "Return only a JSON object with the keys summary, comprehensiveness, "
        "independence, detail."
    )
    # The three axes are named with the 0039 wording, before the format line.
    for axis in ("comprehensiveness", "independence", "detail"):
        assert prompt.index(axis) < prompt.index(lines[-2])


# -- parse_node_object ------------------------------------------------------


def test_parse_accepts_a_plain_object() -> None:
    parsed = parse_node_object(node_reply("A fact.", 10, 20, 30), 120)
    assert parsed == ("A fact.", {"comprehensiveness": 10, "independence": 20, "detail": 30})


def test_parse_tolerates_prose_around_the_object() -> None:
    raw = "Sure! Here you go:\n```json\n" + node_reply("A fact.", 1, 2, 3) + "\n```\n"
    parsed = parse_node_object(raw, 120)
    assert parsed is not None and parsed[0] == "A fact."


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "no json here",
        '{"summary": "a"}',  # missing scores
        '{"summary": "", "comprehensiveness": 1, "independence": 1, "detail": 1}',
        '{"summary": "a", "comprehensiveness": "hi", "independence": 1, "detail": 1}',
        '{"summary": "a", "comprehensiveness": true, "independence": 1, "detail": 1}',
        '["a", "b"]',  # an array is the OLD harness shape, not a node object
        "here is an object: {oops",  # unbalanced: no complete span
    ],
)
def test_parse_rejects_malformed_replies(raw: str) -> None:
    assert parse_node_object(raw, 120) is None


# -- M8: score coercion, nested scores, first balanced object ----------------


def test_parse_accepts_string_scores() -> None:
    """Contract change (M8): a model answering "90" has still ranked the axis."""
    raw = '{"summary": "a", "comprehensiveness": "90", "independence": " 10 ", "detail": 0}'
    assert parse_node_object(raw, 120) == (
        "a", {"comprehensiveness": 90, "independence": 10, "detail": 0}
    )


def test_parse_accepts_float_scores() -> None:
    """Contract change (M8): 1.5 used to be rejected by the integer schema."""
    raw = '{"summary": "a", "comprehensiveness": 1.5, "independence": 1, "detail": 1}'
    assert parse_node_object(raw, 120) == (
        "a", {"comprehensiveness": 1, "independence": 1, "detail": 1}
    )


def test_parse_lifts_a_nested_scores_block() -> None:
    raw = json.dumps(
        {"summary": "a", "scores": {"comprehensiveness": 5, "independence": 60, "detail": 7}}
    )
    assert parse_node_object(raw, 120) == (
        "a", {"comprehensiveness": 5, "independence": 60, "detail": 7}
    )


def test_top_level_scores_win_over_a_nested_block() -> None:
    raw = json.dumps(
        {"summary": "a", "comprehensiveness": 1, "independence": 2, "detail": 3,
         "scores": {"comprehensiveness": 90, "independence": 90, "detail": 90}}
    )
    assert parse_node_object(raw, 120) == (
        "a", {"comprehensiveness": 1, "independence": 2, "detail": 3}
    )


def test_parse_takes_the_first_of_two_objects() -> None:
    """The old first-brace..last-brace rule concatenated both and parsed neither."""
    raw = (
        "Sure: " + node_reply("A fact.", 1, 2, 3)
        + "\nFor reference the example was "
        + '{"summary": "...", "comprehensiveness": 40, "independence": 70, "detail": 20}'
    )
    assert parse_node_object(raw, 120) == (
        "A fact.", {"comprehensiveness": 1, "independence": 2, "detail": 3}
    )


def test_parse_tolerates_trailing_prose_after_the_object() -> None:
    raw = node_reply("A fact.", 4, 5, 6) + "\nI hope that helps! Let me know."
    assert parse_node_object(raw, 120)[0] == "A fact."


def test_parse_survives_a_brace_inside_the_summary() -> None:
    raw = json.dumps(
        {"summary": "the set {a, b}", "comprehensiveness": 1,
         "independence": 2, "detail": 3}
    )
    assert parse_node_object(raw, 120)[0] == "the set {a, b}"


def test_first_json_object_helper() -> None:
    assert first_json_object("no braces") is None
    assert first_json_object('x {"a": 1} y {"b": 2}') == '{"a": 1}'
    assert first_json_object('{"a": {"b": 1}} tail') == '{"a": {"b": 1}}'


def test_parse_truncates_the_summary() -> None:
    parsed = parse_node_object(node_reply("x" * 300, 0, 0, 1), 20)
    assert parsed is not None and parsed[0] == "x" * 20


def test_parse_clamps_out_of_range_scores() -> None:
    parsed = parse_node_object(node_reply("A fact.", 250, -10, 100), 120)
    assert parsed is not None
    assert parsed[1] == {
        "comprehensiveness": SCORE_MAX,
        "independence": SCORE_MIN,
        "detail": 100,
    }


def test_clamp_score_bounds() -> None:
    assert clamp_score(-5) == 0
    assert clamp_score(0) == 0
    assert clamp_score(100) == 100
    assert clamp_score(1000) == 100


# -- the tie table (0040) ---------------------------------------------------


def _scores(c: int, i: int, d: int) -> dict[str, int]:
    return {"comprehensiveness": c, "independence": i, "detail": d}


@pytest.mark.parametrize(
    "c,i,d,level",
    [
        # all six strict orderings of three distinct scores
        (90, 50, 10, SUN),
        (90, 10, 50, SUN),
        (50, 90, 10, PLANET),
        (10, 90, 50, PLANET),
        (50, 10, 90, SATELLITE),
        (10, 50, 90, SATELLITE),
        # ties, resolved detail > independence > comprehensiveness
        (90, 90, 90, SATELLITE),
        (90, 90, 10, PLANET),
        (90, 10, 90, SATELLITE),
        (10, 90, 90, SATELLITE),
        # the 0040 fallback
        (0, 0, 0, SATELLITE),
    ],
)
def test_level_from_scores_tie_table(c: int, i: int, d: int, level: NodeLevel) -> None:
    assert level_from_scores(_scores(c, i, d)) is level


# -- the manager's node call ------------------------------------------------


def test_one_call_per_chunk_on_a_valid_reply() -> None:
    judge = scripted_judge(node=lambda t: node_reply(t, 90, 10, 10))
    manager = make_manager(judge)
    node = manager.node_for_text("The restaurant plan.", turn=3)
    assert node.text == "The restaurant plan."
    assert node.level is SUN
    assert node.mass == 0.0  # normalize() owns the mass (0062)
    assert node.created_turn == 3
    assert manager.runner.calls == {K_NODE: 1}
    assert manager.node_fallback == 0


def test_retry_then_valid() -> None:
    replies = iter(["not json at all", node_reply("A fact.", 0, 0, 80)])

    manager = make_manager(FakeJudge(policy=lambda _p: next(replies, "junk")))
    node = manager.node_for_text("Rent is 500k yen.")
    assert node.text == "A fact." and node.level is SATELLITE
    assert manager.runner.calls[K_NODE] == 2
    assert manager.runner.total_unparsed() == 1
    assert manager.runner.total_retried() == 1
    assert manager.runner.total_defaulted() == 0
    assert manager.node_fallback == 0
    # The retry carries the OBJECT reformat suffix, not the array one.
    assert "Reply with a JSON object only" in judge_prompts(manager)[-1]


def judge_prompts(manager) -> list[str]:
    return manager.runner.judge.prompts


def test_fallback_after_every_attempt_fails() -> None:
    text = "x" * 200
    manager = make_manager(FakeJudge(policy=lambda _p: "still not json"))
    node = manager.node_for_text(text)
    assert node.text == text[:FALLBACK_NODE_CHARS]
    assert node.level is SATELLITE
    # 1 + max_retries attempts
    assert manager.runner.calls[K_NODE] == 2
    assert manager.runner.total_unparsed() == 2
    assert manager.runner.total_defaulted() == 1
    assert manager.node_fallback == 1


def test_max_retries_is_honoured() -> None:
    manager = make_manager(FakeJudge(policy=lambda _p: "junk"), max_retries=3)
    manager.node_for_text("A fact.")
    assert manager.runner.calls[K_NODE] == 4


def test_a_valid_answer_is_cached() -> None:
    judge = scripted_judge(node=lambda t: node_reply(t, 0, 0, 50))
    manager = make_manager(judge)
    manager.node_for_text("Rent is 500k yen.")
    manager.node_for_text("Rent is 500k yen.")
    assert manager.runner.calls[K_NODE] == 1
    assert manager.cache.hits == 1


def test_a_fallback_is_never_cached() -> None:
    """A default is the absence of an answer; caching it would freeze one
    transport hiccup into every later decision about that chunk."""
    manager = make_manager(FakeJudge(policy=lambda _p: "junk"))
    manager.node_for_text("Rent is 500k yen.")
    manager.node_for_text("Rent is 500k yen.")
    assert manager.runner.calls[K_NODE] == 4  # 2 attempts, twice
    assert manager.node_fallback == 2


def test_summary_is_truncated_to_max_node_chars() -> None:
    judge = scripted_judge(node=lambda t: node_reply("y" * 300, 0, 0, 10))
    manager = make_manager(judge, max_node_chars=25)
    node = manager.node_for_text("Rent is 500k yen.")
    assert node.text == "y" * 25


def test_node_prompt_uses_the_configured_max_chars() -> None:
    judge = scripted_judge(node=lambda t: node_reply(t, 0, 0, 10))
    manager = make_manager(judge, max_node_chars=37)
    manager.node_for_text("Rent is 500k yen.")
    assert "37 characters" in judge.prompts[0]


def test_node_call_uses_node_max_tokens() -> None:
    seen: list[int] = []

    class _Spy:
        def complete(self, prompt: str, *, max_tokens: int) -> str:
            seen.append(max_tokens)
            return node_reply("A fact.", 0, 0, 1)

    manager = make_manager(_Spy(), node_max_tokens=333)
    manager.node_for_text("Rent is 500k yen.")
    assert seen == [333]


def test_empty_summary_after_truncation_falls_back() -> None:
    """A summary of only whitespace is not an answer."""
    manager = make_manager(FakeJudge(policy=lambda _p: json.dumps(
        {"summary": "   ", "comprehensiveness": 1, "independence": 1, "detail": 1}
    )))
    node = manager.node_for_text("Rent is 500k yen.")
    assert node.text == "Rent is 500k yen."
    assert manager.node_fallback == 1

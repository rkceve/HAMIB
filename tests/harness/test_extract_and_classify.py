"""B7: extraction (JSON validation, retry counting, fallback) and the
classification precedence table (all 8 boolean combinations)."""

from __future__ import annotations

import json

import pytest
from _helpers import make_manager, scripted_judge

from management.harness.backends import FakeJudge
from management.harness.manager import FALLBACK_STATEMENT_CHARS
from management.harness.prompts import K_EXTRACT, K_SUPPORTED
from models.correlation_diagram import CorrelationDiagram
from models.node import NodeLevel

TEXT = "この店の名前はホシノ亭です。予算は3000円です。"


# -- extraction --------------------------------------------------------------


def test_valid_json_first_try() -> None:
    judge = scripted_judge(extract=lambda t: ["店の名前はホシノ亭である", "予算は3000円である"])
    m = make_manager(judge)
    assert m.extract_statements(TEXT) == ["店の名前はホシノ亭である", "予算は3000円である"]
    assert m.runner.calls[K_EXTRACT] == 1


def test_empty_array_is_valid_and_costs_one_call() -> None:
    judge = scripted_judge(extract=lambda t: [])
    m = make_manager(judge)
    assert m.extract_statements(TEXT) == []
    assert m.runner.calls[K_EXTRACT] == 1


def test_invalid_then_valid_counts_the_retry() -> None:
    replies = iter(["not json at all", json.dumps(["店の名前はホシノ亭である"])])
    judge = FakeJudge(policy=lambda _p: next(replies))
    m = make_manager(judge)
    assert m.extract_statements(TEXT) == ["店の名前はホシノ亭である"]
    assert m.runner.calls[K_EXTRACT] == 2
    assert "JSON array of strings only" in judge.prompts[1]


def test_always_invalid_falls_back_after_max_retries() -> None:
    judge = FakeJudge(default="sorry, I cannot")
    m = make_manager(judge, max_retries=2)
    assert m.extract_statements(TEXT) == [TEXT[:FALLBACK_STATEMENT_CHARS].strip()]
    assert m.runner.calls[K_EXTRACT] == 3  # 1 attempt + 2 retries


def test_schema_rejects_unusable_items() -> None:
    """Numbers are still not statements -> the fallback path."""
    judge = FakeJudge(default="[1, 2, 3]")
    m = make_manager(judge, max_retries=0)
    assert m.extract_statements(TEXT) == [TEXT[:FALLBACK_STATEMENT_CHARS].strip()]
    assert m.runner.calls[K_EXTRACT] == 1
    assert m.extract_fallback == 1


# -- H1: tolerant extraction ------------------------------------------------


def test_dict_element_shapes_are_accepted() -> None:
    """H1 (behaviour change): [{"text": ...}] / [{"statement": ...}] used to be
    rejected by the schema and threw the whole reply away."""
    m = make_manager(FakeJudge(default='[{"text": "x"}, {"statement": "y"}]'))
    assert m.extract_statements(TEXT) == ["x", "y"]
    assert m.extract_fallback == 0


def test_statements_object_shape_is_accepted() -> None:
    m = make_manager(FakeJudge(default='{"statements": ["a", "b"]}'))
    assert m.extract_statements(TEXT) == ["a", "b"]


def test_blank_elements_are_dropped_not_rejected() -> None:
    m = make_manager(FakeJudge(default='["a", "", "   ", "b"]'))
    assert m.extract_statements(TEXT) == ["a", "b"]
    assert m.extract_fallback == 0


def test_truncated_array_is_salvaged() -> None:
    """A reply cut off by the token budget still carries whole statements."""
    m = make_manager(FakeJudge(default='["first fact", "second fact", "third fa'))
    assert m.extract_statements(TEXT) == ["first fact", "second fact"]
    assert m.extract_salvaged == 1
    assert m.extract_fallback == 0
    # Salvage happens on the FIRST reply: retrying a truncation only truncates.
    assert m.runner.calls[K_EXTRACT] == 1


def test_salvage_honours_escapes() -> None:
    m = make_manager(FakeJudge(default=r'["a \"quoted\" fact", "next'))
    assert m.extract_statements(TEXT) == ['a "quoted" fact']


def test_salvage_needs_a_leading_bracket() -> None:
    m = make_manager(FakeJudge(default='"lonely string"'), max_retries=0)
    assert m.extract_statements(TEXT) == [TEXT[:FALLBACK_STATEMENT_CHARS].strip()]
    assert m.extract_fallback == 1


def test_extraction_uses_the_extract_token_budget() -> None:
    """H1: yes/no answers get 64 tokens, a JSON array gets its own budget."""
    seen: list[int] = []

    class _Judge:
        def complete(self, prompt: str, *, max_tokens: int) -> str:
            seen.append(max_tokens)
            return '["a"]'

    m = make_manager(_Judge(), judge_max_tokens=64, extract_max_tokens=999)  # type: ignore[arg-type]
    m.extract_statements(TEXT)
    assert seen == [999]
    m.is_supported("a", TEXT)
    assert seen[-1] == 64


def test_statements_are_truncated_and_blanks_dropped() -> None:
    judge = scripted_judge(extract=lambda t: ["あ" * 300, "  ", " 短い文 "])
    m = make_manager(judge, max_statement_chars=10)
    assert m.extract_statements(TEXT) == ["あ" * 10, "短い文"]


def test_extraction_is_cached() -> None:
    judge = scripted_judge(extract=lambda t: ["文"])
    m = make_manager(judge)
    m.extract_statements(TEXT)
    m.extract_statements(TEXT)
    assert m.runner.calls[K_EXTRACT] == 1
    assert m.cache.hits == 1


# -- classification precedence (all 8 combinations) --------------------------

PRECEDENCE = [
    # (comprehensive, independent, detail) -> level
    ((False, False, False), NodeLevel.SATELLITE),
    ((False, False, True), NodeLevel.SATELLITE),
    ((False, True, False), NodeLevel.PLANET),
    ((False, True, True), NodeLevel.SATELLITE),
    ((True, False, False), NodeLevel.SUN),
    ((True, False, True), NodeLevel.SATELLITE),
    ((True, True, False), NodeLevel.PLANET),
    ((True, True, True), NodeLevel.SATELLITE),
]


@pytest.mark.parametrize("axes,expected", PRECEDENCE)
def test_level_from_axes(axes: tuple[bool, bool, bool], expected: NodeLevel) -> None:
    from management.harness.manager import HarnessManager

    assert HarnessManager.level_from_axes(*axes) is expected


@pytest.mark.parametrize("axes,expected", PRECEDENCE)
def test_classify_statement_matches_the_table(
    axes: tuple[bool, bool, bool], expected: NodeLevel
) -> None:
    comprehensive, independent, detail = axes
    judge = scripted_judge(
        {
            "comprehensive": "yes" if comprehensive else "no",
            "independent": "yes" if independent else "no",
            "detail": "yes" if detail else "no",
        }
    )
    m = make_manager(judge)
    proposal = m.classify_statement("ある文", turn=7)
    assert proposal.node.level is expected
    assert proposal.node.created_turn == 7
    assert proposal.score_comprehensiveness == (1.0 if comprehensive else 0.0)
    assert proposal.score_independence == (1.0 if independent else 0.0)
    assert proposal.score_detail == (1.0 if detail else 0.0)


def test_classification_always_asks_all_three_axes() -> None:
    judge = scripted_judge({"detail": "yes"})
    m = make_manager(judge)
    m.classify_statement("ある文")
    assert m.runner.calls["comprehensive"] == 1
    assert m.runner.calls["independent"] == 1
    assert m.runner.calls["detail"] == 1


# -- faithfulness ------------------------------------------------------------


def test_unsupported_statement_is_dropped() -> None:
    judge = scripted_judge(
        {"supported": "no", "independent": "yes"},
        extract=lambda t: ["捏造された事実"],
    )
    m = make_manager(judge, faithfulness_check=True)
    base = CorrelationDiagram()
    report = m.update(base, TEXT, "", turn=0)
    assert report.statements >= 1
    assert report.dropped_unsupported == report.statements
    assert len(base) == 0
    assert "comprehensive" not in m.runner.calls  # dropped before classification


def test_faithfulness_check_can_be_disabled() -> None:
    judge = scripted_judge({"supported": "no"}, extract=lambda t: ["ある文"])
    m = make_manager(judge, faithfulness_check=False)
    base = CorrelationDiagram()
    report = m.update(base, "ある文。", "", turn=0)
    assert report.dropped_unsupported == 0
    assert K_SUPPORTED not in m.runner.calls
    assert len(base) > 0

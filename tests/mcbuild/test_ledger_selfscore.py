"""Scorer self-test on the mcbuild-bench ledger (DESIGN 0.5, item E): every
question answered with its own gold must pass tier 1 -> 96/96 (H22 (c): the
three retrospective-only facts f082-f084 were dropped with round trip 36)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.bineval.score_binary import (
    judge_none,
    score_condition,
    select_questions,
    tier1_match,
)

QUESTIONS = Path(__file__).resolve().parents[2] / "benchmark" / "mcbuild_bench" / "data" / "questions.json"


@pytest.fixture(scope="module")
def questions() -> list[dict]:
    if not QUESTIONS.exists():
        pytest.skip("ledger not built: %s" % QUESTIONS)
    return select_questions(json.loads(QUESTIONS.read_text(encoding="utf-8")), "all", False)


def test_every_question_passes_on_its_own_gold(questions: list[dict]) -> None:
    assert len(questions) == 96
    answers = {q["qid"]: q["gold_short"] for q in questions}
    res = score_condition(questions, {}, answers, judge_none, max_words=32)
    failed = [it["qid"] for it in res["items"] if it["verdict"] != "pass"]
    assert failed == [], failed
    assert res["aggregate"]["pass"] == 96


def test_no_question_points_at_an_excluded_round_trip(questions: list[dict]) -> None:
    from benchmark.mcbuild_bench.corpus import DEFAULT_EXCLUDE_RT

    assert DEFAULT_EXCLUDE_RT == (36,)
    assert not [q["qid"] for q in questions if q["source_session"] in DEFAULT_EXCLUDE_RT]
    assert not {"f082", "f083", "f084"} & {q["qid"] for q in questions}
    for q in questions:
        for h in q.get("history", []):
            assert h["rt"] not in DEFAULT_EXCLUDE_RT, q["qid"]


def test_ledger_refuses_evidence_that_lies_only_in_an_excluded_round_trip() -> None:
    from benchmark.mcbuild_bench.build_ledger import FACTS, excluded_rt_failures

    assert excluded_rt_failures(FACTS, (36,)) == []
    facts = [
        dict(id="fX", rt=36, evidence="e", history=[]),
        dict(id="fY", rt=3, evidence="e", history=[(36, "v", "ev")]),
        dict(id="fZ", rt=3, evidence="e", history=[]),
    ]
    failures = excluded_rt_failures(facts, (36,))
    assert [f.split(":")[0] for f in failures] == ["fX", "fY"]
    assert all("excluded round trip 36" in f for f in failures)


def test_every_alias_passes_on_its_own_text(questions: list[dict]) -> None:
    """Each alias, answered verbatim, must match (all-of aliases answered with
    their parts joined by a space)."""
    failed = []
    for q in questions:
        for alias in q.get("tier1_aliases", []):
            answer = " ".join(alias.split(" & "))
            ok, _ = tier1_match(q["gold_short"], q["tier1_aliases"], answer)
            if not ok:
                failed.append((q["qid"], alias))
    assert failed == [], failed


def test_absent_questions_fail_on_a_concrete_answer(questions: list[dict]) -> None:
    absent = [q for q in questions if q["kind"] == "absent"]
    assert len(absent) == 8
    for q in absent:
        ok, _ = tier1_match(q["gold_short"], q["tier1_aliases"], "RTX 4090")
        assert not ok

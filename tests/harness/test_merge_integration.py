"""B7: multi-turn merge integration (0042-0062) through HarnessManager.update.

A scripted judge drives every decision, so the whole path is deterministic and
no model is loaded.
"""

from __future__ import annotations

from _helpers import make_manager, scripted_judge, second_fenced_span

from management.harness.backends import first_fenced_span
from models.correlation_diagram import CorrelationDiagram
from models.node import Node, NodeLevel

# statement -> level for the classification axes
LEVELS = {
    "SUN": "sun",
    "P1": "planet",
    "P2": "planet",
    "S1": "satellite",
    "S2": "satellite",
    "ORPHAN": "satellite",
}
# this-turn parent links (0041)
BELONGS = {("SUN", "P1"), ("P1", "S1"), ("P1", "S2")}
# pairs the similarity judge calls "the same matter" (0042), order-insensitive
SAME = {frozenset({"P1", "P1"})}
# turn-4 orphan planet P2 ATTACHES under SUN by Q_BELONGS (0057), not Q_SAME
BELONGS.add(("SUN", "P2"))

TURNS = {
    "T1": ["SUN", "P1", "S1"],
    "T2": ["P1", "S2"],
    "T3": ["ORPHAN"],
    "T4": ["P2"],
}


def _judge():
    def axis(expected: str):
        def fn(prompt: str) -> str:
            statement = first_fenced_span(prompt).strip()
            return "yes" if LEVELS.get(statement) == expected else "no"

        return fn

    def belongs(prompt: str) -> str:
        # Q_BELONGS fences the statement first, then the topic.
        statement = first_fenced_span(prompt).strip()
        topic = second_fenced_span(prompt).strip()
        return "yes" if (topic, statement) in BELONGS else "no"

    def same(prompt: str) -> str:
        a = first_fenced_span(prompt).strip()
        b = second_fenced_span(prompt).strip()
        return "yes" if frozenset({a, b}) in SAME else "no"

    return scripted_judge(
        {
            "supported": "yes",
            "comprehensive": axis("sun"),
            "independent": axis("planet"),
            "detail": axis("satellite"),
            "belongs": belongs,
            "same": same,
        },
        extract=lambda text: TURNS[text],
    )


def _run(n_turns: int):
    m = make_manager(_judge())
    base = CorrelationDiagram()
    reports = []
    for i, key in enumerate(list(TURNS)[:n_turns]):
        reports.append(m.update(base, key, "", turn=i))
    return m, base, reports


def test_turn1_builds_the_tree() -> None:
    _m, base, reports = _run(1)
    assert [se.sun.text for se in base.suns] == ["SUN"]
    se = base.suns[0]
    assert [pe.planet.text for pe in se.planets] == ["P1"]
    assert [s.text for s in se.planets[0].satellites] == ["S1"]
    # 0062: planet mass = number of satellites below it
    assert se.planets[0].planet.mass == 1.0
    assert reports[0].added == 3
    assert reports[0].statements == 3
    assert reports[0].dropped_unsupported == 0


def test_turn2_repeated_planet_vanishes_and_its_satellite_attaches() -> None:
    _m, base, reports = _run(2)
    se = base.suns[0]
    # P1 was NOT duplicated (0056: the incoming planet vanishes)
    assert [pe.planet.text for pe in se.planets] == ["P1"]
    assert [s.text for s in se.planets[0].satellites] == ["S1", "S2"]
    assert se.planets[0].planet.mass == 2.0   # 0062 after normalize
    assert reports[1].added == 1              # only the new satellite
    assert reports[1].merged >= 1             # the P1 == P1 match was counted
    assert reports[1].promoted == 0


def test_turn3_orphan_satellite_similar_to_nothing_is_promoted_to_sun() -> None:
    _m, base, reports = _run(3)
    assert [se.sun.text for se in base.suns] == ["SUN", "ORPHAN"]
    assert reports[2].promoted == 1
    assert reports[2].merged == 0
    # 0058/0062: a sun with no planets keeps the default sun mass
    assert base.suns[1].sun.mass == 1.0


def test_turn4_orphan_planet_similar_to_a_sun_is_attached_under_it() -> None:
    _m, base, reports = _run(4)
    se = base.suns[0]
    assert [pe.planet.text for pe in se.planets] == ["P1", "P2"]
    assert se.planets[1].satellites == []
    assert se.planets[1].planet.mass == 1.0   # 0062 floor: min 1
    assert reports[3].promoted == 0           # attached, not promoted
    assert reports[3].added == 1
    # sun mass = sum of its planet masses (recorded deviation, unchanged here)
    assert se.sun.mass == 3.0


def test_normalize_runs_exactly_once_per_update() -> None:
    m, base, _reports = _run(4)
    assert m.normalize_calls == 4
    # coordinates were recalculated by that normalize
    assert base.suns[0].sun.coordinates.sun_idx == 0
    assert base.suns[1].sun.coordinates.sun_idx == 1


def test_report_counters_and_call_kinds() -> None:
    m, _base, reports = _run(4)
    for r in reports:
        assert r.chunks == 1
        assert r.total_calls() == sum(r.calls.values())
        assert set(r.calls) <= {
            "boundary", "extract", "supported", "comprehensive",
            "independent", "detail", "belongs", "same",
        }
    assert "boundary" not in m.totals            # boundary_check=False here
    assert m.totals["extract"] == 4
    # H9: `totals` now carries the quality counters too, so the call identity is
    # checked on call_totals() (which is what the driver reports).
    assert m.call_totals() == {
        k: sum(r.calls.get(k, 0) for r in reports) for k in m.call_totals()
    }
    assert m.total_cache_hits == sum(r.cache_hits for r in reports)


def test_vanished_and_attached_are_reported_separately() -> None:
    """H14: `merged` conflated the 0056 vanish with the 0057 attach."""
    _m, _base, reports = _run(4)
    for r in reports:
        assert r.merged == r.vanished + r.attached
    assert reports[1].vanished == 1 and reports[1].attached == 0   # P1 == P1
    assert reports[3].attached == 1 and reports[3].vanished == 0   # P2 under SUN


def test_quality_counters_are_clean_on_a_scripted_run() -> None:
    m, _base, reports = _run(4)
    for r in reports:
        assert (r.unparsed, r.defaulted, r.extract_fallback, r.extract_salvaged) == (
            0,
            0,
            0,
            0,
        )
    assert m.quality_totals()["defaulted"] == 0
    assert m.quality_totals()["vanished"] == 1
    assert m.quality_totals()["attached"] == 1


def test_update_normalizes_and_accounts_even_when_a_step_raises() -> None:
    """H4: a judge that dies mid-turn must not leave the diagram un-normalized
    nor lose the calls it already paid for."""
    import pytest

    from management.harness.backends import FakeJudge

    calls = {"n": 0}

    def policy(_prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("judge is down")
        return '["only fact"]'

    m = make_manager(FakeJudge(policy=policy))
    base = CorrelationDiagram()
    with pytest.raises(RuntimeError):
        m.update(base, "T", "", turn=0)
    assert m.normalize_calls == 1
    assert m.totals.get("extract", 0) == 1
    assert m.totals.get("supported", 0) == 1


def test_cache_reuse_across_turns() -> None:
    """Turn 2 re-uses the statement-only answers about P1.

    H6 (behaviour change): Q_COMPREHENSIVE / Q_INDEPENDENT now carry the chunk
    they came from, so their cache key includes that context and a repeated
    statement in a NEW context is asked again.  Q_DETAIL is statement-only and
    is still served from the cache.
    """
    m = make_manager(_judge())
    base = CorrelationDiagram()
    m.update(base, "T1", "", turn=0)
    calls_after_t1 = dict(m.runner.calls)
    r2 = m.update(base, "T2", "", turn=1)
    assert r2.cache_hits > 0
    assert m.runner.calls["detail"] - calls_after_t1["detail"] == 1  # P1 cached
    assert m.runner.calls["comprehensive"] - calls_after_t1["comprehensive"] == 2


def test_context_and_topics_reach_the_classification_prompts() -> None:
    """H6: the two topic-level axes see the discussion and the existing topics."""
    seen: list[str] = []

    def spy(prompt: str) -> str:
        seen.append(prompt)
        return "no"

    m = make_manager(
        scripted_judge({"supported": "yes", "comprehensive": spy}, extract=lambda t: ["P1"])
    )
    base = CorrelationDiagram()
    m.update(base, "SUN", "", turn=0)          # nothing recorded yet
    assert "Current topics:\nnone" in seen[0]
    assert "SUN" in seen[0]                    # the chunk it came from

    base.add_sun(Node(text="A TOPIC", level=NodeLevel.SUN, mass=1.0))
    m.update(base, "P1", "", turn=1)
    assert "- A TOPIC" in seen[-1]


def test_topics_list_is_capped() -> None:
    seen: list[str] = []
    m = make_manager(
        scripted_judge(
            {"supported": "yes", "comprehensive": lambda p: (seen.append(p), "no")[1]},
            extract=lambda t: ["S"],
        ),
        topics_in_prompt=3,
    )
    base = CorrelationDiagram()
    for i in range(6):
        base.add_sun(Node(text="TOPIC%d" % i, level=NodeLevel.SUN, mass=1.0))
    m.update(base, "S", "", turn=0)
    assert seen[0].count("- TOPIC") == 3


def test_within_turn_duplicates_are_dropped() -> None:
    """H8: the same fact stated twice in one turn becomes ONE node."""
    m = make_manager(
        scripted_judge(
            {"supported": "yes", "independent": "yes"},
            extract=lambda t: ["The rent is fixed.", "the rent is fixed", "Other fact"],
        )
    )
    base = CorrelationDiagram()
    report = m.update(base, "T", "", turn=0)
    assert report.statements == 3
    assert report.dedup_dropped == 1          # normalized text collision
    assert len(base) == 2


def test_judge_level_duplicates_are_dropped() -> None:
    """H8: Q_SAME removes near-duplicates of the SAME level within one turn."""
    m = make_manager(
        scripted_judge(
            {"supported": "yes", "independent": "yes", "same": "yes"},
            extract=lambda t: ["Alpha fact", "Beta fact"],
        )
    )
    base = CorrelationDiagram()
    report = m.update(base, "T", "", turn=0)
    assert report.dedup_dropped == 1
    assert len(base) == 1


def test_parallel_stage_matches_the_sequential_one() -> None:
    """H7: max_workers>1 changes the schedule, never the result."""
    statements = ["Fact %d here." % i for i in range(8)]

    def build(workers: int):
        m = make_manager(
            scripted_judge(
                {"supported": "yes", "independent": "yes"},
                extract=lambda t: statements,
            ),
            max_workers=workers,
        )
        base = CorrelationDiagram()
        report = m.update(base, "T", "", turn=0)
        return m, base, report

    m1, base1, r1 = build(1)
    m4, base4, r4 = build(4)
    assert [n.text for n in base1.all_nodes()] == [n.text for n in base4.all_nodes()]
    assert r1.statements == r4.statements == 8
    assert m1.call_totals() == m4.call_totals()

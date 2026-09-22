"""S1.4: a two-turn tree through SpecManager.update() with a scripted judge."""

from __future__ import annotations

import pytest
from _spec_helpers import make_manager, node_reply, same_text_judge, scripted_judge

from management.harness.prompts import K_BELONGS, K_NODE, K_SAME
from models.correlation_diagram import CorrelationDiagram
from models.node import NodeLevel

SUN_TEXT = "The restaurant plan is the topic."
PLANET_TEXT = "The budget matters most."
SAT1 = "Rent is 500k yen."
SAT2 = "The loan is 30M yen."
SAT3 = "Delivery costs 2k yen."
PLANET2_TEXT = "Marketing is a new pillar."

TURN1 = " ".join([SUN_TEXT, PLANET_TEXT, SAT1, SAT2])
TURN2 = " ".join([PLANET_TEXT, PLANET2_TEXT, SAT3])


def _node(text: str) -> str:
    """comprehensive -> sun, independent -> planet, otherwise detail -> satellite."""
    if "restaurant plan" in text:
        return node_reply(text, 90, 10, 10)
    if "budget" in text or "Marketing" in text:
        return node_reply(text, 10, 90, 10)
    return node_reply(text, 10, 10, 90)


def _run_two_turns(**cfg: object):
    manager = make_manager(same_text_judge(node=_node), **cfg)
    base = CorrelationDiagram()
    r1 = manager.update(base, TURN1, "", 0)
    r2 = manager.update(base, TURN2, "", 1)
    return manager, base, r1, r2


def test_turn1_builds_the_tree() -> None:
    manager = make_manager(same_text_judge(node=_node))
    base = CorrelationDiagram()
    r1 = manager.update(base, TURN1, "", 0)
    assert r1.chunks == 4 and r1.nodes == 4
    assert len(base.suns) == 1
    se = base.suns[0]
    assert se.sun.text == SUN_TEXT
    assert [pe.planet.text for pe in se.planets] == [PLANET_TEXT]
    assert [s.text for s in se.planets[0].satellites] == [SAT1, SAT2]
    assert r1.added == 4
    assert manager.normalize_calls == 1  # one per update()


def test_turn2_planet_vanishes_and_the_satellite_lands_under_it() -> None:
    _manager, base, _r1, r2 = _run_two_turns()
    se = base.suns[0]
    assert len(base.suns) == 1  # nothing was promoted to a new sun
    budget = se.planets[0]
    assert budget.planet.text == PLANET_TEXT
    assert [s.text for s in budget.satellites] == [SAT1, SAT2, SAT3]
    # 0062: mass = satellite count.
    assert budget.planet.mass == 3.0
    assert r2.vanished >= 1


def test_turn2_orphan_planet_attaches_to_the_existing_sun_by_q_belongs() -> None:
    _manager, base, _r1, r2 = _run_two_turns()
    se = base.suns[0]
    assert [pe.planet.text for pe in se.planets] == [PLANET_TEXT, PLANET2_TEXT]
    assert se.planets[1].planet.parent_id == se.sun.node_id
    assert r2.attached >= 1
    assert r2.promoted == 0


def test_planet_mass_floor_zero_vs_default() -> None:
    """A planet with no satellites: 0.0 with the spec floor, 1.0 with the
    diagram's own default."""
    _m0, base0, _a, _b = _run_two_turns(planet_mass_floor=0.0)
    assert base0.suns[0].planets[1].planet.mass == 0.0
    assert base0.suns[0].planets[0].planet.mass == 3.0

    _m1, base1, _c, _d = _run_two_turns(planet_mass_floor=1.0)
    assert base1.suns[0].planets[1].planet.mass == 1.0
    # The planet with 3 satellites is identical under either floor.
    assert base1.suns[0].planets[0].planet.mass == 3.0


def test_report_counters_and_call_kinds() -> None:
    manager, _base, r1, r2 = _run_two_turns()
    assert set(manager.call_totals()) <= {K_NODE, K_SAME, K_BELONGS}
    assert manager.call_totals()[K_NODE] > 0
    assert manager.quality_totals() == {
        "unparsed": 0,
        "defaulted": 0,
        "node_fallback": 0,
        "vanished": r1.vanished + r2.vanished,
        "attached": r1.attached + r2.attached,
    }
    assert r2.cache_hits >= 1  # PLANET_TEXT was asked in turn 1 already
    assert sum(manager.totals[k] for k in (K_NODE,)) == manager.runner.calls[K_NODE]


def test_one_chunk_is_one_node() -> None:
    """No statement extraction, no dedup: the node count equals the chunk count
    even when two chunks say the same thing."""
    text = " ".join([SAT1, SAT1, SAT1])
    manager = make_manager(same_text_judge(node=_node))
    report = manager.update(CorrelationDiagram(), text, "", 0)
    assert report.chunks == 3
    assert report.nodes == 3


def test_user_and_assistant_are_both_chunked() -> None:
    manager = make_manager(same_text_judge(node=_node))
    report = manager.update(CorrelationDiagram(), SAT1, SAT2, 0)
    assert report.chunks == 2
    assert [c.source for c in manager.chunk(SAT1, SAT2, 0)] == ["user", "assistant"]


def test_blank_sides_are_skipped() -> None:
    manager = make_manager(same_text_judge(node=_node))
    assert manager.chunk("   ", "", 0) == []


# -- normalize exactly once --------------------------------------------------


class _CountingCD(CorrelationDiagram):
    def __init__(self) -> None:
        super().__init__()
        self.normalize_kwargs: list[float | None] = []

    def normalize(self, planet_mass_floor: float | None = None) -> None:
        self.normalize_kwargs.append(planet_mass_floor)
        super().normalize(planet_mass_floor=planet_mass_floor)


def test_normalize_runs_exactly_once_per_update() -> None:
    """A satellite-only turn never reaches GraphMerger.merge(), so the ONLY
    normalize is the manager's own one in the `finally`."""
    manager = make_manager(scripted_judge(node=lambda t: node_reply(t, 0, 0, 90)))
    base = _CountingCD()
    manager.update(base, SAT1, "", 0)
    assert base.normalize_kwargs == [0.0]
    assert manager.normalize_calls == 1


def test_normalize_runs_even_when_the_judge_raises() -> None:
    class _DyingJudge:
        def complete(self, prompt: str, *, max_tokens: int) -> str:
            raise RuntimeError("judge is down")

    manager = make_manager(_DyingJudge())
    base = _CountingCD()
    with pytest.raises(RuntimeError):
        manager.update(base, SAT1, "", 0)
    assert base.normalize_kwargs == [0.0]
    assert manager.normalize_calls == 1


def test_the_manager_normalize_is_the_last_word_when_the_merger_normalizes() -> None:
    """GraphMerger.merge() normalizes internally with the diagram's own floor;
    update()'s single normalize then re-runs it with the SPEC floor."""
    manager = make_manager(same_text_judge(node=_node))
    base = _CountingCD()
    manager.update(base, TURN1, "", 0)
    assert manager.normalize_calls == 1
    assert base.normalize_kwargs[-1] == 0.0  # the manager's call is last
    assert base.normalize_kwargs[:-1] == [None]  # GraphMerger.merge()'s own


# -- checkpoint resume -------------------------------------------------------


def test_load_totals_restores_counters_and_cache_hits() -> None:
    manager, _base, _r1, _r2 = _run_two_turns()
    resumed = make_manager(same_text_judge(node=_node))
    resumed.load_totals(dict(manager.totals), manager.total_cache_hits)
    assert resumed.call_totals() == manager.call_totals()
    assert resumed.quality_totals() == manager.quality_totals()
    assert resumed.total_cache_hits == manager.total_cache_hits


# -- provisional structure ---------------------------------------------------


def test_unlinked_nodes_become_orphans() -> None:
    """With Q_BELONGS always "no" the planet and the satellites stay orphans."""
    manager = make_manager(scripted_judge({"belongs": "no"}, node=_node))
    nodes = [manager.node_for_text(t) for t in (SUN_TEXT, PLANET_TEXT, SAT1)]
    structure = manager.build_provisional(nodes)
    assert [se.sun.text for se in structure.suns] == [SUN_TEXT]
    assert [pe.planet.text for pe in structure.orphan_planets] == [PLANET_TEXT]
    assert [n.text for n in structure.orphan_satellites] == [SAT1]
    assert all(n.level is NodeLevel.SATELLITE for n in structure.orphan_satellites)

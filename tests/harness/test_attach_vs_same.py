"""Regression: attach decisions (0057/0060/0061) must ask Q_BELONGS, not Q_SAME.

Before the fix GraphMerger used one similarity function for both the vanish
("same matter") and attach ("belongs under") decisions, so an orphan planet
about an existing topic was never attached and got promoted to a new sun.
"""

from __future__ import annotations

import json
import re

from management.graph_merger import GraphMerger
from management.harness import FakeJudge, HarnessConfig, HarnessManager
from management.harness.backends import first_fenced_span
from management.harness.prompts import FENCE_CLOSE, FENCE_OPEN
from models.correlation_diagram import CorrelationDiagram
from models.node import Node, NodeLevel

_FENCES = re.compile(
    re.escape(FENCE_OPEN) + r"\n(.*?)\n" + re.escape(FENCE_CLOSE), re.DOTALL
)


def _policy(prompt: str) -> str | None:
    spans = _FENCES.findall(prompt)
    if "JSON array" in prompt:
        sents = [s for s in first_fenced_span(prompt).split("。") if s.strip()]
        return json.dumps(sents, ensure_ascii=False)
    if "supported by the source" in prompt:
        return "yes"
    if "title or the summary" in prompt:
        return "yes" if "について" in spans[0] else "no"
    if "new pillar of the discussion" in prompt:
        return "yes" if "旅行" in spans[0] and not re.search(r"\d", spans[0]) else "no"
    if "concrete step" in prompt:
        return "yes" if re.search(r"\d", spans[0]) else "no"
    if "belong under this topic" in prompt:
        return "yes" if "旅行" in spans[0] and "旅行" in spans[1] else "no"
    if "state the same matter" in prompt:
        return "yes" if spans[0] == spans[1] else "no"
    return "no"


def test_orphan_planet_attaches_to_existing_sun_via_belongs() -> None:
    mgr = HarnessManager(FakeJudge(policy=_policy), config=HarnessConfig(shortlist_k=0))
    base = CorrelationDiagram()
    mgr.update(base, "夏の旅行について相談したい。", "", turn=0)
    assert len(base.suns) == 1
    # Turn 2 has no sun statement: the planet is an orphan (0055) and must attach
    # under the existing sun by "belongs" (0057), not be promoted to a new sun.
    mgr.update(base, "旅行の行き先は北海道にしたい。", "", turn=1)
    assert len(base.suns) == 1
    assert [pe.planet.text for pe in base.suns[0].planets] == ["旅行の行き先は北海道にしたい"]
    # Turn 3: orphan satellite (has a digit) attaches under the planet by "belongs" (0060).
    mgr.update(base, "旅行の予算は10万円だ。", "", turn=2)
    assert len(base.suns) == 1
    pe = base.suns[0].planets[0]
    assert [s.text for s in pe.satellites] == ["旅行の予算は10万円だ"]
    assert pe.planet.mass == 1.0
    assert "belongs" in mgr.totals  # attach decisions were asked as Q_BELONGS


def test_graph_merger_attach_fn_defaults_to_similarity_fn() -> None:
    calls: list[str] = []

    def same(q: str, c: list[str]) -> tuple[int, float]:
        calls.append("same")
        return 0, 1.0

    base = CorrelationDiagram()
    base.add_sun(Node(text="S", level=NodeLevel.SUN, mass=1.0))
    GraphMerger(similarity_fn=same).merge_case3_satellite(
        base, Node(text="x", level=NodeLevel.SATELLITE, mass=0.1)
    )
    # 0059 skipped (no satellites), 0060 skipped (no planets), 0061 → default attach = same
    assert calls == ["same"]
    assert [pe.planet.text for pe in base.suns[0].planets] == ["x"]


def test_graph_merger_routes_attach_decisions_to_attach_fn() -> None:
    log: list[tuple[str, str]] = []

    def same(q: str, c: list[str]) -> tuple[int, float]:
        log.append(("same", q))
        return 0, 0.0

    def attach(q: str, c: list[str]) -> tuple[int, float]:
        log.append(("attach", q))
        return 0, 1.0

    base = CorrelationDiagram()
    base.add_sun(Node(text="S", level=NodeLevel.SUN, mass=1.0))
    base.add_planet(
        Node(text="P", level=NodeLevel.PLANET, mass=1.0), base.suns[0].sun.node_id
    )
    m = GraphMerger(similarity_fn=same, attach_fn=attach)
    # case 2: planet vs planets = same (no) -> planet vs suns = attach (yes)
    m.merge_case2_planet(base, Node(text="Q", level=NodeLevel.PLANET, mass=1.0), [])
    assert log == [("same", "Q"), ("attach", "Q")]
    assert {pe.planet.text for pe in base.suns[0].planets} == {"P", "Q"}
    log.clear()
    # case 3: sat vs sats (none) -> sat vs planets = attach (yes)
    m.merge_case3_satellite(base, Node(text="d", level=NodeLevel.SATELLITE, mass=0.1))
    assert log == [("attach", "d")]

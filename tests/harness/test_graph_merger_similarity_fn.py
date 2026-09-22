"""B7: GraphMerger's new similarity_fn parameter is backward compatible and is
routed to every internal similarity call site."""

from __future__ import annotations

import management.graph_merger as gm
from models.correlation_diagram import CorrelationDiagram
from models.node import Node, NodeLevel


class Recorder:
    """Records every (query, candidates) pair and never reports a match."""

    def __init__(self, score: float = 0.0) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self.score = score

    def __call__(self, query: str, candidates: list[str]) -> tuple[int, float]:
        self.calls.append((query, list(candidates)))
        return 0, self.score


def _node(text: str, level: NodeLevel) -> Node:
    return Node(text=text, level=level, mass=1.0)


def _base() -> CorrelationDiagram:
    cd = CorrelationDiagram()
    sun = _node("SUN", NodeLevel.SUN)
    cd.add_sun(sun)
    planet = _node("PLANET", NodeLevel.PLANET)
    cd.add_planet(planet, sun.node_id)
    cd.add_satellite(_node("SAT", NodeLevel.SATELLITE), planet.node_id)
    return cd


# -- default behaviour is unchanged ------------------------------------------


def test_default_uses_most_similar_index(monkeypatch) -> None:
    seen: list[tuple[str, list[str]]] = []

    def fake(query: str, candidates: list[str]) -> tuple[int, float]:
        seen.append((query, list(candidates)))
        return 0, 0.0

    monkeypatch.setattr(gm, "most_similar_index", fake)
    merger = gm.GraphMerger()          # no similarity_fn -> default path
    merger.merge_case3_satellite(_base(), _node("X", NodeLevel.SATELLITE))
    assert seen, "the default similarity_fn must call most_similar_index"
    assert seen[0][0] == "X"


def test_default_resolves_most_similar_index_lazily(monkeypatch) -> None:
    """Constructing the merger BEFORE the monkeypatch must still route to it."""
    merger = gm.GraphMerger()
    seen: list[str] = []
    monkeypatch.setattr(
        gm, "most_similar_index", lambda q, c: (seen.append(q), (0, 0.0))[1]
    )
    merger.merge_case3_satellite(_base(), _node("X", NodeLevel.SATELLITE))
    assert seen


# -- custom fn is routed to all five call sites ------------------------------


def test_find_matching_sun_idx_routed() -> None:
    rec = Recorder()
    gm.GraphMerger(similarity_fn=rec)._find_matching_sun_idx("Q", _base())
    assert rec.calls == [("Q", ["SUN"])]


def test_find_matching_planet_id_routed() -> None:
    rec = Recorder()
    base = _base()
    sun_id = base.suns[0].sun.node_id
    gm.GraphMerger(similarity_fn=rec)._find_matching_planet_id("Q", sun_id, base)
    assert rec.calls == [("Q", ["PLANET"])]


def test_find_any_matching_planet_routed() -> None:
    rec = Recorder()
    gm.GraphMerger(similarity_fn=rec)._find_any_matching_planet("Q", _base())
    assert rec.calls == [("Q", ["PLANET"])]


def test_merge_satellite_under_planet_routed() -> None:
    rec = Recorder()
    base = _base()
    planet_id = base.suns[0].planets[0].planet.node_id
    gm.GraphMerger(similarity_fn=rec)._merge_satellite_under_planet(
        base, planet_id, _node("Q", NodeLevel.SATELLITE)
    )
    assert rec.calls == [("Q", ["SAT"])]
    # non-match -> the satellite was added
    assert len(base.suns[0].planets[0].satellites) == 2


def test_merge_case3_satellite_direct_call_routed() -> None:
    rec = Recorder()
    gm.GraphMerger(similarity_fn=rec).merge_case3_satellite(
        _base(), _node("Q", NodeLevel.SATELLITE)
    )
    # satellites, then planets, then suns -- all through the injected fn
    assert [c[1] for c in rec.calls] == [["SAT"], ["PLANET"], ["SUN"]]


def test_matching_fn_makes_the_satellite_vanish() -> None:
    """score 1.0 >= threshold -> 0059 vanish, nothing added."""
    rec = Recorder(score=1.0)
    base = _base()
    before = len(base)
    gm.GraphMerger(similarity_fn=rec).merge_case3_satellite(
        base, _node("Q", NodeLevel.SATELLITE)
    )
    assert len(base) == before
    assert len(rec.calls) == 1

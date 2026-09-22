"""B7: step 5 (0041) -- this turn's nodes linked to each other only."""

from __future__ import annotations

from _helpers import make_manager, scripted_judge, second_fenced_span

from management.harness.backends import first_fenced_span

from management.harness.manager import HarnessManager
from management.node_classifier import Action, NodeProposal
from models.node import Node, NodeLevel


def _proposal(text: str, level: NodeLevel) -> NodeProposal:
    action = {
        NodeLevel.SUN: Action.NEW_SUN,
        NodeLevel.PLANET: Action.NEW_PLANET,
        NodeLevel.SATELLITE: Action.NEW_SATELLITE,
    }[level]
    return NodeProposal(action=action, node=Node(text=text, level=level, mass=1.0))


def _manager_belongs(pairs: set[tuple[str, str]]) -> HarnessManager:
    """Q_BELONGS answers yes only for the given (topic, statement) pairs."""

    def belongs(prompt: str) -> str:
        # Q_BELONGS fences the statement first, then the topic.
        statement = first_fenced_span(prompt).strip()
        topic = second_fenced_span(prompt).strip()
        return "yes" if (topic, statement) in pairs else "no"

    judge = scripted_judge({"belongs": belongs})
    return make_manager(judge)


def test_one_sun_two_planets_three_satellites() -> None:
    m = _manager_belongs(
        {
            ("SUN", "P1"),
            ("SUN", "P2"),
            ("P1", "S1"),
            ("P1", "S2"),
            ("P2", "S3"),
        }
    )
    proposals = [
        _proposal("SUN", NodeLevel.SUN),
        _proposal("P1", NodeLevel.PLANET),
        _proposal("P2", NodeLevel.PLANET),
        _proposal("S1", NodeLevel.SATELLITE),
        _proposal("S2", NodeLevel.SATELLITE),
        _proposal("S3", NodeLevel.SATELLITE),
    ]
    st = m.build_provisional(proposals)

    assert len(st.suns) == 1
    se = st.suns[0]
    assert se.sun.text == "SUN"
    assert [pe.planet.text for pe in se.planets] == ["P1", "P2"]
    assert [s.text for s in se.planets[0].satellites] == ["S1", "S2"]
    assert [s.text for s in se.planets[1].satellites] == ["S3"]
    assert st.orphan_planets == []
    assert st.orphan_satellites == []
    # parent linkage set by CorrelationDiagram.add_*
    assert se.planets[0].planet.parent_id == se.sun.node_id
    assert se.planets[0].satellites[0].parent_id == se.planets[0].planet.node_id


def test_orphan_planet_when_belongs_says_no() -> None:
    m = _manager_belongs({("P1", "S1")})
    proposals = [
        _proposal("SUN", NodeLevel.SUN),
        _proposal("P1", NodeLevel.PLANET),
        _proposal("S1", NodeLevel.SATELLITE),
    ]
    st = m.build_provisional(proposals)
    assert len(st.suns) == 1 and st.suns[0].planets == []
    assert len(st.orphan_planets) == 1
    orphan = st.orphan_planets[0]
    assert orphan.planet.text == "P1"
    assert orphan.planet.parent_id is None
    # the satellite still follows its (orphan) planet
    assert [s.text for s in orphan.satellites] == ["S1"]
    assert st.orphan_satellites == []


def test_orphan_satellite_when_no_planet_matches() -> None:
    m = _manager_belongs(set())
    proposals = [
        _proposal("P1", NodeLevel.PLANET),
        _proposal("S1", NodeLevel.SATELLITE),
    ]
    st = m.build_provisional(proposals)
    assert st.suns == []
    assert [pe.planet.text for pe in st.orphan_planets] == ["P1"]
    assert st.orphan_planets[0].satellites == []
    assert [n.text for n in st.orphan_satellites] == ["S1"]


def test_no_suns_means_no_belongs_call_for_planets() -> None:
    m = _manager_belongs(set())
    st = m.build_provisional([_proposal("P1", NodeLevel.PLANET)])
    assert m.runner.calls.get("belongs", 0) == 0
    assert len(st.orphan_planets) == 1


def test_satellite_only_turn_is_all_orphans() -> None:
    m = _manager_belongs(set())
    st = m.build_provisional(
        [_proposal("S1", NodeLevel.SATELLITE), _proposal("S2", NodeLevel.SATELLITE)]
    )
    assert st.suns == [] and st.orphan_planets == []
    assert [n.text for n in st.orphan_satellites] == ["S1", "S2"]
    assert m.runner.calls.get("belongs", 0) == 0

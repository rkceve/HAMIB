"""S1.2 step 5: `planet_mass_floor` on recalculate_planet_masses / normalize.

The default must be unchanged (floor 1.0) so every existing caller keeps its
behaviour; the spec run passes 0.0, which is 0062 literally.
"""

from __future__ import annotations

from communication.cd_serializer import CDSerializer
from models.correlation_diagram import CorrelationDiagram
from models.node import Node, NodeLevel


def _cd_with(n_satellites: int) -> tuple[CorrelationDiagram, Node]:
    cd = CorrelationDiagram()
    sun = Node(text="Restaurant plan", level=NodeLevel.SUN, mass=0.0)
    cd.add_sun(sun)
    planet = Node(text="Budget", level=NodeLevel.PLANET, mass=0.0)
    cd.add_planet(planet, sun.node_id)
    for i in range(n_satellites):
        cd.add_satellite(
            Node(text="detail %d" % i, level=NodeLevel.SATELLITE, mass=0.0),
            planet.node_id,
        )
    return cd, planet


def test_default_floor_is_one() -> None:
    cd, planet = _cd_with(0)
    cd.normalize()
    assert planet.mass == 1.0


def test_spec_floor_zero_gives_mass_zero() -> None:
    cd, planet = _cd_with(0)
    cd.normalize(planet_mass_floor=0.0)
    assert planet.mass == 0.0


def test_floor_never_lowers_a_real_satellite_count() -> None:
    for floor in (0.0, 1.0, None):
        cd, planet = _cd_with(4)
        cd.normalize(planet_mass_floor=floor)
        assert planet.mass == 4.0


def test_recalculate_planet_masses_takes_the_floor_directly() -> None:
    cd, planet = _cd_with(0)
    cd.recalculate_planet_masses(planet_mass_floor=0.0)
    assert planet.mass == 0.0
    cd.recalculate_planet_masses()
    assert planet.mass == 1.0


def test_sun_mass_follows_the_planet_masses() -> None:
    cd, _planet = _cd_with(0)
    cd.normalize(planet_mass_floor=0.0)
    # A sun with planets takes the planets' mass sum, which is now 0.
    assert cd.suns[0].sun.mass == 0.0
    cd.normalize()
    assert cd.suns[0].sun.mass == 1.0


def test_marker_serialization_shows_mass_on_planets_only() -> None:
    """0062 / 0079: suns and satellites carry no mass number."""
    cd, _planet = _cd_with(2)
    cd.normalize(planet_mass_floor=0.0)
    block = CDSerializer(level_markers=True).to_context_block(cd)
    assert block.splitlines() == [
        "<CONTEXT>",
        "[SN] Restaurant plan",
        "  [PN2.0] Budget",
        "    [RN] detail 0",
        "    [RN] detail 1",
        "</CONTEXT>",
    ]

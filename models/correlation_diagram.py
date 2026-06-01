"""
CorrelationDiagram: model representing the overall structure of the correlation diagram.

Structure:
  CorrelationDiagram
    └── suns: list[SunEntry]
          ├── sun: Node (level=SUN)
          └── planets: list[PlanetEntry]
                ├── planet: Node (level=PLANET)
                └── satellites: list[Node] (level=SATELLITE)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterator
import json
import copy

from models.node import Node, NodeLevel, Coordinates
from utils.config import get


@dataclass
class PlanetEntry:
    planet: Node
    satellites: list[Node] = field(default_factory=list)

    def all_nodes(self) -> Iterator[Node]:
        yield self.planet
        yield from self.satellites

    def to_dict(self) -> dict:
        return {
            "planet": self.planet.to_dict(),
            "satellites": [s.to_dict() for s in self.satellites],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlanetEntry":
        return cls(
            planet=Node.from_dict(d["planet"]),
            satellites=[Node.from_dict(s) for s in d.get("satellites", [])],
        )


@dataclass
class SunEntry:
    sun: Node
    planets: list[PlanetEntry] = field(default_factory=list)

    def all_nodes(self) -> Iterator[Node]:
        yield self.sun
        for pe in self.planets:
            yield from pe.all_nodes()

    def to_dict(self) -> dict:
        return {
            "sun": self.sun.to_dict(),
            "planets": [p.to_dict() for p in self.planets],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SunEntry":
        return cls(
            sun=Node.from_dict(d["sun"]),
            planets=[PlanetEntry.from_dict(p) for p in d.get("planets", [])],
        )


class CorrelationDiagram:
    def __init__(self):
        self.suns: list[SunEntry] = []
        self._max_suns: int = get("graph", "max_sun_nodes", 5)
        self._max_planets: int = get("graph", "max_planet_nodes_per_sun", 10)
        self._max_satellites: int = get("graph", "max_satellite_nodes_per_planet", 5)

    # ── traversal ──────────────────────────────────────────────────────

    def all_nodes(self) -> Iterator[Node]:
        for se in self.suns:
            yield from se.all_nodes()

    def find_node(self, node_id: str) -> Node | None:
        for n in self.all_nodes():
            if n.node_id == node_id:
                return n
        return None

    def find_sun_entry(self, sun_id: str) -> SunEntry | None:
        for se in self.suns:
            if se.sun.node_id == sun_id:
                return se
        return None

    def find_planet_entry(self, planet_id: str) -> tuple[SunEntry, PlanetEntry] | None:
        for se in self.suns:
            for pe in se.planets:
                if pe.planet.node_id == planet_id:
                    return se, pe
        return None

    # ── mutation ───────────────────────────────────────────────────────

    def add_sun(self, node: Node) -> bool:
        if len(self.suns) >= self._max_suns:
            return False
        node.level = NodeLevel.SUN
        node.parent_id = None
        self.suns.append(SunEntry(sun=node))
        return True

    def add_planet(self, node: Node, sun_id: str) -> bool:
        se = self.find_sun_entry(sun_id)
        if se is None or len(se.planets) >= self._max_planets:
            return False
        node.level = NodeLevel.PLANET
        node.parent_id = sun_id
        se.planets.append(PlanetEntry(planet=node))
        return True

    def add_satellite(self, node: Node, planet_id: str) -> bool:
        result = self.find_planet_entry(planet_id)
        if result is None:
            return False
        _, pe = result
        if len(pe.satellites) >= self._max_satellites:
            return False
        node.level = NodeLevel.SATELLITE
        node.parent_id = planet_id
        pe.satellites.append(node)
        return True

    # ── mass and coordinate recalculation ─────────────────────────────

    def recalculate_planet_masses(
        self,
        sun_mass_default: float | None = None,
        satellite_mass_default: float | None = None,
    ) -> None:
        """
        Recalculate the mass of every planet node as the number of satellite
        nodes attached beneath it. The minimum value is 1 (a planet exists as
        a discussion topic even with zero satellites).

        Sun and satellite masses have no explicit formula here: the sun mass is
        the sum of its child planet masses (importance as the basis of the
        discussion), and the satellite mass is a fixed value (weight as detail
        information).
        """
        sun_default = sun_mass_default if sun_mass_default is not None else get(
            "graph", "default_sun_mass", 10.0
        )
        sat_default = satellite_mass_default if satellite_mass_default is not None else get(
            "graph", "default_satellite_mass", 1.0
        )
        for se in self.suns:
            sun_mass_total = 0.0
            for pe in se.planets:
                # mass = number of child satellite nodes (minimum 1)
                pe.planet.mass = float(max(1, len(pe.satellites)))
                sun_mass_total += pe.planet.mass
                for sat in pe.satellites:
                    sat.mass = sat_default
            # sun mass is the sum of its child planet masses (reflects how much
            # discussion has accumulated); keep default_sun_mass if no planets
            if se.planets:
                se.sun.mass = sun_mass_total
            else:
                se.sun.mass = sun_default

    def recalculate_coordinates(self) -> None:
        """
        Reassign numeric coordinates to every node. A coordinate is a 3D tuple
        of (sun_idx, planet_idx, satellite_idx).
        """
        for s_idx, se in enumerate(self.suns):
            se.sun.coordinates = Coordinates(sun_idx=s_idx, planet_idx=-1, satellite_idx=-1)
            for p_idx, pe in enumerate(se.planets):
                pe.planet.coordinates = Coordinates(
                    sun_idx=s_idx, planet_idx=p_idx, satellite_idx=-1
                )
                for sat_idx, sat in enumerate(pe.satellites):
                    sat.coordinates = Coordinates(
                        sun_idx=s_idx, planet_idx=p_idx, satellite_idx=sat_idx
                    )

    def normalize(self) -> None:
        """
        Normalize the diagram. Call after a structural change to recalculate
        mass and coordinates across the whole diagram.
        """
        self.recalculate_planet_masses()
        self.recalculate_coordinates()

    # ── serialisation ──────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {"suns": [se.to_dict() for se in self.suns]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "CorrelationDiagram":
        cd = cls()
        cd.suns = [SunEntry.from_dict(s) for s in d.get("suns", [])]
        return cd

    @classmethod
    def from_json(cls, s: str) -> "CorrelationDiagram":
        return cls.from_dict(json.loads(s))

    def clone(self) -> "CorrelationDiagram":
        return CorrelationDiagram.from_dict(copy.deepcopy(self.to_dict()))

    def __len__(self) -> int:
        return sum(1 for _ in self.all_nodes())

    def __repr__(self) -> str:
        return f"<CorrelationDiagram suns={len(self.suns)} nodes={len(self)}>"

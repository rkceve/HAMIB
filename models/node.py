"""
Node data models for the Correlation Diagram.

  Node = text data + mass + coordinates
  Hierarchy:
    SunNode  → top-level concept (sun node)
      PlanetNode → intermediate concept (planet node)
        SatelliteNode → detail information (satellite node)

Mass:
  Defined as the number of satellite nodes attached beneath a single planet
  node. A natural number that directly indicates the depth of the user's
  interest. Added as a weight to the attention M matrix (scores += w * M).

Coordinates:
  Each node carries a mass and a coordinate as numeric data. This
  implementation uses a logical 3D coordinate (sun_idx, planet_idx, sat_idx)
  to express the geometric position of the node within the diagram.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import uuid


class NodeLevel(str, Enum):
    SUN = "sun"
    PLANET = "planet"
    SATELLITE = "satellite"


@dataclass
class Coordinates:
    """
    Node coordinate expressing the geometric position within the diagram.

    sun_idx: index of the owning sun node (own index when the node is a sun)
    planet_idx: index of the owning planet node (-1 for a sun, own index for a planet)
    satellite_idx: index of the satellite (-1 for a sun or planet)
    """
    sun_idx: int = -1
    planet_idx: int = -1
    satellite_idx: int = -1

    def to_dict(self) -> dict:
        return {
            "sun_idx": self.sun_idx,
            "planet_idx": self.planet_idx,
            "satellite_idx": self.satellite_idx,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Coordinates":
        return cls(
            sun_idx=d.get("sun_idx", -1),
            planet_idx=d.get("planet_idx", -1),
            satellite_idx=d.get("satellite_idx", -1),
        )

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.sun_idx, self.planet_idx, self.satellite_idx)


@dataclass
class Node:
    text: str
    level: NodeLevel
    mass: float
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_id: Optional[str] = None      # None for a SunNode
    coordinates: Coordinates = field(default_factory=Coordinates)

    def __post_init__(self):
        if self.mass < 0:
            raise ValueError("mass must be non-negative")

    def token_repr(self, precision: int = 1) -> str:
        """Return the [PN{mass}] token representation."""
        return f"[PN{round(self.mass, precision)}] {self.text}"

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "text": self.text,
            "level": self.level.value,
            "mass": self.mass,
            "parent_id": self.parent_id,
            "coordinates": self.coordinates.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        return cls(
            text=d["text"],
            level=NodeLevel(d["level"]),
            mass=d["mass"],
            node_id=d["node_id"],
            parent_id=d.get("parent_id"),
            coordinates=Coordinates.from_dict(d.get("coordinates", {})),
        )

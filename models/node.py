"""Node data model for the correlation diagram.

A node is a piece of text plus a mass and coordinates. Nodes form a
three-level hierarchy:
  sun        top-level topic
    planet     sub-topic
      satellite  detail

Mass: for a planet, the number of satellites beneath it, i.e. how deeply the
user has gone into that sub-topic. The server adds it to the attention scores
(scores += w * M).

Coordinates: the node's (sun_idx, planet_idx, satellite_idx) position in the
diagram.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import uuid


class NodeLevel(str, Enum):
    SUN = "sun"
    PLANET = "planet"
    SATELLITE = "satellite"


@dataclass
class Coordinates:
    """Position of a node in the diagram; -1 means "does not apply".

    sun_idx: index of the node's sun (a sun's own index)
    planet_idx: index of the node's planet (-1 for a sun)
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
    # First 12 characters of a UUID (11 hex digits and a hyphen). The old
    # 8-character ids had a 0.24% chance of a collision at ~4.5k nodes per run.
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    parent_id: str | None = None  # None for a sun
    coordinates: Coordinates = field(default_factory=Coordinates)
    # Turn of the chunk the node was created from (-1 = unknown). The serializer's
    # "recency" policy and its tie-breaks use it.
    created_turn: int = -1

    def __post_init__(self):
        if self.mass < 0:
            raise ValueError("mass must be non-negative")

    def token_repr(self, precision: int = 1) -> str:
        """Return ``"[PN{mass}] {text}"``."""
        return f"[PN{round(self.mass, precision)}] {self.text}"

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "text": self.text,
            "level": self.level.value,
            "mass": self.mass,
            "parent_id": self.parent_id,
            "coordinates": self.coordinates.to_dict(),
            "created_turn": self.created_turn,
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
            created_turn=d.get("created_turn", -1),
        )

"""
Node data models for the Correlation Diagram.

特許 第1実施形態 §0030-§0035 準拠:
  Node = テキストデータ + 質量(mass) + 座標(coordinates)
  Hierarchy:
    SunNode  → 最上位概念（太陽ノード／§0031）
      PlanetNode → 中間概念（惑星ノード／§0031）
        SatelliteNode → 詳細情報（衛星ノード／§0031）

質量 (§0062, §0079, §0083):
  「一つの惑星ノードの下に連なる衛星ノードの数」として定義。
  ユーザーの関心の深さを直接的に示す自然数。
  Attention Mマトリクスへ加算される重み（§0082: scores += w * M）。

座標 (§0030):
  各ノードは「数値データとしてのノードの質量と座標」を含む。
  本実装では論理的な (sun_idx, planet_idx, sat_idx) 3次元座標を採用し、
  相関図内におけるノードの幾何学的位置を表現する。
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
    特許§0030 ノード座標。
    相関図内における幾何学的位置を表現する。

    sun_idx: 所属する太陽ノードのインデックス（自身が太陽の場合は自インデックス）
    planet_idx: 所属する惑星ノードのインデックス（太陽は -1、惑星は自身）
    satellite_idx: 衛星のインデックス（太陽・惑星は -1）
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
    # H17: 12 hex chars, not 8.  With 4.5k nodes in one run the birthday
    # collision probability drops from 0.24% (8 chars) to ~1e-5 (12 chars);
    # nothing in the codebase depends on the id length.
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    parent_id: Optional[str] = None      # SunNode なら None
    coordinates: Coordinates = field(default_factory=Coordinates)
    # -1 = 未設定。ノード生成時の chunk.turn を刻印し、recency ポリシーの
    # eviction 判定・同順位時の tie-break に用いる（後方互換のため末尾に追加）。
    created_turn: int = -1

    def __post_init__(self):
        if self.mass < 0:
            raise ValueError("mass must be non-negative")

    def token_repr(self, precision: int = 1) -> str:
        """案C: [PN{mass}] トークン表現を返す。"""
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

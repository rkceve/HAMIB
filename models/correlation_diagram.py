"""
CorrelationDiagram: 相関図データの全体構造を表すモデル。

構造:
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

    def capacities(self) -> dict[str, int]:
        """The effective insertion limits (config.yaml ``graph.*``), for manifests.

        ``add_sun`` / ``add_planet`` / ``add_satellite`` return False at these
        limits; callers that must not lose a node check the result
        (management.graph_merger raises).
        """
        return {
            "max_sun_nodes": self._max_suns,
            "max_planet_nodes_per_sun": self._max_planets,
            "max_satellite_nodes_per_planet": self._max_satellites,
        }

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

    # ── 特許準拠: 質量再計算 (§0062) と座標再計算 (§0030) ─────────────

    def recalculate_planet_masses(
        self,
        sun_mass_default: float | None = None,
        satellite_mass_default: float | None = None,
        planet_mass_floor: float | None = None,
    ) -> None:
        """
        特許§0062 準拠: 「質量は『一つの惑星ノードの下に連なる衛星ノードの数』として定義」

        本メソッドは相関図内の全惑星ノードの質量を、その下に連なる衛星ノードの数に
        再設定する。下限は planet_mass_floor（既定 1.0: 衛星0個でも惑星自体は
        議論の柱として存在するという従来の解釈）。

        planet_mass_floor=0.0 は SPEC_FAITHFUL_DESIGN.md S1.2 step 5 の
        「§0062 の字義どおり」設定で、衛星0個の惑星は質量0になる。
        既定値は config.yaml graph.planet_mass_floor（未設定なら 1.0）なので、
        呼び出し側が指定しない限り従来の挙動は変わらない。

        サン・衛星ノードの質量については特許に明示的な計算式がないため、
        サンは「下位惑星の質量総和」（議論の根底としての重要度）、
        衛星は固定値（詳細情報としての重み）とする。
        level_markers 直列化ではサン・衛星の質量は出力されない（§0062/§0079）。
        """
        floor = (
            planet_mass_floor
            if planet_mass_floor is not None
            else float(get("graph", "planet_mass_floor", 1.0))
        )
        sun_default = sun_mass_default if sun_mass_default is not None else get(
            "graph", "default_sun_mass", 10.0
        )
        sat_default = satellite_mass_default if satellite_mass_default is not None else get(
            "graph", "default_satellite_mass", 1.0
        )
        for se in self.suns:
            sun_mass_total = 0.0
            for pe in se.planets:
                # §0062: 質量 = 下位衛星ノード数（下限 floor）
                pe.planet.mass = float(max(floor, float(len(pe.satellites))))
                sun_mass_total += pe.planet.mass
                for sat in pe.satellites:
                    sat.mass = sat_default
            # サンノードの質量は下位惑星の総合計（議論の蓄積量を反映）
            # 惑星が0個の場合は default_sun_mass を維持
            if se.planets:
                se.sun.mass = sun_mass_total
            else:
                se.sun.mass = sun_default

    def recalculate_coordinates(self) -> None:
        """
        特許§0030 準拠: 各ノードに数値データとしての座標を再付与する。
        座標は (sun_idx, planet_idx, satellite_idx) の3次元タプルで表現される。
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

    def normalize(self, planet_mass_floor: float | None = None) -> None:
        """
        特許§0030, §0062 準拠の「正規化」。
        相関図の構造変更後に呼び出し、質量と座標を全体に再計算する。

        planet_mass_floor は惑星質量の下限。None（既定）なら
        recalculate_planet_masses の既定（config.yaml graph.planet_mass_floor、
        未設定なら 1.0）を使うので、従来の呼び出しは挙動が変わらない。
        """
        self.recalculate_planet_masses(planet_mass_floor=planet_mass_floor)
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

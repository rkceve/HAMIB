"""
GraphMerger: 仮構造（incoming CD）を既存の相関図データ（base CD）に統合する。

特許§0041-§0061 準拠の詳細統合ロジック:

  ケース1 (§0043-§0054): 仮構造に太陽ノードが含まれる場合
    - 太陽B が既存太陽A と類似 → B消滅、A残存。配下を A に統合
      - 1) B の下に惑星のみ
      - 2) B の下に惑星と衛星
      - 3) B の下に衛星のみ → 衛星を昇格判定し統合
    - 太陽B が独立（下位なし） → 既存太陽との類似度判定
    - 太陽B が既存いずれにも非類似 → 新規太陽として追加

  ケース2 (§0055-§0058): 仮構造に太陽なし、惑星のみ/惑星+衛星
    - 惑星A が既存惑星B と類似 → A消滅、衛星のみ A→B 配下に追加
    - 惑星A が非類似 → 全太陽との類似度判定
      - 類似太陽あり → A をその下に新規惑星として追加
      - 類似太陽なし → A を太陽に昇格

  ケース3 (§0059-§0061): 仮構造に衛星のみ
    - 衛星 vs 既存衛星 → 類似なら消滅
    - 衛星 vs 既存惑星 → 類似なら下に追加
    - 衛星 vs 既存太陽 → 類似なら惑星に昇格して追加
    - 全て非類似 → 太陽に昇格して追加

統合完了後、§0062 に従い質量を「衛星ノード総数」で再計算する。
"""
from __future__ import annotations
import copy
from typing import Callable, Optional

from models.correlation_diagram import CorrelationDiagram, SunEntry, PlanetEntry
from models.node import Node, NodeLevel
from utils.similarity import most_similar_index
from utils.config import get

# 類似度判定関数の型: (query, candidates) -> (index, score)
SimilarityFn = Callable[[str, list[str]], tuple[int, float]]


def _require(inserted: bool, node: Node, parent_id: str | None) -> None:
    """Every insertion into the base diagram must succeed (mcbuild_bench Astra
    round 3 item 2): ``add_sun`` / ``add_planet`` / ``add_satellite`` return
    False at the configured capacity (config.yaml ``graph.max_*``) or for a
    missing parent, and a node must never be dropped silently."""
    if not inserted:
        raise RuntimeError(
            "capacity exceeded: could not add %s %r under %s (parent missing or at "
            "its configured maximum, see config.yaml graph.max_*)"
            % (node.level.value, node.text[:60], parent_id)
        )


def _default_similarity_fn(text: str, candidates: list[str]) -> tuple[int, float]:
    """既定の類似度判定（埋め込みコサイン）。

    モジュール属性 ``most_similar_index`` を呼び出し時に解決するため、
    テストからの monkeypatch が構築順序に依存しない。
    """
    return most_similar_index(text, candidates)


class GraphMerger:
    def __init__(
        self,
        similarity_fn: Optional[SimilarityFn] = None,
        attach_fn: Optional[SimilarityFn] = None,
    ):
        """similarity_fn: 特許§0042 準拠の 1対1 判定を差し替えるためのフック。
        None（既定）なら従来どおり utils.similarity.most_similar_index を使う。
        harness は SimilarityJudge.most_similar を渡す（後方互換の追加引数）。
        """
        self._sim_threshold: float = get("management", "similarity_threshold", 0.75)
        self._similarity_fn: SimilarityFn = (
            similarity_fn if similarity_fn is not None else _default_similarity_fn
        )
        # attach_fn: 「同じ事柄か（消滅判定）」ではなく「その上位ノードに属するか
        # （連結判定）」を問う場面 (§0057 惑星→太陽, §0060 衛星→惑星, §0061 衛星→太陽)
        # で使う判定関数。None なら similarity_fn と同じ（埋め込み経路の従来動作）。
        self._attach_fn: SimilarityFn = (
            attach_fn if attach_fn is not None else self._similarity_fn
        )

    # ── エントリポイント ───────────────────────────────────────────────

    def merge(
        self, base: CorrelationDiagram, incoming: CorrelationDiagram
    ) -> CorrelationDiagram:
        """
        incoming の各サブツリーを 3 ケース別に分類して base に統合する。

        incoming.suns に太陽がある場合 → ケース1 (§0043-§0054)
        incoming に太陽がない場合 → ケース2/3 ユーティリティを呼ぶ
          - 遊離惑星 → ケース2 (§0055-§0058)
          - 遊離衛星 → ケース3 (§0059-§0061)

        統合後、§0062 質量再計算 + §0030 座標再計算 を実行。
        """
        # ケース1: 太陽を含む仮構造
        for se_in in list(incoming.suns):
            self._merge_case1(base, se_in)

        # ケース2/3 は merge_case2_planet / merge_case3_satellite として外部公開する
        # NodeClassifier の現実装では孤児惑星/衛星はサンに昇格されてケース1 に流れるため、
        # ここでは incoming.suns 以外を直接トラバースする経路は不要。

        # §0062 質量再計算 + §0030 座標再計算
        base.normalize()
        return base

    # ── ケース1: 仮構造に太陽ノードが含まれる場合 (§0043-§0054) ────────

    def _merge_case1(self, base: CorrelationDiagram, se_in: SunEntry) -> None:
        """
        新規仮太陽ノード B(=se_in.sun) が含まれる場合の統合。
        """
        sun_b = se_in.sun
        match_idx = self._find_matching_sun_idx(sun_b.text, base)

        if match_idx is None:
            # §0054: B がどの A とも非類似 → B を新太陽として追加し、配下も丸ごと持ち込む
            self._add_full_subtree_as_new_sun(base, se_in)
            return

        # §0045: B が既存太陽 A に類似 → B 消滅、A 残存。B の配下を A 側に統合
        sun_a_id = base.suns[match_idx].sun.node_id

        if not se_in.planets:
            # §0053: B が独立（下位なし） → 何もせず終了（A は変化なし）
            return

        # B 配下に惑星があり、衛星もぶら下がっている可能性がある
        for pe_in in se_in.planets:
            self._merge_planet_into_sun(base, sun_a_id, pe_in)

    def _add_full_subtree_as_new_sun(
        self, base: CorrelationDiagram, se_in: SunEntry
    ) -> None:
        """B が新太陽として追加される場合、配下の惑星・衛星も同じ形で追加する。"""
        new_sun = copy.deepcopy(se_in.sun)
        _require(base.add_sun(new_sun), new_sun, None)
        new_sun_id = new_sun.node_id
        for pe_in in se_in.planets:
            new_planet = copy.deepcopy(pe_in.planet)
            _require(base.add_planet(new_planet, new_sun_id), new_planet, new_sun_id)
            for sat_in in pe_in.satellites:
                new_sat = copy.deepcopy(sat_in)
                _require(base.add_satellite(new_sat, new_planet.node_id), new_sat, new_planet.node_id)

    def _merge_planet_into_sun(
        self, base: CorrelationDiagram, sun_a_id: str, pe_in: PlanetEntry
    ) -> None:
        """
        §0046-§0049: 惑星 C(=pe_in.planet) を既存太陽A の下に統合する。

        - C が既存惑星 D に類似 → C 消滅、衛星は D 配下に追加（類似衛星があれば消滅）
        - C が非類似 → A 配下に新規惑星として追加し、配下衛星も持ち込む
        """
        planet_c = pe_in.planet
        match_planet_id = self._find_matching_planet_id(planet_c.text, sun_a_id, base)

        if match_planet_id is not None:
            # §0046: C が D に類似 → C 消滅、D 残存。配下衛星のみ D 側へ
            for sat_in in pe_in.satellites:
                self._merge_satellite_under_planet(base, match_planet_id, sat_in)
            return

        # §0049: C が非類似 → A 配下に新規惑星として追加
        new_planet = copy.deepcopy(planet_c)
        _require(base.add_planet(new_planet, sun_a_id), new_planet, sun_a_id)
        new_planet_id = new_planet.node_id
        for sat_in in pe_in.satellites:
            new_sat = copy.deepcopy(sat_in)
            _require(base.add_satellite(new_sat, new_planet_id), new_sat, new_planet_id)

    def _merge_satellite_under_planet(
        self, base: CorrelationDiagram, planet_id: str, satellite: Node
    ) -> None:
        """
        §0048: 衛星 E が既存惑星 D 配下の衛星 F と類似なら消滅、非類似なら追加。
        """
        result = base.find_planet_entry(planet_id)
        if result is None:
            return
        _, pe = result
        existing_sat_texts = [s.text for s in pe.satellites]
        if existing_sat_texts:
            idx, score = self._similarity_fn(satellite.text, existing_sat_texts)
            if score >= self._sim_threshold:
                # §0048: E が F と類似 → E 消滅、F 残存（何もしない）
                return
        # §0049: E は非類似 → 新規衛星として D 配下に追加
        new_sat = copy.deepcopy(satellite)
        _require(base.add_satellite(new_sat, planet_id), new_sat, planet_id)

    # ── ケース2: 太陽なし、惑星のみ/惑星+衛星 (§0055-§0058) ────────────

    def merge_case2_planet(
        self, base: CorrelationDiagram, planet_a: Node, satellites: list[Node]
    ) -> None:
        """
        §0055-§0058: 仮構造に太陽がなく、独立惑星 A（と配下衛星）のみがある場合。
        """
        # §0056: A が既存惑星 B と類似 → A 消滅、衛星のみ B 配下へ
        match_planet = self._find_any_matching_planet(planet_a.text, base)
        if match_planet is not None:
            sun_idx, planet_idx = match_planet
            target_planet_id = base.suns[sun_idx].planets[planet_idx].planet.node_id
            for sat in satellites:
                self._merge_satellite_under_planet(base, target_planet_id, sat)
            return

        # §0057: A が非類似 → 全太陽との類似度判定（連結判定: attach_fn）
        match_sun_idx = self._find_matching_sun_idx(planet_a.text, base, attach=True)
        if match_sun_idx is not None:
            # 類似太陽あり → 新規惑星として追加
            target_sun_id = base.suns[match_sun_idx].sun.node_id
            new_planet = copy.deepcopy(planet_a)
            _require(base.add_planet(new_planet, target_sun_id), new_planet, target_sun_id)
            for sat in satellites:
                new_sat = copy.deepcopy(sat)
                _require(base.add_satellite(new_sat, new_planet.node_id), new_sat, new_planet.node_id)
            return

        # §0058: 類似太陽なし → A を太陽に昇格、衛星は惑星に昇格
        promoted_sun = copy.deepcopy(planet_a)
        promoted_sun.level = NodeLevel.SUN
        _require(base.add_sun(promoted_sun), promoted_sun, None)
        for sat in satellites:
            promoted_planet = copy.deepcopy(sat)
            promoted_planet.level = NodeLevel.PLANET
            _require(base.add_planet(promoted_planet, promoted_sun.node_id), promoted_planet, promoted_sun.node_id)

    # ── ケース3: 衛星のみ (§0059-§0061) ──────────────────────────────

    def merge_case3_satellite(
        self, base: CorrelationDiagram, satellite: Node
    ) -> None:
        """
        §0059-§0061: 仮構造に衛星ノードのみが含まれる場合。
        """
        # §0059: 既存全衛星との類似度判定
        all_satellites: list[tuple[str, Node]] = []
        for se in base.suns:
            for pe in se.planets:
                for sat in pe.satellites:
                    all_satellites.append((sat.text, sat))
        if all_satellites:
            texts = [t for t, _ in all_satellites]
            idx, score = self._similarity_fn(satellite.text, texts)
            if score >= self._sim_threshold:
                return  # 類似衛星あり → 消滅

        # §0060: 既存惑星との類似度判定（連結判定: attach_fn）
        match_planet = self._find_any_matching_planet(satellite.text, base, attach=True)
        if match_planet is not None:
            sun_idx, planet_idx = match_planet
            target_planet_id = base.suns[sun_idx].planets[planet_idx].planet.node_id
            new_sat = copy.deepcopy(satellite)
            _require(base.add_satellite(new_sat, target_planet_id), new_sat, target_planet_id)
            return

        # §0061: 既存太陽との類似度判定（衛星を惑星に昇格、連結判定: attach_fn）
        match_sun_idx = self._find_matching_sun_idx(satellite.text, base, attach=True)
        if match_sun_idx is not None:
            promoted_planet = copy.deepcopy(satellite)
            promoted_planet.level = NodeLevel.PLANET
            target_sun_id = base.suns[match_sun_idx].sun.node_id
            _require(base.add_planet(promoted_planet, target_sun_id), promoted_planet, target_sun_id)
            return

        # 全て非類似 → 太陽に昇格
        promoted_sun = copy.deepcopy(satellite)
        promoted_sun.level = NodeLevel.SUN
        _require(base.add_sun(promoted_sun), promoted_sun, None)

    # ── 類似度判定ヘルパー ─────────────────────────────────────────────

    def _find_matching_sun_idx(
        self, text: str, cd: CorrelationDiagram, *, attach: bool = False
    ) -> int | None:
        sun_texts = [se.sun.text for se in cd.suns]
        if not sun_texts:
            return None
        fn = self._attach_fn if attach else self._similarity_fn
        idx, score = fn(text, sun_texts)
        if score >= self._sim_threshold:
            return idx
        return None

    def _find_matching_planet_id(
        self, text: str, sun_id: str, cd: CorrelationDiagram
    ) -> str | None:
        se = cd.find_sun_entry(sun_id)
        if se is None or not se.planets:
            return None
        planet_texts = [pe.planet.text for pe in se.planets]
        idx, score = self._similarity_fn(text, planet_texts)
        if score >= self._sim_threshold:
            return se.planets[idx].planet.node_id
        return None

    def _find_any_matching_planet(
        self, text: str, cd: CorrelationDiagram, *, attach: bool = False
    ) -> tuple[int, int] | None:
        """全太陽配下の全惑星から最も類似する惑星の (sun_idx, planet_idx) を返す。

        attach=True のときは連結判定 (attach_fn) を使う (§0060)。"""
        all_planet_texts: list[str] = []
        index_map: list[tuple[int, int]] = []
        for s_idx, se in enumerate(cd.suns):
            for p_idx, pe in enumerate(se.planets):
                all_planet_texts.append(pe.planet.text)
                index_map.append((s_idx, p_idx))
        if not all_planet_texts:
            return None
        fn = self._attach_fn if attach else self._similarity_fn
        idx, score = fn(text, all_planet_texts)
        if score >= self._sim_threshold:
            return index_map[idx]
        return None

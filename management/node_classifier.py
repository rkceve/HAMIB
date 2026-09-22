"""
NodeClassifier: テキストチャンクからノード候補を生成する。

特許§0039-§0040 準拠:
  各チャンクを以下の3項目で100点満点でスコアリング:
    - 包括性 (comprehensiveness): sun ノードに対応
    - 独立性 (independence):       planet ノードに対応
    - 詳細度 (detail):             satellite ノードに対応
  最高得点の項目に対応するノードレベルに分類する。

特許§0042 準拠:
  ノード同士の類似度判定は学習済みLLMで1対1で実行。
  本実装では SentenceTransformer 埋め込み（all-MiniLM-L6-v2）を
  デフォルトとし、LLM ベース類似度判定もオプションとして提供する。
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

from models.node import Node, NodeLevel
from models.correlation_diagram import CorrelationDiagram
from management.text_chunker import Chunk
from utils.config import get
from utils.similarity import most_similar_index


class Action(str, Enum):
    NEW_SUN = "new_sun"
    NEW_PLANET = "new_planet"
    NEW_SATELLITE = "new_satellite"
    UPDATE_MASS = "update_mass"
    SKIP = "skip"


@dataclass
class NodeProposal:
    action: Action
    node: Node
    parent_id: str | None = None
    # 特許§0039: 3項目スコアを保持（後段の評価部2やデバッグで利用）
    score_comprehensiveness: float = 0.0
    score_independence: float = 0.0
    score_detail: float = 0.0


class NodeClassifier:
    def __init__(self):
        self._sim_threshold: float = get("management", "similarity_threshold", 0.75)
        self._default_sun_mass: float = get("graph", "default_sun_mass", 10.0)
        self._default_planet_mass: float = get("graph", "default_planet_mass", 5.0)
        self._default_satellite_mass: float = get("graph", "default_satellite_mass", 1.0)

    def classify(
        self, chunk: Chunk, cd: CorrelationDiagram, llm_extract_fn,
        builder=None,
    ) -> list[NodeProposal]:
        """
        chunk のテキストから概念を抽出し、NodeProposal のリストを返す。
        llm_extract_fn(text) -> list[dict] は、特許§0039 準拠で
            [{"text": "...",
              "level": "sun"|"planet"|"satellite",
              "score_comprehensiveness": 0-100,
              "score_independence": 0-100,
              "score_detail": 0-100,
              "parent_hint": "...(parent text, optional)"}]
        を返す関数（server 側で実装）。

        旧フォーマット（スコアなし）も後方互換で受け付ける。

        §94.2 (2026-05-20) fix: ``builder`` 引数を渡すと、 各 item の proposal
        を生成直後に provisional CD へ <strong>incremental に apply</strong> する。
        これにより、 同一 chunk 内で extractor が返す entity と
        satellite(parent_hint=entity) のような親子関係を保てる。

        旧挙動 (builder=None): 全 proposal をリストに溜めて呼出側で一括 apply。
            問題: 後続 item から見た provisional CD に先行 item が居ないため、
            satellite が parent_hint で entity を探しても見つからず、
            最終 fallback で NEW_SUN に強制昇格 → 親子関係喪失。

        新挙動 (builder=GraphBuilder): item ごとに classify→apply を回す。
        """
        raw = llm_extract_fn(chunk.text)
        proposals: list[NodeProposal] = []

        for item in raw:
            text = item.get("text", "").strip()
            if not text:
                continue

            score_c = float(item.get("score_comprehensiveness", 0))
            score_i = float(item.get("score_independence", 0))
            score_d = float(item.get("score_detail", 0))

            # 特許§0040: スコアが一番高かった項目に対応するノードに分類
            # ただしスコアが全て0（旧フォーマット）の場合は item["level"] を使用
            if score_c == 0 and score_i == 0 and score_d == 0:
                level_str = item.get("level", "satellite")
                try:
                    level = NodeLevel(level_str)
                except ValueError:
                    level = NodeLevel.SATELLITE
            else:
                level = self._level_from_scores(score_c, score_i, score_d)

            parent_hint = item.get("parent_hint", "")

            item_proposals: list[NodeProposal] = []
            for p in self._build_proposals(text, level, parent_hint, cd, chunk.turn):
                p.score_comprehensiveness = score_c
                p.score_independence = score_i
                p.score_detail = score_d
                item_proposals.append(p)
                proposals.append(p)

            # §94.2 fix: apply each item's proposals immediately so the next
            # item in the same chunk can see them in `cd`. Without this,
            # extractor-supplied parent_hint chains within the same chunk
            # always fall through to NEW_SUN promotion (parent never found).
            if builder is not None and item_proposals:
                builder.apply(cd, item_proposals)

        return proposals

    @staticmethod
    def _level_from_scores(
        score_c: float, score_i: float, score_d: float
    ) -> NodeLevel:
        """特許§0040: スコアが一番高かった項目に対応するノードを返す。"""
        scores = [
            (score_c, NodeLevel.SUN),
            (score_i, NodeLevel.PLANET),
            (score_d, NodeLevel.SATELLITE),
        ]
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[0][1]

    def _build_proposals(
        self,
        text: str,
        level: NodeLevel,
        parent_hint: str,
        cd: CorrelationDiagram,
        created_turn: int = -1,
    ) -> list[NodeProposal]:
        existing_texts = [n.text for n in cd.all_nodes()]

        if existing_texts:
            best_idx, score = most_similar_index(text, existing_texts)
            if score >= self._sim_threshold:
                matched = list(cd.all_nodes())[best_idx]
                # 類似ノードが既にある → mass を増加させる提案
                # 注: 質量は GraphMerger の最終正規化で衛星数ベースに再計算される
                updated = Node(
                    text=matched.text,
                    level=matched.level,
                    mass=matched.mass + self._mass_for(matched.level),
                    node_id=matched.node_id,
                    parent_id=matched.parent_id,
                )
                return [NodeProposal(action=Action.UPDATE_MASS, node=updated)]

        # 新規ノード候補
        # created_turn は生成時の chunk.turn を刻印する (recency ポリシー用)。
        # 以降のレベル昇格 (§76 fallback) は同一 Node を mutate するだけなので
        # created_turn はそのまま引き継がれ、ノード誕生時に一度だけ刻印される。
        # UPDATE_MASS 経路は既存ノードの .mass のみを更新し created_turn は触らない。
        mass = self._mass_for(level)
        new_node = Node(text=text, level=level, mass=mass, created_turn=created_turn)

        if level == NodeLevel.SUN:
            return [NodeProposal(action=Action.NEW_SUN, node=new_node)]

        # parent_hint から親ノードを探す
        parent_id = self._find_parent(parent_hint, level, cd)
        if parent_id is None:
            # §76 (2026-05-12): 旧版は強制 SUN 昇格していたが、 これが原因で全 fact が
            # SUN になり CD cap (旧 5) に当たって fact 5+ が drop していた。
            # 修正: 親不在時は <strong>同レベルで親なし保存</strong>を試みる。
            # PLANET なら直近 SUN を仮親、 SATELLITE なら直近 PLANET を仮親に設定。
            # 仮親も無ければ NEW_SUN に fallback (旧動作維持) — ただし上限解除済なので drop しない。
            if level == NodeLevel.PLANET and cd.suns:
                # 直近の SUN を仮親に
                parent_id = cd.suns[-1].sun.node_id
            elif level == NodeLevel.SATELLITE:
                # PLANET ノードがあるならその直近を仮親に
                all_planets = [pe.planet for se in cd.suns for pe in se.planets]
                if all_planets:
                    parent_id = all_planets[-1].node_id
                elif cd.suns:
                    # PLANET が無ければ SUN 直下に PLANET として保存
                    new_node.level = NodeLevel.PLANET
                    new_node.mass = self._mass_for(NodeLevel.PLANET)
                    parent_id = cd.suns[-1].sun.node_id
            if parent_id is None:
                # 真に何も無い (CD 空) → NEW_SUN (起動初期のみ)
                new_node.level = NodeLevel.SUN
                new_node.mass = self._default_sun_mass
                return [NodeProposal(action=Action.NEW_SUN, node=new_node)]

        action = Action.NEW_PLANET if new_node.level == NodeLevel.PLANET else Action.NEW_SATELLITE
        return [NodeProposal(action=action, node=new_node, parent_id=parent_id)]

    def _find_parent(
        self, hint: str, level: NodeLevel, cd: CorrelationDiagram
    ) -> str | None:
        if level == NodeLevel.PLANET:
            candidates = [se.sun for se in cd.suns]
        elif level == NodeLevel.SATELLITE:
            candidates = [pe.planet for se in cd.suns for pe in se.planets]
        else:
            return None

        if not candidates:
            return None

        if not hint:
            # ヒントなし → 最初の候補
            return candidates[0].node_id

        texts = [c.text for c in candidates]
        idx, score = most_similar_index(hint, texts)
        if score >= self._sim_threshold:
            return candidates[idx].node_id
        return candidates[0].node_id

    def _mass_for(self, level: NodeLevel) -> float:
        if level == NodeLevel.SUN:
            return self._default_sun_mass
        elif level == NodeLevel.PLANET:
            return self._default_planet_mass
        return self._default_satellite_mass

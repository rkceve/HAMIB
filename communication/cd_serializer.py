"""
CDSerializer: CorrelationDiagram をサーバー向けのペイロードに変換する。

「案C: [PN{mass}] 特殊トークン」方式:
  相関図の各ノードを [PN{mass}] プレフィックス付きのテキストとして
  プロンプト先頭に埋め込む。サーバー側はそのトークン位置を検出して
  Mマトリクスを構築する。

フォーマット（プロンプト先頭に挿入するブロック）:
  <CONTEXT>
  [PN10.0] 機械学習
    [PN5.0] ニューラルネットワーク
      [PN1.0] バックプロパゲーション
  ...
  </CONTEXT>
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Callable, Literal

from models.correlation_diagram import CorrelationDiagram
from models.node import Node
from utils.config import get


@dataclass
class _NodeRecord:
    """予算付きシリアライズ用の走査レコード（挿入順・親・インデント付き）。"""
    node: Node
    parent_id: str | None
    indent: int
    order_index: int


class CDSerializer:
    def __init__(self, level_markers: bool | None = None):
        self._prefix: str = get("tokenization", "prefix", "[PN")
        self._suffix: str = get("tokenization", "suffix", "]")
        self._precision: int = get("tokenization", "mass_precision", 1)
        # Spec-faithful format (2026-09-07): mass is defined for PLANET nodes
        # only (spec 0062/0079). With level_markers=True the lines are
        #   [SN] sun text / [PN{mass}] planet text / [RN] satellite text
        # (SN/PN/RN = the labels of spec Fig. 2), so suns and satellites carry
        # NO mass number. False keeps the legacy "[PN{mass}] on every line".
        self._level_markers: bool = (
            bool(get("tokenization", "level_markers", False))
            if level_markers is None
            else level_markers
        )

    def to_context_block(self, cd: CorrelationDiagram) -> str:
        """
        CDを <CONTEXT>...</CONTEXT> ブロック文字列に変換する。
        プロンプト先頭に付加して送信する。
        """
        lines = ["<CONTEXT>"]
        for se in cd.suns:
            lines.append(self._node_line(se.sun.text, se.sun.mass, indent=0))
            for pe in se.planets:
                lines.append(self._node_line(pe.planet.text, pe.planet.mass, indent=1))
                for sat in pe.satellites:
                    lines.append(self._node_line(sat.text, sat.mass, indent=2))
        lines.append("</CONTEXT>")
        return "\n".join(lines)

    def _node_line(self, text: str, mass: float, indent: int) -> str:
        if self._level_markers:
            if indent == 0:
                token = "[SN]"
            elif indent == 1:
                token = f"[PN{round(mass, self._precision)}]"
            else:
                token = "[RN]"
        else:
            token = f"{self._prefix}{round(mass, self._precision)}{self._suffix}"
        return "  " * indent + f"{token} {text}"

    # ── 予算付きシリアライズ (WO-1) ────────────────────────────────────

    def to_context_block_budgeted(
        self,
        cd: CorrelationDiagram,
        budget_tokens: int,
        policy: Literal["mass", "random", "recency"],
        token_counter: Callable[[str], int],
        seed: int = 0,
    ) -> str:
        """予算 budget_tokens 内に収まるよう eviction をかけたコンテキストブロック。

        3つの eviction ポリシー:
          - "mass":   質量の降順で残す。同順位は created_turn が新しい方、
                      さらに同じなら相関図の挿入順を優先する tie-break。
          - "recency": created_turn の降順で残す（新しいノードを優先）。
                      同順位は挿入順。
          - "random": random.Random(seed) による一様サンプリング。

        祖先不変条件 (ancestor invariant): あるノードを残せるのは、
        その親ノード（さらにその親…）が全て残っている場合のみ。
        太陽ノードは親を持たないため常に候補になる。

        貪欲規則 (greedy rule):
          1. 候補をポリシー順に並べる（"random" は毎回、残り候補から一様抽選）。
          2. 各候補について、その親が既に kept なら追加を試みる。
          3. 追加後のブロック全体（<CONTEXT>/</CONTEXT> ラッパー行と
             インデントを含む）の token_counter によるトークン数が
             budget_tokens を<strong>超える</strong>なら、そのノードは<strong>飛ばして</strong>
             （skip）残りの候補を試し続ける（より小さいノードは入り得る）。
          4. あるノードを kept にすると、その子ノードが新たに候補になり得る。
             また budget 超過で飛ばしたノードは恒久的に除外はせず、
             （親が後から kept になった等で）状況が変われば再度候補になる。
          5. 1 パスで 1 件も追加できなくなったら終了（不動点）。

        予算 = 完成したブロック文字列全体のトークン数
        （token_counter(text)）。<CONTEXT> ラッパー行とインデントも数える。

        出力順序: kept 集合が確定した後、元の to_context_block と同一の
        走査順（sun→planet→satellite、挿入順）で emit する。したがって
        予算が十分で全ノードが残るケースでは、どのポリシーでも
        to_context_block(cd) とバイト単位で一致する。

        policy が {"mass","random","recency"} 以外なら ValueError。
        budget_tokens がラッパー行 (<CONTEXT> + </CONTEXT>) だけのトークン数を
        下回る場合も ValueError（予算超過ブロックを返さない。mcbuild_bench
        Astra round 3 item 7）。
        """
        if policy not in ("mass", "random", "recency"):
            raise ValueError(
                f"policy must be one of 'mass','random','recency', got {policy!r}"
            )
        wrapper_tokens = token_counter(self._emit_kept([], set()))
        if budget_tokens < wrapper_tokens:
            raise ValueError(
                f"budget_tokens={budget_tokens} is smaller than the wrapper cost of "
                f"{wrapper_tokens} tokens (<CONTEXT> + </CONTEXT> lines): no block "
                "within the budget exists"
            )

        # 全ノードを挿入順（= to_context_block の走査順）で列挙し、
        # 各ノードに (node, parent_id, indent, insertion_index) を付与する。
        records: list[_NodeRecord] = []
        order_index = 0
        for se in cd.suns:
            records.append(_NodeRecord(se.sun, None, 0, order_index))
            order_index += 1
            for pe in se.planets:
                records.append(_NodeRecord(pe.planet, se.sun.node_id, 1, order_index))
                order_index += 1
                for sat in pe.satellites:
                    records.append(_NodeRecord(sat, pe.planet.node_id, 2, order_index))
                    order_index += 1

        kept_ids: set[str] = set()
        rejected_ids: set[str] = set()  # budget 超過で今回飛ばしたもの

        # Spec-faithful mode (level_markers): the ONLY mass that exists is the
        # planet mass (spec 0062/0079). The eviction priority must therefore
        # not read the internal sun Σ-mass or the fixed satellite mass:
        #   planet    -> its own mass
        #   satellite -> its parent planet's mass
        #   sun       -> the largest planet mass beneath it (0 if none)
        # Legacy mode keeps node.mass for every level (published arms).
        priority: dict[str, float] = {}
        if self._level_markers:
            for se in cd.suns:
                best = 0.0
                for pe in se.planets:
                    priority[pe.planet.node_id] = pe.planet.mass
                    best = max(best, pe.planet.mass)
                    for sat in pe.satellites:
                        priority[sat.node_id] = pe.planet.mass
                priority[se.sun.node_id] = best

        def mass_key(r: "_NodeRecord") -> float:
            return priority.get(r.node.node_id, r.node.mass) if self._level_markers else r.node.mass

        def parent_kept(rec: "_NodeRecord") -> bool:
            return rec.parent_id is None or rec.parent_id in kept_ids

        def block_tokens(keep: set[str]) -> int:
            return token_counter(self._emit_kept(records, keep))

        rng = random.Random(seed)

        while True:
            # このパスで追加可能な候補: 未決定 かつ 親が kept。
            eligible = [
                r for r in records
                if r.node.node_id not in kept_ids
                and r.node.node_id not in rejected_ids
                and parent_kept(r)
            ]
            if not eligible:
                break

            if policy == "random":
                candidates = list(eligible)
                rng.shuffle(candidates)
            elif policy == "mass":
                # 質量降順 → created_turn 降順 → 挿入順昇順
                candidates = sorted(
                    eligible,
                    key=lambda r: (-mass_key(r), -r.node.created_turn, r.order_index),
                )
            else:  # recency
                # created_turn 降順 → 挿入順昇順
                candidates = sorted(
                    eligible,
                    key=lambda r: (-r.node.created_turn, r.order_index),
                )

            added_any = False
            for r in candidates:
                trial = kept_ids | {r.node.node_id}
                if block_tokens(trial) <= budget_tokens:
                    kept_ids.add(r.node.node_id)
                    added_any = True
                    # 1 件足したら候補集合（親が kept になった子など）が
                    # 変わり得るので、このパスを打ち切り再走査する。
                    break
                else:
                    # budget 超過 → このパスでは飛ばす。より小さい候補は入り得る。
                    rejected_ids.add(r.node.node_id)

            if not added_any:
                # このパスで 1 件も追加できなかった → 不動点。
                break
            # 次パスに向けて rejected をクリア（親が増えて配置が変わり得るため
            # 再評価する）。kept は保持。
            rejected_ids.clear()

        return self._emit_kept(records, kept_ids)

    def _emit_kept(self, records: list["_NodeRecord"], keep: set[str]) -> str:
        """kept 集合を元の走査順で <CONTEXT> ブロック文字列に整形する。

        keep が全ノードを含む場合、to_context_block と同一出力になるよう
        同じ _node_line / ラッパー行を用いる。
        """
        lines = ["<CONTEXT>"]
        for r in records:
            if r.node.node_id in keep:
                lines.append(self._node_line(r.node.text, r.node.mass, r.indent))
        lines.append("</CONTEXT>")
        return "\n".join(lines)

    def to_api_payload(self, cd: CorrelationDiagram) -> dict:
        """
        サーバーAPIへ送信するペイロード。
        node_list: サーバーがMマトリクスを構築するために使う構造化データ。
        """
        node_list = []
        for n in cd.all_nodes():
            node_list.append({
                "node_id": n.node_id,
                "text": n.text,
                "level": n.level.value,
                "mass": n.mass,
                "token_repr": n.token_repr(self._precision),
            })
        return {
            "node_list": node_list,
            "context_block": self.to_context_block(cd),
        }

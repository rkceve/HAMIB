"""
GraphBuilder: NodeProposal のリストを受け取り、
CorrelationDiagram に実際にノードを追加/更新するクラス。
"""
from __future__ import annotations

from models.correlation_diagram import CorrelationDiagram
from models.node import NodeLevel
from management.node_classifier import NodeProposal, Action


class GraphBuilder:
    def apply(self, cd: CorrelationDiagram, proposals: list[NodeProposal]) -> CorrelationDiagram:
        """proposals を順番に cd に適用し、更新された cd を返す。"""
        for p in proposals:
            self._apply_one(cd, p)
        return cd

    def _apply_one(self, cd: CorrelationDiagram, p: NodeProposal) -> None:
        if p.action == Action.SKIP:
            return

        if p.action == Action.NEW_SUN:
            cd.add_sun(p.node)

        elif p.action == Action.NEW_PLANET:
            if p.parent_id:
                cd.add_planet(p.node, p.parent_id)

        elif p.action == Action.NEW_SATELLITE:
            if p.parent_id:
                cd.add_satellite(p.node, p.parent_id)

        elif p.action == Action.UPDATE_MASS:
            existing = cd.find_node(p.node.node_id)
            if existing is not None:
                existing.mass = p.node.mass

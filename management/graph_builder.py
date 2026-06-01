"""
GraphBuilder: takes a list of NodeProposal and applies them to a
CorrelationDiagram, adding or updating nodes accordingly.
"""
from __future__ import annotations

from models.correlation_diagram import CorrelationDiagram
from management.node_classifier import NodeProposal, Action


class GraphBuilder:
    def apply(self, cd: CorrelationDiagram, proposals: list[NodeProposal]) -> CorrelationDiagram:
        """Applies the proposals to cd in order and returns the updated cd."""
        for p in proposals:
            self._apply_one(cd, p)
        return cd

    def _apply_one(self, cd: CorrelationDiagram, p: NodeProposal) -> None:
        """Applies a single proposal to cd based on its action type."""
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

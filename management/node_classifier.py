"""NodeClassifier: turn a text chunk into proposals for new or updated nodes.

An extractor function (an LLM, supplied by the caller) lists the concepts in
a chunk and scores each 0-100 on three axes. The highest score picks the
level: comprehensiveness -> sun, independence -> planet, detail -> satellite.

A concept similar enough to an existing node (utils.similarity, threshold
config ``management.similarity_threshold``) becomes a mass update of that
node instead of a new node.
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
    # The extractor's three scores, kept for later evaluation and debugging.
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
        """Extract the concepts in ``chunk`` and return the proposals for them.

        ``llm_extract_fn(text)`` returns a list of dicts like
            {"text": "...",
             "score_comprehensiveness": 0-100,
             "score_independence": 0-100,
             "score_detail": 0-100,
             "parent_hint": "<parent's text, optional>"}
        An older format without scores, carrying
        "level": "sun" | "planet" | "satellite" instead, is also accepted.

        If ``builder`` (a GraphBuilder) is given, each concept's proposals are
        applied to ``cd`` straight away, so a later concept from the same chunk
        can name an earlier one as its parent. Without it, the caller applies
        the returned list afterwards and such links within one chunk are lost.
        """
        proposals: list[NodeProposal] = []

        for item in llm_extract_fn(chunk.text):
            text = item.get("text", "").strip()
            if not text:
                continue

            score_c = float(item.get("score_comprehensiveness", 0))
            score_i = float(item.get("score_independence", 0))
            score_d = float(item.get("score_detail", 0))

            if score_c == 0 and score_i == 0 and score_d == 0:
                # Old format without scores: use the given level.
                try:
                    level = NodeLevel(item.get("level", "satellite"))
                except ValueError:
                    level = NodeLevel.SATELLITE
            else:
                level = self._level_from_scores(score_c, score_i, score_d)

            parent_hint = item.get("parent_hint", "")

            item_proposals = self._build_proposals(text, level, parent_hint, cd, chunk.turn)
            for p in item_proposals:
                p.score_comprehensiveness = score_c
                p.score_independence = score_i
                p.score_detail = score_d
            proposals.extend(item_proposals)

            if builder is not None and item_proposals:
                builder.apply(cd, item_proposals)

        return proposals

    @staticmethod
    def _level_from_scores(
        score_c: float, score_i: float, score_d: float
    ) -> NodeLevel:
        """Highest score wins; a tie goes to the higher level (sun > planet > satellite)."""
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
        existing = list(cd.all_nodes())
        if existing:
            best_idx, score = most_similar_index(text, [n.text for n in existing])
            if score >= self._sim_threshold:
                # Already in the diagram: propose raising that node's mass.
                # (GraphMerger later recomputes all masses from satellite counts.)
                matched = existing[best_idx]
                updated = Node(
                    text=matched.text,
                    level=matched.level,
                    mass=matched.mass + self._mass_for(matched.level),
                    node_id=matched.node_id,
                    parent_id=matched.parent_id,
                )
                return [NodeProposal(action=Action.UPDATE_MASS, node=updated)]

        # created_turn is stamped once, here; the level changes below modify
        # this same object, so the stamp survives them.
        new_node = Node(text=text, level=level, mass=self._mass_for(level), created_turn=created_turn)

        if level == NodeLevel.SUN:
            return [NodeProposal(action=Action.NEW_SUN, node=new_node)]

        parent_id = self._find_parent(parent_hint, level, cd)
        if parent_id is None:
            # No parent: hang the node under the most recent node one level up
            # instead of making it a sun (turning every orphan into a sun used
            # to hit the sun limit and drop facts).
            if level == NodeLevel.PLANET and cd.suns:
                parent_id = cd.suns[-1].sun.node_id
            elif level == NodeLevel.SATELLITE:
                all_planets = [pe.planet for se in cd.suns for pe in se.planets]
                if all_planets:
                    parent_id = all_planets[-1].node_id
                elif cd.suns:
                    # No planets yet: keep it as a planet under the latest sun.
                    new_node.level = NodeLevel.PLANET
                    new_node.mass = self._mass_for(NodeLevel.PLANET)
                    parent_id = cd.suns[-1].sun.node_id
            if parent_id is None:
                # Empty diagram: the node starts a new sun.
                new_node.level = NodeLevel.SUN
                new_node.mass = self._default_sun_mass
                return [NodeProposal(action=Action.NEW_SUN, node=new_node)]

        action = Action.NEW_PLANET if new_node.level == NodeLevel.PLANET else Action.NEW_SATELLITE
        return [NodeProposal(action=action, node=new_node, parent_id=parent_id)]

    def _find_parent(
        self, hint: str, level: NodeLevel, cd: CorrelationDiagram
    ) -> str | None:
        """Id of the node one level up that best matches ``hint``, else the
        first such node; None if there is none."""
        if level == NodeLevel.PLANET:
            candidates = [se.sun for se in cd.suns]
        elif level == NodeLevel.SATELLITE:
            candidates = [pe.planet for se in cd.suns for pe in se.planets]
        else:
            return None

        if not candidates:
            return None
        if not hint:
            return candidates[0].node_id

        idx, score = most_similar_index(hint, [c.text for c in candidates])
        if score >= self._sim_threshold:
            return candidates[idx].node_id
        return candidates[0].node_id

    def _mass_for(self, level: NodeLevel) -> float:
        if level == NodeLevel.SUN:
            return self._default_sun_mass
        elif level == NodeLevel.PLANET:
            return self._default_planet_mass
        return self._default_satellite_mass

"""
NodeClassifier: generates node candidates from text chunks.

Each chunk is scored out of 100 on three dimensions:
  - comprehensiveness: corresponds to a sun node
  - independence:      corresponds to a planet node
  - detail:            corresponds to a satellite node
The chunk is classified at the node level of the highest-scoring dimension.

Node-to-node similarity is computed pairwise. This implementation uses
SentenceTransformer embeddings (all-MiniLM-L6-v2) by default and also offers
an optional LLM-based similarity comparison.
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
    # Retains the three dimension scores (used by later evaluation and debugging)
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
        Extracts concepts from the chunk text and returns a list of NodeProposal.

        llm_extract_fn(text) -> list[dict] is a function (implemented on the
        server side) that returns:
            [{"text": "...",
              "level": "sun"|"planet"|"satellite",
              "score_comprehensiveness": 0-100,
              "score_independence": 0-100,
              "score_detail": 0-100,
              "parent_hint": "...(parent text, optional)"}]

        The older format (without scores) is also accepted for backward
        compatibility.

        When a ``builder`` argument is passed, each item's proposals are applied
        incrementally to the provisional CD right after they are generated. This
        preserves parent-child relationships within the same chunk, such as an
        entity and a satellite(parent_hint=entity) returned by the extractor.

        With builder=None, all proposals are accumulated and applied in bulk by
        the caller. In that mode a satellite cannot find an earlier entity in
        the provisional CD via parent_hint, so it falls through to forced
        NEW_SUN promotion and loses its parent-child relationship.
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

            # Classify at the node level of the highest-scoring dimension.
            # If all scores are 0 (old format), use item["level"] instead.
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
            for p in self._build_proposals(text, level, parent_hint, cd):
                p.score_comprehensiveness = score_c
                p.score_independence = score_i
                p.score_detail = score_d
                item_proposals.append(p)
                proposals.append(p)

            # Apply each item's proposals immediately so the next item in the
            # same chunk can see them in `cd`. Without this,
            # extractor-supplied parent_hint chains within the same chunk
            # always fall through to NEW_SUN promotion (parent never found).
            if builder is not None and item_proposals:
                builder.apply(cd, item_proposals)

        return proposals

    @staticmethod
    def _level_from_scores(
        score_c: float, score_i: float, score_d: float
    ) -> NodeLevel:
        """Returns the node level of the highest-scoring dimension."""
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
    ) -> list[NodeProposal]:
        existing_texts = [n.text for n in cd.all_nodes()]

        if existing_texts:
            best_idx, score = most_similar_index(text, existing_texts)
            if score >= self._sim_threshold:
                matched = list(cd.all_nodes())[best_idx]
                # A similar node already exists -> propose increasing its mass.
                # Note: mass is recomputed from satellite count during
                # GraphMerger's final normalization.
                updated = Node(
                    text=matched.text,
                    level=matched.level,
                    mass=matched.mass + self._mass_for(matched.level),
                    node_id=matched.node_id,
                    parent_id=matched.parent_id,
                )
                return [NodeProposal(action=Action.UPDATE_MASS, node=updated)]

        # New node candidate
        mass = self._mass_for(level)
        new_node = Node(text=text, level=level, mass=mass)

        if level == NodeLevel.SUN:
            return [NodeProposal(action=Action.NEW_SUN, node=new_node)]

        # Look up the parent node from parent_hint
        parent_id = self._find_parent(parent_hint, level, cd)
        if parent_id is None:
            # When no parent is found, try to keep the node at its own level by
            # attaching it to a fallback parent: for a PLANET use the most
            # recent SUN, for a SATELLITE use the most recent PLANET. Fall back
            # to NEW_SUN only when no fallback parent exists.
            if level == NodeLevel.PLANET and cd.suns:
                # Use the most recent SUN as a fallback parent
                parent_id = cd.suns[-1].sun.node_id
            elif level == NodeLevel.SATELLITE:
                # If a PLANET node exists, use the most recent one as parent
                all_planets = [pe.planet for se in cd.suns for pe in se.planets]
                if all_planets:
                    parent_id = all_planets[-1].node_id
                elif cd.suns:
                    # No PLANET available -> store directly under a SUN as a PLANET
                    new_node.level = NodeLevel.PLANET
                    new_node.mass = self._mass_for(NodeLevel.PLANET)
                    parent_id = cd.suns[-1].sun.node_id
            if parent_id is None:
                # Truly nothing exists (empty CD) -> NEW_SUN (initial startup only)
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
            # No hint -> first candidate
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

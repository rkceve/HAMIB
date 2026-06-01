"""
EvalGraphBuilder: builds an evaluation CorrelationDiagram from the most recent
five round trips of conversation.

Partial rebuild:
  - start from a full clone of base_cd
  - rebuild and update only the sun nodes mentioned in the recent turns
  - sun nodes not mentioned keep their state from base_cd
  → this keeps unmentioned-node information from being lost even after the
    evaluation CD is swapped in
"""
from __future__ import annotations
import copy

from models.correlation_diagram import CorrelationDiagram
from management.text_chunker import TextChunker
from management.node_classifier import NodeClassifier
from management.graph_builder import GraphBuilder
from utils.similarity import most_similar_index
from utils.config import get


class EvalGraphBuilder:
    def __init__(self):
        self._chunker = TextChunker()
        self._classifier = NodeClassifier()
        self._builder = GraphBuilder()
        self._sim_threshold: float = get("management", "similarity_threshold", 0.75)

    def build(
        self,
        recent_turns: list[tuple[str, str]],  # [(user, assistant), ...]
        base_cd: CorrelationDiagram,
        llm_extract_fn,
        start_turn: int = 0,
    ) -> CorrelationDiagram:
        """
        Build the evaluation CD from the recent_turns conversation.

        Partial rebuild:
          1. clone base_cd into eval_cd
          2. build a temporary CD (temp_cd) from recent_turns to identify the
             active sun nodes
          3. replace only the subtrees of the active sun nodes in eval_cd
          4. keep unmentioned sun nodes as they are in base_cd
        """
        # Step 1: clone base_cd as the starting point of the evaluation CD
        eval_cd = base_cd.clone()

        # Step 2: build a temp CD from recent_turns (used to identify active sun nodes)
        temp_cd = CorrelationDiagram()
        for i, (user_text, assistant_text) in enumerate(recent_turns):
            turn = start_turn + i
            chunks = self._chunker.chunk_turn(user_text, assistant_text, turn)
            for chunk in chunks:
                # Pass builder for incremental apply (see NodeClassifier.classify
                # docstring). Without this, satellite items whose parent_hint
                # points to entities in the same chunk are demoted to NEW_SUN,
                # breaking attribution.
                self._classifier.classify(
                    chunk, temp_cd, llm_extract_fn, builder=self._builder,
                )

        # Step 3: for each sun node in temp_cd, replace the subtree of the
        # corresponding sun node in eval_cd
        for temp_se in temp_cd.suns:
            sun_text = temp_se.sun.text
            match_idx = self._find_matching_sun_idx(sun_text, eval_cd)
            if match_idx is not None:
                # replace the existing sun node's planet subtree with new info
                eval_cd.suns[match_idx].planets = copy.deepcopy(temp_se.planets)
                # keep the larger of the two sun node masses
                eval_cd.suns[match_idx].sun.mass = max(
                    eval_cd.suns[match_idx].sun.mass, temp_se.sun.mass
                )
            # do not add new sun nodes (not present in base_cd), since the
            # evaluation CD aims to improve the existing CD

        return eval_cd

    def _find_matching_sun_idx(self, text: str, cd: CorrelationDiagram) -> int | None:
        """Return the index of the sun node in eval_cd closest to text."""
        sun_texts = [se.sun.text for se in cd.suns]
        if not sun_texts:
            return None
        idx, score = most_similar_index(text, sun_texts)
        if score >= self._sim_threshold:
            return idx
        return None

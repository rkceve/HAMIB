"""
Scorer: scores a CorrelationDiagram on two metrics.

1. contradiction_score: how free the nodes are of semantic contradiction
2. concentration_score: how concentrated the mass is (inverse of entropy)

total score = w1 * contradiction_score + w2 * concentration_score
"""
from __future__ import annotations
import math
import numpy as np

from models.correlation_diagram import CorrelationDiagram
from utils.config import get
from utils.similarity import embed


class Scorer:
    def __init__(self):
        self._w_contradiction: float = get("scoring", "contradiction_weight", 0.6)
        self._w_concentration: float = get("scoring", "concentration_weight", 0.4)

    def score(self, cd: CorrelationDiagram) -> dict:
        nodes = list(cd.all_nodes())
        if not nodes:
            return {"total": 0.0, "contradiction": 0.0, "concentration": 0.0}

        contradiction = self._contradiction_score(nodes)
        concentration = self._concentration_score(nodes)
        total = self._w_contradiction * contradiction + self._w_concentration * concentration

        return {
            "total": round(total, 4),
            "contradiction": round(contradiction, 4),
            "concentration": round(concentration, 4),
            "node_count": len(nodes),
        }

    def _contradiction_score(self, nodes) -> float:
        """
        Estimate freedom from contradiction from the average cosine similarity
        between node texts. Both too-high similarity (duplication) and too-low
        similarity (contradiction) are treated as problems.
        Prototype: a smaller variance of similarity is considered better.
        """
        if len(nodes) < 2:
            return 1.0
        texts = [n.text for n in nodes]
        vecs = embed(texts)
        # cosine similarity over all pairs
        sim_matrix = vecs @ vecs.T
        n = len(nodes)
        upper = [sim_matrix[i, j] for i in range(n) for j in range(i + 1, n)]
        if not upper:
            return 1.0
        mean_sim = float(np.mean(upper))
        # treat 0.3-0.7 mid-range similarity as ideal; penalize by distance from it
        ideal = 0.5
        deviation = abs(mean_sim - ideal)
        return max(0.0, 1.0 - deviation * 2)

    def _concentration_score(self, nodes) -> float:
        """
        The lower the entropy of the mass distribution, the more concentrated
        the information, and the higher the score.
        """
        masses = np.array([n.mass for n in nodes], dtype=float)
        total = masses.sum()
        if total == 0:
            return 0.0
        probs = masses / total
        # normalized entropy (0 = concentrated, 1 = uniform)
        n = len(probs)
        entropy = -float(np.sum(probs * np.log(probs + 1e-9)))
        max_entropy = math.log(n) if n > 1 else 1.0
        normalized_entropy = entropy / max_entropy
        return max(0.0, 1.0 - normalized_entropy)

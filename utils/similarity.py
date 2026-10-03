"""Text similarity, used to decide whether two nodes mean the same thing.

By default the score is the cosine similarity of SentenceTransformer
embeddings (model from config ``management.embedding_model``).

The spec has an LLM compare nodes one pair at a time. To do that, register a
pairwise scorer; ``cosine_similarity`` and ``most_similar_index`` then use it
instead of embeddings (scores are clamped to [0, 1]). Pass None to switch back.

    def my_llm_sim(text_a: str, text_b: str) -> float:
        ...  # score in [0.0, 1.0]

    set_llm_similarity_fn(my_llm_sim)
"""
from __future__ import annotations
import numpy as np
from functools import lru_cache
from typing import Callable

from utils.config import get

_MODEL_NAME: str = get("management", "embedding_model", "all-MiniLM-L6-v2")

# Pairwise scorer registered with set_llm_similarity_fn; None = use embeddings.
_llm_similarity_fn: Callable[[str, str], float] | None = None


def set_llm_similarity_fn(fn: Callable[[str, str], float] | None) -> None:
    """Use ``fn(text_a, text_b) -> score`` for all similarity calls (None = embeddings)."""
    global _llm_similarity_fn
    _llm_similarity_fn = fn


def get_llm_similarity_fn() -> Callable[[str, str], float] | None:
    return _llm_similarity_fn


def _clamp01(score) -> float:
    return max(0.0, min(1.0, float(score)))


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(_MODEL_NAME)


def embed(texts: list[str]) -> np.ndarray:
    """Return L2-normalised embedding matrix of shape (N, dim)."""
    return _get_model().encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def cosine_similarity(a: str, b: str) -> float:
    """Similarity of two texts (the registered LLM scorer, if any, else embeddings)."""
    if _llm_similarity_fn is not None:
        return _clamp01(_llm_similarity_fn(a, b))
    vecs = embed([a, b])
    return float(np.dot(vecs[0], vecs[1]))


def most_similar_index(query: str, candidates: list[str]) -> tuple[int, float]:
    """Return (index, score) of the candidate most similar to ``query``; (-1, 0.0) if none."""
    if not candidates:
        return -1, 0.0

    if _llm_similarity_fn is not None:
        raw = [_llm_similarity_fn(query, c) for c in candidates]
        scores = np.array([_clamp01(s) for s in raw])
    else:
        vecs = embed([query] + candidates)
        scores = vecs[1:] @ vecs[0]
    best = int(np.argmax(scores))
    return best, float(scores[best])

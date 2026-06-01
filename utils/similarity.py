"""
Node similarity utilities.

Two similarity backends are supported:
  - Default: cosine similarity of SentenceTransformer (all-MiniLM-L6-v2)
    embeddings (practical in terms of compute cost and speed).
  - Optional: LLM-based similarity scoring, which compares nodes strictly
    in a one-to-one fashion. Registered externally via
    set_llm_similarity_fn(fn).

Usage of the LLM-based backend:
  from utils.similarity import set_llm_similarity_fn

  def my_llm_sim(text_a: str, text_b: str) -> float:
      # Function returning a 0.0-1.0 score from an LLM
      ...
  set_llm_similarity_fn(my_llm_sim)

  # Subsequent similarity calls switch to the LLM-based backend
"""
from __future__ import annotations
import numpy as np
from functools import lru_cache
from typing import Callable, Optional

from utils.config import get

_MODEL_NAME: str = get("management", "embedding_model", "all-MiniLM-L6-v2")

# Optional LLM-based similarity scoring function.
# When registered, cosine_similarity / most_similar_index prefer it.
_llm_similarity_fn: Optional[Callable[[str, str], float]] = None


def set_llm_similarity_fn(fn: Optional[Callable[[str, str], float]]) -> None:
    """
    Register the LLM-based similarity scoring function.
    Passing None reverts to the embedding-based backend.
    """
    global _llm_similarity_fn
    _llm_similarity_fn = fn


def get_llm_similarity_fn() -> Optional[Callable[[str, str], float]]:
    return _llm_similarity_fn


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(_MODEL_NAME)


def embed(texts: list[str]) -> np.ndarray:
    """Return L2-normalised embedding matrix of shape (N, dim)."""
    model = _get_model()
    vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vecs


def cosine_similarity(a: str, b: str) -> float:
    """
    Cosine similarity in [0, 1] between two texts.
    Prefers the LLM similarity function when one is registered.
    """
    if _llm_similarity_fn is not None:
        score = _llm_similarity_fn(a, b)
        return max(0.0, min(1.0, float(score)))
    vecs = embed([a, b])
    return float(np.dot(vecs[0], vecs[1]))


def most_similar_index(query: str, candidates: list[str]) -> tuple[int, float]:
    """
    Return (index, score) of the most similar candidate.

    When the LLM similarity function is registered, similarity against
    every candidate is scored one-to-one.
    """
    if not candidates:
        return -1, 0.0

    if _llm_similarity_fn is not None:
        # Score nodes strictly one-to-one against every candidate.
        scores = [_llm_similarity_fn(query, c) for c in candidates]
        scores_arr = np.array([max(0.0, min(1.0, float(s))) for s in scores])
        best = int(np.argmax(scores_arr))
        return best, float(scores_arr[best])

    all_texts = [query] + candidates
    vecs = embed(all_texts)
    q_vec = vecs[0]
    c_vecs = vecs[1:]
    scores = c_vecs @ q_vec
    best = int(np.argmax(scores))
    return best, float(scores[best])

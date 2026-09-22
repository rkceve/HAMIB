"""
ノード類似度判定ユーティリティ。

特許§0042 準拠:
  「ノード同士の類似度の判定には、学習済みのLLMが用いられ、
   必ずノードを1対1の形で判定する」

実装方針:
  - デフォルト: SentenceTransformer (all-MiniLM-L6-v2) 埋め込みコサイン類似度
    （計算量・速度の観点で実用的）
  - オプション: LLM ベース類似度判定（特許準拠の厳密実装）
    set_llm_similarity_fn(fn) で外部から登録可能

LLM ベース判定の使い方:
  from utils.similarity import set_llm_similarity_fn

  def my_llm_sim(text_a: str, text_b: str) -> float:
      # LLM で 0.0-1.0 のスコアを返す関数
      ...
  set_llm_similarity_fn(my_llm_sim)

  # 以降の similarity 呼び出しは LLM ベースに切り替わる
"""
from __future__ import annotations
import numpy as np
from functools import lru_cache
from typing import Callable, Optional

from utils.config import get

_MODEL_NAME: str = get("management", "embedding_model", "all-MiniLM-L6-v2")

# 特許§0042 準拠の LLM ベース類似度判定関数（オプション）
# 登録された場合、cosine_similarity / most_similar_index がこれを優先する
_llm_similarity_fn: Optional[Callable[[str, str], float]] = None


def set_llm_similarity_fn(fn: Optional[Callable[[str, str], float]]) -> None:
    """
    特許§0042 準拠の LLM ベース類似度判定関数を登録する。
    None を渡すと埋め込みベースに戻る。
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
    LLM 類似度関数が登録されていればそれを優先（特許§0042 準拠）。
    """
    if _llm_similarity_fn is not None:
        score = _llm_similarity_fn(a, b)
        return max(0.0, min(1.0, float(score)))
    vecs = embed([a, b])
    return float(np.dot(vecs[0], vecs[1]))


def most_similar_index(query: str, candidates: list[str]) -> tuple[int, float]:
    """
    Return (index, score) of the most similar candidate.

    LLM 類似度関数が登録されていれば、特許§0042 準拠で
    1対1 で全候補との類似度を判定する。
    """
    if not candidates:
        return -1, 0.0

    if _llm_similarity_fn is not None:
        # 特許§0042: 「必ずノードを1対1の形で判定する」
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

"""B7: SimilarityJudge shortlist ordering, first-yes stop, scores, cache.

A fake embed_fn is injected everywhere -- no SBERT model is ever loaded.
"""

from __future__ import annotations

import numpy as np

from management.harness.backends import FakeJudge, first_fenced_span
from management.harness.judge import JudgeCache, JudgeRunner
from management.harness.prompts import K_SAME
from management.harness.similarity_judge import SimilarityJudge

# Deterministic 2-D "embeddings": each text maps to a fixed unit vector.
_VECTORS = {
    "q": (1.0, 0.0),
    "c0": (0.0, 1.0),      # cosine 0.0
    "c1": (0.6, 0.8),      # cosine 0.6
    "c2": (0.8, 0.6),      # cosine 0.8
    "c3": (0.28, 0.96),    # cosine 0.28
    "c4": (0.96, 0.28),    # cosine 0.96
    "c5": (0.5, 0.866),    # cosine 0.5
}


def fake_embed(texts: list[str]) -> np.ndarray:
    return np.array([_VECTORS[t] for t in texts], dtype=float)


def _judge_saying_yes_to(*targets: str) -> FakeJudge:
    """Answers "yes" only when the SECOND fenced span is one of the targets."""

    def policy(prompt: str) -> str:
        spans = prompt.split(">>>")
        # Q_SAME layout: A fenced first, B fenced second.
        candidate = first_fenced_span(">>>".join(spans[1:]) + ">>>")
        return "yes" if candidate.strip() in targets else "no"

    return FakeJudge(policy=policy)


def _sj(judge: FakeJudge, **kw: object) -> SimilarityJudge:
    params: dict = {"shortlist_k": 5, "embed_fn": fake_embed, "cache": JudgeCache()}
    params.update(kw)
    return SimilarityJudge(judge, **params)  # type: ignore[arg-type]


def test_shortlist_keeps_top_k_in_cosine_order() -> None:
    judge = FakeJudge(default="no")
    sj = _sj(judge, shortlist_k=3)
    candidates = ["c0", "c1", "c2", "c3", "c4", "c5"]
    idx, score = sj.most_similar("q", candidates)
    assert score == 0.0
    # Only 3 of the 6 candidates were judged, in descending cosine order.
    assert len(judge.prompts) == 3
    asked = [first_fenced_span(">>>".join(p.split(">>>")[1:]) + ">>>").strip()
             for p in judge.prompts]
    assert asked == ["c4", "c2", "c1"]
    # No yes -> best-cosine candidate index with score 0.0.
    assert candidates[idx] == "c4"


def test_stops_at_first_yes() -> None:
    # Both c2 and c1 would say yes; the shortlist order reaches c2 first.
    judge = _judge_saying_yes_to("c2", "c1")
    sj = _sj(judge, shortlist_k=5)
    candidates = ["c0", "c1", "c2", "c3"]
    idx, score = sj.most_similar("q", candidates)
    assert (candidates[idx], score) == ("c2", 1.0)
    # c4 is absent; order is c2(0.8), c1(0.6), ... -> asked c2 first and stopped.
    assert len(judge.prompts) == 1


def test_returns_original_index_not_shortlist_index() -> None:
    judge = _judge_saying_yes_to("c1")
    sj = _sj(judge, shortlist_k=2)
    candidates = ["c0", "c1", "c2"]
    idx, score = sj.most_similar("q", candidates)
    assert idx == 1 and score == 1.0


def test_shortlist_k_zero_asks_every_candidate_in_original_order() -> None:
    judge = FakeJudge(default="no")
    sj = _sj(judge, shortlist_k=0)
    candidates = ["c0", "c1", "c2", "c3", "c4", "c5"]
    idx, score = sj.most_similar("q", candidates)
    assert len(judge.prompts) == 6
    asked = [first_fenced_span(">>>".join(p.split(">>>")[1:]) + ">>>").strip()
             for p in judge.prompts]
    assert asked == candidates          # original order, no embedding used
    assert (idx, score) == (0, 0.0)     # index 0 fallback without a shortlist


def test_shortlist_k_zero_needs_no_embed_fn() -> None:
    judge = FakeJudge(default="no")
    sj = SimilarityJudge(judge, shortlist_k=0, embed_fn=None, cache=JudgeCache())
    assert sj.most_similar("q", ["c0", "c1"]) == (0, 0.0)


def test_cache_prevents_repeat_calls() -> None:
    judge = FakeJudge(default="no")
    cache = JudgeCache()
    sj = SimilarityJudge(judge, shortlist_k=0, embed_fn=fake_embed, cache=cache)
    sj.most_similar("q", ["c0", "c1"])
    n_first = len(judge.prompts)
    sj.most_similar("q", ["c0", "c1"])
    assert len(judge.prompts) == n_first
    assert cache.hits == 2


def test_empty_candidates() -> None:
    judge = FakeJudge(default="yes")
    sj = _sj(judge)
    assert sj.most_similar("q", []) == (-1, 0.0)
    assert judge.prompts == []


def test_embeddings_are_cached_per_text() -> None:
    """H7: the base CD must not be re-encoded on every single decision."""
    encoded: list[list[str]] = []

    def counting_embed(texts: list[str]):
        encoded.append(list(texts))
        return fake_embed(texts)

    judge = FakeJudge(default="no")
    sj = _sj(judge, shortlist_k=3, embed_fn=counting_embed)
    candidates = ["c0", "c1", "c2", "c3"]
    sj.most_similar("q", candidates)
    assert encoded == [["q", "c0", "c1", "c2", "c3"]]
    assert sj.embed_calls == 5

    sj.most_similar("q", candidates)
    # Second decision: nothing new to encode.
    assert len(encoded) == 1
    assert sj.embed_calls == 5
    assert sj.embed_cache_hits == 5

    sj.most_similar("c5", candidates)
    assert encoded[-1] == ["c5"]
    assert sj.embed_calls == 6


def test_cached_embeddings_give_the_same_ranking() -> None:
    judge = FakeJudge(default="no")
    sj = _sj(judge, shortlist_k=2)
    first = sj.most_similar("q", ["c0", "c1", "c2", "c4"])
    second = sj.most_similar("q", ["c0", "c1", "c2", "c4"])
    assert first == second


def test_runner_is_shared_when_injected() -> None:
    judge = FakeJudge(default="no")
    runner = JudgeRunner(judge, JudgeCache())
    sj = SimilarityJudge(judge, shortlist_k=0, embed_fn=fake_embed, runner=runner)
    sj.most_similar("q", ["c0", "c1"])
    assert runner.calls[K_SAME] == 2

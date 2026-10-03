"""SimilarityJudge: asks the LLM about one candidate at a time whether two nodes
are the same matter (or whether a node belongs under a topic).

``most_similar`` has the ``(query, candidates) -> (index, score)`` signature
GraphMerger expects for its ``similarity_fn``.  The score is 1.0 when the judge
says yes and 0.0 otherwise, so it passes or fails any threshold in (0, 1]
(config.yaml uses 0.92).

With ``shortlist_k > 0`` only the ``shortlist_k`` candidates closest by
embedding are asked about, which saves calls but means the others are never
judged.  ``shortlist_k=0`` asks about every candidate, as the spec describes.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from management.harness.judge import JudgeCache, JudgeLLM, JudgeRunner
from management.harness.prompts import K_SAME, PAIRWISE_PROMPTS

# Scores for a "yes" and a "no" answer.
MATCH_SCORE = 1.0
NO_MATCH_SCORE = 0.0

EmbedFn = Callable[[list[str]], np.ndarray]


class SimilarityJudge:
    def __init__(
        self,
        judge: JudgeLLM,
        *,
        shortlist_k: int = 5,
        use_embedding_shortlist: bool = True,
        embed_fn: EmbedFn | None = None,
        cache: JudgeCache | None = None,
        max_tokens: int = 256,
        runner: JudgeRunner | None = None,
    ) -> None:
        self._shortlist_k = shortlist_k
        self._use_shortlist = use_embedding_shortlist and shortlist_k > 0
        self._embed_fn = embed_fn
        self._runner = (
            runner
            if runner is not None
            else JudgeRunner(judge, cache, max_tokens=max_tokens)
        )
        # The diagram's texts are re-ranked on every decision; without this
        # cache a 4,500-node diagram is re-encoded thousands of times per run.
        self._embed_cache: dict[str, np.ndarray] = {}
        self.embed_calls: int = 0        # texts actually sent to the encoder
        self.embed_cache_hits: int = 0   # texts served from the cache

    # -- embedding helper ----------------------------------------------------

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self._embed_fn is not None:
            return self._embed_fn(texts)
        # Imported lazily: tests always inject embed_fn so no model is loaded.
        from utils.similarity import embed

        return embed(texts)

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Encode ``texts``, reusing vectors already computed."""
        missing = [t for t in dict.fromkeys(texts) if t not in self._embed_cache]
        self.embed_cache_hits += len(texts) - len(missing)
        if missing:
            self.embed_calls += len(missing)
            vecs = self._encode(missing)
            for text, vec in zip(missing, vecs):
                self._embed_cache[text] = np.asarray(vec)
        return np.array([self._embed_cache[t] for t in texts])

    def _order_candidates(self, query: str, candidates: Sequence[str]) -> list[int]:
        """Candidate indices in the order to ask: closest first and at most
        ``shortlist_k`` of them, or all in original order without a shortlist.
        """
        if not self._use_shortlist:
            return list(range(len(candidates)))
        vecs = self._embed([query] + list(candidates))
        scores = vecs[1:] @ vecs[0]
        order = [int(i) for i in np.argsort(-scores, kind="stable")]
        return order[: self._shortlist_k]

    # -- public API ----------------------------------------------------------

    def most_similar(
        self, query: str, candidates: list[str], kind: str = K_SAME
    ) -> tuple[int, float]:
        """Ask about each candidate in turn and stop at the first "yes".

        Returns that candidate's index with MATCH_SCORE.  With no "yes", returns
        the closest candidate (index 0 without a shortlist) with NO_MATCH_SCORE,
        which fails every threshold; ``(-1, NO_MATCH_SCORE)`` for no candidates.
        """
        if not candidates:
            return -1, NO_MATCH_SCORE
        template = PAIRWISE_PROMPTS[kind]
        order = self._order_candidates(query, candidates)
        for idx in order:
            prompt = template.format(a=query, b=candidates[idx])
            if self._runner.ask_yes_no(kind, prompt, query, candidates[idx]):
                return idx, MATCH_SCORE
        return order[0], NO_MATCH_SCORE

    # -- introspection (report plumbing) -------------------------------------

    @property
    def runner(self) -> JudgeRunner:
        return self._runner

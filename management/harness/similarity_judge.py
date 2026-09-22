"""SimilarityJudge: 0042 "always judge nodes one-to-one", with an embedding shortlist.

Design: HARNESS_DESIGN.md Stream B, B3.

``most_similar`` has the signature GraphMerger expects
(``(query, candidates) -> (index, score)``) so it can be injected as the
merger's ``similarity_fn``.  The score is binary: 1.0 when the judge says the
pair is the same matter, 0.0 otherwise.  GraphMerger compares against
``similarity_threshold`` (0.92 in config.yaml), so 1.0 passes and 0.0 fails.

Documented deviation from 0042: with ``shortlist_k > 0`` the candidates outside
the embedding top-k are never judged by the LLM.  ``shortlist_k=0`` disables the
shortlist and restores the fully spec-faithful O(n)-calls behaviour.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from management.harness.judge import JudgeCache, JudgeLLM, JudgeRunner
from management.harness.prompts import K_SAME, PAIRWISE_PROMPTS

# The binary score a "yes" answer maps to.  Any threshold in (0.0, 1.0] passes.
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
        # H7: the base CD's texts are re-ranked on every single decision.  Without
        # this cache a 4.5k-node diagram is re-encoded thousands of times per run.
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
        """Encode ``texts``, reusing per-text vectors already computed."""
        missing = [t for t in dict.fromkeys(texts) if t not in self._embed_cache]
        self.embed_cache_hits += len(texts) - len(missing)
        if missing:
            self.embed_calls += len(missing)
            vecs = self._encode(missing)
            for text, vec in zip(missing, vecs):
                self._embed_cache[text] = np.asarray(vec)
        return np.array([self._embed_cache[t] for t in texts])

    def _order_candidates(self, query: str, candidates: Sequence[str]) -> list[int]:
        """Return candidate indices in ask-order.

        Shortlist enabled  -> descending cosine, truncated to k when longer.
        Shortlist disabled -> original order, all candidates.
        """
        if not self._use_shortlist:
            return list(range(len(candidates)))
        vecs = self._embed([query] + list(candidates))
        scores = vecs[1:] @ vecs[0]
        order = [int(i) for i in np.argsort(-scores, kind="stable")]
        if len(order) > self._shortlist_k:
            order = order[: self._shortlist_k]
        return order

    # ── public API ──────────────────────────────────────────────────────────

    def most_similar(
        self, query: str, candidates: list[str], kind: str = K_SAME
    ) -> tuple[int, float]:
        """0042 one-to-one judgment over (a shortlist of) the candidates.

        Asks the pairwise question for each kept candidate in ask-order and STOPS
        at the first "yes", returning that candidate's ORIGINAL index with score
        1.0.  With no "yes" the best-cosine candidate (or index 0 without a
        shortlist) is returned with score 0.0, which fails every threshold.
        """
        if not candidates:
            return -1, NO_MATCH_SCORE
        template = PAIRWISE_PROMPTS[kind]
        order = self._order_candidates(query, candidates)
        for idx in order:
            prompt = template.format(a=query, b=candidates[idx])
            if self._runner.ask_yes_no(kind, prompt, query, candidates[idx]):
                return idx, MATCH_SCORE
        best = order[0] if order else 0
        return best, NO_MATCH_SCORE

    # ── introspection (report plumbing) ─────────────────────────────────────

    @property
    def runner(self) -> JudgeRunner:
        return self._runner

"""
Replacer: performs the swap when the evaluation CD scores higher than the
current CD.

The swap happens only when the score difference exceeds replace_score_margin.
After the swap, the evaluation CD is discarded.
"""
from __future__ import annotations

from evaluation.scorer import Scorer
from store.cd_store import CDStore
from utils.config import get


class Replacer:
    def __init__(self, store: CDStore, scorer=None):
        """
        Args:
            store: CDStore instance
            scorer: scorer (defaults to the v1 Scorer when omitted). Can be
                    swapped for the v2 LLM-based scorer.
        """
        self._store = store
        self._scorer = scorer if scorer is not None else Scorer()
        self._margin: float = get("evaluation", "replace_score_margin", 0.05)

    def evaluate_and_replace(self) -> dict:
        """
        Score the evaluation CD and the current CD, and swap if the evaluation
        CD is better by at least the margin. Returns a result dict.
        """
        current_cd = self._store.get_current()
        eval_cd = self._store.get_eval()

        if eval_cd is None:
            return {"replaced": False, "reason": "no eval CD"}

        current_score = self._scorer.score(current_cd)
        eval_score = self._scorer.score(eval_cd)

        diff = eval_score["total"] - current_score["total"]
        replaced = diff > self._margin

        if replaced:
            self._store.replace_with_eval()
        else:
            self._store.discard_eval()

        return {
            "replaced": replaced,
            "current_score": current_score,
            "eval_score": eval_score,
            "score_diff": round(diff, 4),
            "margin": self._margin,
        }

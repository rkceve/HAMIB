"""
ScorerLLM: LLM-based evaluation scorer (v2).

Difference from the existing Scorer (v1):
  v1 used metrics that diverged from the intended definitions:
    - contradiction = mid-range cosine similarity (no LLM)
    - concentration = inverse of mass entropy
  v2 follows the intended definitions:
    - freedom from contradiction = the LLM judges whether "opposite events"
      are present and scores out of 100
    - information concentration = the LLM scores out of 100 either the
      difficulty of summarizing after node deletion OR the density of proper
      nouns and predicates

Interface:
  Returns the same score(cd) -> dict as the existing Scorer, so it can be
  swapped into Replacer interchangeably.

LLM call:
  llm_judge_fn(prompt: str) -> str is passed as an argument. The return value
  is expected to be of the form "integer + newline" ("75\n..."), and the first
  integer is extracted. On parse failure the median value 50 is returned
  (prioritizing not crashing).

Call count:
  2 LLM calls per score() (contradiction + concentration). The evaluation unit
  runs every eval_interval=5 turns, so a 30-turn session adds 6 x 2 = 12 LLM
  calls. Lightweight compared to the management unit.
"""
from __future__ import annotations
import re
from typing import Callable

from models.correlation_diagram import CorrelationDiagram
from utils.config import get


def _parse_first_int(text: str, default: int = 50) -> int:
    """Extract the first integer near the start of the LLM response. Clip to 0-100."""
    if not text:
        return default
    m = re.search(r"-?\d+", text)
    if not m:
        return default
    try:
        v = int(m.group(0))
    except ValueError:
        return default
    return max(0, min(100, v))


class ScorerLLM:
    """LLM-based evaluation scorer."""

    def __init__(self, llm_judge_fn: Callable[[str], str]):
        """
        Args:
            llm_judge_fn: function that takes a prompt and returns an LLM
                          response. Usually MassWeightedGemma.generate is passed.
        """
        if llm_judge_fn is None:
            raise ValueError("ScorerLLM requires llm_judge_fn (LLM-based scoring is its core)")
        self._judge = llm_judge_fn
        self._w_contradiction: float = get("scoring", "contradiction_weight", 0.6)
        self._w_concentration: float = get("scoring", "concentration_weight", 0.4)

    def score(self, cd: CorrelationDiagram) -> dict:
        """Return the same dict format as the existing Scorer (for compatibility)."""
        nodes = list(cd.all_nodes())
        if not nodes:
            return {
                "total": 0.0, "contradiction": 0.0, "concentration": 0.0,
                "node_count": 0, "scorer_version": "v2_llm",
            }

        # format the node list as text for use in the prompt
        node_texts = []
        for n in nodes:
            level = getattr(n.level, "value", str(n.level))
            node_texts.append(f"[{level}] {n.text}")
        nodes_dump = "\n".join(node_texts)

        # freedom-from-contradiction score (whether "opposite events" are present)
        contradiction_raw = self._judge(self._build_contradiction_prompt(nodes_dump))
        contradiction_100 = _parse_first_int(contradiction_raw, default=50)
        contradiction = contradiction_100 / 100.0

        # information-concentration score (summarization difficulty on deletion or
        # proper-noun/predicate density)
        concentration_raw = self._judge(self._build_concentration_prompt(nodes_dump))
        concentration_100 = _parse_first_int(concentration_raw, default=50)
        concentration = concentration_100 / 100.0

        total = self._w_contradiction * contradiction + self._w_concentration * concentration

        return {
            "total": round(total, 4),
            "contradiction": round(contradiction, 4),
            "concentration": round(concentration, 4),
            "node_count": len(nodes),
            "scorer_version": "v2_llm",
            "raw_contradiction_response": (contradiction_raw or "")[:80],
            "raw_concentration_response": (concentration_raw or "")[:80],
        }

    @staticmethod
    def _build_contradiction_prompt(nodes_dump: str) -> str:
        # English translation of the Japanese prompt below (reference only —
        # the Japanese original is the live prompt; translating it would
        # shift the LLM's scoring. See README "A note on language"):
        #   The following is a list of nodes extracted as a correlation
        #   diagram from a conversation.
        #   Evaluate whether the nodes contain "opposite events", "logical
        #   contradictions", or "conflicting statements".
        #   Fewer contradictions = higher score; more = lower.
        #
        #   Node list:
        #   {nodes_dump}
        #
        #   Answer the absence of contradictions as a single integer 0-100
        #   (0 = full of contradictions, 100 = none).
        #   Numbers only:
        return (
            "以下は会話から抽出された相関図のノード一覧です。\n"
            "ノード間に「逆の事象」「論理的矛盾」「相反する記述」が含まれているか評価してください。\n"
            "矛盾が少ないほど高スコア、多いほど低スコアです。\n\n"
            f"ノード一覧:\n{nodes_dump}\n\n"
            "矛盾の少なさを 0〜100 の整数 1 つで答えてください（0=矛盾だらけ、100=矛盾なし）。\n"
            "回答は数字のみ:"
        )

    @staticmethod
    def _build_concentration_prompt(nodes_dump: str) -> str:
        # English translation of the Japanese prompt below (reference only —
        # the Japanese original is the live prompt; translating it would
        # shift the LLM's scoring. See README "A note on language"):
        #   The following is a list of nodes extracted as a correlation
        #   diagram from a conversation. Evaluate the "information
        #   concentration" along these axes:
        #     - if each node were deleted, how much harder would summarizing
        #       the original conversation become?
        #     - how dense are proper nouns and predicates relative to token
        #       count?
        #   Higher concentration -> higher score.
        #
        #   Node list:
        #   {nodes_dump}
        #
        #   Answer the information concentration as a single integer 0-100
        #   (0 = sparse, 100 = extremely concentrated).
        #   Numbers only:
        return (
            "以下は会話から抽出された相関図のノード一覧です。\n"
            "「情報の濃縮度」を評価してください。具体的には:\n"
            "  - 各ノードを削除したら、元の会話の要約がどれだけ困難になるか\n"
            "  - トークン数に対する固有名詞・述語の密度がどれだけ高いか\n"
            "の観点で総合判定してください。濃縮度が高いほど高スコアです。\n\n"
            f"ノード一覧:\n{nodes_dump}\n\n"
            "情報の濃縮度を 0〜100 の整数 1 つで答えてください（0=希薄、100=極めて濃縮）。\n"
            "回答は数字のみ:"
        )

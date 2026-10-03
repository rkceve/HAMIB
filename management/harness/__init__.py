"""Manager harness: the correlation-diagram manager written as plain code, with
an LLM that only answers small, checkable yes/no and JSON questions.

Design notes: HARNESS_DESIGN.md and SPEC_FAITHFUL_DESIGN.md.
"""

from __future__ import annotations

from management.harness.backends import (
    AnthropicJudge,
    FakeJudge,
    OpenAICompatJudge,
    make_driver_fake_judge,
    strip_think_blocks,
)
from management.harness.chunking import is_query_turn_en, split_candidates
from management.harness.judge import JudgeCache, JudgeLLM, JudgeRunner, parse_yes_no
from management.harness.manager import (
    HarnessConfig,
    HarnessManager,
    HarnessTurnReport,
    ProvisionalStructure,
)
from management.harness.similarity_judge import SimilarityJudge
from management.harness.spec_manager import (
    SpecChunk,
    SpecConfig,
    SpecManager,
    SpecTurnReport,
    make_spec_fake_judge,
)

__all__ = [
    "AnthropicJudge",
    "FakeJudge",
    "HarnessConfig",
    "HarnessManager",
    "HarnessTurnReport",
    "JudgeCache",
    "JudgeLLM",
    "JudgeRunner",
    "OpenAICompatJudge",
    "ProvisionalStructure",
    "SimilarityJudge",
    "SpecChunk",
    "SpecConfig",
    "SpecManager",
    "SpecTurnReport",
    "is_query_turn_en",
    "make_driver_fake_judge",
    "make_spec_fake_judge",
    "parse_yes_no",
    "split_candidates",
    "strip_think_blocks",
]

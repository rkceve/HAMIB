"""Shared fakes for the harness tests.  No model is ever loaded."""

from __future__ import annotations

import json
from typing import Callable, Iterable

import numpy as np

from management.harness.backends import FakeJudge, first_fenced_span
from management.harness.manager import HarnessConfig, HarnessManager

# Distinctive substrings that identify each harness question in a raw prompt.
MARKERS = {
    "boundary": "Does the topic change",
    "extract": "JSON array",
    "supported": "supported by the source",
    "comprehensive": "title or the summary",
    "independent": "new pillar of the discussion",
    "detail": "concrete step",
    "belongs": "belong under this topic",
    "same": "state the same matter",
}


def kind_of(prompt: str) -> str:
    """Which harness question a raw prompt encodes."""
    for kind, marker in MARKERS.items():
        if marker in prompt:
            return kind
    return "unknown"


def second_fenced_span(prompt: str) -> str:
    parts = prompt.split(">>>")
    if len(parts) < 3:
        return ""
    return first_fenced_span(">>>".join(parts[1:]) + ">>>")


def scripted_judge(
    answers: dict[str, object] | None = None,
    *,
    extract: Callable[[str], Iterable[str]] | None = None,
    default: str = "no",
) -> FakeJudge:
    """A FakeJudge driven by question KIND rather than by regex.

    ``answers`` maps a kind to a fixed string answer, or to a callable
    ``(prompt) -> str``.  ``extract`` maps the chunk text to statements.
    """
    answers = dict(answers or {})

    def policy(prompt: str) -> str:
        kind = kind_of(prompt)
        if kind == "extract" and "extract" not in answers:
            text = first_fenced_span(prompt).strip()
            statements = list(extract(text)) if extract is not None else [text]
            return json.dumps(statements, ensure_ascii=False)
        rule = answers.get(kind)
        if rule is None:
            return default
        if callable(rule):
            return str(rule(prompt))
        return str(rule)

    return FakeJudge(policy=policy)


def fixed_embed(texts: list[str]) -> np.ndarray:
    """Length-based unit vectors: deterministic, never loads a model."""
    vecs = []
    for t in texts:
        a = float(len(t) % 7) + 1.0
        b = float(len(t) % 5) + 1.0
        v = np.array([a, b], dtype=float)
        vecs.append(v / np.linalg.norm(v))
    return np.array(vecs)


def make_manager(judge: FakeJudge, **cfg: object) -> HarnessManager:
    params: dict = {
        "boundary_check": False,
        "faithfulness_check": True,
        "shortlist_k": 0,
        "max_retries": 2,
        "max_statement_chars": 120,
        "judge_max_tokens": 64,
        "extract_max_tokens": 256,
        "chunk_max_chars": 800,
        "topics_in_prompt": 8,
        "max_workers": 1,
    }
    params.update(cfg)
    return HarnessManager(
        judge, config=HarnessConfig(**params), embed_fn=fixed_embed
    )

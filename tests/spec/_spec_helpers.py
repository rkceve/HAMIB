"""Shared fakes for the spec-manager tests.  No model is ever loaded.

Named ``_spec_helpers`` rather than ``_helpers`` on purpose: pytest puts every
rootless test directory on sys.path, so a second module called ``_helpers``
would shadow ``tests/harness/_helpers.py``.
"""

from __future__ import annotations

import json
from typing import Callable

import numpy as np

from management.harness.backends import FakeJudge, first_fenced_span
from management.harness.spec_manager import SpecConfig, SpecManager

# Distinctive substrings that identify each spec-mode question in a raw prompt.
MARKERS = {
    "node": "JSON object with the keys summary",
    "belongs": "belong under this topic",
    "same": "state the same matter",
}


def kind_of(prompt: str) -> str:
    for kind, marker in MARKERS.items():
        if marker in prompt:
            return kind
    return "unknown"


def second_fenced_span(prompt: str) -> str:
    parts = prompt.split(">>>")
    if len(parts) < 3:
        return ""
    return first_fenced_span(">>>".join(parts[1:]) + ">>>")


def node_reply(
    summary: str,
    comprehensiveness: int = 0,
    independence: int = 0,
    detail: int = 0,
) -> str:
    return json.dumps(
        {
            "summary": summary,
            "comprehensiveness": comprehensiveness,
            "independence": independence,
            "detail": detail,
        },
        ensure_ascii=False,
    )


def scripted_judge(
    answers: dict[str, object] | None = None,
    *,
    node: Callable[[str], str] | None = None,
    default: str = "no",
) -> FakeJudge:
    """A FakeJudge driven by question KIND rather than by regex.

    ``node`` maps the fenced fragment to the RAW Q_NODE reply; without it the
    fragment itself becomes the summary with all three scores 0 (-> satellite).
    ``answers`` maps a yes/no kind to a fixed answer or to a callable.
    """
    answers = dict(answers or {})

    def policy(prompt: str) -> str:
        kind = kind_of(prompt)
        if kind == "node" and "node" not in answers:
            text = first_fenced_span(prompt).strip()
            return node(text) if node is not None else node_reply(text)
        rule = answers.get(kind)
        if rule is None:
            return default
        if callable(rule):
            return str(rule(prompt))
        return str(rule)

    return FakeJudge(policy=policy)


def same_text_judge(node: Callable[[str], str] | None = None) -> FakeJudge:
    """Q_SAME answers yes only for two IDENTICAL texts; Q_BELONGS always yes."""

    def same(prompt: str) -> str:
        a = first_fenced_span(prompt).strip()
        b = second_fenced_span(prompt).strip()
        return "yes" if a and a == b else "no"

    return scripted_judge({"same": same, "belongs": "yes"}, node=node)


def fixed_embed(texts: list[str]) -> np.ndarray:
    """Length-based unit vectors: deterministic, never loads a model."""
    vecs = []
    for t in texts:
        a = float(len(t) % 7) + 1.0
        b = float(len(t) % 5) + 1.0
        v = np.array([a, b], dtype=float)
        vecs.append(v / np.linalg.norm(v))
    return np.array(vecs)


def make_manager(judge, **cfg: object) -> SpecManager:
    params: dict = {
        "chunk_max_chars": 400,
        "max_node_chars": 120,
        "shortlist_k": 0,
        "judge_max_tokens": 64,
        "node_max_tokens": 200,
        "max_retries": 1,
        "planet_mass_floor": 0.0,
    }
    params.update(cfg)
    return SpecManager(judge, config=SpecConfig(**params), embed_fn=fixed_embed)

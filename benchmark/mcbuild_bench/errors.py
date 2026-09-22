"""Stop conditions of the mcbuild-bench manager phase (DECISIONS D1 / D2).

There is no fallback judge: when Jev cannot answer, the run STOPS.  A summarizer
that cannot be reached stops the run too; a summarizer answer that merely breaks
the node-text rule falls back the way the spec manager always did (H28).  Both exceptions are
defined here so that the Jev client, the summarizer client, the judge adapters
and build_cd.py all raise / catch the same classes.
"""

from __future__ import annotations


class JevStop(RuntimeError):
    """Jev (TypeSafe) could not deliver a usable answer: HTTP failure after the
    backoff attempts, a validation error, an answer missing a required field,
    or a choice question with more than 254 suns (D1: no defaults, ever)."""


class SummarizerStop(RuntimeError):
    """The local summarizer is unreachable or answers outside its contract
    (transport error, non-2xx, unparseable body)."""


class SummarizerNodeTextUnusable(SummarizerStop):
    """The summarizer answered, but not with a usable node text (over 120
    characters, multi-line, marker syntax, or cut off) after the one retry.
    H28: ``JevNodeFn`` turns this into ``None`` so that ``SpecManager`` takes
    its pre-existing node_fallback path (truncated chunk text, counted in
    ``harness_quality.node_fallback``) instead of stopping the run."""

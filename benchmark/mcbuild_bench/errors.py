"""Exceptions that stop the manager phase (build_cd).

There is no fallback judge: when Jev cannot give a usable answer the run stops
instead of inventing a default, and an unreachable summarizer stops it too.
The one soft case is a summarizer that answers but cannot produce a usable node
text; the spec manager has always handled that by using the truncated chunk
text, so it has its own subclass that ``JevNodeFn`` catches.  All three classes
live here so that the clients, the judge adapters and build_cd raise and catch
the same types.
"""

from __future__ import annotations


class JevStop(RuntimeError):
    """Jev could not deliver a usable answer: an HTTP failure after the retries,
    a validation error, a missing or malformed field, or the spending budget is
    used up.  There is never a default answer in its place."""


class SummarizerStop(RuntimeError):
    """The local summarizer is unreachable or answers outside its contract
    (transport error, non-2xx status, unparseable body)."""


class SummarizerNodeTextUnusable(SummarizerStop):
    """The summarizer answered, but even after its one retry the text is not a
    usable node text (over 120 characters, multi-line, contains marker syntax,
    or was cut off).  ``JevNodeFn`` turns this into ``None``, so ``SpecManager``
    falls back to the truncated chunk text (counted in
    ``harness_quality.node_fallback``) and the run continues."""

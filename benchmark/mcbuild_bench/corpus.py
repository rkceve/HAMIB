"""corpus.py — the experiment corpus = the redacted session minus excluded round trips.

DECISIONS H22 (c) (Ryosuke, 2026-09-20): round trip 36 — the session
retrospective, whose text restates many facts of the whole session — is EXCLUDED
from the experiment corpus.  The exclusion is a documented, CHECKED filter
applied at load time by every consumer (build_cd, run_arms, compaction_c, the
ledger); ``data/session_redacted.json`` itself is never edited.

    corpus = load_corpus(session_path, exclude_idx=DEFAULT_EXCLUDE_RT)
    corpus.round_trips       # the filtered list, session order
    corpus.sha256            # sha256 of the FILTERED content (artifact binding)
    corpus.session_file_sha256  # sha256 of the raw file bytes (recorded only)

Checked means: every excluded ``idx`` must exist in the session (a default of
``(36,)`` applied to a session without a round trip 36 is an error, not a
no-op), duplicates are refused, and the result must be non-empty.

``sha256`` hashes the canonical JSON of the filtered round-trip list
(``sort_keys``, compact separators, ``ensure_ascii=False``), so two consumers
with the same session file and the same exclusion set agree byte for byte, and a
session that never contained the excluded round trips hashes identically to one
that had them filtered out.  The CLIs take ``--exclude-rt 36`` (comma-separated
indices; ``none`` or the empty string disables the filter) and record the list
in every manifest / meta / checkpoint header.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

# H22 (c): the session retrospective.
DEFAULT_EXCLUDE_RT: tuple[int, ...] = (36,)
DEFAULT_EXCLUDE_RT_CLI = ",".join(str(i) for i in DEFAULT_EXCLUDE_RT)


@dataclass(frozen=True)
class Corpus:
    round_trips: list[dict]
    sha256: str
    exclude_rt: tuple[int, ...]
    session_path: str
    session_file_sha256: str
    n_session_round_trips: int


def corpus_sha256(round_trips: list[dict]) -> str:
    """sha256 of the canonical JSON of ``round_trips`` (the filtered content)."""
    canonical = json.dumps(round_trips, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_exclude_rt(text: str) -> tuple[int, ...]:
    """``"36"`` -> (36,), ``"36,10"`` -> (10, 36), ``"none"`` / ``""`` -> ()."""
    text = (text or "").strip()
    if text == "" or text.lower() == "none":
        return ()
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part.isdigit():
            raise ValueError(
                "--exclude-rt expects comma-separated non-negative round-trip indices or "
                "'none', got %r" % (text,)
            )
        out.append(int(part))
    return _checked_exclusion(out)


def _checked_exclusion(exclude_idx) -> tuple[int, ...]:
    values = [int(i) for i in exclude_idx]
    if len(set(values)) != len(values):
        raise ValueError("exclude_idx has duplicates: %r" % (values,))
    if any(v < 0 for v in values):
        raise ValueError("exclude_idx must be non-negative: %r" % (values,))
    return tuple(sorted(values))


def load_corpus(session_path: str | Path, exclude_idx=DEFAULT_EXCLUDE_RT) -> Corpus:
    """Read ``session_path`` and drop the round trips whose ``idx`` is in
    ``exclude_idx`` (checked: every index must exist; the result must be
    non-empty)."""
    exclude = _checked_exclusion(exclude_idx)
    path = Path(session_path)
    raw = path.read_bytes()
    session = json.loads(raw.decode("utf-8"))
    round_trips = list(session["round_trips"])
    present = {int(rt["idx"]) for rt in round_trips}
    missing = [i for i in exclude if i not in present]
    if missing:
        raise ValueError(
            "exclude_idx %r names round trips absent from %s (present idx: %d..%d): the "
            "exclusion is a checked filter, not a no-op" % (
                missing, path.as_posix(), min(present) if present else -1,
                max(present) if present else -1)
        )
    kept = [rt for rt in round_trips if int(rt["idx"]) not in exclude]
    if not kept:
        raise ValueError("the corpus is empty after excluding %r" % (exclude,))
    return Corpus(
        round_trips=kept,
        sha256=corpus_sha256(kept),
        exclude_rt=exclude,
        session_path=str(path),
        session_file_sha256=hashlib.sha256(raw).hexdigest(),
        n_session_round_trips=len(round_trips),
    )

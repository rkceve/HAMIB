"""The experiment corpus: the redacted session minus the excluded round trips.

Every stage (build_cd, compaction_c, run_arms, build_ledger) loads the session
through ``load_corpus`` so that they all work on the same round trips.  By
default round trip 36 is excluded: it is the session retrospective, which
restates facts from the whole session and would put many answers into any
window that contains it.  ``data/session_redacted.json`` itself is never edited.

    corpus = load_corpus(session_path, exclude_idx=DEFAULT_EXCLUDE_RT)
    corpus.round_trips          # the kept round trips, in session order
    corpus.sha256               # hash of the kept content; artifacts are bound to it
    corpus.session_file_sha256  # hash of the raw file bytes (recorded only)

The exclusion is strict: every excluded index must exist in the session (so a
typo cannot silently exclude nothing), duplicates are refused, and the result
must not be empty.

``sha256`` hashes the canonical JSON of the kept round trips (``sort_keys``,
compact separators, ``ensure_ascii=False``).  Two runs with the same session
file and the same exclusion therefore agree byte for byte, and a session that
never had the excluded round trips hashes the same as one that had them
filtered out.  The CLIs take ``--exclude-rt`` (comma-separated indices; ``none``
or the empty string disables the filter) and record the list in their outputs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

# Round trip 36 is the session retrospective (see the module docstring).
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
    ``exclude_idx``.  Every excluded index must exist and the result must not
    be empty; otherwise ValueError."""
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

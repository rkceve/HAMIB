"""Judge protocol, yes/no parsing, answer cache and the counting call runner.

Design: HARNESS_DESIGN.md Stream B, B1 + B4.

The backend is deliberately dumb (~20 lines, see ``backends.py``): every piece of
policy -- yes/no parsing, the one-shot reformat retry, the per-question default,
caching and call counting -- lives here so that all backends behave identically.
"""

from __future__ import annotations

import re
import threading
from typing import Any, Protocol, runtime_checkable

from management.harness.prompts import DEFAULT_ANSWERS, RETRY_YES_NO_SUFFIX


@runtime_checkable
class JudgeLLM(Protocol):
    """Minimal backend contract: one prompt in, raw text out."""

    def complete(self, prompt: str, *, max_tokens: int) -> str: ...


# -- yes/no parsing ----------------------------------------------------------

# ASCII tokens are matched on word boundaries so that "nothing" is not a "no".
_YES_RE = re.compile(r"\b(?:yes|y|true)\b", re.IGNORECASE)
_NO_RE = re.compile(r"\b(?:no|n|false)\b", re.IGNORECASE)

# Japanese has no word boundaries, so these are matched as substrings.
_YES_SUBSTRINGS: tuple[str, ...] = ("はい", "同じ")
_NO_SUBSTRINGS: tuple[str, ...] = ("いいえ", "異なる")

# H2: phrases that negate an otherwise affirmative-looking line.  Checked FIRST,
# because "同じではない" contains the yes substring "同じ" and "not the same"
# contains no negative token at all.
_NEGATIONS: tuple[str, ...] = (
    # Japanese
    "ではない",
    "では ない",
    "ではありません",
    "とは言えません",
    "とは言えない",
    "違います",
    "違う",
    "異なります",
    "異なり",
    "いや",
    # English (lower-cased comparison)
    "not the same",
    "does not",
    "doesn't",
    "do not",
    "don't",
    "isn't",
    "is not",
    "are not",
    "aren't",
    "no.",
)

# Leading list markers / labels stripped before parsing.
_MARKER_RE = re.compile(r"^\s*(?:[-*•・]\s+|\d+\s*[.)]\s+|>\s+)")
_LABEL_RE = re.compile(
    r"^\s*(?:answer|response|reply|ans|a|回答|答え|返答)\s*[:：]\s*", re.IGNORECASE
)


def _first_meaningful_line(text: str) -> str:
    """First non-empty, non-fence line with list markers and labels removed."""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("```"):
            # A bare fence or a ```json opener carries no answer.
            if line.strip("`").strip().isalpha() or line.strip("`").strip() == "":
                continue
            line = line.strip("`").strip()
        previous = None
        while previous != line:
            previous = line
            line = _MARKER_RE.sub("", line)
            line = _LABEL_RE.sub("", line)
            line = line.strip()
        if line:
            return line
    return ""


def parse_yes_no(text: str) -> bool | None:
    """Return True/False for an affirmative/negative answer, None if unparsable.

    Procedure (B4 + H2):
      1. take the first meaningful line (blank lines, code fences, list markers
         such as "- " / "1. " and labels such as "Answer:" / "回答:" removed);
      2. an explicit NEGATION phrase in that line -> False;
      3. a line carrying BOTH a yes and a no token -> None (e.g. "はい/いいえ");
      4. NO is checked before YES, then the Japanese substrings.
    """
    if not text:
        return None
    line = _first_meaningful_line(text)
    if not line:
        return None
    low = line.lower()
    for phrase in _NEGATIONS:
        if phrase in low:
            return False
    has_yes = bool(_YES_RE.search(low)) or any(s in line for s in _YES_SUBSTRINGS)
    has_no = bool(_NO_RE.search(low)) or any(s in line for s in _NO_SUBSTRINGS)
    if has_yes and has_no:
        return None
    if has_no:
        return False
    if has_yes:
        return True
    return None


# -- cache -------------------------------------------------------------------


class JudgeCache:
    """Answer cache keyed by ``(kind, a, b)``.

    The kind is part of the key: the same pair of texts asked as Q_SAME and as
    Q_BELONGS are two different questions and must not share an answer.

    Thread-safe (H7): the per-statement stage may run on a ThreadPoolExecutor.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, str, str], Any] = {}
        self._lock = threading.Lock()
        self.hits: int = 0
        self.misses: int = 0

    def get(self, kind: str, a: str, b: str = "") -> Any | None:
        key = (kind, a, b)
        with self._lock:
            if key in self._store:
                self.hits += 1
                return self._store[key]
            self.misses += 1
            return None

    def put(self, kind: str, a: str, b: str, value: Any) -> None:
        with self._lock:
            self._store[(kind, a, b)] = value

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# -- call runner -------------------------------------------------------------


class JudgeRunner:
    """Counts calls by kind and applies the yes/no retry + default policy.

    Shared by :class:`~management.harness.manager.HarnessManager` and
    :class:`~management.harness.similarity_judge.SimilarityJudge` so that one
    ``calls`` / ``cache`` pair covers the whole turn.

    Observability (H9), all keyed by question kind:
      ``unparsed``  the FIRST answer could not be parsed (a retry was needed);
      ``retried``   a reformat retry was actually issued;
      ``defaulted`` neither answer parsed, so the per-kind default was used.
    A defaulted answer is NOT cached: it is not an answer, and caching it would
    freeze one transport hiccup into every later decision about that pair.
    """

    def __init__(
        self,
        judge: JudgeLLM,
        cache: JudgeCache | None = None,
        *,
        max_tokens: int = 256,
    ) -> None:
        self.judge = judge
        self.cache = cache if cache is not None else JudgeCache()
        self.max_tokens = max_tokens
        self._lock = threading.Lock()
        self.calls: dict[str, int] = {}
        self.unparsed: dict[str, int] = {}
        self.retried: dict[str, int] = {}
        self.defaulted: dict[str, int] = {}

    # -- counters -----------------------------------------------------------

    def _bump(self, counter: dict[str, int], kind: str) -> None:
        with self._lock:
            counter[kind] = counter.get(kind, 0) + 1

    def total_calls(self) -> int:
        return sum(self.calls.values())

    def total_unparsed(self) -> int:
        return sum(self.unparsed.values())

    def total_defaulted(self) -> int:
        return sum(self.defaulted.values())

    def total_retried(self) -> int:
        return sum(self.retried.values())

    # -- questions ----------------------------------------------------------

    def ask_raw(self, kind: str, prompt: str, *, max_tokens: int | None = None) -> str:
        """One uncached backend call, counted under ``kind``."""
        self._bump(self.calls, kind)
        return self.judge.complete(
            prompt, max_tokens=self.max_tokens if max_tokens is None else max_tokens
        )

    def ask_yes_no(self, kind: str, prompt: str, key_a: str, key_b: str = "") -> bool:
        """Cached yes/no question with one reformat retry and a per-kind default."""
        cached = self.cache.get(kind, key_a, key_b)
        if cached is not None:
            return bool(cached)
        answer = parse_yes_no(self.ask_raw(kind, prompt))
        if answer is None:
            self._bump(self.unparsed, kind)
            self._bump(self.retried, kind)
            answer = parse_yes_no(
                self.ask_raw(kind, prompt + "\n" + RETRY_YES_NO_SUFFIX)
            )
        if answer is None:
            self._bump(self.defaulted, kind)
            # Not cached on purpose: a default is the absence of an answer.
            return DEFAULT_ANSWERS[kind]
        self.cache.put(kind, key_a, key_b, answer)
        return answer

"""Split a message into chunk candidates, for English and Japanese text.

``management.text_chunker.TextChunker`` only knows Japanese sentence enders
(``。！？``) and connectives, so on the English benchmark conversations it
returned a whole turn (up to 5799 characters) as one chunk.

Candidate boundaries:
  * sentence ends: ``.``/``!``/``?`` followed by whitespace or the end of the
    text, and ``。``/``！``/``？`` anywhere (Japanese puts no space between
    sentences).  Decimals ("3.5") and common abbreviations ("e.g.", "Mr.",
    ...) are not boundaries.
  * a connective at the start of a clause (English: after a comma, semicolon,
    colon, dash or newline; Japanese: the ``text_chunker`` list, whose entries
    include the trailing ``、``).

The resulting units become chunks of at most ``max_chars``.  A sentence is cut
only when it alone is longer than ``max_chars`` (at the last space before the
limit), and a fragment shorter than ``MIN_CHUNK_CHARS`` is glued to the
previous chunk when the result still fits.

Whole sentences are deliberately not packed together by length: that would
put boundaries at arbitrary offsets.  HarnessManager merges neighbours on the
same topic by asking the Q_BOUNDARY question instead.
"""

from __future__ import annotations

import re

# -- connectives -------------------------------------------------------------

# Copied from management/text_chunker.py::_CONNECTIVES rather than imported, so
# this module does not depend on the old chunker.  Keep the two lists in sync.
JA_CONNECTIVES: tuple[str, ...] = (
    "また、", "また,",
    "例えば、", "例えば,",
    "ところで、", "ところで,",
    "しかし、", "しかし,",
    "ただし、", "ただし,",
    "なお、", "なお,",
    "一方、", "一方,",
    "さらに、", "さらに,",
    "そして、", "そして,",
    "次に、", "次に,",
    "つまり、", "つまり,",
    "従って、", "従って,",
    "したがって、", "したがって,",
    "そのため、", "そのため,",
)

# English connectives that start a new clause.
EN_CONNECTIVES: tuple[str, ...] = (
    "however",
    "for example",
    "also",
    "but",
    "on the other hand",
    "in addition",
    "meanwhile",
    "next",
    "then",
    "therefore",
    "so",
    "first",
    "second",
    "finally",
)

# An English connective starts a clause when the previous non-space character
# is one of these.
_CLAUSE_OPENERS = ",;:—–-\n"

_EN_CONNECTIVE_RE = re.compile(
    r"(?<![\w])(?:" + "|".join(re.escape(c) for c in EN_CONNECTIVES) + r")(?![\w])",
    re.IGNORECASE,
)

# -- sentence ends -----------------------------------------------------------

_JA_ENDERS = "。！？"
_ASCII_ENDERS = ".!?"

# Abbreviations that end in '.' but do not end a sentence, compared against the
# lower-cased word (letters and inner dots) right before the period.
_ABBREVIATIONS: frozenset[str] = frozenset(
    {
        "e.g", "i.e", "etc", "vs", "cf", "al", "approx", "est",
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "rev", "gen",
        "inc", "ltd", "co", "corp", "dept", "univ", "fig", "no", "vol",
        "a.m", "p.m", "u.s", "u.k", "e.u",
    }
)

_WORD_BEFORE_DOT = re.compile(r"([A-Za-z](?:[A-Za-z]|\.(?=[A-Za-z]))*)\.$")

# Shorter fragments are glued to the previous chunk (same value as
# TextChunker._min_chunk_chars).
MIN_CHUNK_CHARS = 12


def _is_abbreviation(text: str, dot_idx: int) -> bool:
    """True when the '.' at ``dot_idx`` closes a known abbreviation."""
    m = _WORD_BEFORE_DOT.search(text[: dot_idx + 1])
    return m is not None and m.group(1).lower() in _ABBREVIATIONS


def _is_decimal_point(text: str, dot_idx: int) -> bool:
    """True for the '.' of a number such as 3.5 (digit on both sides)."""
    if dot_idx == 0 or dot_idx + 1 >= len(text):
        return False
    return text[dot_idx - 1].isdigit() and text[dot_idx + 1].isdigit()


def sentence_end_positions(text: str) -> list[int]:
    """End offsets (exclusive) of the sentences in ``text``."""
    ends: list[int] = []
    n = len(text)
    for i, ch in enumerate(text):
        if ch in _JA_ENDERS:
            # A run such as "！？" is one boundary, after its last character.
            if i + 1 < n and text[i + 1] in _JA_ENDERS:
                continue
            ends.append(i + 1)
            continue
        if ch not in _ASCII_ENDERS:
            continue
        # ASCII enders count only before whitespace or the end of the text.
        if i + 1 < n and not text[i + 1].isspace():
            continue
        if ch == "." and (_is_decimal_point(text, i) or _is_abbreviation(text, i)):
            continue
        ends.append(i + 1)
    return ends


def _connective_positions(text: str) -> list[int]:
    """Start offsets of clause-initial connectives (Japanese and English)."""
    cuts = [
        m.start()
        for conn in JA_CONNECTIVES
        for m in re.finditer(re.escape(conn), text)
        if m.start() > 0
    ]
    for m in _EN_CONNECTIVE_RE.finditer(text):
        before = text[: m.start()].rstrip()
        if before and before[-1] in _CLAUSE_OPENERS:
            cuts.append(m.start())
    return cuts


def split_units(text: str) -> list[str]:
    """Split ``text`` into sentences, further cut before clause connectives."""
    text = text.strip()
    if not text:
        return []
    cuts = set(sentence_end_positions(text)) | set(_connective_positions(text))
    cuts.add(len(text))
    units: list[str] = []
    prev = 0
    for c in sorted(cuts):
        if c <= prev:
            continue
        piece = text[prev:c].strip()
        if piece:
            units.append(piece)
        prev = c
    return units


def _hard_split(unit: str, max_chars: int) -> list[str]:
    """Cut an over-long sentence at the last space before each limit."""
    pieces: list[str] = []
    rest = unit
    while len(rest) > max_chars:
        cut = rest.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        head = rest[:cut].strip()
        if head:
            pieces.append(head)
        rest = rest[cut:].strip()
    if rest:
        pieces.append(rest)
    return pieces


def _is_cjk(ch: str) -> bool:
    return ord(ch) > 0x2E80


def _is_cjk_join(left: str, right: str) -> bool:
    """True when no space belongs between ``left`` and ``right`` (both CJK)."""
    return _is_cjk(left[-1]) and _is_cjk(right[0])


def split_candidates(text: str, max_chars: int = 800) -> list[str]:
    """Chunk candidates for ``text``: each at most ``max_chars`` long, and cut
    mid-sentence only when a single sentence is longer than that.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    chunks: list[str] = []
    for unit in split_units(text):
        for piece in _hard_split(unit, max_chars):
            if (
                chunks
                and len(chunks[-1]) < MIN_CHUNK_CHARS
                and len(chunks[-1]) + 1 + len(piece) <= max_chars
            ):
                joiner = "" if _is_cjk_join(chunks[-1], piece) else " "
                chunks[-1] = chunks[-1] + joiner + piece
            else:
                chunks.append(piece)
    return chunks


# -- English query-turn detection --------------------------------------------

_INTERROGATIVES: tuple[str, ...] = (
    "what", "when", "where", "who", "which", "how", "why",
    "did", "do", "does", "is", "are", "can", "could", "would", "will",
    "tell me", "remind me", "recall",
)

_QUERY_MAX_CHARS = 300


def is_query_turn_en(text: str) -> bool:
    """True for a message that only asks something and adds no new facts.

    English version of ``build_cd_offline._is_query_turn``: a short message
    (<= 300 chars) that ends with '?', starts with a question word, or asks
    for a one-word answer.
    """
    stripped = text.strip()
    if not stripped or len(stripped) > _QUERY_MAX_CHARS:
        return False
    if stripped.endswith("?"):
        return True
    low = stripped.lower()
    if "one word" in low:
        return True
    for word in _INTERROGATIVES:
        # Whole word only: "can" must not match "candle" or "can't".
        next_char = low[len(word) : len(word) + 1]
        if low.startswith(word) and not (next_char.isalpha() or next_char == "'"):
            return True
    return False

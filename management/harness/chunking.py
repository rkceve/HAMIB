"""Language-agnostic chunk candidates for the manager harness (H3).

Design: HARNESS_DESIGN.md Stream B, step 1 (0037-0038).

``management.text_chunker.TextChunker`` only knows Japanese sentence enders
(``。！？``) and Japanese connectives, so on the ENGLISH benchmark corpus
(benchmark/longchat/restaurant_chat_v2.json: 832k chars, zero ``。``, 6498 ``.``)
it returned the whole turn as ONE chunk -- up to 5799 characters -- which made
Q_BOUNDARY dead and Q_EXTRACT a summarisation task over a whole essay.

This module produces the boundary CANDIDATES of 0038 for both languages:

  * sentence ends: ``.``/``!``/``?`` followed by whitespace or end of text, and
    ``。``/``！``/``？`` anywhere (Japanese has no inter-sentence space).
    Decimals ("3.5") and the common abbreviations ("e.g.", "i.e.", "Mr.",
    "Dr.", ...) are NOT boundaries.
  * discourse connectives at a clause start (English: preceded by a comma,
    semicolon, colon, dash or newline; Japanese: the ``text_chunker``
    connective list, which carries its own trailing ``、``).

The candidates are then packed into chunks of at most ``max_chars`` characters.
Packing never cuts a sentence in half: a sentence longer than ``max_chars`` (only
possible for a single unbroken sentence) is cut at the last whitespace before the
limit, and a fragment shorter than ``MIN_CHUNK_CHARS`` is glued to the previous
chunk when the result still fits (the same "coalesce short fragments" rule
``TextChunker._coalesce_short_chunks`` applies).

Deliberate deviation, documented for review: consecutive FULL sentences are NOT
greedily packed together here.  Joining same-topic neighbours is the job of the
Q_BOUNDARY merge in ``HarnessManager._chunk`` (capped by ``chunk_max_chars``);
doing it blindly by length would place chunk boundaries at arbitrary character
offsets and would make the 0038 "meaning shift" question unanswerable.
"""

from __future__ import annotations

import re

# -- connectives -------------------------------------------------------------

# Copied verbatim from management/text_chunker.py::_CONNECTIVES (0038 examples).
# Copied rather than imported so this module stays independent of the legacy
# chunker; keep the two lists in sync if the legacy one ever changes.
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

# English discourse connectives that mark a new clause (H3b).
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

# A clause start for an English connective: right after one of these characters
# (whitespace between them is ignored).
_CLAUSE_OPENERS = ",;:—–-\n"

_EN_CONNECTIVE_RE = re.compile(
    r"(?<![\w])(?:" + "|".join(re.escape(c) for c in EN_CONNECTIVES) + r")(?![\w])",
    re.IGNORECASE,
)

# -- sentence ends -----------------------------------------------------------

_JA_ENDERS = "。！？"
_ASCII_ENDERS = ".!?"

# Abbreviations that end in '.' but do not end a sentence.  Compared against the
# lower-cased word (letters and inner dots) immediately before the period.
_ABBREVIATIONS: frozenset[str] = frozenset(
    {
        "e.g", "i.e", "etc", "vs", "cf", "al", "approx", "est",
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "rev", "gen",
        "inc", "ltd", "co", "corp", "dept", "univ", "fig", "no", "vol",
        "a.m", "p.m", "u.s", "u.k", "e.u",
    }
)

_WORD_BEFORE_DOT = re.compile(r"([A-Za-z](?:[A-Za-z]|\.(?=[A-Za-z]))*)\.$")

# Minimum length of a standalone chunk; shorter fragments are glued to the
# previous chunk (mirrors TextChunker._min_chunk_chars).
MIN_CHUNK_CHARS = 12


def _is_abbreviation(text: str, dot_idx: int) -> bool:
    """True when the '.' at ``dot_idx`` closes a known abbreviation."""
    m = _WORD_BEFORE_DOT.search(text[: dot_idx + 1])
    if m is None:
        return False
    return m.group(1).lower() in _ABBREVIATIONS


def _is_decimal_point(text: str, dot_idx: int) -> bool:
    """True for the '.' of a number such as 3.5 (digit on both sides)."""
    if dot_idx == 0 or dot_idx + 1 >= len(text):
        return False
    return text[dot_idx - 1].isdigit() and text[dot_idx + 1].isdigit()


def sentence_end_positions(text: str) -> list[int]:
    """Indices (exclusive end offsets) at which a sentence ends."""
    ends: list[int] = []
    n = len(text)
    for i, ch in enumerate(text):
        if ch in _JA_ENDERS:
            # Collapse runs such as "！？" into a single boundary.
            if i + 1 < n and text[i + 1] in _JA_ENDERS:
                continue
            ends.append(i + 1)
            continue
        if ch not in _ASCII_ENDERS:
            continue
        # Only a terminator when followed by whitespace or the end of the text.
        if i + 1 < n and not text[i + 1].isspace():
            continue
        if ch == "." and (_is_decimal_point(text, i) or _is_abbreviation(text, i)):
            continue
        ends.append(i + 1)
    return ends


def _connective_positions(text: str) -> list[int]:
    """Start offsets of clause-initial connectives (Japanese and English)."""
    cuts: list[int] = []
    for conn in JA_CONNECTIVES:
        start = 0
        while True:
            idx = text.find(conn, start)
            if idx == -1:
                break
            if idx > 0:
                cuts.append(idx)
            start = idx + len(conn)
    for m in _EN_CONNECTIVE_RE.finditer(text):
        idx = m.start()
        if idx == 0:
            continue
        before = text[:idx].rstrip()
        if before and before[-1] in _CLAUSE_OPENERS:
            cuts.append(idx)
    return cuts


def split_units(text: str) -> list[str]:
    """The 0038 meaning units: sentences, further cut at clause connectives."""
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
    """Cut an over-long single sentence at the last whitespace before the limit."""
    pieces: list[str] = []
    rest = unit
    while len(rest) > max_chars:
        window = rest[:max_chars]
        cut = window.rfind(" ")
        if cut <= 0:
            cut = max_chars
        head = rest[:cut].strip()
        if head:
            pieces.append(head)
        rest = rest[cut:].strip()
        if not rest:
            break
    if rest:
        pieces.append(rest)
    return pieces


def _is_cjk(ch: str) -> bool:
    return ord(ch) > 0x2E80


def _is_cjk_join(left: str, right: str) -> bool:
    """No space is inserted between two CJK fragments."""
    return _is_cjk(left[-1]) and _is_cjk(right[0])


def split_candidates(text: str, max_chars: int = 800) -> list[str]:
    """Chunk candidates for ``text``: never longer than ``max_chars``, never cut
    mid-sentence unless a single sentence exceeds ``max_chars``.
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


# -- English query-turn detection (H3c / D-7) --------------------------------

_INTERROGATIVES: tuple[str, ...] = (
    "what", "when", "where", "who", "which", "how", "why",
    "did", "do", "does", "is", "are", "can", "could", "would", "will",
    "tell me", "remind me", "recall",
)

_QUERY_MAX_CHARS = 300


def is_query_turn_en(text: str) -> bool:
    """English counterpart of ``build_cd_offline._is_query_turn`` (D-7).

    A short user message (<= 300 chars) that ends with '?', opens with an
    interrogative, or asks for a one-word answer introduces no new facts.
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
        if low.startswith(word) and (
            len(low) == len(word)
            or not (low[len(word)].isalpha() or low[len(word)] == "'")
        ):
            return True
    return False

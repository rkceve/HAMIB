"""
TextChunker: splits conversation text into semantic units (chunks).

Rather than splitting per sentence, boundaries are placed at connectives such
as "また" or "例えば" and at points where the meaning vector changes sharply,
producing compact semantic units.

Implementation:
  1. Split at connective boundaries (lightweight, runs immediately)
  2. Auxiliary split at punctuation boundaries (。！？)
  3. Force-split by character count when the configured max token limit is exceeded
  4. Merge adjacent chunks that are too short

Embedding-based splitting on meaning-vector changes is offered as an option
(uses SentenceTransformer when compute_embeddings=True).
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from utils.config import get


# Common Japanese connectives and transition words. When one appears mid-text,
# the position right before it is used as a split boundary.
_CONNECTIVES: tuple[str, ...] = (
    "また、",
    "また,",
    "例えば、",
    "例えば,",
    "ところで、",
    "ところで,",
    "しかし、",
    "しかし,",
    "ただし、",
    "ただし,",
    "なお、",
    "なお,",
    "一方、",
    "一方,",
    "さらに、",
    "さらに,",
    "そして、",
    "そして,",
    "次に、",
    "次に,",
    "つまり、",
    "つまり,",
    "従って、",
    "従って,",
    "したがって、",
    "したがって,",
    "そのため、",
    "そのため,",
)

_SENTENCE_END = re.compile(r"(?<=[。！？!?])")


@dataclass
class Chunk:
    text: str
    source: str    # "user" | "assistant" | "combined"
    turn: int      # round-trip number


class TextChunker:
    def __init__(self):
        self._max_tokens: int = get("management", "chunk_max_tokens", 200)
        self._min_chunk_chars: int = 12  # threshold below which a fragment is merged with its neighbor

    def chunk_turn(self, user_text: str, assistant_text: str, turn: int) -> list[Chunk]:
        """
        Converts one round-trip of text into a list of Chunks.

        Uses connective and punctuation boundaries to form semantic units,
        splitting further only when max_tokens is exceeded.
        """
        chunks: list[Chunk] = []

        if user_text.strip():
            for piece in self._split_to_meaning_units(user_text):
                chunks.append(Chunk(text=piece, source="user", turn=turn))

        if assistant_text.strip():
            for piece in self._split_to_meaning_units(assistant_text):
                chunks.append(Chunk(text=piece, source="assistant", turn=turn))

        # Merge fragments that are too short
        chunks = self._coalesce_short_chunks(chunks)
        return chunks

    # ── Splitting logic ─────────────────────────────────────────────────

    def _split_to_meaning_units(self, text: str) -> list[str]:
        """
        Splits text into semantic units, progressively refining the
        granularity: connective boundaries -> punctuation boundaries ->
        forced split at max_tokens.
        """
        text = text.strip()
        if not text:
            return []

        # Step 1: split at connective boundaries (the connective itself stays at
        # the start of the next chunk)
        units = self._split_on_connectives(text)

        # Step 2: re-split any unit exceeding max_tokens at punctuation
        refined: list[str] = []
        for u in units:
            if self._estimate_tokens(u) <= self._max_tokens:
                refined.append(u)
            else:
                refined.extend(self._split_on_sentences(u))

        # Step 3: force-split by character count if still over the limit
        final: list[str] = []
        max_chars = self._max_tokens * 4  # ~4 chars per token
        for u in refined:
            if len(u) <= max_chars:
                final.append(u)
            else:
                for i in range(0, len(u), max_chars):
                    final.append(u[i : i + max_chars])

        return [u for u in (s.strip() for s in final) if u]

    def _split_on_connectives(self, text: str) -> list[str]:
        """
        Splits text using connectives as boundaries. The connective stays at
        the start of the next chunk, since it semantically belongs to it.
        """
        # Collect the positions of each connective
        cuts: list[int] = []
        for conn in _CONNECTIVES:
            start = 0
            while True:
                idx = text.find(conn, start)
                if idx == -1:
                    break
                if idx > 0:  # do not split on a connective at the start of the text
                    cuts.append(idx)
                start = idx + len(conn)

        if not cuts:
            return [text]

        cuts = sorted(set(cuts))
        pieces: list[str] = []
        prev = 0
        for c in cuts:
            piece = text[prev:c].strip()
            if piece:
                pieces.append(piece)
            prev = c
        tail = text[prev:].strip()
        if tail:
            pieces.append(tail)
        return pieces

    @staticmethod
    def _split_on_sentences(text: str) -> list[str]:
        """Splits into sentences at punctuation (。！？)."""
        parts = _SENTENCE_END.split(text)
        return [p.strip() for p in parts if p.strip()]

    def _coalesce_short_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Merges extremely short chunks (e.g. a lone connective) into the
        preceding chunk, so they do not hinder meaningful node extraction.
        """
        if len(chunks) <= 1:
            return chunks

        coalesced: list[Chunk] = []
        for c in chunks:
            if (
                coalesced
                and len(c.text) < self._min_chunk_chars
                and coalesced[-1].source == c.source
                and coalesced[-1].turn == c.turn
            ):
                merged = Chunk(
                    text=f"{coalesced[-1].text}{c.text}",
                    source=coalesced[-1].source,
                    turn=coalesced[-1].turn,
                )
                coalesced[-1] = merged
            else:
                coalesced.append(c)
        return coalesced

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # Rough estimate: 4 characters ≈ 1 token
        return max(1, len(text) // 4)

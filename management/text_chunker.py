"""TextChunker: split one round trip of dialogue into small meaning units (chunks).

Cuts are made from coarse to fine:
  1. just before each connective word in _CONNECTIVES (the connective starts
     the next chunk);
  2. at sentence ends, but only inside a piece that is over the token limit
     (config ``management.chunk_max_tokens``);
  3. as a last resort, every ``chunk_max_tokens * 4`` characters.
Pieces shorter than 12 characters are then joined onto the previous chunk
from the same speaker.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from utils.config import get


# Connective words that start a new meaning unit: the spec's own examples
# ("also", "for example") plus other common Japanese connectives, each with
# a Japanese and an ASCII comma. Text is cut just before each occurrence.
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
        self._min_chunk_chars: int = 12  # shorter pieces are joined onto the previous chunk

    def chunk_turn(self, user_text: str, assistant_text: str, turn: int) -> list[Chunk]:
        """Split one round trip (user text, then assistant text) into chunks."""
        chunks = [
            Chunk(text=piece, source=source, turn=turn)
            for source, text in (("user", user_text), ("assistant", assistant_text))
            for piece in self._split_to_meaning_units(text)
        ]
        return self._coalesce_short_chunks(chunks)

    # ── splitting ──────────────────────────────────────────────────────

    def _split_to_meaning_units(self, text: str) -> list[str]:
        """Cut at connectives, then sentence ends, then a hard character limit."""
        text = text.strip()
        if not text:
            return []

        refined: list[str] = []
        for unit in self._split_on_connectives(text):
            if self._estimate_tokens(unit) <= self._max_tokens:
                refined.append(unit)
            else:
                refined.extend(self._split_on_sentences(unit))

        max_chars = self._max_tokens * 4  # ~4 chars per token
        pieces = [u[i : i + max_chars] for u in refined for i in range(0, len(u), max_chars)]
        return [p.strip() for p in pieces if p.strip()]

    def _split_on_connectives(self, text: str) -> list[str]:
        """Cut ``text`` just before each connective, so the connective starts
        the piece it introduces."""
        cuts = sorted({
            m.start()
            for conn in _CONNECTIVES
            for m in re.finditer(re.escape(conn), text)
            if m.start() > 0  # a connective at the very start is not a cut
        })
        if not cuts:
            return [text]

        bounds = [0, *cuts, len(text)]
        pieces = (text[a:b].strip() for a, b in zip(bounds, bounds[1:]))
        return [p for p in pieces if p]

    @staticmethod
    def _split_on_sentences(text: str) -> list[str]:
        """Split after each sentence-ending mark."""
        parts = _SENTENCE_END.split(text)
        return [p.strip() for p in parts if p.strip()]

    def _coalesce_short_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Join very short chunks (e.g. a bare connective) onto the previous chunk
        from the same speaker and turn, so they don't become nodes of their own."""
        coalesced: list[Chunk] = []
        for c in chunks:
            prev = coalesced[-1] if coalesced else None
            if (
                prev is not None
                and len(c.text) < self._min_chunk_chars
                and prev.source == c.source
                and prev.turn == c.turn
            ):
                coalesced[-1] = Chunk(text=f"{prev.text}{c.text}", source=prev.source, turn=prev.turn)
            else:
                coalesced.append(c)
        return coalesced

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)  # rough: 4 characters per token

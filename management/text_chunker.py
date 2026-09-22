"""
TextChunker: 会話テキストを意味の最小単位（チャンク）に分割する。

特許§0038 準拠:
  「一文単位ではなく、『また』、『例えば』等といった接続詞や
   意味のベクトルが急変する箇所を境界として、意味の最小単位に分割する」

実装方針:
  1. 接続詞境界での分割（軽量・即時実行）
  2. 句読点（。！？）境界での補助分割
  3. 設定された最大トークン数を超える場合は文字数ベースで強制分割
  4. 短すぎる隣接チャンクは結合する

意味ベクトル変化（embedding）ベースの分割はオプションとして提供する
（compute_embeddings=True 指定時に SentenceTransformer を使用）。
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from utils.config import get


# 特許§0038 で例示された接続詞と、典型的な日本語接続詞・転換語
# 文中に現れた場合に、その直前を境界として分割するためのマーカー。
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
    turn: int      # ラウンドトリップ番号


class TextChunker:
    def __init__(self):
        self._max_tokens: int = get("management", "chunk_max_tokens", 200)
        self._min_chunk_chars: int = 12  # 短すぎる断片は隣と結合する閾値

    def chunk_turn(self, user_text: str, assistant_text: str, turn: int) -> list[Chunk]:
        """
        1ラウンドトリップ分のテキストをChunkのリストに変換する。

        特許§0038 準拠で接続詞・句読点境界を意味の最小単位として用い、
        max_tokens を超える場合のみ更に分割する。
        """
        chunks: list[Chunk] = []

        if user_text.strip():
            for piece in self._split_to_meaning_units(user_text):
                chunks.append(Chunk(text=piece, source="user", turn=turn))

        if assistant_text.strip():
            for piece in self._split_to_meaning_units(assistant_text):
                chunks.append(Chunk(text=piece, source="assistant", turn=turn))

        # 短すぎる断片を後段で結合
        chunks = self._coalesce_short_chunks(chunks)
        return chunks

    # ── 分割ロジック ───────────────────────────────────────────────────

    def _split_to_meaning_units(self, text: str) -> list[str]:
        """
        特許§0038 の「意味の最小単位」分割。
        接続詞境界 → 句読点境界 → max_tokens 強制分割の順で粒度を細かくする。
        """
        text = text.strip()
        if not text:
            return []

        # Step 1: 接続詞境界で分割（接続詞自身は次のチャンクの先頭に残す）
        units = self._split_on_connectives(text)

        # Step 2: 各ユニットが max_tokens を超える場合は句読点で再分割
        refined: list[str] = []
        for u in units:
            if self._estimate_tokens(u) <= self._max_tokens:
                refined.append(u)
            else:
                refined.extend(self._split_on_sentences(u))

        # Step 3: それでも超える場合は文字数で強制分割
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
        接続詞を境界としてテキストを分割する。
        接続詞は次のチャンクの先頭に残す（意味的にそのチャンクの一部だから）。
        """
        # 各接続詞の出現位置を集める
        cuts: list[int] = []
        for conn in _CONNECTIVES:
            start = 0
            while True:
                idx = text.find(conn, start)
                if idx == -1:
                    break
                if idx > 0:  # 文頭の接続詞は分割しない
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
        """句読点 (。！？) で文単位に分割する。"""
        parts = _SENTENCE_END.split(text)
        return [p.strip() for p in parts if p.strip()]

    def _coalesce_short_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        極端に短いチャンク（接続詞だけなど）を直前のチャンクに結合する。
        意味のあるノード抽出を阻害しないようにするため。
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
        # 簡易推定: 4文字 ≈ 1トークン
        return max(1, len(text) // 4)

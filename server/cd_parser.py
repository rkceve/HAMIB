"""Parse the client's node list and locate mass markers in tokenized prompts.

Also builds the LLM prompt used by the /extract_nodes endpoint to extract
concept nodes from text.
"""
from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass
class ParsedNode:
    node_id: str
    text: str
    level: str
    mass: float
    token_repr: str


def parse_node_list(node_list: list[dict]) -> list[ParsedNode]:
    return [
        ParsedNode(
            node_id=n["node_id"],
            text=n["text"],
            level=n["level"],
            mass=float(n["mass"]),
            token_repr=n["token_repr"],
        )
        for n in node_list
    ]


# ── Marker detection ─────────────────────────────────────────────────────
_PN_PATTERN = re.compile(r"\[PN([\d.]+)\]")
# Level markers written by CDSerializer(level_markers=True):
#   [SN] sun / [PN{mass}] planet / [RN] satellite. Only planets carry a mass.
_MARKER_PATTERN = re.compile(r"\[(SN|RN|PN([\d.]+))\]")
_MARKER_STARTS = ("[PN", "[SN", "[RN")
_MARKER_LEVEL = {"S": "sun", "P": "planet", "R": "satellite"}


def _marker_start(text: str, level_markers: bool) -> int:
    """Index of the first marker start in text, or -1."""
    if not level_markers:
        return text.find("[PN")
    hits = [text.find(m) for m in _MARKER_STARTS]
    hits = [h for h in hits if h != -1]
    return min(hits) if hits else -1

# Decoded characters kept while outside a concept. It only has to exceed the
# longest possible marker match (~12 characters); without a limit the buffer
# grows to the whole input and the scan becomes O(n^2).
_SCAN_TAIL_CHARS = 32

# CDSerializer writes each node as "  " * indent + "[PN{mass}] text"
# (_node_line in communication/cd_serializer.py); indent 0/1/2 is
# sun/planet/satellite.
LEVEL_BY_INDENT = {0: "sun", 1: "planet", 2: "satellite"}


def _level_from_leading_spaces(n_spaces: int) -> str:
    """Level from leading spaces: 0-1 sun, 2-3 planet, 4 or more satellite."""
    return LEVEL_BY_INDENT[min(n_spaces // 2, 2)]


def _scan_pn_spans(
    token_ids: list[int], tokenizer, *, level_markers: bool = False
) -> list[tuple[float, str, list[int]]]:
    """Scan the tokens once; return ``(mass, level, concept_token_positions)`` per marker.

    A concept runs from the end of its marker to the next newline or the next
    marker. Every find_* function in this module is derived from this one
    scan, so their results always agree.

    Without level markers the level comes from the indentation of the line the
    ``[PN`` appears on. That is unreliable with SentencePiece tokenizers, whose
    ``decode([id])`` drops leading spaces; to filter by level use
    levels_from_context_block + positions_for_levels instead.
    """
    # Decodes one token at a time on purpose. The scan is a character-level
    # state machine over the decoded stream (a marker can straddle two tokens,
    # and BPE can fuse "]" with the following text), so a batched decode would
    # still need the per-token boundaries. It runs once per question.
    spans: list[tuple[float, str, list[int]]] = []
    decoded_so_far = ""
    line_text = ""  # text decoded since the last '\n'
    in_concept_mass: float | None = None
    in_concept_level: str = LEVEL_BY_INDENT[0]
    current: list[int] = []

    def _close() -> None:
        nonlocal in_concept_mass, current
        if in_concept_mass is not None:
            spans.append((in_concept_mass, in_concept_level, current))
        in_concept_mass = None
        current = []

    for i, tid in enumerate(token_ids):
        token_decoded = tokenizer.decode([tid], skip_special_tokens=False)

        # Track the current line (used only for the indentation-based level).
        line_text += token_decoded
        if "\n" in line_text:
            line_text = line_text[line_text.rindex("\n") + 1:]

        if in_concept_mass is not None:
            # Inside a concept: a newline or the next marker ends it.
            nxt = _marker_start(token_decoded, level_markers)
            if "\n" in token_decoded or nxt != -1:
                if "\n" in token_decoded:
                    # The closing token may carry concept text before its
                    # newline (e.g. " beta\n"); it belongs to the span.
                    if token_decoded[:token_decoded.index("\n")].strip():
                        current.append(i)
                    tail = token_decoded[token_decoded.rindex("\n") + 1:]
                else:
                    tail = token_decoded[nxt:]
                _close()
                decoded_so_far = tail
            else:
                current.append(i)
        else:
            # Keep only the last _SCAN_TAIL_CHARS characters plus the whole
            # current token, so no marker can be missed.
            decoded_so_far = decoded_so_far[-_SCAN_TAIL_CHARS:] + token_decoded
            m = (_MARKER_PATTERN if level_markers else _PN_PATTERN).search(decoded_so_far)
            if m:
                if level_markers:
                    # [SN]/[RN] carry no mass (0.0); [PN{mass}] carries the planet mass.
                    in_concept_mass = float(m.group(2)) if m.group(2) else 0.0
                    in_concept_level = _MARKER_LEVEL[m.group(1)[0]]
                else:
                    in_concept_mass = float(m.group(1))
                    # Level from the indentation of the line this [PN is on.
                    head = line_text[:line_text.rindex("[PN")] if "[PN" in line_text else line_text
                    in_concept_level = _level_from_leading_spaces(len(head) - len(head.lstrip(" ")))
                current = []
                residual = decoded_so_far[m.end():]
                decoded_so_far = residual
                # BPE may fuse "]" with the following text into one token
                # (e.g. "] AAA"); then token i itself is part of the concept.
                nxt_r = _marker_start(residual, level_markers)
                if "\n" in residual:
                    concept_part, tail = residual.split("\n", 1)
                    terminated = True
                elif nxt_r != -1:
                    concept_part, tail = residual[:nxt_r], residual[nxt_r:]
                    terminated = True
                else:
                    concept_part, tail = residual, residual
                    terminated = False
                if concept_part.strip():
                    current.append(i)
                if terminated:
                    # The concept also ends inside this token: close it now,
                    # otherwise its mass would leak onto the next line.
                    decoded_so_far = tail
                    _close()

    _close()
    return spans


def find_marker_spans(
    token_ids: list[int], tokenizer
) -> list[tuple[str, float, list[int]]]:
    """``(level, mass, concept_token_positions)`` per [SN]/[PN{mass}]/[RN] marker, in prompt order.

    Suns and satellites have mass 0.0 (mass is defined for planets only). The
    level comes from the marker itself, so it does not depend on the
    tokenizer keeping indentation.
    """
    return [(level, mass, pos) for mass, level, pos in _scan_pn_spans(
        token_ids, tokenizer, level_markers=True
    )]


def marker_positions(
    spans: list[tuple[str, float, list[int]]],
    *,
    inject_levels: set[str] | None = None,
    satellite_inherit: bool = False,
) -> list[tuple[int, float]]:
    """Turn find_marker_spans output into ``(position, mass)`` pairs for the mass vector.

    ``inject_levels`` defaults to ``{"planet"}``. With ``satellite_inherit=True``
    the satellites after a planet take that planet's mass (an experiment
    switch; satellites have no mass of their own). Suns never get mass.
    """
    levels = inject_levels if inject_levels else {"planet"}
    out: list[tuple[int, float]] = []
    planet_mass = 0.0
    for level, mass, positions in spans:
        if level == "planet":
            planet_mass = mass
            eff = mass
        elif level == "satellite":
            eff = planet_mass if satellite_inherit else mass
        else:
            # A sun (or anything else) has no mass and ends the previous
            # planet's scope, so a satellite between a sun and that sun's
            # first planet does not inherit an unrelated planet's mass.
            planet_mass = 0.0
            eff = 0.0
        if level not in levels and not (level == "satellite" and satellite_inherit):
            continue
        if eff <= 0.0:
            continue
        out.extend((p, eff) for p in positions)
    return out


def find_pn_positions(token_ids: list[int], tokenizer) -> list[tuple[int, float]]:
    """``(token_position, mass)`` for every token of the concept text after each ``[PN{mass}]``.

    The concept runs to the next newline or the next ``[PN``. Every one of its
    tokens gets the mass, not only the first (BPE may split a concept such as
    "CRANE-164" into several tokens), including a token in which BPE fused
    "]" with the text.
    """
    return [
        (pos, mass)
        for mass, _level, positions in _scan_pn_spans(token_ids, tokenizer)
        for pos in positions
    ]


def find_pn_spans(token_ids: list[int], tokenizer) -> list[tuple[float, list[int]]]:
    """``(mass, [token_position, ...])`` per ``[PN]`` marker, in prompt order.

    Lets callers filter by level without relying on indentation: the
    serializers (CDSerializer.to_context_block and to_context_block_budgeted)
    write nodes in depth-first sun -> planet -> satellite order, so
    ``find_pn_spans(...)[k]`` and ``levels_from_context_block(block)[k]``
    are the same node. positions_for_levels relies on that pairing.
    """
    return [
        (mass, positions)
        for mass, _level, positions in _scan_pn_spans(token_ids, tokenizer)
    ]


def levels_from_context_block(block: str) -> list[str]:
    """Node levels in output order, read from a ``<CONTEXT>`` block string.

    The level is the line's leading spaces divided by 2 (0 sun, 1 planet,
    2 or more satellite). Wrapper lines (``<CONTEXT>`` / ``</CONTEXT>``) and
    lines without ``[PN`` are ignored. It reads the raw string, so tokenizers
    that drop leading spaces do not affect it.
    """
    levels: list[str] = []
    for line in block.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("<"):
            continue
        if "[PN" not in line:
            continue
        head = line[:line.index("[PN")]
        levels.append(_level_from_leading_spaces(len(head) - len(head.lstrip(" "))))
    return levels


def inherit_satellite_mass(
    spans: list[tuple[float, list[int]]],
    node_levels: list[str],
) -> list[tuple[float, list[int]]]:
    """Return spans in which each satellite takes the mass of the most recent planet.

    Most answers sit on satellite lines, which otherwise get only a small fixed
    mass (graph.default_satellite_mass); this experiment mode
    (attention.satellite_mass_mode = "inherit") gives them their planet's mass.
    The ``[PN]`` numbers in the prompt text are left unchanged. A satellite
    before any planet keeps its own mass. Raises ValueError if ``spans`` and
    ``node_levels`` differ in length.
    """
    if len(spans) != len(node_levels):
        raise ValueError(
            f"span/level count mismatch: {len(spans)} [PN] spans "
            f"but {len(node_levels)} node levels"
        )
    out: list[tuple[float, list[int]]] = []
    planet_mass: float | None = None
    for (mass, positions), level in zip(spans, node_levels):
        if level == "planet":
            planet_mass = mass
            out.append((mass, positions))
        elif level == "satellite" and planet_mass is not None:
            out.append((planet_mass, positions))
        else:
            out.append((mass, positions))
    return out


def positions_for_levels(
    spans: list[tuple[float, list[int]]],
    node_levels: list[str],
    levels: set[str] | None,
) -> list[tuple[int, float]]:
    """Filter find_pn_spans output by node level; return ``(position, mass)`` pairs.

    ``node_levels`` should come from levels_from_context_block on the block
    that was actually put into the prompt. ``levels`` None or empty keeps every
    level. Raises ValueError if ``spans`` and ``node_levels`` differ in length.
    """
    if len(spans) != len(node_levels):
        raise ValueError(
            f"span/level count mismatch: {len(spans)} [PN] spans "
            f"but {len(node_levels)} node levels"
        )
    out: list[tuple[int, float]] = []
    for (mass, positions), level in zip(spans, node_levels):
        if levels and level not in levels:
            continue
        out.extend((pos, mass) for pos in positions)
    return out


# ── [PN] positions with an indentation-based level ──────────────────────

def find_pn_positions_with_level(
    token_ids: list[int], tokenizer
) -> list[tuple[int, float, str]]:
    """find_pn_positions plus the level of each position: ``(position, mass, level)``.

    The level comes from the leading spaces of the line the ``[PN`` is on:
    0-1 "sun", 2-3 "planet", 4 or more "satellite". It is the same scan as
    find_pn_positions, so dropping the level gives exactly that function's
    output.

    Unreliable with SentencePiece tokenizers (``decode([id])`` drops leading
    spaces); filter with levels_from_context_block + positions_for_levels
    instead. Kept for character-level tokenizers and older callers.
    """
    return [
        (pos, mass, level)
        for mass, level, positions in _scan_pn_spans(token_ids, tokenizer)
        for pos in positions
    ]


def filter_positions_by_level(
    positions: list[tuple[int, float, str]], levels: set[str] | None
) -> list[tuple[int, float]]:
    """Keep entries whose level is in ``levels`` (None or empty keeps all) and drop the level."""
    if not levels:
        return [(pos, mass) for pos, mass, _ in positions]
    return [(pos, mass) for pos, mass, level in positions if level in levels]


def extract_nodes_prompt(text: str) -> str:
    """LLM prompt that extracts concept nodes from ``text`` as a JSON list.

    Each piece of information is scored 0-100 on three criteria and gets the
    level of its best score: comprehensiveness -> sun (could be a title or
    summary), independence -> planet (a new fact or line of discussion),
    detail -> satellite (numbers, proper nouns, concrete steps). An ID and its
    value or attribute must form one node (e.g. "ALPHA" -> "CRANE-1"), so the
    [PN{mass}] marker the serializer writes covers the whole relation.
    """
    return (
        "以下のテキストから重要な情報を抽出し、JSON形式で返してください。\n"
        "【3項目スコアリング】各情報を以下の3項目で100点満点で評価し、"
        "最高得点の項目に対応する level に分類してください:\n"
        "  - 包括性 (comprehensiveness): 全体トピックの要約・表題となり得るか → sun\n"
        "  - 独立性 (independence): 新しい事実や議論の柱か → planet\n"
        "  - 詳細度 (detail): 数値・固有名詞・具体的手順か → satellite\n"
        "【ルール】ID・コード・名前とその対応値がある場合は、"
        "関係全体を1つのノードに記述すること。\n"
        "  良い例: {\"text\": \"「ALPHA」の対応値は「CRANE-1」\", \"level\": \"sun\", "
        "\"score_comprehensiveness\": 90, \"score_independence\": 60, "
        "\"score_detail\": 30, \"parent_hint\": \"\"}\n"
        "  悪い例: {\"text\": \"ALPHA\", ...} と {\"text\": \"CRANE-1\", ...} を別々に返す\n"
        "フォーマット: [{\"text\": \"...\", \"level\": \"sun|planet|satellite\", "
        "\"score_comprehensiveness\": 0-100, \"score_independence\": 0-100, "
        "\"score_detail\": 0-100, \"parent_hint\": \"親概念名（なければ空文字）\"}]\n\n"
        f"テキスト:\n{text}\n\n"
        "JSON:"
    )

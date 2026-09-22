"""Spec-faithful serialization: [SN]/[PN{mass}]/[RN] markers, mass on planets only."""

from __future__ import annotations

from communication.cd_serializer import CDSerializer
from models.correlation_diagram import CorrelationDiagram
from models.node import Node, NodeLevel
from server.cd_parser import find_marker_spans, find_pn_positions, marker_positions


class _CharTok:
    def __init__(self, text: str) -> None:
        self.text = text

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.text[i] for i in ids)


class _PieceTok:
    def __init__(self, pieces):
        self.p = pieces

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.p[i] for i in ids)


def _cd() -> CorrelationDiagram:
    cd = CorrelationDiagram()
    s = Node(text="Restaurant plan", level=NodeLevel.SUN, mass=1.0)
    cd.add_sun(s)
    p = Node(text="Budget", level=NodeLevel.PLANET, mass=1.0)
    cd.add_planet(p, s.node_id)
    cd.add_satellite(Node(text="Rent is 500k", level=NodeLevel.SATELLITE, mass=0.1), p.node_id)
    cd.add_satellite(Node(text="Loan is 30M", level=NodeLevel.SATELLITE, mass=0.1), p.node_id)
    cd.normalize()
    return cd


def test_serializer_level_markers_mass_on_planets_only() -> None:
    block = CDSerializer(level_markers=True).to_context_block(_cd())
    assert block.splitlines() == [
        "<CONTEXT>",
        "[SN] Restaurant plan",
        "  [PN2.0] Budget",
        "    [RN] Rent is 500k",
        "    [RN] Loan is 30M",
        "</CONTEXT>",
    ]
    legacy = CDSerializer(level_markers=False).to_context_block(_cd())
    assert "[SN]" not in legacy and legacy.count("[PN") == 4


def test_marker_spans_and_positions_char_tokenizer() -> None:
    block = CDSerializer(level_markers=True).to_context_block(_cd())
    tok = _CharTok(block)
    ids = list(range(len(block)))
    spans = find_marker_spans(ids, tok)
    assert [(lvl, m) for lvl, m, _ in spans] == [
        ("sun", 0.0), ("planet", 2.0), ("satellite", 0.0), ("satellite", 0.0)
    ]
    planet_text = "".join(block[i] for i in spans[1][2]).strip()
    assert planet_text == "Budget"
    # default: planets only
    pos = marker_positions(spans)
    assert {m for _, m in pos} == {2.0}
    assert set(p for p, _ in pos) == set(spans[1][2])
    # inherit: satellites take the planet mass; suns never
    pos2 = marker_positions(spans, satellite_inherit=True)
    assert set(p for p, _ in pos2) == set(spans[1][2] + spans[2][2] + spans[3][2])
    assert {m for _, m in pos2} == {2.0}
    # legacy scanner does not see [SN]/[RN] lines but still finds the planet
    legacy = find_pn_positions(ids, tok)
    assert {m for _, m in legacy} == {2.0}


def test_marker_spans_merged_bpe_pieces() -> None:
    pieces = ["[SN", "] Plan", "\n  ", "[PN", "3.0", "] Budget", "\n    [RN]", " Rent", "\n"]
    tok = _PieceTok(pieces)
    spans = find_marker_spans(list(range(len(pieces))), tok)
    assert [(lvl, m, pos) for lvl, m, pos in spans] == [
        ("sun", 0.0, [1]), ("planet", 3.0, [5]), ("satellite", 0.0, [7])
    ]
    assert marker_positions(spans) == [(5, 3.0)]

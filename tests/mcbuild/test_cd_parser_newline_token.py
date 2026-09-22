"""Item B: a token that carries concept text BEFORE its newline belongs to the span."""

from __future__ import annotations

from server.cd_parser import find_marker_spans, find_pn_spans


class PieceTok:
    def __init__(self, pieces: list[str]) -> None:
        self.p = pieces

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return "".join(self.p[i] for i in ids)


def test_text_before_newline_in_the_closing_token_is_part_of_the_span() -> None:
    pieces = ["[PN2.0]", " alpha", " beta\n", "  [PN1.0]", " gamma", "。\n", "[SN]", " tail"]
    spans = find_marker_spans(list(range(len(pieces))), PieceTok(pieces))
    assert [(lvl, m, pos) for lvl, m, pos in spans] == [
        ("planet", 2.0, [1, 2]),
        ("planet", 1.0, [4, 5]),
        ("sun", 0.0, [7]),
    ]


def test_a_bare_newline_token_is_not_added() -> None:
    pieces = ["[PN2.0]", " alpha", "\n", "[PN1.0]", " gamma", " \n"]
    spans = find_pn_spans(list(range(len(pieces))), PieceTok(pieces))
    assert spans == [(2.0, [1]), (1.0, [4])]

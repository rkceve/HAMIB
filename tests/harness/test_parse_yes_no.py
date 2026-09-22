"""B7 + H2: table-driven parse_yes_no tests (English / Japanese / garbage)."""

from __future__ import annotations

import pytest

from management.harness.judge import parse_yes_no

TABLE = [
    ("yes", True),
    ("YES", True),
    ("Yes.", True),
    ("y", True),
    ("true", True),
    ("はい", True),
    ("はい。", True),
    ("はい、同じです。", True),
    ("同じ", True),
    ("**yes**", True),
    ("yes, they describe the same matter", True),
    ("yes\nbecause both mention the price", True),
    ("no", False),
    ("NO", False),
    ("no.", False),
    ("n", False),
    ("false", False),
    ("いいえ", False),
    ("いいえ。", False),
    ("いいえ、異なります。", False),
    ("異なる", False),
    ("「no」", False),
    ("", None),
    ("   ", None),
    ("maybe", None),
    ("nothing in common", None),  # ASCII words are never prefix-matched
    ("わかりません", None),
    ("42", None),
    ("\n\nyes", True),
    # -- H2: negation guards (the yes token appears inside a negative answer) --
    ("同じではない", False),
    ("同じではありません", False),
    ("同じ事柄ではない", False),
    ("同じとは言えません", False),
    ("異なります", False),
    ("違います", False),
    ("違う", False),
    ("いや、別の話です", False),
    ("not the same", False),
    ("It does not belong", False),
    ("that doesn't match", False),
    ("it isn't the same topic", False),
    # -- H2: labels, list markers and code fences --
    ("Answer: yes", True),
    ("回答: いいえ", False),
    ("A: yes", True),
    ("```\nno\n```", False),
    ("```json\nyes\n```", True),
    ("1. yes", True),
    ("- no", False),
    ("* yes", True),
    ("no（属しません）", False),
    # -- H2: both tokens on the first line is not an answer --
    ("はい/いいえ", None),
    ("yes or no", None),
    # -- English full-sentence answers --
    ("Yes, it belongs", True),
    ("No, different", False),
]


@pytest.mark.parametrize("text,expected", TABLE)
def test_parse_yes_no_table(text: str, expected: bool | None) -> None:
    assert parse_yes_no(text) is expected


def test_first_line_only() -> None:
    # A trailing "no" on a later line must not override the first line.
    assert parse_yes_no("yes\nno") is True


def test_no_is_checked_before_yes() -> None:
    """H2: a line carrying only a negative token never reads as affirmative."""
    assert parse_yes_no("no, they are not connected") is False

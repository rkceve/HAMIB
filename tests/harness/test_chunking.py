"""H3b/H3c: English + Japanese chunk candidates and the English query-turn rule."""

from __future__ import annotations

import pytest

from management.harness.chunking import (
    is_query_turn_en,
    sentence_end_positions,
    split_candidates,
    split_units,
)


# -- English sentence boundaries --------------------------------------------


def test_english_sentences_are_split() -> None:
    text = "I worked at Marubeni. The margins were thin. I left in 2019."
    assert split_candidates(text, 800) == [
        "I worked at Marubeni.",
        "The margins were thin.",
        "I left in 2019.",
    ]


def test_decimals_are_not_sentence_ends() -> None:
    text = "The rent is 3.5 million yen. The margin is 8.25 percent."
    assert split_candidates(text, 800) == [
        "The rent is 3.5 million yen.",
        "The margin is 8.25 percent.",
    ]


@pytest.mark.parametrize(
    "abbrev", ["e.g.", "i.e.", "Mr.", "Dr.", "etc.", "Inc.", "vs.", "approx."]
)
def test_abbreviations_are_not_sentence_ends(abbrev: str) -> None:
    text = "We met %s the owner yesterday. Then we left." % abbrev
    assert len(split_candidates(text, 800)) == 2


def test_question_and_exclamation_end_sentences() -> None:
    text = "Is the rent fixed for five years? It is definitely not! We renegotiated it."
    assert len(split_candidates(text, 800)) == 3


def test_no_split_without_trailing_space() -> None:
    """A period glued to the next word (a URL, a version) is not a boundary."""
    assert split_candidates("see config.yaml for the value", 800) == [
        "see config.yaml for the value"
    ]


# -- connectives -------------------------------------------------------------


def test_english_connective_at_a_clause_start_cuts() -> None:
    text = "The rent is high, however the location is excellent."
    assert split_candidates(text, 800) == [
        "The rent is high,",
        "however the location is excellent.",
    ]


def test_english_connective_inside_a_clause_does_not_cut() -> None:
    # "so" here is not preceded by a clause opener -> no boundary.
    assert split_candidates("It was so expensive that we left.", 800) == [
        "It was so expensive that we left."
    ]


def test_connective_word_must_be_whole() -> None:
    assert split_candidates("The number is high, buttoned down.", 800) == [
        "The number is high, buttoned down."
    ]


def test_japanese_connectives_still_cut() -> None:
    text = "昨日は駅前の和食の店に行きました。また、店主は元築地の仲卸だそうです。"
    assert split_candidates(text, 800) == [
        "昨日は駅前の和食の店に行きました。",
        "また、店主は元築地の仲卸だそうです。",
    ]


def test_japanese_sentence_end_needs_no_space() -> None:
    assert sentence_end_positions("あ。い。") == [2, 4]


# -- packing / hard split ----------------------------------------------------

def test_no_chunk_exceeds_max_chars() -> None:
    text = " ".join("word%d" % i for i in range(200))
    for chunk in split_candidates(text, 60):
        assert len(chunk) <= 60


def test_long_sentence_is_cut_at_whitespace() -> None:
    text = "alpha beta gamma delta epsilon zeta"
    chunks = split_candidates(text, 12)
    assert chunks == ["alpha beta", "gamma delta", "epsilon zeta"]
    assert all(" " not in c[-1] for c in chunks)


def test_short_fragment_is_glued_to_the_previous_chunk() -> None:
    # "OK." is shorter than MIN_CHUNK_CHARS and is absorbed by its neighbour.
    assert split_candidates("OK. The rent is fixed for five years.", 800) == [
        "OK. The rent is fixed for five years."
    ]


def test_empty_text() -> None:
    assert split_candidates("", 800) == []
    assert split_units("   ") == []


def test_max_chars_must_be_positive() -> None:
    with pytest.raises(ValueError):
        split_candidates("x", 0)


# -- English query turns (H3c / D-7) ----------------------------------------

QUERY_TABLE = [
    ("What is the name of the restaurant?", True),
    ("Tell me the budget.", True),
    ("Remind me what the rent was", True),
    ("Recall the opening date", True),
    ("Answer in one word: the chef's name", True),
    ("Did I mention the landlord", True),
    ("How much was it", True),
    ("Is the lease signed", True),
    ("I signed the lease yesterday.", False),
    ("Whatever happens, I will sign it.", False),  # "whatever" is not "what"
    ("The answer is 12.", False),
    ("", False),
]


@pytest.mark.parametrize("text,expected", QUERY_TABLE)
def test_is_query_turn_en(text: str, expected: bool) -> None:
    assert is_query_turn_en(text) is expected


def test_long_text_is_never_a_query_turn() -> None:
    assert is_query_turn_en("What " + "x" * 400 + "?") is False

"""B7 (bug-hunt addition): step 1 chunking + the Q_BOUNDARY merge (0038)."""

from __future__ import annotations

from _helpers import make_manager, scripted_judge

from management.harness.prompts import K_BOUNDARY

# TextChunker splits this at the connective, giving two chunks.
USER = "昨日は駅前の和食の店に行きました。また、店主は元築地の仲卸だそうです。"
PIECE_A = "昨日は駅前の和食の店に行きました。"
PIECE_B = "また、店主は元築地の仲卸だそうです。"


def test_chunker_precondition() -> None:
    m = make_manager(scripted_judge(), boundary_check=False)
    chunks = m._chunk(USER, "", 0)
    assert [c.text for c in chunks] == [PIECE_A, PIECE_B]
    assert K_BOUNDARY not in m.runner.calls


def test_topic_change_keeps_the_split() -> None:
    m = make_manager(scripted_judge({"boundary": "yes"}), boundary_check=True)
    chunks = m._chunk(USER, "", 0)
    assert [c.text for c in chunks] == [PIECE_A, PIECE_B]
    assert m.runner.calls[K_BOUNDARY] == 1


def test_no_topic_change_merges_the_pair() -> None:
    m = make_manager(scripted_judge({"boundary": "no"}), boundary_check=True)
    chunks = m._chunk(USER, "", 0)
    # H1 (behaviour change): the merge joins with a single space -- the two
    # chunks are separate sentences and gluing them produced run-on text.
    assert [c.text for c in chunks] == [PIECE_A + " " + PIECE_B]
    assert chunks[0].source == "user"
    assert chunks[0].turn == 0


def test_unparsable_boundary_answer_keeps_the_split() -> None:
    """K_BOUNDARY's default is True: never lose a topic to a bad answer."""
    m = make_manager(scripted_judge({"boundary": "たぶん"}), boundary_check=True)
    chunks = m._chunk(USER, "", 0)
    assert [c.text for c in chunks] == [PIECE_A, PIECE_B]
    assert m.runner.calls[K_BOUNDARY] == 2  # original + reformat retry


def test_no_question_across_a_source_change() -> None:
    m = make_manager(scripted_judge({"boundary": "no"}), boundary_check=True)
    chunks = m._chunk("ユーザーの発言です。", "アシスタントの返答です。", 0)
    assert [c.source for c in chunks] == ["user", "assistant"]
    assert K_BOUNDARY not in m.runner.calls


def test_single_chunk_turn_asks_nothing() -> None:
    m = make_manager(scripted_judge({"boundary": "no"}), boundary_check=True)
    chunks = m._chunk("短い一文です。", "", 0)
    assert len(chunks) == 1
    assert K_BOUNDARY not in m.runner.calls


def test_three_chunks_merge_transitively() -> None:
    text = "第一の文をここに書きます。また、第二の文を書きます。さらに、第三の文を書きます。"
    m_split = make_manager(scripted_judge({"boundary": "yes"}), boundary_check=True)
    parts = [c.text for c in m_split._chunk(text, "", 0)]
    assert len(parts) == 3

    m = make_manager(scripted_judge({"boundary": "no"}), boundary_check=True)
    chunks = m._chunk(text, "", 0)
    assert [c.text for c in chunks] == [" ".join(parts)]
    assert m.runner.calls[K_BOUNDARY] == 2  # two adjacent pairs


def test_merge_stops_at_the_chunk_size_cap() -> None:
    """H1: the merge chain never rebuilds the whole turn as one giant chunk."""
    text = "First sentence here. Second sentence here. Third sentence here."
    m = make_manager(
        scripted_judge({"boundary": "no"}), boundary_check=True, chunk_max_chars=45
    )
    chunks = m._chunk(text, "", 0)
    assert [len(c.text) <= 45 for c in chunks] == [True] * len(chunks)
    assert len(chunks) == 2  # 1+2 fits, adding the third would exceed the cap


def test_english_turn_is_split_into_sentences() -> None:
    """H3b: the corpus is English; TextChunker returned it as ONE chunk."""
    m = make_manager(scripted_judge(), boundary_check=False)
    text = "I worked at Marubeni for 11 years. I want to open a restaurant."
    assert [c.text for c in m._chunk(text, "", 0)] == [
        "I worked at Marubeni for 11 years.",
        "I want to open a restaurant.",
    ]

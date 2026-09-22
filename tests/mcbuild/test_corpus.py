"""corpus.load_corpus: the experiment corpus is the session minus the excluded
round trips (DECISIONS H22 (c): round trip 36, the retrospective), as a
documented and CHECKED filter — never by editing the data file."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmark.mcbuild_bench.corpus import (
    DEFAULT_EXCLUDE_RT,
    Corpus,
    corpus_sha256,
    load_corpus,
    parse_exclude_rt,
)

REAL_SESSION = Path("benchmark/mcbuild_bench/data/session_redacted.json")


def _session(idxs: list[int]) -> dict:
    return {"round_trips": [{"idx": i, "human": "h%d" % i, "events": []} for i in idxs]}


def test_default_exclusion_is_round_trip_36() -> None:
    assert DEFAULT_EXCLUDE_RT == (36,)


def test_load_corpus_filters_and_hashes_the_filtered_content(tmp_path: Path) -> None:
    p = tmp_path / "s.json"
    p.write_text(json.dumps(_session([0, 1, 36])), encoding="utf-8")
    c = load_corpus(p)
    assert isinstance(c, Corpus)
    assert [rt["idx"] for rt in c.round_trips] == [0, 1]
    assert c.exclude_rt == (36,)
    assert c.n_session_round_trips == 3
    assert c.session_file_sha256 == hashlib.sha256(p.read_bytes()).hexdigest()
    # the corpus sha covers the FILTERED content only: it differs from the file's
    assert c.sha256 == corpus_sha256(c.round_trips) != c.session_file_sha256
    # ... and equals the sha of a session that never had round trip 36
    q = tmp_path / "t.json"
    q.write_text(json.dumps(_session([0, 1])), encoding="utf-8")
    assert load_corpus(q, exclude_idx=()).sha256 == c.sha256
    # a different exclusion set changes the sha
    assert load_corpus(p, exclude_idx=(1, 36)).sha256 != c.sha256


def test_load_corpus_refuses_an_exclusion_that_is_not_in_the_session(tmp_path: Path) -> None:
    p = tmp_path / "s.json"
    p.write_text(json.dumps(_session([0, 1])), encoding="utf-8")
    with pytest.raises(ValueError, match="36"):
        load_corpus(p)  # the default (36,) is checked, not silently ignored
    assert [rt["idx"] for rt in load_corpus(p, exclude_idx=()).round_trips] == [0, 1]
    with pytest.raises(ValueError, match="empty"):
        load_corpus(p, exclude_idx=(0, 1))
    with pytest.raises(ValueError):
        load_corpus(p, exclude_idx=(0, 0))


def test_parse_exclude_rt_cli_syntax() -> None:
    assert parse_exclude_rt("36") == (36,)
    assert parse_exclude_rt("36,10") == (10, 36)
    assert parse_exclude_rt("none") == () and parse_exclude_rt("") == ()
    with pytest.raises(ValueError):
        parse_exclude_rt("abc")
    with pytest.raises(ValueError):
        parse_exclude_rt("-1")


def test_real_session_loses_exactly_the_retrospective() -> None:
    if not REAL_SESSION.exists():
        pytest.skip("real session not present")
    c = load_corpus(REAL_SESSION)
    assert c.n_session_round_trips == 37 and len(c.round_trips) == 36
    assert [rt["idx"] for rt in c.round_trips] == list(range(36))

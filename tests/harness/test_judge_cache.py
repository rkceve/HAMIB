"""B7: JudgeCache hit counting; the kind is part of the key."""

from __future__ import annotations

from management.harness.backends import FakeJudge
from management.harness.judge import JudgeCache, JudgeRunner
from management.harness.prompts import K_BELONGS, K_SAME


def test_miss_then_hit() -> None:
    cache = JudgeCache()
    assert cache.get(K_SAME, "a", "b") is None
    assert cache.misses == 1
    assert cache.hits == 0
    cache.put(K_SAME, "a", "b", True)
    assert cache.get(K_SAME, "a", "b") is True
    assert cache.hits == 1
    assert len(cache) == 1


def test_key_includes_kind() -> None:
    cache = JudgeCache()
    cache.put(K_SAME, "a", "b", True)
    # Same texts, different question -> different key, so still a miss.
    assert cache.get(K_BELONGS, "a", "b") is None
    cache.put(K_BELONGS, "a", "b", False)
    assert cache.get(K_SAME, "a", "b") is True
    assert cache.get(K_BELONGS, "a", "b") is False
    assert len(cache) == 2


def test_key_includes_both_texts() -> None:
    cache = JudgeCache()
    cache.put(K_SAME, "a", "b", True)
    assert cache.get(K_SAME, "b", "a") is None


def test_runner_caches_and_counts() -> None:
    judge = FakeJudge(default="yes")
    runner = JudgeRunner(judge, JudgeCache(), max_tokens=32)
    assert runner.ask_yes_no(K_SAME, "prompt", "a", "b") is True
    assert runner.ask_yes_no(K_SAME, "prompt", "a", "b") is True
    assert runner.calls[K_SAME] == 1  # second question served from the cache
    assert runner.cache.hits == 1
    assert len(judge.prompts) == 1


def test_runner_retries_once_then_falls_back_to_default() -> None:
    judge = FakeJudge(default="perhaps")
    runner = JudgeRunner(judge, JudgeCache())
    # K_SAME default is False (prefer adding over a wrong merge).
    assert runner.ask_yes_no(K_SAME, "p", "a", "b") is False
    assert runner.calls[K_SAME] == 2  # original + one reformat retry
    assert "single word yes" in judge.prompts[1]
    # H9: the failure is visible, and a default is NOT cached.
    assert runner.unparsed[K_SAME] == 1
    assert runner.retried[K_SAME] == 1
    assert runner.defaulted[K_SAME] == 1
    assert len(runner.cache) == 0


def test_defaulted_answers_are_not_cached() -> None:
    """H9: a default is the ABSENCE of an answer; caching it freezes one
    transport hiccup into every later decision about the same pair."""
    judge = FakeJudge(default="perhaps")
    runner = JudgeRunner(judge, JudgeCache())
    runner.ask_yes_no(K_SAME, "p", "a", "b")
    runner.ask_yes_no(K_SAME, "p", "a", "b")
    assert runner.calls[K_SAME] == 4  # both questions really went to the judge
    assert runner.defaulted[K_SAME] == 2


def test_quality_counters_stay_zero_on_a_clean_answer() -> None:
    runner = JudgeRunner(FakeJudge(default="yes"), JudgeCache())
    assert runner.ask_yes_no(K_SAME, "p", "a", "b") is True
    assert runner.total_unparsed() == 0
    assert runner.total_defaulted() == 0
    assert runner.total_retried() == 0


def test_counters_are_thread_safe() -> None:
    """H7 smoke: concurrent askers must not lose counter increments."""
    from concurrent.futures import ThreadPoolExecutor

    runner = JudgeRunner(FakeJudge(default="yes"), JudgeCache())
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: runner.ask_yes_no(K_SAME, "p", "a%d" % i), range(200)))
    assert runner.calls[K_SAME] == 200
    assert len(runner.cache) == 200


def test_runner_retry_answer_is_used() -> None:
    answers = iter(["hmm", "yes"])
    judge = FakeJudge(policy=lambda _p: next(answers))
    runner = JudgeRunner(judge, JudgeCache())
    assert runner.ask_yes_no(K_SAME, "p", "a", "b") is True
    assert runner.calls[K_SAME] == 2

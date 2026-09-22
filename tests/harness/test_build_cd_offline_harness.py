"""B7: the driver's `manager=` path (build_cd with a HarnessManager)."""

from __future__ import annotations

import json

import pytest
from _helpers import make_manager, scripted_judge

from benchmark.bineval.build_cd_offline import (
    _load_checkpoint,
    _write_checkpoint,
    build_cd,
    cd_to_records,
    summarize,
)
from management.harness.backends import FakeJudge, make_driver_fake_judge
from management.harness.manager import HarnessConfig, HarnessManager

CHAT = {
    "sessions": [
        {
            "turns": [
                {"role": "user", "content": "この店の名前はホシノ亭です"},
                {"role": "assistant", "content": "ホシノ亭ですね、覚えました"},
                {"role": "user", "content": "店の名前は何ですか？"},  # D-7 query turn
            ]
        },
        {
            "turns": [
                {"role": "user", "content": "予算は3000円です"},
                {"role": "assistant", "content": "予算は3000円ですね"},
            ]
        },
    ]
}

# The real corpus is English (H3): same shape, English content.
CHAT_EN = {
    "sessions": [
        {
            "turns": [
                {"role": "user", "content": "My name is Kenta Morishita."},
                {"role": "assistant", "content": "Noted, Kenta."},
                # English query turn WITHOUT a question mark: only the English
                # rule catches it (the Japanese rule keys on "?" / JP phrases).
                {"role": "user", "content": "Tell me my name"},
            ]
        },
        {
            "turns": [
                {"role": "user", "content": "The budget is 3000 yen."},
                {"role": "assistant", "content": "The budget is 3000 yen, noted."},
            ]
        },
    ]
}


def _manager() -> HarnessManager:
    judge = scripted_judge(
        {"supported": "yes", "independent": "yes"},
        extract=lambda text: [text],
    )
    return make_manager(judge)


def test_build_cd_with_manager_returns_a_cd() -> None:
    manager = _manager()
    cd, n_turns, failed = build_cd(CHAT, None, apply_d7=True, manager=manager)
    assert n_turns == 5
    assert failed == 0
    assert len(cd) > 0
    # every processed turn ran exactly one normalize
    assert manager.normalize_calls == 4  # 5 turns minus the D-7 query turn
    records = cd_to_records(cd)
    assert all(r["created_turn"] >= 0 for r in records)


def test_d7_skip_still_applies() -> None:
    manager = _manager()
    build_cd(CHAT, None, apply_d7=True, manager=manager)
    skipped = manager.normalize_calls
    manager2 = _manager()
    build_cd(CHAT, None, apply_d7=False, manager=manager2)
    assert manager2.normalize_calls == skipped + 1
    # the query turn's text never reached the judge under D-7
    assert not any("店の名前は何ですか" in p for p in manager.runner.judge.prompts)
    assert any("店の名前は何ですか" in p for p in manager2.runner.judge.prompts)


def test_english_query_turn_is_skipped_for_the_harness_arm() -> None:
    """H3c: the corpus is English, so D-7 needs the English rule too."""
    manager = _manager()
    _cd, n_turns, _failed = build_cd(CHAT_EN, None, apply_d7=True, manager=manager)
    assert n_turns == 5
    assert manager.normalize_calls == 4
    assert not any("Tell me my name" in p for p in manager.runner.judge.prompts)


def test_english_query_rule_is_harness_only() -> None:
    """The sbert/gemma arms keep the Japanese rule only, so their published
    numbers stay reproducible: the driver ORs in `is_query_turn_en` only when a
    harness manager is driving the turn."""
    from benchmark.bineval.build_cd_offline import _is_query_turn
    from management.harness.chunking import is_query_turn_en

    assert _is_query_turn("Tell me my name") is False
    assert is_query_turn_en("Tell me my name") is True


def test_max_sessions_truncates() -> None:
    manager = _manager()
    _cd, n_turns, _f = build_cd(
        CHAT, None, apply_d7=True, max_sessions=1, manager=manager
    )
    assert n_turns == 3


def test_summary_includes_harness_totals() -> None:
    manager = _manager()
    cd, n_turns, failed = build_cd(CHAT, None, apply_d7=True, manager=manager)
    summary = summarize(cd, n_turns, 0, failed)
    summary["harness_calls"] = manager.call_totals()
    summary["harness_quality"] = manager.quality_totals()
    assert summary["harness_calls"]["extract"] > 0
    assert sum(summary["harness_calls"].values()) == sum(manager.runner.calls.values())
    assert summary["failed_turns"] == 0
    # H9: the quality block is always present, with every key.
    assert set(summary["harness_quality"]) >= {
        "unparsed",
        "defaulted",
        "extract_fallback",
        "extract_salvaged",
    }


def test_driver_fake_judge_path() -> None:
    """The exact configuration `--judge fake` builds (H10)."""
    config = HarnessConfig.from_config()
    config.shortlist_k = 0  # what main() forces for --judge fake
    manager = HarnessManager(
        make_driver_fake_judge(config.max_statement_chars), config=config
    )
    cd, n_turns, failed = build_cd(CHAT, None, apply_d7=True, manager=manager)
    assert n_turns == 5
    assert failed == 0
    # H10 (behaviour change): the faithfulness answer is now "yes", so the
    # statements survive, become satellites and are promoted by 0061 -> the
    # smoke run really exercises the merge + normalize + serialization path.
    assert len(cd) > 0
    assert cd.suns
    assert manager.totals["extract"] > 0
    assert manager.totals["supported"] > 0


# -- H4: failure counting and the abort ------------------------------------


class _DyingJudge:
    def __init__(self, fail_from: int = 0) -> None:
        self.n = 0
        self.fail_from = fail_from

    def complete(self, prompt: str, *, max_tokens: int) -> str:
        self.n += 1
        if self.n > self.fail_from:
            raise RuntimeError("judge is down")
        return "no"


def test_failed_turns_are_counted() -> None:
    manager = make_manager(_DyingJudge())  # type: ignore[arg-type]
    cd, n_turns, failed = build_cd(
        CHAT, None, apply_d7=True, manager=manager, max_consecutive_failures=100
    )
    assert n_turns == 5
    assert failed == 4  # every non-query turn failed
    assert len(cd) == 0


def test_consecutive_failures_abort_the_run(tmp_path) -> None:
    manager = make_manager(_DyingJudge())  # type: ignore[arg-type]
    ckpt = tmp_path / "out.json.ckpt"
    with pytest.raises(SystemExit) as excinfo:
        build_cd(
            CHAT,
            None,
            apply_d7=True,
            manager=manager,
            ckpt_path=ckpt,
            max_consecutive_failures=2,
        )
    assert "consecutive failed turns" in str(excinfo.value)
    # a checkpoint was written before giving up
    assert ckpt.exists()
    assert "resume" in json.loads(ckpt.read_text(encoding="utf-8"))


def test_a_recovered_turn_resets_the_consecutive_counter() -> None:
    """One flaky turn in a healthy run must not abort it."""
    replies = iter(["boom", '["a fact"]', "yes", "no", "no", "no"])

    def policy(_p: str) -> str:
        value = next(replies, "no")
        if value == "boom":
            raise RuntimeError("transient")
        return value

    manager = make_manager(FakeJudge(policy=policy))
    _cd, _n, failed = build_cd(
        CHAT, None, apply_d7=True, manager=manager, max_consecutive_failures=2
    )
    assert failed == 1


# -- H12: checkpoint carries the counters ----------------------------------


def test_checkpoint_round_trips_the_harness_totals(tmp_path) -> None:
    manager = _manager()
    cd, n_turns, _f = build_cd(CHAT, None, apply_d7=True, manager=manager)
    ckpt = tmp_path / "cd.json.ckpt"
    _write_checkpoint(ckpt, cd, n_turns, 1, manager=manager)

    state = _load_checkpoint(ckpt)
    assert state["harness_totals"] == manager.totals
    assert state["harness_cache_hits"] == manager.total_cache_hits

    resumed = _manager()
    resumed.load_totals(state["harness_totals"])
    assert resumed.call_totals() == manager.call_totals()
    assert resumed.quality_totals() == manager.quality_totals()

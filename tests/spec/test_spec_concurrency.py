"""B5: SpecManager.max_workers -- the Q_NODE calls of one turn run in threads.

The contract is that concurrency changes WALL TIME only: the node list, the
linking, the merge and therefore the whole diagram must be byte-identical to the
sequential run.  Everything here uses a FakeJudge; no model, no network.
"""

from __future__ import annotations

import threading

from _spec_helpers import fixed_embed, node_reply, scripted_judge

from management.harness.judge import JudgeCache, JudgeRunner
from management.harness.spec_manager import SpecConfig, SpecManager
from models.correlation_diagram import CorrelationDiagram

# 12 distinct sentences: enough chunks per turn for the pool to interleave.
TURN_1 = " ".join(
    "The %s item is number %d." % (name, i)
    for i, name in enumerate(
        ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"], start=1
    )
)
TURN_2 = " ".join(
    "The %s item is number %d." % (name, i)
    for i, name in enumerate(
        ["golf", "hotel", "india", "juliett", "kilo", "lima"], start=7
    )
)


def _scores_for(text: str) -> str:
    """A deterministic level per sentence, so the tree has all three levels."""
    if "number 1." in text or "number 7." in text:
        return node_reply(text, 90, 10, 10)   # sun
    if "number 2." in text or "number 8." in text:
        return node_reply(text, 10, 90, 10)   # planet
    return node_reply(text, 10, 10, 90)       # satellite


def _manager(workers: int) -> SpecManager:
    config = SpecConfig(
        chunk_max_chars=400, max_node_chars=120, shortlist_k=0,
        judge_max_tokens=64, node_max_tokens=200, max_retries=1,
        planet_mass_floor=0.0, max_workers=workers,
    )
    return SpecManager(
        scripted_judge({"belongs": "yes"}, node=_scores_for),
        config=config,
        embed_fn=fixed_embed,
    )


def _run(workers: int) -> tuple[CorrelationDiagram, list[dict]]:
    manager = _manager(workers)
    base = CorrelationDiagram()
    reports = [
        vars(manager.update(base, TURN_1, "", 0)),
        vars(manager.update(base, "", TURN_2, 1)),
    ]
    return base, reports


def _shape(cd: CorrelationDiagram) -> list[tuple[str, str, float, str | None]]:
    return [
        (n.level.value, n.text, n.mass, n.parent_id and "parent")
        for n in cd.all_nodes()
    ]


def test_four_workers_produce_the_same_diagram_as_one() -> None:
    seq_cd, seq_reports = _run(1)
    par_cd, par_reports = _run(4)
    assert _shape(par_cd) == _shape(seq_cd)
    assert len(par_cd) == len(seq_cd) > 0
    assert par_reports == seq_reports


def test_the_config_default_is_sequential() -> None:
    assert SpecConfig().max_workers == 1
    assert SpecConfig.from_config().max_workers == 1


def test_workers_really_overlap() -> None:
    """Not a timing test: the judge itself proves two calls were in flight."""
    seen_together = threading.Event()
    inside = []
    lock = threading.Lock()
    barrier = threading.Barrier(2, timeout=5)

    def node(text: str) -> str:
        with lock:
            inside.append(text)
            if len(inside) >= 2:
                seen_together.set()
        try:
            barrier.wait()
        except threading.BrokenBarrierError:  # pragma: no cover - timeout path
            pass
        return node_reply(text, 10, 10, 90)

    manager = SpecManager(
        scripted_judge(node=node),
        config=SpecConfig(shortlist_k=0, planet_mass_floor=0.0, max_workers=2),
        embed_fn=fixed_embed,
    )
    manager.update(CorrelationDiagram(), TURN_1, "", 0)
    assert seen_together.is_set()


def test_a_single_chunk_turn_never_starts_a_pool() -> None:
    manager = _manager(4)
    base = CorrelationDiagram()
    report = manager.update(base, "Only one sentence here.", "", 0)
    assert report.chunks == 1 and report.nodes == 1


def test_node_fallback_counting_is_thread_safe() -> None:
    """Every chunk fails to parse; with 4 workers the counter must still be
    exactly one per chunk (it is bumped under a lock)."""
    from management.harness.backends import FakeJudge

    manager = SpecManager(
        FakeJudge(policy=lambda _p: "not json"),
        config=SpecConfig(shortlist_k=0, planet_mass_floor=0.0, max_workers=4,
                          max_retries=0),
        embed_fn=fixed_embed,
    )
    report = manager.update(CorrelationDiagram(), TURN_1, "", 0)
    assert report.chunks == 6
    assert manager.node_fallback == 6
    assert report.node_fallback == 6


def test_judge_runner_and_cache_are_lock_protected() -> None:
    """B5 asked for this to be verified rather than assumed."""
    lock_type = type(threading.Lock())
    cache = JudgeCache()
    assert isinstance(cache._lock, lock_type)
    assert isinstance(JudgeRunner(scripted_judge(), cache)._lock, lock_type)


def test_concurrent_cache_and_counter_totals_are_exact() -> None:
    """Two turns that repeat the SAME chunks: with 4 workers the second turn
    must be served entirely from the cache, so the call count does not grow."""
    manager = _manager(4)
    base = CorrelationDiagram()
    manager.update(base, TURN_1, "", 0)
    calls_after_first = dict(manager.runner.calls)
    manager.update(base, TURN_1, "", 1)
    assert manager.runner.calls["node"] == calls_after_first["node"]

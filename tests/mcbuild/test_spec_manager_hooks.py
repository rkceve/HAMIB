"""SpecManager hooks (DESIGN.md 5.1): node_fn, keep_fn, similarity.

Defaults must reproduce today's behaviour exactly; each hook is exercised on
its own with a deterministic fake.  No model, no network.
"""

from __future__ import annotations

from management.harness.prompts import K_BELONGS, K_NODE, K_SAME
from management.harness.spec_manager import (
    SpecConfig,
    SpecManager,
    SpecTurnReport,
    make_spec_fake_judge,
)
from models.correlation_diagram import CorrelationDiagram

T1_USER = "Hackathon plan overview for the Minecraft build agent.\nThe server runs Paper 1.21.8 on the demo machine."
T1_ASSISTANT = "The RCON bridge sends one packet per command.\nSpark profiling is disabled on the demo server."
T2_USER = "Change the scale of the build to two to one.\nThe scale decision replaces the earlier one to one."


def _cfg(**over) -> SpecConfig:
    params = {"shortlist_k": 0, "max_workers": 1}
    params.update(over)
    return SpecConfig(**params)


def _records(cd: CorrelationDiagram) -> list[tuple[str, str, float, int]]:
    return [(n.text, n.level.value, n.mass, n.created_turn) for n in cd.all_nodes()]


def _run(manager: SpecManager) -> tuple[CorrelationDiagram, list[SpecTurnReport]]:
    cd = CorrelationDiagram()
    reports = [
        manager.update(cd, T1_USER, T1_ASSISTANT, turn=0),
        manager.update(cd, T2_USER, "", turn=1),
    ]
    return cd, reports


# -- defaults ---------------------------------------------------------------------


def test_defaults_reproduce_previous_behaviour() -> None:
    cd_a, reps_a = _run(SpecManager(make_spec_fake_judge(), config=_cfg()))
    cd_b, reps_b = _run(
        SpecManager(make_spec_fake_judge(), config=_cfg(), node_fn=None, keep_fn=None, similarity=None)
    )
    assert _records(cd_a) == _records(cd_b)
    assert len(cd_a) == 6  # 1 chunk = 1 node, nothing vanishes with the fake judge
    assert [r.dropped for r in reps_a] == [0, 0]
    assert [r.calls for r in reps_a] == [r.calls for r in reps_b]
    assert K_NODE in reps_a[0].calls  # the text prompt path was used


def test_report_has_dropped_field_default_zero() -> None:
    assert SpecTurnReport().dropped == 0


# -- node_fn ------------------------------------------------------------------------


class _RecordingJudge:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, max_tokens: int) -> str:
        self.prompts.append(prompt)
        return "no"


def test_node_fn_replaces_the_text_prompt() -> None:
    seen: list[str] = []

    def node_fn(text: str):
        seen.append(text)
        return "N:" + text[:30], {"comprehensiveness": 10, "independence": 10, "detail": 90}

    judge = _RecordingJudge()
    manager = SpecManager(judge, config=_cfg(), node_fn=node_fn)
    cd = CorrelationDiagram()
    report = manager.update(cd, T1_USER, "", turn=0)
    assert report.chunks == 2 and report.nodes == 2
    assert len(seen) == 2
    assert all(n.text.startswith("N:") for n in cd.all_nodes())
    assert report.node_fallback == 0
    # No Q_NODE prompt reached the judge; only same/belongs yes/no prompts did.
    assert not any("JSON object with the keys summary" in p for p in judge.prompts)
    assert K_NODE not in report.calls


def test_node_fn_none_is_a_node_fallback_like_today() -> None:
    manager = SpecManager(_RecordingJudge(), config=_cfg(), node_fn=lambda text: None)
    cd = CorrelationDiagram()
    report = manager.update(cd, T1_USER, "", turn=0)
    assert report.node_fallback == 2
    assert manager.node_fallback == 2
    # fallback node text = text[:80], all scores zero -> satellite
    assert all(n.text == n.text[:80] for n in cd.all_nodes())


def test_node_fn_is_asked_again_for_a_repeated_text() -> None:
    """Astra round 3 item 3: with a node_fn the K_NODE cache is bypassed, so a
    repeated chunk is asked again (fresh Jev answers).  This replaces the
    earlier test that asserted the cached behaviour."""
    calls: list[str] = []

    def node_fn(text: str):
        calls.append(text)
        return text[:20], {"comprehensiveness": 10, "independence": 10, "detail": 90}

    manager = SpecManager(_RecordingJudge(), config=_cfg(), node_fn=node_fn)
    cd = CorrelationDiagram()
    manager.update(cd, T1_USER, "", turn=0)
    manager.update(cd, T1_USER, "", turn=1)
    assert len(calls) == 4  # second turn asked node_fn again
    # (The report's cache_hits still counts the same/belongs yes/no prompts of
    # the text judge, which go through the cache; K_NODE entries are never written.)
    assert all(manager.cache.get(K_NODE, text, "") is None for text in calls)


# -- keep_fn -----------------------------------------------------------------------


def _keep_fn(text: str) -> bool:
    return "Spark" not in text and "RCON" not in text


def test_keep_fn_drops_chunks_and_reports_per_turn() -> None:
    manager = SpecManager(make_spec_fake_judge(), config=_cfg(), keep_fn=_keep_fn)
    cd = CorrelationDiagram()
    r1 = manager.update(cd, T1_USER, T1_ASSISTANT, turn=0)
    assert r1.chunks == 4
    assert r1.dropped == 2
    assert r1.nodes == 2
    r2 = manager.update(cd, T2_USER, "", turn=1)
    assert r2.dropped == 0  # per turn, not cumulative
    assert r2.nodes == 2
    assert manager.totals["dropped"] == 2
    assert len(cd) == 4
    assert not any("Spark" in n.text or "RCON" in n.text for n in cd.all_nodes())
    # quality_totals() keeps its documented shape.
    assert set(manager.quality_totals()) == {"unparsed", "defaulted", "node_fallback", "vanished", "attached"}


def test_keep_fn_on_the_thread_pool_path() -> None:
    seq = SpecManager(make_spec_fake_judge(), config=_cfg(max_workers=1), keep_fn=_keep_fn)
    par = SpecManager(make_spec_fake_judge(), config=_cfg(max_workers=3), keep_fn=_keep_fn)
    cd_s, cd_p = CorrelationDiagram(), CorrelationDiagram()
    rs = seq.update(cd_s, T1_USER, T1_ASSISTANT, turn=0)
    rp = par.update(cd_p, T1_USER, T1_ASSISTANT, turn=0)
    assert rp.dropped == rs.dropped == 2
    assert _records(cd_p) == _records(cd_s)


# -- similarity ------------------------------------------------------------------


class _Counters:
    def __init__(self) -> None:
        self.calls: dict[str, int] = {}
        self.unparsed: dict[str, int] = {}
        self.defaulted: dict[str, int] = {}
        self.retried: dict[str, int] = {}


class _FakeSimilarity:
    """Says 'same' never and 'belongs' to the first candidate; counts by kind."""

    def __init__(self) -> None:
        self.runner = _Counters()
        self.asked: list[tuple[str, str, int]] = []

    def most_similar(self, query: str, candidates: list[str], kind: str = K_SAME) -> tuple[int, float]:
        self.asked.append((kind, query, len(candidates)))
        self.runner.calls[kind] = self.runner.calls.get(kind, 0) + 1
        if not candidates:
            return -1, 0.0
        if kind == K_BELONGS:
            return 0, 1.0
        return -1, 0.0


def test_injected_similarity_is_used_and_accounted() -> None:
    sim = _FakeSimilarity()
    manager = SpecManager(make_spec_fake_judge(), config=_cfg(), similarity=sim)
    assert manager.similarity is sim
    cd = CorrelationDiagram()
    r1 = manager.update(cd, T1_USER, T1_ASSISTANT, turn=0)
    kinds = {k for k, _, _ in sim.asked}
    assert K_BELONGS in kinds
    # The injected object's counters appear in the turn report and the totals.
    assert r1.calls.get(K_BELONGS) == sim.runner.calls[K_BELONGS]
    assert manager.call_totals().get(K_BELONGS) == sim.runner.calls[K_BELONGS]
    # belongs -> first candidate: the turn's planet sits under the turn's sun
    # (within-turn linking; `attached` counts merge-time attaches only).
    assert len(cd.suns) >= 1 and any(se.planets for se in cd.suns)
    # Turn 2 delta is per turn, not cumulative.
    before = dict(sim.runner.calls)
    r2 = manager.update(cd, T2_USER, "", turn=1)
    for kind, total in sim.runner.calls.items():
        assert r2.calls.get(kind, 0) == total - before.get(kind, 0)
    assert K_NODE in r2.calls  # the manager's own runner is still accounted too
    assert manager.quality_totals()["defaulted"] == 0


# -- item 3 (Astra round 3): node_fn bypasses the K_NODE answer cache -------------


def test_node_fn_bypasses_the_node_cache_and_uses_fresh_answers() -> None:
    """Same text twice with a node_fn whose scores change: the second node
    reflects the second answer (the default text-prompt path keeps caching)."""
    answers = iter(
        [
            ("first", {"comprehensiveness": 90, "independence": 10, "detail": 10}),  # sun
            ("second", {"comprehensiveness": 10, "independence": 10, "detail": 90}),  # satellite
        ]
    )
    calls: list[str] = []

    def node_fn(text: str):
        calls.append(text)
        return next(answers)

    manager = SpecManager(_RecordingJudge(), config=_cfg(), node_fn=node_fn)
    n1 = manager.node_for_text("the same chunk", turn=0)
    n2 = manager.node_for_text("the same chunk", turn=1)
    assert calls == ["the same chunk", "the same chunk"]
    assert (n1.text, n1.level.value) == ("first", "sun")
    assert (n2.text, n2.level.value) == ("second", "satellite")
    assert manager.cache.hits == 0
    assert manager.cache.get(K_NODE, "the same chunk", "") is None


def test_default_text_prompt_path_still_caches() -> None:
    manager = SpecManager(make_spec_fake_judge(), config=_cfg())
    manager.node_for_text("cached chunk", turn=0)
    manager.node_for_text("cached chunk", turn=1)
    assert manager.cache.hits == 1

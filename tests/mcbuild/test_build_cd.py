"""build_cd.build() end to end on two synthetic round trips with fake Jev + summarizer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.mcbuild_bench import build_cd
from benchmark.mcbuild_bench.build_cd import (
    build,
    make_manager,
    payload_of,
    resume_state,
    round_trip_texts,
    session_sha256,
    summarizer_totals,
    with_run_id,
)
from benchmark.mcbuild_bench.errors import JevStop
from benchmark.mcbuild_bench.jev_judge import JevCounters
from management.harness.spec_manager import SpecTurnReport
from models.correlation_diagram import CorrelationDiagram
from tests.mcbuild._fakes import TAG_PLANET, TAG_SUN, FakeJev, FakeSummarizer

SESSION = {
    "round_trips": [
        {
            "idx": 0,
            "ts_start": "2026-09-12T15:20:20.511Z",
            "ts_end": "2026-09-12T15:24:47.946Z",
            "human": f"{TAG_SUN} Hackathon planning for the Minecraft build agent.",
            "events": [
                {"kind": "text", "text": f"{TAG_PLANET} Hackathon server runs Paper 1.21.8."},
                {"kind": "tool_use", "text": "Read C:\\Users\\user\\Downloads\\hackathon_plan.md"},
                {"kind": "tool_result", "text": "line 1 of a file listing\nline 2 of a file listing"},
                {"kind": "text", "text": "Hackathon RCON port is 25575."},
            ],
        },
        {
            "idx": 1,
            "ts_start": "2026-09-12T15:30:00.000Z",
            "ts_end": "2026-09-12T15:31:00.000Z",
            "human": "Hackathon scale is two to one.",
            "events": [
                {"kind": "harness_note", "text": "Hackathon build agent uses codex exec."},
            ],
        },
    ],
    "dropped_human": [],
    "stats": {},
}


def test_round_trip_texts_prefix_each_event_with_its_kind() -> None:
    user, assistant = round_trip_texts(SESSION["round_trips"][0])
    assert user == SESSION["round_trips"][0]["human"]
    assert assistant.startswith(f"[text]\n{TAG_PLANET} Hackathon server runs Paper 1.21.8.\n[tool_use]\nRead ")
    assert assistant == (
        f"[text]\n{TAG_PLANET} Hackathon server runs Paper 1.21.8.\n"
        "[tool_use]\nRead C:\\Users\\user\\Downloads\\hackathon_plan.md\n"
        "[tool_result]\nline 1 of a file listing\nline 2 of a file listing\n"
        "[text]\nHackathon RCON port is 25575."
    )


def test_build_two_round_trips_end_to_end() -> None:
    jev, summ = FakeJev(), FakeSummarizer()
    manager = make_manager(jev, summ)
    payload = build(SESSION, manager)

    # JSON round trip and record shape (build_cd_offline._node_record).
    text = json.dumps(payload, ensure_ascii=False)
    parsed = json.loads(text)
    assert set(parsed) == {"nodes", "summary"}
    assert parsed["nodes"], "the diagram must not be empty"
    for rec in parsed["nodes"]:
        assert set(rec) == {"node_id", "text", "level", "mass", "parent_id", "created_turn"}
        assert len(rec["text"]) <= 120
        assert TAG_SUN not in rec["text"] and TAG_PLANET not in rec["text"]

    s = parsed["summary"]
    assert set(s) == {"sun", "planet", "satellite", "total", "turns", "dropped_chunks",
                      "harness_calls", "harness_quality"}
    assert s["turns"] == 2
    assert s["total"] == len(parsed["nodes"]) == s["sun"] + s["planet"] + s["satellite"]
    assert s["sun"] >= 1 and s["planet"] >= 1 and s["satellite"] >= 1
    # The tool_result chunk (and only it) fails `keep`.
    assert s["dropped_chunks"] == 1
    assert s["harness_quality"]["defaulted"] == 0
    assert s["harness_quality"]["unparsed"] == 0
    assert s["harness_quality"]["node_fallback"] == 0
    # H1: ONE Jev request per chunk ({keep, 3 axes}), counted under "node";
    # a kept chunk = 1 node, nothing vanished, so node calls = nodes + dropped.
    assert s["harness_calls"].get("node") == s["total"] + s["dropped_chunks"]
    assert "keep" not in manager.totals
    # the summarizer ran for kept chunks only
    assert len(summ.calls) == s["total"]

    # 0062 after normalize: planet mass = satellite count (floor 0.0).
    by_id = {r["node_id"]: r for r in parsed["nodes"]}
    for rec in parsed["nodes"]:
        if rec["level"] == "planet":
            n_sats = sum(1 for r in parsed["nodes"] if r["level"] == "satellite" and r["parent_id"] == rec["node_id"])
            assert rec["mass"] == float(n_sats)
        if rec["level"] == "satellite":
            assert by_id[rec["parent_id"]]["level"] == "planet"
    assert any(r["mass"] >= 1.0 for r in parsed["nodes"] if r["level"] == "planet")
    assert {r["created_turn"] for r in parsed["nodes"]} == {0, 1}
    # Every Jev request carried the raw chunk (tags intact), never a summary.
    assert all(isinstance(state, str) and state for state, _ in jev.requests)


def test_max_round_trips_limits_the_loop() -> None:
    manager = make_manager(FakeJev(), FakeSummarizer())
    payload = build(SESSION, manager, max_round_trips=1)
    assert payload["summary"]["turns"] == 1
    assert {r["created_turn"] for r in payload["nodes"]} == {0}


def test_jevstop_propagates_with_partial_state() -> None:
    class StopOnSecond(FakeJev):
        def ask(self, state, questions):
            if "scale" in state:
                raise JevStop("429 after 3 attempts")
            return super().ask(state, questions)

    counters = JevCounters()
    manager = make_manager(StopOnSecond(), FakeSummarizer(), counters)
    cd = CorrelationDiagram()
    reports: list[SpecTurnReport] = []
    with pytest.raises(JevStop, match="429"):
        build(SESSION, manager, cd=cd, reports=reports)
    assert len(reports) == 1 and len(cd) > 0  # round trip 0 completed
    partial = payload_of(cd, reports, manager, stopped={"reason": "JevStop: 429 after 3 attempts"})
    assert partial["stopped"] == {"reason": "JevStop: 429 after 3 attempts"}
    assert partial["summary"]["turns"] == 1
    json.dumps(partial)


def test_summarizer_totals_from_accounting_file(tmp_path) -> None:
    p = tmp_path / "summarizer_calls.jsonl"
    p.write_text(
        '{"ts": 1, "prompt_tokens": 50, "completion_tokens": 20, "latency_ms": 3.0, "retried": false, "chars": 80}\n'
        '{"ts": 2, "prompt_tokens": 60, "completion_tokens": 25, "latency_ms": 3.0, "retried": true, "chars": 90}\n',
        encoding="utf-8",
    )
    assert summarizer_totals(p) == {
        "summarizer_calls": 2,
        "summarizer_prompt_tokens": 110,
        "summarizer_completion_tokens": 45,
    }
    assert summarizer_totals(tmp_path / "absent.jsonl")["summarizer_calls"] == 0


# -- F1 wiring: the similarity sees the suns of the diagram being built --------------


def test_make_manager_wires_sun_texts_of_the_built_cd() -> None:
    cd = CorrelationDiagram()
    manager = make_manager(FakeJev(), FakeSummarizer(), cd=cd)
    assert manager.similarity._sun_texts_fn() == set()
    build(SESSION, manager, cd=cd)
    suns = {se.sun.text for se in cd.suns}
    assert suns and manager.similarity._sun_texts_fn() == suns
    # without cd there is no Choice routing at all (tests / offline use)
    assert make_manager(FakeJev(), FakeSummarizer()).similarity._sun_texts_fn is None


# -- F3 / F12 / F2: main() ---------------------------------------------------------------


class _NoGpuSampler:
    def __init__(self, path) -> None:
        self.path = path

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


def _wire_main(monkeypatch, tmp_path: Path, jev, summarizer) -> list[str]:
    """Point main()'s lazily imported clients at fakes; return the base argv."""
    from benchmark.mcbuild_bench import gpu_sampler, jev_client, summarizer_client

    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-test")
    monkeypatch.setattr(jev_client, "JevClient", lambda **kw: jev)
    monkeypatch.setattr(summarizer_client, "SummarizerClient", lambda **kw: summarizer)
    monkeypatch.setattr(gpu_sampler, "GpuSampler", _NoGpuSampler)
    session_path = tmp_path / "session.json"
    session_path.write_text(json.dumps(SESSION), encoding="utf-8")
    return [
        "--session", str(session_path),
        "--out", str(tmp_path / "cd.json"),
        "--jev-accounting", str(tmp_path / "jev_calls.jsonl"),
        "--summarizer-accounting", str(tmp_path / "summarizer_calls.jsonl"),
        "--gpu-csv", str(tmp_path / "gpu.csv"),
        # H22 (c): the default exclusion (round trip 36) is CHECKED against the
        # session; these two-round-trip fixtures opt out explicitly.
        "--exclude-rt", "none",
        # item 1 (Astra round 3): both budgets are required; generous defaults here
        # (kept LAST: the budget tests slice them off with argv[:-4])
        "--max-jev-requests", "100000",
        "--max-jev-input-tokens", "100000000",
    ]


def _node_keys(payload: dict) -> list[tuple]:
    """(text, level, mass, parent text): node ids are random uuid slices."""
    by_id = {r["node_id"]: r for r in payload["nodes"]}
    return [
        (r["text"], r["level"], r["mass"], by_id[r["parent_id"]]["text"] if r["parent_id"] else None)
        for r in payload["nodes"]
    ]


def test_null_noul_is_jevstop_and_main_writes_the_partial(monkeypatch, tmp_path: Path) -> None:
    jev = FakeJev(override={"keep": lambda s, q: {"type": "noul", "noul": None}})
    with pytest.raises(JevStop):  # not TypeError
        build(SESSION, make_manager(jev, FakeSummarizer()))

    argv = _wire_main(monkeypatch, tmp_path, jev, FakeSummarizer())
    rc = build_cd.main(argv + ["--run-id", "run1"])
    assert rc == 1
    out = json.loads((tmp_path / "cd.json").read_text(encoding="utf-8"))
    assert out["stopped"]["type"] == "JevStop"
    assert out["stopped"]["reason"].startswith("JevStop(")
    assert out["summary"]["turns"] == 0
    m = out["manifest"]
    assert m["session_file_sha256"] == session_sha256(tmp_path / "session.json")
    assert m["session_sha256"] != m["session_file_sha256"]  # H22 (c): corpus hash
    assert m["exclude_rt"] == [] and m["n_round_trips"] == 2  # corpus size, not turns processed
    assert m["run_id"] == "run1"
    assert m["jev_accounting"].endswith("jev_calls.run1.jsonl")
    assert m["summarizer_accounting"].endswith("summarizer_calls.run1.jsonl")
    assert m["gpu_csv"].endswith("gpu.run1.csv")  # item 10: tagged like the accounting files
    assert "resumed_from" not in m and "prior_manifest" not in m


def test_main_writes_the_partial_on_any_exception_then_reraises(monkeypatch, tmp_path: Path) -> None:
    class Boom(FakeSummarizer):
        def summarize(self, excerpt: str) -> str:
            if "scale" in excerpt:
                raise RuntimeError("boom")
            return super().summarize(excerpt)

    argv = _wire_main(monkeypatch, tmp_path, FakeJev(), Boom())
    with pytest.raises(RuntimeError, match="boom"):
        build_cd.main(argv)
    out = json.loads((tmp_path / "cd.json").read_text(encoding="utf-8"))
    assert out["stopped"] == {"reason": "RuntimeError('boom')", "type": "RuntimeError"}
    assert out["summary"]["turns"] == 1 and out["nodes"]
    assert "manifest" in out


def test_with_run_id_goes_before_the_extension() -> None:
    assert with_run_id(Path("data/jev_calls.jsonl"), "20260917T101010Z") == Path(
        "data/jev_calls.20260917T101010Z.jsonl"
    )


def test_resume_equals_an_uninterrupted_build(monkeypatch, tmp_path: Path) -> None:
    class StopOnSecond(FakeJev):
        def ask(self, state, questions):
            if "scale" in state:
                raise JevStop("429 after 3 attempts")
            return super().ask(state, questions)

    argv = _wire_main(monkeypatch, tmp_path, StopOnSecond(), FakeSummarizer())
    assert build_cd.main(argv + ["--run-id", "r1"]) == 1
    partial_path = tmp_path / "cd.json"
    partial = json.loads(partial_path.read_text(encoding="utf-8"))
    assert partial["summary"]["turns"] == 1 and partial["stopped"]["type"] == "JevStop"
    resumed_from = tmp_path / "cd.partial.json"
    partial_path.rename(resumed_from)

    # the resumed run uses a healthy Jev and continues from round trip idx 1
    jev2 = FakeJev()
    argv = _wire_main(monkeypatch, tmp_path, jev2, FakeSummarizer())
    assert build_cd.main(argv + ["--run-id", "r2", "--resume", str(resumed_from)]) == 0
    final = json.loads(partial_path.read_text(encoding="utf-8"))
    assert "stopped" not in final
    assert final["summary"]["turns"] == 2
    assert {r["created_turn"] for r in final["nodes"]} == {0, 1}
    # only round trip 1 was processed by the resumed run
    assert all("scale" in s or "codex" in s or "\nB: " in s for s, _ in jev2.requests)
    m = final["manifest"]
    assert m["resumed_from"] == str(resumed_from)
    assert m["prior_manifest"] == partial["manifest"]
    assert m["run_id"] == "r2" and m["jev_accounting"].endswith("jev_calls.r2.jsonl")
    assert (tmp_path / "jev_calls.r1.jsonl").exists() is False  # fakes write no accounting
    assert m["session_sha256"] == partial["manifest"]["session_sha256"]

    cd = CorrelationDiagram()
    full = build(SESSION, make_manager(FakeJev(), FakeSummarizer(), cd=cd), cd=cd)
    assert _node_keys(final) == _node_keys(full)
    assert final["summary"]["dropped_chunks"] == full["summary"]["dropped_chunks"]


def test_resume_refuses_a_different_session(monkeypatch, tmp_path: Path, capsys) -> None:
    argv = _wire_main(monkeypatch, tmp_path, FakeJev(), FakeSummarizer())
    assert build_cd.main(argv + ["--max-round-trips", "1"]) == 0
    out_path = tmp_path / "cd.json"
    partial = json.loads(out_path.read_text(encoding="utf-8"))
    assert partial["summary"]["turns"] == 1
    prior = out_path.with_name("prior.json")
    out_path.rename(prior)
    # a different session file (one more space in the human text)
    other = json.loads(json.dumps(SESSION))
    other["round_trips"][0]["human"] += " "
    session_path = tmp_path / "session.json"
    session_path.write_text(json.dumps(other), encoding="utf-8")
    rc = build_cd.main(argv + ["--resume", str(prior)])
    assert rc == 2
    assert "session_sha256" in capsys.readouterr().err
    assert not out_path.exists()
    with pytest.raises(ValueError, match="session_sha256"):
        resume_state(partial, "deadbeef")


def test_resume_state_reads_nodes_and_turns() -> None:
    cd = CorrelationDiagram()
    payload = build(SESSION, make_manager(FakeJev(), FakeSummarizer(), cd=cd), cd=cd)
    payload["manifest"] = {"session_sha256": "abc"}
    loaded, start_idx, manifest = resume_state(payload, "abc")
    assert start_idx == 2 and manifest == {"session_sha256": "abc"}
    assert [n.text for n in loaded.all_nodes()] == [n.text for n in cd.all_nodes()]
    assert [n.node_id for n in loaded.all_nodes()] == [n.node_id for n in cd.all_nodes()]
    with pytest.raises(ValueError, match="session_sha256"):
        resume_state({"nodes": [], "summary": {"turns": 0}}, "abc")


# -- item 3: transactional round trips ---------------------------------------------------


class StopOnSecondNoulOfRoundTrip1(FakeJev):
    """Raises JevStop on the SECOND noul question asked for round trip 1."""

    def __init__(self) -> None:
        super().__init__()
        self.rt1_nouls = 0

    def ask(self, state, questions):
        in_rt1 = "scale" in state or "codex" in state
        if in_rt1:
            for q in questions.values():
                if q.get("type") == "noul":
                    self.rt1_nouls += 1
                    if self.rt1_nouls == 2:
                        raise JevStop("429 after 3 attempts")
        return super().ask(state, questions)


def test_partial_after_failure_in_round_trip_1_is_exactly_round_trip_0(monkeypatch, tmp_path: Path) -> None:
    jev = StopOnSecondNoulOfRoundTrip1()
    argv = _wire_main(monkeypatch, tmp_path, jev, FakeSummarizer())
    assert build_cd.main(argv + ["--run-id", "r1"]) == 1
    assert jev.rt1_nouls == 2
    partial = json.loads((tmp_path / "cd.json").read_text(encoding="utf-8"))
    assert partial["summary"]["turns"] == 1
    assert {r["created_turn"] for r in partial["nodes"]} == {0}
    # exactly the nodes of an uninterrupted round-trip-0 build
    cd0 = CorrelationDiagram()
    rt0 = build(SESSION, make_manager(FakeJev(), FakeSummarizer(), cd=cd0), cd=cd0, max_round_trips=1)
    assert _node_keys(partial) == _node_keys(rt0)

    # resuming from it with a healthy fake equals the uninterrupted 2-round-trip build
    resumed_from = tmp_path / "cd.partial.json"
    (tmp_path / "cd.json").rename(resumed_from)
    argv = _wire_main(monkeypatch, tmp_path, FakeJev(), FakeSummarizer())
    assert build_cd.main(argv + ["--run-id", "r2", "--resume", str(resumed_from)]) == 0
    final = json.loads((tmp_path / "cd.json").read_text(encoding="utf-8"))
    assert final["summary"]["turns"] == 2 and "stopped" not in final
    cd = CorrelationDiagram()
    full = build(SESSION, make_manager(FakeJev(), FakeSummarizer(), cd=cd), cd=cd)
    assert _node_keys(final) == _node_keys(full)


def test_partial_is_the_pre_turn_snapshot_even_if_update_mutated_before_failing() -> None:
    """The exception fires AFTER manager.update mutated cd: the caller's cd must
    still hold only the completed round trip (rolled back in place)."""
    cd = CorrelationDiagram()
    manager = make_manager(FakeJev(), FakeSummarizer(), cd=cd)
    real_update = manager.update

    def late_failure(base, user_text, assistant_text, turn):
        report = real_update(base, user_text, assistant_text, turn=turn)
        if turn == 1:
            assert {n.created_turn for n in base.all_nodes()} == {0, 1}  # mutated
            raise RuntimeError("late")
        return report

    manager.update = late_failure  # type: ignore[method-assign]
    reports: list[SpecTurnReport] = []
    with pytest.raises(RuntimeError, match="late"):
        build(SESSION, manager, cd=cd, reports=reports)
    assert len(reports) == 1
    assert {n.created_turn for n in cd.all_nodes()} == {0}
    partial = payload_of(cd, reports, manager)
    assert partial["summary"]["turns"] == 1
    cd0 = CorrelationDiagram()
    rt0 = build(SESSION, make_manager(FakeJev(), FakeSummarizer(), cd=cd0), cd=cd0, max_round_trips=1)
    assert _node_keys(partial) == _node_keys(rt0)
    # the same object is still the one the sun_texts_fn closes over
    assert manager.similarity._sun_texts_fn() == {se.sun.text for se in cd.suns}


# -- H22 (a): more than 254 suns no longer stop the build (Choice batches) ------------


def test_more_than_254_suns_is_not_a_stop_any_more() -> None:
    from models.node import Node, NodeLevel

    class ManySuns:
        def update(self, base, user_text, assistant_text, turn):
            for k in range(255):
                base.add_sun(Node(text=f"topic {turn}-{k}", level=NodeLevel.SUN, mass=1.0))
            return SpecTurnReport()

        def call_totals(self):
            return {}

        def quality_totals(self):
            return {}

    cd = CorrelationDiagram()
    reports: list[SpecTurnReport] = []
    payload = build(SESSION, ManySuns(), cd=cd, reports=reports, max_round_trips=1)
    assert len(reports) == 1 and len(cd.suns) == 255
    assert payload["summary"]["sun"] == 255
    assert not hasattr(build_cd, "MAX_JEV_CHOICE_SUNS")


# -- H22 (c): the corpus excludes round trip 36 through a checked filter ----------------


def _session_with_36() -> dict:
    other = json.loads(json.dumps(SESSION))
    other["round_trips"].append({
        "idx": 36, "ts_start": "x", "ts_end": "y",
        "human": f"{TAG_SUN} Retrospective of the whole session.",
        "events": [{"kind": "text", "text": "Retrospective timeline took 85 minutes."}],
    })
    return other


def test_main_excludes_round_trip_36_by_default_and_records_it(monkeypatch, tmp_path: Path) -> None:
    from benchmark.mcbuild_bench.corpus import load_corpus

    jev = FakeJev()
    argv = _wire_main(monkeypatch, tmp_path, jev, FakeSummarizer())
    session_path = tmp_path / "session.json"
    session_path.write_text(json.dumps(_session_with_36()), encoding="utf-8")
    argv = argv[:-6] + argv[-4:]  # drop the opt-out: the default (36) applies
    assert build_cd.main(argv + ["--run-id", "x1"]) == 0
    out = json.loads((tmp_path / "cd.json").read_text(encoding="utf-8"))
    assert out["summary"]["turns"] == 2
    assert {r["created_turn"] for r in out["nodes"]} == {0, 1}
    assert not any("Retrospective" in s for s, _ in jev.requests)
    m = out["manifest"]
    corpus = load_corpus(session_path)
    assert m["exclude_rt"] == [36]
    assert m["n_round_trips"] == 2
    assert m["session_sha256"] == corpus.sha256  # sha of the FILTERED content
    assert m["session_file_sha256"] == session_sha256(session_path) != m["session_sha256"]
    # the same content without round trip 36 hashes identically (data-independent filter)
    plain = tmp_path / "plain.json"
    plain.write_text(json.dumps(SESSION), encoding="utf-8")
    assert load_corpus(plain, exclude_idx=()).sha256 == m["session_sha256"]


def test_main_refuses_a_default_exclusion_missing_from_the_session(monkeypatch, tmp_path: Path) -> None:
    jev = FakeJev()
    argv = _wire_main(monkeypatch, tmp_path, jev, FakeSummarizer())
    argv = argv[:-6] + argv[-4:]  # SESSION has round trips 0 and 1 only
    with pytest.raises(SystemExit, match="36"):
        build_cd.main(argv)
    assert jev.requests == [] and not (tmp_path / "cd.json").exists()
    with pytest.raises(SystemExit):
        build_cd.main(argv + ["--exclude-rt", "abc"])


def test_resume_skips_the_processed_count_not_the_idx() -> None:
    """F2 with an excluded middle round trip: ``summary.turns`` counts processed
    round trips, so resume skips that many POSITIONS of the filtered list."""
    session = {"round_trips": [SESSION["round_trips"][0], dict(SESSION["round_trips"][1], idx=5)]}
    cd = CorrelationDiagram()
    reports: list[SpecTurnReport] = []
    payload = build(session, make_manager(FakeJev(), FakeSummarizer(), cd=cd), cd=cd,
                    reports=reports, start_idx=1)
    assert payload["summary"]["turns"] == 1
    assert {r["created_turn"] for r in payload["nodes"]} == {5}


# -- item 10: run id -------------------------------------------------------------------------


def test_default_run_id_is_utc_timestamp_plus_six_hex() -> None:
    import re

    from benchmark.mcbuild_bench.build_cd import default_run_id

    a, b = default_run_id(), default_run_id()
    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{6}", a), a
    assert a != b


def test_reused_run_id_refuses_to_start(monkeypatch, tmp_path: Path) -> None:
    argv = _wire_main(monkeypatch, tmp_path, FakeJev(), FakeSummarizer())
    (tmp_path / "gpu.r1.csv").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="r1"):
        build_cd.main(argv + ["--run-id", "r1"])
    assert not (tmp_path / "cd.json").exists()
    (tmp_path / "gpu.r1.csv").unlink()
    (tmp_path / "jev_calls.r1.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="jev_calls.r1.jsonl"):
        build_cd.main(argv + ["--run-id", "r1"])
    assert not (tmp_path / "cd.json").exists()


# -- item G: the CD file is written atomically after EVERY completed round trip ----


def test_cd_file_exists_after_each_round_trip_and_is_atomic(monkeypatch, tmp_path: Path) -> None:
    out_path = tmp_path / "cd.json"
    seen: list[dict] = []

    class Peek(FakeSummarizer):
        """At the first chunk of round trip 1, read what is on disk, then die."""

        def summarize(self, excerpt: str) -> str:
            if "scale" in excerpt:
                seen.append(json.loads(out_path.read_text(encoding="utf-8")))
                raise RuntimeError("boom after peek")
            return super().summarize(excerpt)

    argv = _wire_main(monkeypatch, tmp_path, FakeJev(), Peek())
    with pytest.raises(RuntimeError, match="boom after peek"):
        build_cd.main(argv + ["--run-id", "g1"])
    assert len(seen) == 1
    checkpoint = seen[0]
    assert checkpoint["summary"]["turns"] == 1 and checkpoint["nodes"]
    assert "stopped" not in checkpoint
    assert checkpoint["manifest"]["run_id"] == "g1"
    assert checkpoint["manifest"]["session_file_sha256"] == session_sha256(tmp_path / "session.json")
    # no temp file left behind (tmp + os.replace)
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("cd.json")) == ["cd.json"]
    # the final partial is the exception path's output
    final = json.loads(out_path.read_text(encoding="utf-8"))
    assert final["stopped"]["type"] == "RuntimeError" and final["summary"]["turns"] == 1


def test_write_json_is_tmp_plus_replace(monkeypatch, tmp_path: Path) -> None:
    import os

    calls: list[tuple[str, str]] = []
    real_replace = os.replace

    def spy(src, dst):
        calls.append((str(src), str(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(build_cd.os, "replace", spy)
    target = tmp_path / "x.json"
    build_cd.write_json(target, {"a": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    assert len(calls) == 1 and calls[0][1] == str(target) and calls[0][0] != str(target)
    assert not Path(calls[0][0]).exists()


# -- item 1 (Astra round 3): Jev request / input-token budget guard ----------------------


def test_budgeted_jev_raises_before_the_request_that_exceeds_the_request_budget() -> None:
    from benchmark.mcbuild_bench.build_cd import BudgetedJev

    inner = FakeJev()
    jev = BudgetedJev(inner, max_requests=2, max_input_tokens=10**9)
    q = {"same": {"type": "noul", "instructions": "x"}}
    jev.ask("A: a\nB: b", q)
    jev.ask("A: a\nB: c", q)
    with pytest.raises(JevStop, match=r"^budget exceeded: requests 3 > max 2$"):
        jev.ask("A: a\nB: d", q)
    assert len(inner.requests) == 2  # the third request was never sent
    assert jev.requests_sent == 2
    assert jev.input_tokens == sum(100 + len(s) // 4 for s, _ in inner.requests)


def test_budgeted_jev_raises_before_the_request_once_input_tokens_reach_the_budget() -> None:
    from benchmark.mcbuild_bench.build_cd import BudgetedJev

    inner = FakeJev()
    q = {"same": {"type": "noul", "instructions": "x"}}
    first_cost = 100 + len("A: a\nB: b") // 4
    jev = BudgetedJev(inner, max_requests=10, max_input_tokens=first_cost)
    jev.ask("A: a\nB: b", q)
    with pytest.raises(JevStop, match=r"^budget exceeded: input_tokens %d >= max %d$" % (first_cost, first_cost)):
        jev.ask("A: a\nB: c", q)
    assert len(inner.requests) == 1
    # a budget above the tokens used so far lets the next request through
    jev2 = BudgetedJev(FakeJev(), max_requests=10, max_input_tokens=first_cost + 1)
    jev2.ask("A: a\nB: b", q)
    jev2.ask("A: a\nB: c", q)
    with pytest.raises(ValueError):
        BudgetedJev(FakeJev(), max_requests=0, max_input_tokens=1)


def test_main_requires_both_budgets(monkeypatch, tmp_path: Path) -> None:
    argv = _wire_main(monkeypatch, tmp_path, FakeJev(), FakeSummarizer())
    i = argv.index("--max-jev-requests")
    without = argv[:i] + argv[i + 2:]
    with pytest.raises(SystemExit):
        build_cd.main(without)
    assert not (tmp_path / "cd.json").exists()


def test_request_budget_stops_the_run_and_writes_the_partial(monkeypatch, tmp_path: Path) -> None:
    jev = FakeJev()
    argv = _wire_main(monkeypatch, tmp_path, jev, FakeSummarizer())
    # round trip 0 alone needs more than 3 Jev requests (5 chunks -> 5 classifications)
    argv = argv[:-4] + ["--max-jev-requests", "3", "--max-jev-input-tokens", "100000000"]
    assert build_cd.main(argv + ["--run-id", "b1"]) == 1
    assert len(jev.requests) == 3
    out = json.loads((tmp_path / "cd.json").read_text(encoding="utf-8"))
    assert out["stopped"]["type"] == "JevStop"
    assert "budget exceeded: requests 4 > max 3" in out["stopped"]["reason"]
    assert out["summary"]["turns"] == 0 and out["nodes"] == []  # rolled back round trip 0
    m = out["manifest"]
    assert m["jev_budget"] == {
        "max_requests": 3,
        "max_input_tokens": 100000000,
        "requests_sent": 3,
        "input_tokens": sum(100 + len(s) // 4 for s, _ in jev.requests),
    }


def test_input_token_budget_stops_the_run_after_round_trip_0(monkeypatch, tmp_path: Path) -> None:
    jev = FakeJev()
    argv = _wire_main(monkeypatch, tmp_path, jev, FakeSummarizer())
    # First learn what round trip 0 costs with an unlimited budget ...
    probe = FakeJev()
    cd0 = CorrelationDiagram()
    build(SESSION, make_manager(probe, FakeSummarizer(), cd=cd0), cd=cd0, max_round_trips=1)
    rt0_tokens = sum(100 + len(s) // 4 for s, _ in probe.requests)
    # ... then a budget that is exactly reached at the end of round trip 0:
    # the first request of round trip 1 is refused, round trip 0 is kept.
    argv = argv[:-4] + ["--max-jev-requests", "100000", "--max-jev-input-tokens", str(rt0_tokens)]
    assert build_cd.main(argv + ["--run-id", "b2"]) == 1
    out = json.loads((tmp_path / "cd.json").read_text(encoding="utf-8"))
    assert out["summary"]["turns"] == 1 and {r["created_turn"] for r in out["nodes"]} == {0}
    assert "budget exceeded: input_tokens %d >= max %d" % (rt0_tokens, rt0_tokens) in out["stopped"]["reason"]
    assert out["manifest"]["jev_budget"]["requests_sent"] == len(probe.requests)
    assert len(jev.requests) == len(probe.requests)


def test_projection_cli_on_the_real_session(capsys) -> None:
    """--project: chunk counts per round trip + worst-case totals, no Jev, no other args."""
    from benchmark.mcbuild_bench.jev_judge import project_requests

    session_path = Path("benchmark/mcbuild_bench/data/session_redacted.json")
    rc = build_cd.main(["--project", "--session", str(session_path)])
    assert rc == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    rt_lines = [ln for ln in lines if ln.startswith("rt ")]
    session = json.loads(session_path.read_text(encoding="utf-8"))
    # H22 (c): the projection runs on the corpus = session minus round trip 36
    assert len(session["round_trips"]) == 37 and len(rt_lines) == 36
    assert "exclude_rt=[36]" in out
    chunks = [int(ln.split("chunks=")[1].split()[0]) for ln in rt_lines]
    assert sum(chunks) == 3388 and max(chunks) == 423
    # the totals line is the sum of project_requests over the round trips with
    # the CD bounded by the cumulative chunk count (no sun cap: Choice batches, H22 (a))
    expected, seen = 0, 0
    for n in chunks:
        expected += project_requests(n, seen, seen, n_satellites=seen)
        seen += n
    assert "classification floor (1 request per chunk): 3388" in out
    assert "worst-case total requests: %d" % expected in out
    assert "chunk_max_chars=400" in out


# -- item 2 (Astra round 3): capacity exhaustion raises, the round trip rolls back --------


def test_capacity_exceeded_raises_and_rolls_back(monkeypatch) -> None:
    import models.correlation_diagram as cd_module

    real_get = cd_module.get

    def tiny(section, key, default=None):
        if (section, key) == ("graph", "max_satellite_nodes_per_planet"):
            return 2
        return real_get(section, key, default)

    monkeypatch.setattr(cd_module, "get", tiny)
    session = {
        "round_trips": [
            {
                "idx": 0,
                "human": f"{TAG_SUN} Hackathon overview.",
                "events": [
                    {"kind": "text", "text": f"{TAG_PLANET} Hackathon server configuration."},
                    # H23: the satellites' first capitalised words differ (Port / Seed /
                    # Version) so FakeJev's `same` Choice (topic-word rule, keys mN) answers
                    # none and all three reach the planet; "Hackathon" keeps them attached.
                    {"kind": "text", "text": "Port 25575 for the Hackathon."},
                    {"kind": "text", "text": "Seed 42 for the Hackathon."},
                    {"kind": "text", "text": "Version 1.21.8 for the Hackathon."},
                ],
            }
        ]
    }
    cd = CorrelationDiagram()
    assert cd.capacities() == {"max_sun_nodes": 10000, "max_planet_nodes_per_sun": 1000,
                               "max_satellite_nodes_per_planet": 2}
    manager = make_manager(FakeJev(), FakeSummarizer(), cd=cd)
    reports: list[SpecTurnReport] = []
    with pytest.raises(RuntimeError, match="capacity exceeded"):
        build(session, manager, cd=cd, reports=reports)
    assert reports == [] and len(cd) == 0  # transaction rolled back
    # the same session fits when the cap is the config default (3 satellites)
    monkeypatch.setattr(cd_module, "get", real_get)
    cd2 = CorrelationDiagram()
    payload = build(session, make_manager(FakeJev(), FakeSummarizer(), cd=cd2), cd=cd2)
    assert payload["summary"]["satellite"] == 3


def test_graph_merger_never_drops_silently(monkeypatch) -> None:
    import models.correlation_diagram as cd_module
    from management.graph_merger import GraphMerger
    from models.node import Node, NodeLevel

    real_get = cd_module.get
    capped = {("graph", "max_planet_nodes_per_sun"): 1, ("graph", "max_satellite_nodes_per_planet"): 1}
    monkeypatch.setattr(cd_module, "get", lambda s, k, d=None: capped.get((s, k), real_get(s, k, d)))
    base = CorrelationDiagram()
    sun = Node(text="Topic", level=NodeLevel.SUN, mass=1.0)
    assert base.add_sun(sun)
    planet = Node(text="Topic first", level=NodeLevel.PLANET, mass=0.0)
    assert base.add_planet(planet, sun.node_id)
    assert base.add_satellite(Node(text="Topic first detail", level=NodeLevel.SATELLITE, mass=0.0), planet.node_id)
    merger = GraphMerger(similarity_fn=lambda q, c: (-1, 0.0), attach_fn=lambda q, c: (0, 1.0))
    # 0057: attach under the (full) sun -> add_planet False -> raise, never drop
    with pytest.raises(RuntimeError, match="capacity exceeded: could not add planet 'Topic second'"):
        merger.merge_case2_planet(base, Node(text="Topic second", level=NodeLevel.PLANET, mass=0.0), [])
    # 0060: attach under the (full) planet -> add_satellite False -> raise
    with pytest.raises(RuntimeError, match="capacity exceeded: could not add satellite"):
        merger.merge_case3_satellite(base, Node(text="Topic detail", level=NodeLevel.SATELLITE, mass=0.0))
    assert len(base) == 3  # nothing was added on either path


def test_manifest_records_the_effective_capacities(monkeypatch, tmp_path: Path) -> None:
    argv = _wire_main(monkeypatch, tmp_path, FakeJev(), FakeSummarizer())
    assert build_cd.main(argv + ["--max-round-trips", "1"]) == 0
    out = json.loads((tmp_path / "cd.json").read_text(encoding="utf-8"))
    assert out["manifest"]["capacities"] == {
        "max_sun_nodes": 10000, "max_planet_nodes_per_sun": 1000, "max_satellite_nodes_per_planet": 1000,
    }


# -- item 4 (Astra round 3): manifest totals come from THIS run's accounting JSONL ---------


def test_jev_totals_from_accounting_file(tmp_path: Path) -> None:
    from benchmark.mcbuild_bench.build_cd import jev_totals

    p = tmp_path / "jev_calls.jsonl"
    p.write_text(
        '{"ts": 1, "input_tokens": 312, "output_tokens": 48, "http_status": 200, "answers": {}}\n'
        '{"ts": 2, "input_tokens": null, "output_tokens": null, "http_status": 429, "answers": null}\n'
        '\n'
        '{"ts": 3, "input_tokens": 100, "output_tokens": 2, "http_status": 200, "answers": null}\n',
        encoding="utf-8",
    )
    assert jev_totals(p) == {
        "jev_requests": 3,  # every line (the 429 attempt included), blank lines ignored
        "jev_input_tokens_total": 412,
        "jev_output_tokens_total": 50,
        "jev_cost_usd": 412 * 0.042 / 1e6,
    }
    assert jev_totals(tmp_path / "absent.jsonl") == {
        "jev_requests": 0, "jev_input_tokens_total": 0, "jev_output_tokens_total": 0, "jev_cost_usd": 0.0,
    }


def _wire_real_jev(monkeypatch, tmp_path: Path, transport, summarizer) -> list[str]:
    """main() with the REAL JevClient over ``transport`` and a fake summarizer."""
    from benchmark.mcbuild_bench import jev_client

    real_client = jev_client.JevClient  # captured BEFORE _wire_main replaces the name
    argv = _wire_main(monkeypatch, tmp_path, None, summarizer)
    monkeypatch.setattr(
        jev_client, "JevClient",
        lambda **kw: real_client(transport=transport, sleep_fn=lambda s: None, **kw),
    )
    return argv


def test_missing_answer_id_is_jevstop_and_the_partial_manifest_counts_its_tokens(monkeypatch, tmp_path: Path) -> None:
    def transport(url, headers, body, timeout):
        return 200, json.dumps({"model": "jev-latest", "answers": {},
                                "usage": {"input_tokens": 312, "output_tokens": 48}}).encode("utf-8")

    argv = _wire_real_jev(monkeypatch, tmp_path, transport, FakeSummarizer())
    assert build_cd.main(argv + ["--run-id", "t1"]) == 1
    out = json.loads((tmp_path / "cd.json").read_text(encoding="utf-8"))
    assert out["stopped"]["type"] == "JevStop" and "missing answers" in out["stopped"]["reason"]
    m = out["manifest"]
    assert m["jev_requests"] == 1
    assert m["jev_input_tokens_total"] == 312 and m["jev_output_tokens_total"] == 48
    assert m["jev_cost_usd"] == pytest.approx(312 * 0.042 / 1e6)
    lines = (tmp_path / "jev_calls.t1.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["input_tokens"] == 312
    # the budget wrapper saw NO completed request: totals must not come from it
    assert m["jev_budget"]["requests_sent"] == 1 and m["jev_budget"]["input_tokens"] == 0


def test_resume_reconciles_the_prior_manifest_from_its_accounting_files(monkeypatch, tmp_path: Path) -> None:
    argv = _wire_main(monkeypatch, tmp_path, FakeJev(), FakeSummarizer())
    assert build_cd.main(argv + ["--run-id", "r1", "--max-round-trips", "1"]) == 0
    prior_path = tmp_path / "cd.partial.json"
    (tmp_path / "cd.json").rename(prior_path)
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    assert prior["manifest"]["jev_requests"] == 0  # fakes write no accounting
    # Write the prior run's accounting files AFTER the fact: reconciliation must re-sum them.
    Path(prior["manifest"]["jev_accounting"]).write_text(
        '{"ts": 1, "input_tokens": 200, "output_tokens": 10, "http_status": 200, "answers": {}}\n'
        '{"ts": 2, "input_tokens": 100, "output_tokens": 5, "http_status": 200, "answers": {}}\n',
        encoding="utf-8",
    )
    Path(prior["manifest"]["summarizer_accounting"]).write_text(
        '{"ts": 1, "prompt_tokens": 50, "completion_tokens": 20, "latency_ms": 3.0, "retried": false, "chars": 80}\n',
        encoding="utf-8",
    )
    argv = _wire_main(monkeypatch, tmp_path, FakeJev(), FakeSummarizer())
    assert build_cd.main(argv + ["--run-id", "r2", "--resume", str(prior_path)]) == 0
    final = json.loads((tmp_path / "cd.json").read_text(encoding="utf-8"))
    m = final["manifest"]
    assert m["prior_manifest_reconciled"] is True
    pm = m["prior_manifest"]
    assert pm["jev_requests"] == 2 and pm["jev_input_tokens_total"] == 300 and pm["jev_output_tokens_total"] == 15
    assert pm["jev_cost_usd"] == pytest.approx(300 * 0.042 / 1e6)
    assert pm["summarizer_calls"] == 1 and pm["summarizer_prompt_tokens"] == 50
    assert pm["run_id"] == "r1"  # everything else is kept

    # absent accounting files: stored values kept, flagged as not reconciled
    Path(prior["manifest"]["jev_accounting"]).unlink()
    (tmp_path / "cd.json").rename(tmp_path / "cd.final.json")
    argv = _wire_main(monkeypatch, tmp_path, FakeJev(), FakeSummarizer())
    assert build_cd.main(argv + ["--run-id", "r3", "--resume", str(prior_path)]) == 0
    m3 = json.loads((tmp_path / "cd.json").read_text(encoding="utf-8"))["manifest"]
    assert m3["prior_manifest_reconciled"] is False
    assert m3["prior_manifest"] == prior["manifest"]

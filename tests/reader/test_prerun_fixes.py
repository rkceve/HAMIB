"""Pre-GPU defect fixes F1-F9 / A1-A5 (2026-09-07 review round).

CPU only: no model is loaded, no network is touched, ``modal`` is never
installed here (a fake module is injected where the plumbing needs one).
"""

from __future__ import annotations

import json
import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from experiments import modal_spec_run as msr

# --------------------------------------------------------------------------
# A1: REPO_ROOT inside the Modal container
# --------------------------------------------------------------------------


def test_repo_path_follows_the_current_repo_root(tmp_path, monkeypatch) -> None:
    """A1: ``_enter_repo`` rebinds REPO_ROOT, and repo_path must see it.

    Modal runs this file as ``__main__`` from ``/root/modal_spec_run.py``, where
    ``Path(__file__).resolve().parents[1]`` is ``/`` -- so every repo-relative
    path became ``/benchmark/...`` and the reader phase died on a
    FileNotFoundError AFTER the weights were downloaded.
    """
    fake_repo = tmp_path / "cms-prototype"
    (fake_repo / "benchmark").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(msr, "REPO_ROOT", msr.REPO_ROOT)  # restored afterwards
    saved_path = list(sys.path)
    try:
        msr._enter_repo(str(fake_repo))
        assert msr.REPO_ROOT == Path(fake_repo)
        assert msr.repo_path("benchmark/x.json") == fake_repo / "benchmark/x.json"
        assert str(fake_repo) in sys.path
    finally:
        sys.path[:] = saved_path


def test_repo_path_resolves_under_the_real_repo_here() -> None:
    assert msr.repo_path(msr.DEFAULT_QUESTIONS).exists()
    assert msr.repo_path(msr.DEFAULT_CHAT).exists()


# --------------------------------------------------------------------------
# F2: the manager's embedding shortlist needs sentence-transformers
# --------------------------------------------------------------------------


def test_the_shortlist_decision_is_pure_and_covers_the_three_cases() -> None:
    assert msr.needs_sentence_transformers("local", 5) is True
    # build_cd_offline forces shortlist_k = 0 for a fake judge
    assert msr.needs_sentence_transformers("fake", 5) is False
    assert msr.needs_sentence_transformers("local", 0) is False


def test_require_sentence_transformers_fails_fast_when_it_is_missing() -> None:
    def boom(name: str) -> Any:
        raise ImportError("No module named %r" % name)

    with pytest.raises(SystemExit) as excinfo:
        msr.require_sentence_transformers("local", 5, import_fn=boom)
    msg = str(excinfo.value)
    assert "sentence_transformers" in msg and "shortlist_k=5" in msg


def test_require_sentence_transformers_is_a_no_op_without_the_shortlist() -> None:
    def boom(_name: str) -> Any:  # pragma: no cover - must never be called
        raise AssertionError("import attempted with shortlist_k=0")

    assert msr.require_sentence_transformers("local", 0, import_fn=boom) is False


def test_the_vllm_image_carries_sentence_transformers_and_one_torch() -> None:
    """F2: the package must be in the image, and NOT a second torch pin.

    sentence-transformers depends on torch; vLLM's image already carries the
    exact torch vLLM was built against. Pinning torch again here lets pip
    resolve one of the two away and breaks the server.
    """
    recorded: list[tuple] = []

    class FakeImage:
        def pip_install(self, *pkgs, **_kw):
            recorded.append(("pip_install", pkgs))
            return self

        def env(self, mapping):
            recorded.append(("env", mapping))
            return self

        def add_local_dir(self, *_a, **kw):
            recorded.append(("add_local_dir", kw))
            return self

    fake_modal = types.SimpleNamespace(
        Image=types.SimpleNamespace(debian_slim=lambda **_kw: FakeImage())
    )
    msr._build_images.__globals__["modal"] = fake_modal
    try:
        msr._build_images()
    finally:
        msr._build_images.__globals__["modal"] = None

    pip_calls = [pkgs for kind, pkgs in recorded if kind == "pip_install"]
    assert len(pip_calls) == 2
    vllm_pkgs, hf_pkgs = pip_calls
    assert msr.SENTENCE_TRANSFORMERS_PIN in vllm_pkgs
    assert msr.VLLM_PIN in vllm_pkgs
    assert not any(str(p).startswith("torch") for p in vllm_pkgs)

    # A3 (2026-09-07): the reader is the hybrid, so image_hf carries the
    # linear-attention kernels; causal-conv1d is deliberately absent (it
    # compiles CUDA sources and debian_slim has no nvcc).
    assert msr.FLA_PIN in hf_pkgs
    assert msr.FLA_PIN == "flash-linear-attention==0.5.2"
    assert msr.TORCH_PIN in hf_pkgs and msr.TRANSFORMERS_PIN in hf_pkgs
    assert not any("causal" in str(pkg) for pkg in hf_pkgs)
    # a backend extra would pull its own torch and resolve TORCH_PIN away
    assert "[" not in msr.FLA_PIN
    # and vLLM's image does NOT get it: only the reader loads via transformers
    assert msr.FLA_PIN not in vllm_pkgs


def test_both_images_point_hf_home_at_the_shared_cache() -> None:
    """F6: one download for the whole app instead of one per phase."""
    envs: list[dict] = []

    class FakeImage:
        def pip_install(self, *_pkgs, **_kw):
            return self

        def env(self, mapping):
            envs.append(dict(mapping))
            return self

        def add_local_dir(self, *_a, **_kw):
            return self

    fake_modal = types.SimpleNamespace(
        Image=types.SimpleNamespace(debian_slim=lambda **_kw: FakeImage())
    )
    msr._build_images.__globals__["modal"] = fake_modal
    try:
        msr._build_images()
    finally:
        msr._build_images.__globals__["modal"] = None

    assert len(envs) == 2
    assert all(e == {"HF_HOME": msr.HF_CACHE_MOUNT} for e in envs)
    assert msr.HF_CACHE_MOUNT == "/root/.cache/huggingface"
    assert msr.HF_CACHE_VOLUME_NAME != msr.VOLUME_NAME


def test_the_manager_phase_checks_the_import_before_starting_vllm(
    tmp_path, monkeypatch
) -> None:
    """F2: the gate runs BEFORE the 56 GB download, not inside the pilot."""
    import subprocess

    order: list[str] = []

    def gate(judge, shortlist_k, **_kw):
        order.append("gate")
        raise SystemExit("sentence_transformers missing")

    monkeypatch.setattr(msr, "require_sentence_transformers", gate)
    # git_sha() shells out, which would itself trip the Popen spy below
    monkeypatch.setattr(msr, "git_sha", lambda default="unknown": "abc1234")
    monkeypatch.setattr(
        subprocess, "Popen",
        lambda *_a, **_kw: order.append("vllm") or SimpleNamespace(),
    )
    with pytest.raises(SystemExit):
        msr._phase_manager_impl("fake/model", run_id="rid", vol=str(tmp_path))
    assert order == ["gate"]


# --------------------------------------------------------------------------
# F3: stale volume state
# --------------------------------------------------------------------------


def test_run_root_is_one_segment_under_the_volume() -> None:
    assert msr.run_root("/vol", "abc") == Path("/vol") / "abc"
    for bad in ("", "a/b", "..", "."):
        with pytest.raises(ValueError):
            msr.run_root("/vol", bad)


def test_run_info_records_what_produced_the_run(tmp_path) -> None:
    info = msr.build_run_info(
        "rid1", model_id="judge/m", reader_model_id="reader/m",
        arms=["floor"], w_grid=[0.0, 0.1], allow_linear_layers=True,
        versions={"transformers": "5.8.0"},
    )
    assert info["run_id"] == "rid1"
    assert info["pairing"] == "round_trip"
    assert info["arms"] == ["floor"] and info["w_grid"] == [0.0, 0.1]
    assert info["versions"] == {"transformers": "5.8.0"}
    assert isinstance(info["git_sha"], str) and info["git_sha"]

    msr.write_run_info(tmp_path, **info)
    merged = msr.write_run_info(tmp_path, phase_reader={"cells": 38})
    assert merged["run_id"] == "rid1" and merged["phase_reader"] == {"cells": 38}
    on_disk = json.loads((tmp_path / "run_info.json").read_text(encoding="utf-8"))
    assert on_disk == merged


def test_two_dry_runs_do_not_share_a_directory(tmp_path) -> None:
    """F3: the whole point -- a second run must not inherit the first's cells."""
    cells = [{"cell": "floor__w0", "arm": "floor", "w": 0.0, "inject": "none",
              "prefill_scale": 0.0, "max_questions": 1}]
    a = msr.dry_run(tmp_path, cells=cells, run_id="run-a", with_tokens=False)
    b = msr.dry_run(tmp_path, cells=cells, run_id="run-b", with_tokens=False)
    assert Path(a["out"]) != Path(b["out"])
    assert (tmp_path / "run-a" / "answers" / "floor__w0.json").exists()
    assert (tmp_path / "run-b" / "answers" / "floor__w0.json").exists()


# -- the checkpoint's pairing field ----------------------------------------


def test_checkpoint_records_the_pairing_mode_and_code_version(tmp_path) -> None:
    from benchmark.bineval import build_cd_offline as bco
    from models.correlation_diagram import CorrelationDiagram

    ckpt = tmp_path / "cd.json.ckpt"
    bco._write_checkpoint(
        ckpt, CorrelationDiagram(), 25, 0,
        pairing=bco.PAIRING_ROUND_TRIP, code_version="abc1234",
    )
    resume = json.loads(ckpt.read_text(encoding="utf-8"))["resume"]
    assert resume["pairing"] == "round_trip"
    assert resume["code_version"] == "abc1234"
    assert resume["turn"] == 25


def test_resuming_across_pairing_modes_is_refused(tmp_path) -> None:
    from benchmark.bineval import build_cd_offline as bco
    from models.correlation_diagram import CorrelationDiagram

    ckpt = tmp_path / "cd.json.ckpt"
    bco._write_checkpoint(
        ckpt, CorrelationDiagram(), 25, 0, pairing=bco.PAIRING_MESSAGE
    )
    with pytest.raises(SystemExit) as excinfo:
        bco._load_checkpoint(ckpt, expected_pairing=bco.PAIRING_ROUND_TRIP)
    msg = str(excinfo.value)
    # both values are named, so the operator can see which is which
    assert "'message'" in msg and "'round_trip'" in msg


def test_resuming_the_same_pairing_mode_is_allowed(tmp_path) -> None:
    from benchmark.bineval import build_cd_offline as bco
    from models.correlation_diagram import CorrelationDiagram

    ckpt = tmp_path / "cd.json.ckpt"
    bco._write_checkpoint(
        ckpt, CorrelationDiagram(), 25, 0, pairing=bco.PAIRING_ROUND_TRIP
    )
    state = bco._load_checkpoint(ckpt, expected_pairing=bco.PAIRING_ROUND_TRIP)
    assert state["turn"] == 25 and state["pairing"] == "round_trip"


def test_a_pre_f3_checkpoint_without_the_field_is_refused(tmp_path) -> None:
    """A checkpoint left on the volume by an older build says nothing about how
    its turns were counted, which is exactly the case F3 is about."""
    from benchmark.bineval import build_cd_offline as bco

    ckpt = tmp_path / "old.ckpt"
    ckpt.write_text(
        json.dumps({"nodes": [], "summary": {}, "resume": {"turn": 300}}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="unknown"):
        bco._load_checkpoint(ckpt, expected_pairing=bco.PAIRING_ROUND_TRIP)


def test_loading_without_an_expectation_still_works(tmp_path) -> None:
    from benchmark.bineval import build_cd_offline as bco

    ckpt = tmp_path / "old.ckpt"
    ckpt.write_text(
        json.dumps({"nodes": [], "summary": {}, "resume": {"turn": 7}}),
        encoding="utf-8",
    )
    assert bco._load_checkpoint(ckpt)["turn"] == 7


def test_the_spec_arm_maps_to_round_trip_pairing() -> None:
    from benchmark.bineval import build_cd_offline as bco

    assert bco.pairing_mode(True) == "round_trip"
    assert bco.pairing_mode(False) == "message"


# --------------------------------------------------------------------------
# F4: scoring uses --subset generated
# --------------------------------------------------------------------------


def test_every_documented_score_command_pins_the_subset() -> None:
    from benchmark.bineval import run_reader as rr

    for doc in (msr.__doc__, rr.__doc__):
        assert doc is not None
        for line in doc.splitlines():
            if "score_binary" in line or "--max-words" in line:
                pass
        assert "--subset generated" in doc, doc[:80]


def test_the_question_counts_the_defect_is_about() -> None:
    """173 generated vs 189 non-excluded: an 8-point uniform error."""
    from benchmark.bineval.run_reader import load_questions

    path = msr.repo_path(msr.DEFAULT_QUESTIONS)
    assert len(load_questions(path, subset="generated")) == 173
    assert len(load_questions(path, subset="all")) == 189


def _write_answers(root: Path, cell: str, answers: dict, meta: dict | None = None):
    (root / "answers").mkdir(parents=True, exist_ok=True)
    (root / "answers" / ("%s.json" % cell)).write_text(
        json.dumps(answers), encoding="utf-8"
    )
    if meta is not None:
        (root / "answers" / ("%s.meta.json" % cell)).write_text(
            json.dumps(meta), encoding="utf-8"
        )


def _questions_file(tmp_path: Path) -> Path:
    items = [
        {"qid": "g1", "question": "rent?", "gold_short": "800,000 yen",
         "tier1_aliases": []},
        {"qid": "g2", "question": "deposit?", "gold_short": "4,800,000 yen",
         "tier1_aliases": []},
        {"qid": "g3", "question": "seats?", "gold_short": "24 seats",
         "tier1_aliases": []},
        {"qid": "L1", "question": "legacy?", "gold_short": "legacy answer",
         "tier1_aliases": [], "legacy": True},
        {"qid": "x1", "question": "excluded?", "gold_short": "nope",
         "tier1_aliases": [], "excluded": True},
    ]
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(items), encoding="utf-8")
    return path


def test_score_run_dir_scores_every_cell_into_a_csv(tmp_path) -> None:
    from experiments.score_spec_run import score_run_dir

    qs = _questions_file(tmp_path)
    root = tmp_path / "rid1"
    _write_answers(
        root, "floor__w0",
        {"g1": "800,000 yen", "g2": "unknown", "g3": "24 seats"},
        {"cell": "floor__w0", "arm": "floor", "w": 0.0, "inject": "none",
         "prefill_scale": 0.0},
    )
    _write_answers(
        root, "cd_mass_6x__w0.1__planet",
        {"g1": "800,000 yen", "g2": "4,800,000 yen", "g3": "24 seats"},
        {"cell": "cd_mass_6x__w0.1__planet", "arm": "cd_mass_6x", "w": 0.1,
         "inject": "planet", "prefill_scale": 0.0},
    )
    res = score_run_dir(root, questions=qs)
    rows = {r["cell"]: r for r in res["rows"]}
    # the legacy and excluded questions are NOT in the denominator
    assert rows["floor__w0"]["total"] == 3
    assert rows["floor__w0"]["pass"] == 2
    assert rows["cd_mass_6x__w0.1__planet"]["pass"] == 3
    assert rows["cd_mass_6x__w0.1__planet"]["pass_rate"] == 1.0
    assert rows["cd_mass_6x__w0.1__planet"]["w"] == 0.1
    assert rows["cd_mass_6x__w0.1__planet"]["arm"] == "cd_mass_6x"

    csv_text = Path(res["csv"]).read_text(encoding="utf-8")
    assert csv_text.splitlines()[0].startswith("cell,arm,w,inject,prefill")
    assert (root / "scores" / "floor__w0.json").exists()


def test_the_exploratory_cell_is_scored_on_the_questions_it_answered(
    tmp_path,
) -> None:
    """F4: 20 answers must not be scored against 173 questions."""
    from experiments.score_spec_run import score_run_dir

    qs = _questions_file(tmp_path)
    root = tmp_path / "rid1"
    _write_answers(root, "cd_mass_6x__w0.1__planet__pf1", {"g1": "800,000 yen"})
    only = score_run_dir(root, questions=qs)["rows"][0]
    assert only["total"] == 1 and only["pass"] == 1 and only["pass_rate"] == 1.0

    every = score_run_dir(root, questions=qs, only_answered=False)["rows"][0]
    assert every["total"] == 3 and every["pass"] == 1


def test_cell_fields_fall_back_to_the_cell_name(tmp_path) -> None:
    from experiments.score_spec_run import cell_fields

    root = tmp_path / "answers"
    root.mkdir(parents=True)
    path = root / "cd_mass_6x__w0.25__planet-satellites__pf1.json"
    path.write_text("{}", encoding="utf-8")
    fields = cell_fields(path)
    assert fields["arm"] == "cd_mass_6x"
    assert fields["w"] == 0.25
    assert fields["inject"] == "planet+satellites"
    assert fields["prefill"] == 1.0


def test_the_meta_sidecar_is_never_scored_as_a_cell(tmp_path) -> None:
    from experiments.score_spec_run import answer_files

    root = tmp_path / "answers"
    root.mkdir(parents=True)
    (root / "a.json").write_text("{}", encoding="utf-8")
    (root / "a.meta.json").write_text("{}", encoding="utf-8")
    assert [p.name for p in answer_files(root)] == ["a.json"]


def test_score_defaults_are_the_documented_ones() -> None:
    from experiments import score_spec_run as ssr

    assert ssr.DEFAULT_SUBSET == "generated"
    assert ssr.DEFAULT_MAX_WORDS == 32


# --------------------------------------------------------------------------
# F5 / A2: detached execution and client output
# --------------------------------------------------------------------------


class _FakeCall:
    def __init__(self, object_id: str, result: Any = None) -> None:
        self.object_id = object_id
        self._result = result

    def get(self) -> Any:
        return self._result


class _FakeFunction:
    def __init__(self, name: str, recorder: list) -> None:
        self.name = name
        self._recorder = recorder

    def spawn(self, **kwargs) -> _FakeCall:
        self._recorder.append(("spawn", self.name, kwargs))
        return _FakeCall("fc-123")


def _fake_modal(recorder: list) -> Any:
    import contextlib

    @contextlib.contextmanager
    def enable_output():
        recorder.append(("enable_output", None, None))
        yield

    return types.SimpleNamespace(
        Function=types.SimpleNamespace(
            from_name=lambda app, fn: _FakeFunction("%s.%s" % (app, fn), recorder)
        ),
        FunctionCall=types.SimpleNamespace(
            from_id=lambda cid: _FakeCall(cid, {"cells": 38})
        ),
        enable_output=enable_output,
    )


def _args(argv: list[str]):
    args = msr.build_parser().parse_args(argv)
    args.run_id = args.run_id or "rid-test"
    return args


def test_spawn_records_the_call_id_and_wraps_output(tmp_path) -> None:
    recorder: list = []
    args = _args(["--spawn", "reader", "--run-id", "rid1",
                  "--call-dir", str(tmp_path), "--w-grid", "0,0.1"])
    record = msr.spawn_phase("reader", args, modal_mod=_fake_modal(recorder))

    assert record["call_id"] == "fc-123" and record["run_id"] == "rid1"
    # A2: the client sees the container's output for the whole call
    assert ("enable_output", None, None) in recorder
    name, kwargs = [r for r in recorder if r[0] == "spawn"][0][1:]
    assert name == "cms-spec-run.phase_reader"
    assert kwargs["run_id"] == "rid1"
    assert kwargs["model_id"] == msr.DEFAULT_READER_MODEL_ID
    assert kwargs["gpu_mem_gb"] == msr.GPU_MEM_GB[msr.GPU_SPEC]
    assert record["n_cells"] == len(kwargs["cells"])

    on_disk = json.loads(
        (tmp_path / "spawned_reader.json").read_text(encoding="utf-8")
    )
    assert on_disk["call_id"] == "fc-123"
    # the cell list is NOT duplicated into the record
    assert "cells" not in on_disk["kwargs"]


def test_wait_reads_the_recorded_call_id(tmp_path) -> None:
    recorder: list = []
    args = _args(["--wait", "reader", "--run-id", "rid1",
                  "--call-dir", str(tmp_path)])
    (tmp_path / "spawned_reader.json").write_text(
        json.dumps({"phase": "reader", "call_id": "fc-999"}), encoding="utf-8"
    )
    result = msr.wait_phase("reader", args, modal_mod=_fake_modal(recorder))
    assert result == {"cells": 38}


def test_wait_without_a_spawn_record_says_what_to_do(tmp_path) -> None:
    args = _args(["--wait", "manager", "--call-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="--spawn manager"):
        msr.wait_phase("manager", args, modal_mod=_fake_modal([]))


def test_phase_call_kwargs_are_the_documented_ones() -> None:
    args = _args(["--run-id", "rid1", "--allow-linear-layers",
                  "--shortlist-k", "3", "--w-grid", "0"])
    assert msr.phase_call_kwargs("manager", args) == {
        "model_id": msr.DEFAULT_MODEL_ID, "run_id": "rid1", "shortlist_k": 3,
    }
    inst = msr.phase_call_kwargs("instrument", args)
    assert inst["model_id"] == msr.DEFAULT_READER_MODEL_ID
    assert inst["allow_linear_layers"] is True
    with pytest.raises(SystemExit, match="unknown phase"):
        msr.phase_call_kwargs("nope", args)


def test_output_ctx_degrades_to_a_no_op(tmp_path) -> None:
    """A2: an older modal client without enable_output must still work."""
    with msr._output_ctx(types.SimpleNamespace()):
        pass


def test_the_docstring_warns_that_run_is_tied_to_the_client() -> None:
    assert msr.__doc__ is not None
    assert "--spawn" in msr.__doc__ and "TIED TO" in msr.__doc__.upper()


def test_download_refuses_without_a_run_id(monkeypatch) -> None:
    """F3: pulling '/' would merge every run that ever used the volume."""
    monkeypatch.setattr(msr, "require_modal", lambda: None)
    with pytest.raises(SystemExit, match="--download needs --run-id"):
        msr.main(["--download"])


# --------------------------------------------------------------------------
# A3: the model ids
# --------------------------------------------------------------------------


def test_judge_and_reader_default_to_the_same_hybrid_model() -> None:
    """2026-09-07 owner decision: BOTH roles are Qwen/Qwen3.8-27B."""
    args = _args([])
    assert msr.resolve_judge_model_id(args) == msr.DEFAULT_MODEL_ID
    assert msr.resolve_reader_model_id(args) == msr.DEFAULT_READER_MODEL_ID
    # Directive 5: same model for both roles; A3: it is the hybrid, on purpose.
    assert msr.DEFAULT_READER_MODEL_ID == msr.DEFAULT_MODEL_ID
    assert msr.DEFAULT_MODEL_ID == msr.HYBRID_MODEL_ID == "Qwen/Qwen3.8-27B"
    # the dense alternative is still reachable and is NOT the hybrid
    assert msr.DENSE_MODEL_ID == "Qwen/Qwen3-32B" != msr.HYBRID_MODEL_ID


def test_the_hybrid_layer_constants_agree_across_the_two_modules() -> None:
    """modal_spec_run duplicates them (no repo import at module level)."""
    from benchmark.bineval import run_reader as rr

    assert msr.HYBRID_MODEL_ID == rr.HYBRID_MODEL_ID
    assert msr.HYBRID_N_SDPA_LAYERS == rr.HYBRID_N_SDPA_LAYERS == 16
    assert msr.HYBRID_N_LINEAR_LAYERS == rr.HYBRID_N_LINEAR_LAYERS == 48
    assert rr.HYBRID_NUM_HIDDEN_LAYERS == 64
    assert msr.is_hybrid_model_id("Qwen/Qwen3.8-27B") is True
    assert rr.is_hybrid_model_id("qwen/qwen3.8-27b") is True
    assert msr.is_hybrid_model_id("Qwen/Qwen3-32B") is False
    assert rr.is_hybrid_model_id(None) is False


def test_allow_linear_layers_defaults_to_on_for_the_hybrid_reader() -> None:
    """The decided reader must not need a flag on every invocation."""
    assert msr.resolve_allow_linear_layers(_args([])) is True
    assert _args([]).allow_linear_layers is None  # nothing forced on the CLI


def test_allow_linear_layers_defaults_to_off_for_a_dense_reader() -> None:
    args = _args(["--reader-model-id", msr.DENSE_MODEL_ID])
    assert msr.resolve_allow_linear_layers(args) is False


def test_no_allow_linear_layers_forces_the_dense_only_rule() -> None:
    """The explicit flag wins over the model-id default, both ways."""
    args = _args(["--no-allow-linear-layers"])
    assert args.allow_linear_layers is False
    assert msr.resolve_allow_linear_layers(args) is False
    args = _args(["--reader-model-id", "some/other-hybrid",
                  "--allow-linear-layers"])
    assert msr.resolve_allow_linear_layers(args) is True


def test_run_info_records_the_expected_sdpa_layer_count() -> None:
    info = msr.build_run_info("rid", reader_model_id=msr.HYBRID_MODEL_ID)
    assert info["n_sdpa_layers_expected"] == 16
    # a dense reader has no expectation to assert
    dense = msr.build_run_info("rid", reader_model_id=msr.DENSE_MODEL_ID)
    assert dense["n_sdpa_layers_expected"] is None
    assert msr.expected_n_sdpa_layers(msr.HYBRID_MODEL_ID) == 16
    assert msr.expected_n_sdpa_layers(None) is None


def test_the_default_dry_run_records_the_16_layer_expectation(tmp_path) -> None:
    """The gate command `--dry-run` must show the 16/64 fact on disk."""
    assert msr.main(["--dry-run", "--out", str(tmp_path), "--run-id", "rid-dry",
                     "--w-grid", "0", "--no-tokens"]) == 0
    payload = json.loads(
        (Path(tmp_path) / "rid-dry" / "run_info.json").read_text(encoding="utf-8")
    )
    assert payload["reader_model_id"] == msr.HYBRID_MODEL_ID
    assert payload["model_id"] == msr.HYBRID_MODEL_ID
    assert payload["n_sdpa_layers_expected"] == 16


def test_phase_kwargs_carry_the_hybrid_default_without_a_flag() -> None:
    args = _args(["--run-id", "rid1", "--w-grid", "0"])
    assert msr.phase_call_kwargs("instrument", args)["allow_linear_layers"] is True
    assert msr.phase_call_kwargs("reader", args)["allow_linear_layers"] is True
    dense = _args(["--run-id", "rid1", "--w-grid", "0",
                   "--reader-model-id", msr.DENSE_MODEL_ID])
    assert msr.phase_call_kwargs("reader", dense)["allow_linear_layers"] is False


def test_both_model_ids_can_be_overridden() -> None:
    args = _args(["--judge-model-id", "j/m", "--reader-model-id", "r/m"])
    assert msr.resolve_judge_model_id(args) == "j/m"
    assert msr.resolve_reader_model_id(args) == "r/m"


def test_model_id_still_drives_the_judge() -> None:
    args = _args(["--model-id", "old/way"])
    assert msr.resolve_judge_model_id(args) == "old/way"


def test_the_docstring_states_the_hybrid_fact_and_both_configurations() -> None:
    doc = msr.__doc__ or ""
    assert "48 linear and 16 full of 64 layers" in doc
    assert "--allow-linear-layers" in doc
    assert "--no-allow-linear-layers" in doc
    assert msr.DEFAULT_READER_MODEL_ID in doc
    # the default is the hybrid and the dense model is named as the alternative
    assert "MEASURED CONFIGURATION" in doc
    assert msr.DENSE_MODEL_ID in doc
    # the kernel decision is written down where the image is read
    assert "flash-linear-attention" in doc and "causal-conv1d" in doc


def test_the_reader_cli_exposes_the_flag() -> None:
    from benchmark.bineval.run_reader import build_parser

    args = build_parser().parse_args(
        ["--model-id", "m", "--questions", "q.json", "--out", "o.json",
         "--arm", "floor"]
    )
    # unset on the CLI: the model id decides (run_reader.resolve_allow_...)
    assert args.allow_linear_layers is None
    args = build_parser().parse_args(
        ["--model-id", "m", "--questions", "q.json", "--out", "o.json",
         "--arm", "floor", "--allow-linear-layers"]
    )
    assert args.allow_linear_layers is True
    args = build_parser().parse_args(
        ["--model-id", "m", "--questions", "q.json", "--out", "o.json",
         "--arm", "floor", "--no-allow-linear-layers"]
    )
    assert args.allow_linear_layers is False


def test_the_reader_cli_default_follows_the_model_id() -> None:
    from benchmark.bineval.run_reader import (
        HYBRID_MODEL_ID,
        build_parser,
        resolve_allow_linear_layers,
    )

    def _parse(*extra: str):
        return build_parser().parse_args(
            ["--questions", "q.json", "--out", "o.json", "--arm", "floor",
             *extra]
        )

    assert resolve_allow_linear_layers(_parse("--model-id", HYBRID_MODEL_ID)) is True
    assert resolve_allow_linear_layers(
        _parse("--model-id", HYBRID_MODEL_ID, "--no-allow-linear-layers")
    ) is False
    assert resolve_allow_linear_layers(_parse("--model-id", "Qwen/Qwen3-32B")) is False
    assert resolve_allow_linear_layers(
        _parse("--model-id", "Qwen/Qwen3-32B", "--allow-linear-layers")
    ) is True


def test_run_meta_carries_the_reachable_layer_count() -> None:
    pytest.importorskip("torch")
    from benchmark.bineval.run_reader import check_model_supported, run_meta

    cfg = {
        "model_type": "qwen3_5_text", "num_hidden_layers": 64,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 16,
        "max_position_embeddings": 262144,
    }
    info = check_model_supported(cfg, allow_linear_layers=True)
    meta = run_meta(
        model_id="m", layer_info=info, w=0.1, inject="planet",
        prefill_scale=0.0, bias_cap=None, context_tokens=100, arm="cd_mass_6x",
    )
    assert meta["n_sdpa_layers"] == 16
    assert meta["n_linear_layers"] == 48
    assert meta["allow_linear_layers"] is True
    assert meta["layer_types_summary"]["full_attention"] == 16


# --------------------------------------------------------------------------
# A4: tiktoken budgets vs the model tokenizer
# --------------------------------------------------------------------------

CTX_CONFIG = SimpleNamespace(
    model_type="qwen3",
    layer_types=["full_attention"] * 4,
    use_sliding_window=False,
    max_position_embeddings=100_000,
    num_hidden_layers=4,
    num_key_value_heads=4,
    head_dim=128,
)


class _RatioTok:
    """A tokenizer that produces ``ratio`` ids per tiktoken-ish word."""

    def __init__(self, ratio: float) -> None:
        self.ratio = ratio

    def __call__(self, text: str, **_kw) -> dict:
        return {"input_ids": list(range(int(len(text.split()) * self.ratio)))}


def test_context_stage_reports_both_token_counts(tmp_path) -> None:
    from experiments import gpu_preflight as pf

    text = " ".join(["word"] * 1000)
    res = pf.stage_context(
        CTX_CONFIG, gpu_mem_gb=141.0, weight_bytes=1, arm_tokens={"full": 1000},
        tokenizer=_RatioTok(1.1), arm_text=lambda _arm: text,
    )
    assert res.ok, res.message
    assert res.detail["tiktoken_tokens"] == 1000
    assert res.detail["real_tokenizer_tokens"] == 1100
    assert res.detail["real_over_tiktoken"] == pytest.approx(1.1)
    assert "1100 real" in res.message


def test_context_stage_fails_when_the_real_count_overflows(tmp_path) -> None:
    from experiments import gpu_preflight as pf

    text = " ".join(["word"] * 60_000)
    res = pf.stage_context(
        CTX_CONFIG, gpu_mem_gb=141.0, weight_bytes=1, arm_tokens={"full": 60_000},
        tokenizer=_RatioTok(2.0), arm_text=lambda _arm: text,
    )
    assert not res.ok
    assert "MODEL tokenizer" in res.message
    assert res.detail["real_tokenizer_tokens"] == 120_000


def test_context_stage_survives_an_arm_whose_text_is_missing() -> None:
    from experiments import gpu_preflight as pf

    def boom(_arm: str) -> str:
        raise FileNotFoundError("the stored artifact is gone")

    res = pf.stage_context(
        CTX_CONFIG, gpu_mem_gb=141.0, weight_bytes=1, arm_tokens={"full": 1000},
        tokenizer=_RatioTok(1.0), arm_text=boom,
    )
    assert res.ok, res.message
    assert "FileNotFoundError" in res.detail["real_tokenizer_error"]


def test_context_stage_without_a_tokenizer_is_the_old_behaviour() -> None:
    from experiments import gpu_preflight as pf

    res = pf.stage_context(CTX_CONFIG, gpu_mem_gb=141.0, weight_bytes=1, arm_tokens={"full": 1000})
    assert res.ok and "real" not in res.message


# --------------------------------------------------------------------------
# F8: the mass vector is built on the CPU
# --------------------------------------------------------------------------


def _old_positions_to_mass_vector(positions, seq_len, *, cap, scale=1.0, device=None):
    """The pre-F8 implementation, verbatim, as the equality reference."""
    import torch

    if not positions:
        return None
    vec = torch.zeros(seq_len, dtype=torch.float32, device=device)
    for pos, mass in positions:
        if 0 <= pos < seq_len:
            value = min(cap, float(mass) * scale)
            if value > float(vec[pos]):
                vec[pos] = value
    return vec


def test_the_vectorised_mass_vector_equals_the_old_loop() -> None:
    torch = pytest.importorskip("torch")
    from server.mass_vector import positions_to_mass_vector

    rng = random.Random(20260907)
    for _ in range(50):
        seq_len = rng.randint(1, 64)
        n = rng.randint(0, 40)
        positions = [
            # deliberately includes out-of-range and duplicate positions
            (rng.randint(-3, seq_len + 3), rng.uniform(0.0, 12.0))
            for _ in range(n)
        ]
        cap = rng.choice([float("inf"), 3.0, 8.0])
        scale = rng.choice([1.0, 0.5, 2.0])
        new = positions_to_mass_vector(
            positions, seq_len, cap=cap, scale=scale
        )
        old = _old_positions_to_mass_vector(
            positions, seq_len, cap=cap, scale=scale
        )
        if old is None:
            assert new is None
            continue
        assert torch.equal(new, old), (positions, seq_len, cap, scale)


def test_the_mass_vector_still_takes_the_max_on_a_collision() -> None:
    torch = pytest.importorskip("torch")
    from server.mass_vector import positions_to_mass_vector

    vec = positions_to_mass_vector(
        [(2, 1.0), (2, 5.0), (2, 3.0)], 4, cap=float("inf")
    )
    assert torch.equal(vec, torch.tensor([0.0, 0.0, 5.0, 0.0]))


def test_the_mass_vector_is_built_on_the_cpu_before_any_move() -> None:
    """F8: no per-element host<->device sync. The device move happens once."""
    torch = pytest.importorskip("torch")
    from server import mass_vector as mv

    moved: list = []

    class _Recorder(torch.Tensor):
        pass

    real_to = torch.Tensor.to

    def spy_to(self, *a, **kw):
        moved.append(a)
        return real_to(self, *a, **kw)

    original = torch.Tensor.to
    try:
        torch.Tensor.to = spy_to  # type: ignore[method-assign]
        vec = mv.positions_to_mass_vector([(1, 2.0)], 4, cap=3.0, device="cpu")
    finally:
        torch.Tensor.to = original  # type: ignore[method-assign]
    assert vec is not None
    assert moved == [("cpu",)]  # exactly one move, after the build


def test_no_device_means_no_move_at_all() -> None:
    pytest.importorskip("torch")
    from server.mass_vector import positions_to_mass_vector

    vec = positions_to_mass_vector([(0, 1.0)], 2, cap=3.0, device=None)
    assert vec is not None and str(vec.device) == "cpu"


# --------------------------------------------------------------------------
# F9 / A5: the judge's reply parsing
# --------------------------------------------------------------------------


def _judge_with_reply(monkeypatch, message: dict):
    """An OpenAICompatJudge whose HTTP round trip returns ``message``."""
    import contextlib
    import urllib.request

    from management.harness.backends import OpenAICompatJudge

    payload = json.dumps({"choices": [{"message": message}]}).encode("utf-8")

    @contextlib.contextmanager
    def fake_urlopen(_req, timeout=None):
        yield SimpleNamespace(read=lambda: payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return OpenAICompatJudge("http://127.0.0.1:8000", "m")


def test_a_null_content_is_the_empty_string_not_the_word_none(monkeypatch) -> None:
    """F9: ``str(None)`` produced a plausible-looking, unparseable "None"."""
    judge = _judge_with_reply(monkeypatch, {"content": None})
    assert judge._once("hi", 16) == ""


def test_a_list_content_is_the_empty_string(monkeypatch) -> None:
    judge = _judge_with_reply(
        monkeypatch, {"content": [{"type": "text", "text": "yes"}]}
    )
    assert judge._once("hi", 16) == ""


def test_a_reasoning_only_reply_is_counted_and_not_mined(monkeypatch) -> None:
    """A5: vLLM 0.28 renamed reasoning_content -> reasoning. Neither is an
    answer, so the reply stays empty and only the fact is recorded."""
    judge = _judge_with_reply(
        monkeypatch, {"content": None, "reasoning": "the deposit is 4,800,000"}
    )
    assert judge._once("hi", 16) == ""
    assert judge.reasoning_only_replies == 1


def test_the_old_reasoning_content_key_is_recognised_too(monkeypatch) -> None:
    judge = _judge_with_reply(
        monkeypatch, {"content": None, "reasoning_content": "thinking"}
    )
    assert judge._once("hi", 16) == ""
    assert judge.reasoning_only_replies == 1


def test_a_normal_reply_is_unchanged(monkeypatch) -> None:
    judge = _judge_with_reply(monkeypatch, {"content": "<think>x</think>yes"})
    assert judge._once("hi", 16) == "yes"
    assert judge.reasoning_only_replies == 0

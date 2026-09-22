"""run_arms: argparse + pre-model validation (item 12), artifact binding (item 2),
full-prompt budgeting (item 9), planet guards (items 1, 13), E1 timing hook
(item 6) and sampler coverage (item 7) — with fakes; no model is ever loaded."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark.bineval.run_reader import ReaderRun, build_prompt
from benchmark.mcbuild_bench import run_arms
from benchmark.mcbuild_bench.run_arms import (
    attn_flops_decode,
    attn_flops_prefill,
    budget_question,
    build_parser,
    cell_round_trips,
    check_all_questions_fit,
    check_bias_counters,
    check_cd_artifact,
    check_compaction_artifact,
    check_compaction_window,
    check_per_question,
    effective_prefill_chunk,
    expected_bias_counters,
    file_sha256,
    n_prefill_forwards,
    parse_W,
    read_checkpoint,
    resolve_inject,
    validate_args,
)
from benchmark.mcbuild_bench.windows import count_planet_lines

MODEL_ID = "Qwen/Qwen3.8-27B"

# ``source_session`` = the round trip whose statement the question asks for
# (ledger field name kept from bineval).
QUESTIONS = [
    {"qid": "f001", "question": "What is the RCON port?", "gold_short": "25575",
     "tier1_aliases": [], "kind": "value", "excluded": False, "legacy": False,
     "source_session": 1},
    {"qid": "f002", "question": "Which server software version runs on the hackathon box exactly?",
     "gold_short": "Paper 1.21.8", "tier1_aliases": [], "kind": "value", "excluded": False,
     "legacy": False, "source_session": 0},
]
ABSENT_QUESTION = {
    "qid": "a001", "question": "What GPU was in the dev PC?", "gold_short": "unknown",
    "tier1_aliases": ["not in context"], "kind": "absent", "excluded": False, "legacy": False,
    "source_session": -1,
}
SESSION = {
    "round_trips": [
        {"idx": 0, "human": "Plan the hackathon build agent.",
         "events": [{"kind": "text", "text": "The server runs Paper 1.21.8."}]},
        {"idx": 1, "human": "What port?",
         "events": [{"kind": "text", "text": "RCON port is 25575."}]},
    ]
}
# Round trip 0 is too large for W=8000 (whitespace tokens), so an 8k window holds
# round trip 1 only: first_recent_rt == 1, f002 (rt 0) lies outside the window,
# f001 (rt 1) inside (H22 (d)).
SESSION_LONG = {
    "round_trips": [
        {"idx": 0, "human": "Plan the hackathon build agent.",
         "events": [{"kind": "text", "text": "The server runs Paper 1.21.8. " + " ".join(
             "pad%d" % k for k in range(9000))}]},
        {"idx": 1, "human": "What port?",
         "events": [{"kind": "text", "text": "RCON port is 25575."}]},
    ]
}
CD_RECORDS = [
    {"node_id": "s1", "text": "Hackathon build agent", "level": "sun", "mass": 1.0,
     "parent_id": None, "created_turn": 0},
    {"node_id": "p1", "text": "RCON port is 25575", "level": "planet", "mass": 1.0,
     "parent_id": "s1", "created_turn": 1},
    {"node_id": "r1", "text": "Server runs Paper 1.21.8", "level": "satellite", "mass": 0.0,
     "parent_id": "p1", "created_turn": 0},
]


class WsTok:
    """Whitespace tokenizer with the two call shapes run_arms / run_reader use."""

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        return {"input_ids": text.split()}


class NonAdditiveTok:
    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        return {"input_ids": list(range(len(text.replace(" ", "")) // 3))}


ZERO_STATS = {"bias_applied_calls": 0, "bias_skipped_prefill_calls": 0,
              "bias_skipped_sliding_calls": 0}


class FakeLLM:
    """The item-6 attributes + the surface run_reader.run_reader touches.

    Counters follow the real patch (F1): with a mass vector set, one prefill
    forward per chunk (16 skipped-prefill calls each) and 16 applied calls per
    DECODE forward (= generated tokens - 1).  ``fail_on`` makes the k-th
    generate() raise (F3 checkpoint / resume tests).
    """

    n_sdpa_layers = 16

    def __init__(self, tokenizer, *, n_tokens: int = 3, fail_on: int | None = None) -> None:
        self.tokenizer = tokenizer
        self.n_tokens = n_tokens
        self.fail_on = fail_on
        self.last_generated_tokens = None
        self.last_decode_forwards = None
        self.last_prefill_ms = None
        self.last_decode_ms = None
        self.generated: list[str] = []
        self.loaded_class_name = "Qwen3_5ForCausalLM"
        self.loading_info = {"missing": 0, "unexpected": 3, "mismatched": 0}
        self._model = SimpleNamespace(generation_config=SimpleNamespace(prefill_chunk_size=None))
        self._vec = None
        self._stats = dict(ZERO_STATS)
        self.chunks_seen: list[int | None] = []

    def set_mass_vector(self, v) -> None:
        self._vec = v
        self._stats = dict(ZERO_STATS)

    def clear_mass_vector(self) -> None:
        self._vec = None
        self._stats = dict(ZERO_STATS)

    def mass_injection_stats(self) -> dict:
        return dict(self._stats)

    def generate(self, prompt: str) -> str:
        self.generated.append(prompt)
        if self.fail_on is not None and len(self.generated) == self.fail_on:
            raise RuntimeError("generate failed on call %d" % self.fail_on)
        time.sleep(0.02)
        n = self.n_tokens
        self.last_generated_tokens = n
        self.last_decode_forwards = n - 1
        self.last_prefill_ms = 12.5
        self.last_decode_ms = 4.5
        chunk = self._model.generation_config.prefill_chunk_size
        self.chunks_seen.append(chunk)
        L = len(prompt.split())
        n_prefill = 1 if chunk is None else math.ceil(L / chunk)
        if self._vec is not None:
            self._stats = {
                "bias_applied_calls": self.n_sdpa_layers * (n - 1),
                "bias_skipped_prefill_calls": self.n_sdpa_layers * n_prefill,
                "bias_skipped_sliding_calls": 0,
            }
        return "25575 is it\nextra line"


class FakeSampler:
    """Writes a CSV covering [now - 3 s, now + 60 s] at 0.5 s so every wait is
    satisfied immediately and every question span is integrable."""

    empty = False

    def __init__(self, path) -> None:
        self.path = Path(path)
        self.started = self.stopped = False

    def start(self) -> None:
        self.started = True
        lines = ["timestamp, utilization.gpu [%], utilization.memory [%], memory.used [MiB], power.draw [W]"]
        if not FakeSampler.empty:
            now = time.time()
            t = now - 3.0
            while t < now + 60.0:
                dt = datetime.fromtimestamp(t)
                lines.append("%s.%03d, 50 %%, 10 %%, 100 MiB, 200.00 W"
                             % (dt.strftime("%Y/%m/%d %H:%M:%S"), dt.microsecond // 1000))
                t += 0.5
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def stop(self) -> None:
        self.stopped = True


def _files(tmp_path: Path, *, questions=QUESTIONS, session=SESSION) -> dict[str, Path]:
    q = tmp_path / "q.json"
    s = tmp_path / "s.json"
    q.write_text(json.dumps(questions), encoding="utf-8")
    s.write_text(json.dumps(session), encoding="utf-8")
    return {"questions": q, "session": s}


def _base(tmp_path: Path, **kw) -> list[str]:
    f = _files(tmp_path, **kw)
    return [
        "--model-id", MODEL_ID,
        "--questions", str(f["questions"]),
        "--session", str(f["session"]),
        "--out", str(tmp_path / "out"),
        "--gpu-csv", str(tmp_path / "gpu.csv"),
        # H22 (c): the default exclusion (round trip 36) is checked against the
        # session; these fixtures opt out explicitly.
        "--exclude-rt", "none",
    ]


def _cd_json(tmp_path: Path, *, session_sha: str | None, nodes=CD_RECORDS, drop_nodes=False) -> Path:
    payload: dict = {"summary": {"turns": 2}, "manifest": {}}
    if not drop_nodes:
        payload["nodes"] = nodes
    if session_sha is not None:
        payload["manifest"]["session_sha256"] = session_sha
    p = tmp_path / "cd.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _wire_gpu(monkeypatch, *, llm: FakeLLM, run_reader_fn=None) -> None:
    import torch
    import transformers

    from benchmark.bineval import run_reader

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda i: SimpleNamespace(total_memory=48 * 1024 ** 3))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda mid: llm.tokenizer)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda mid: object())
    monkeypatch.setattr(run_reader, "check_model_supported",
                        lambda cfg, allow_linear_layers: {"n_sdpa_layers": 16})
    monkeypatch.setattr(run_reader, "check_context_fits", lambda *a, **k: {"fits": True})
    monkeypatch.setattr(run_reader, "load_reader", lambda *a, **k: llm)
    monkeypatch.setattr(run_arms, "safetensors_total_bytes", lambda mid: 55_600_000_000)
    if run_reader_fn is not None:
        monkeypatch.setattr(run_reader, "run_reader", run_reader_fn)
    monkeypatch.setattr(run_arms, "GpuSampler", FakeSampler)
    monkeypatch.setattr(run_arms, "SAMPLER_WARMUP_TIMEOUT_S", 0.3)
    monkeypatch.setattr(run_arms, "SAMPLER_TAIL_TIMEOUT_S", 0.3)
    monkeypatch.setattr(run_arms, "SAMPLER_POLL_S", 0.01)
    FakeSampler.empty = False


def _forbid_model_load(monkeypatch) -> None:
    from benchmark.bineval import run_reader

    def boom(*a, **k):
        raise AssertionError("load_reader must not be called")

    monkeypatch.setattr(run_reader, "load_reader", boom)


# -------------------------------------------------------------------- parser / pure


def test_parser_defaults_and_choices(tmp_path: Path) -> None:
    args = build_parser().parse_args(["--arm", "B", "--W", "8000"] + _base(tmp_path))
    assert args.arm == "B" and args.W == 8000
    assert args.w == 0.0
    assert args.prefill_chunk == 8192
    assert args.inject is None  # resolved by validate_args (H11)
    assert args.prefill_scale == 0.0
    assert args.max_new_tokens == 48
    assert args.cd is None and args.compaction_summary is None
    with pytest.raises(SystemExit):  # item 4: choices = run_reader.INJECT_MODES
        build_parser().parse_args(["--arm", "B", "--W", "8000", "--inject", "all"] + _base(tmp_path))


def test_W_full_and_bad_values(tmp_path: Path) -> None:
    base = _base(tmp_path)
    args = build_parser().parse_args(["--arm", "A", "--W", "full"] + base)
    assert args.W is None
    assert parse_W("16000") == 16000
    with pytest.raises(argparse.ArgumentTypeError):
        parse_W("big")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_W("0")
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--arm", "D", "--W", "8000"] + base)
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--arm", "B", "--W", "8000"])  # missing required


def test_validate_args_cross_rules(tmp_path: Path) -> None:
    p = build_parser()
    base = _base(tmp_path)
    (tmp_path / "cd.json").write_text("{}", encoding="utf-8")
    (tmp_path / "c.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        validate_args(p.parse_args(["--arm", "A", "--W", "8000"] + base))
    with pytest.raises(SystemExit):
        validate_args(p.parse_args(["--arm", "B", "--W", "full"] + base))
    with pytest.raises(SystemExit):
        validate_args(p.parse_args(["--arm", "proposed", "--W", "8000"] + base))
    with pytest.raises(SystemExit):
        validate_args(p.parse_args(["--arm", "C", "--W", "8000"] + base))
    a = p.parse_args(["--arm", "proposed", "--W", "8000", "--cd", str(tmp_path / "cd.json")] + base)
    validate_args(a)
    assert a.inject == "planet"
    a = p.parse_args(["--arm", "C", "--W", "8000", "--compaction-summary", str(tmp_path / "c.json")] + base)
    validate_args(a)
    assert a.inject == "none"
    a = p.parse_args(["--arm", "A", "--W", "full"] + base)
    validate_args(a)
    assert a.inject == "none"


@pytest.mark.parametrize(
    "extra",
    [
        ["--arm", "B", "--W", "9000"],                       # W not in the C6 grid
        ["--arm", "B", "--W", "full"],                       # full only with A
        ["--arm", "A", "--W", "8000"],                       # A needs full
        ["--arm", "proposed", "--W", "8000", "--w", "-1"],   # w < 0
        ["--arm", "B", "--W", "8000", "--w", "0.5"],         # baselines need w == 0
        ["--arm", "B", "--W", "8000", "--prefill-chunk", "0"],
        ["--arm", "B", "--W", "8000", "--max-new-tokens", "0"],
        ["--arm", "B", "--W", "8000", "--inject", "planet"],  # H11: baselines never scan
        ["--arm", "proposed", "--W", "8000"],                # --cd required
        ["--arm", "C", "--W", "8000"],                       # --compaction-summary required
        ["--arm", "proposed", "--W", "8000", "--cd", "missing_cd.json"],
    ],
)
def test_invalid_cli_values_exit_before_any_model_import(monkeypatch, tmp_path: Path, extra) -> None:
    """Item 12: every invalid combination is a SystemExit before torch/transformers."""
    _forbid_model_load(monkeypatch)
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(SystemExit):
        run_arms.main(extra + _base(tmp_path))


def test_missing_input_paths_exit_before_model_import(monkeypatch, tmp_path: Path) -> None:
    _forbid_model_load(monkeypatch)
    for missing in ("--questions", "--session"):
        argv = ["--arm", "B", "--W", "8000"] + _base(tmp_path)
        argv[argv.index(missing) + 1] = str(tmp_path / "absent.json")
        with pytest.raises(SystemExit, match="file not found"):
            run_arms.main(argv)


def test_resolve_inject_forces_none_on_baselines() -> None:
    assert resolve_inject("A", None) == "none"
    assert resolve_inject("B", "none") == "none"
    assert resolve_inject("C", None) == "none"
    assert resolve_inject("proposed", None) == "planet"
    assert resolve_inject("proposed", "planet+satellites") == "planet+satellites"
    with pytest.raises(SystemExit, match="H11"):
        resolve_inject("B", "planet")


def test_flops_formula_known_numbers() -> None:
    # 16 layers x 2 x 24 heads x L^2 x 256, L = 10
    assert attn_flops_prefill(10) == 16 * 2 * 24 * 100 * 256 == 19_660_800
    # decode (F1): n tokens = 1 prefill + (n - 1) decode forwards, so
    # sum_{t=1..n-1} 16*2*24*(10+t)*256 = 196608 * 11 for n = 2
    assert attn_flops_decode(10, 2) == 196_608 * 11 == 2_162_688
    assert attn_flops_decode(10, 3) == 196_608 * (11 + 12)
    assert attn_flops_decode(10, 1) == 0
    assert attn_flops_decode(10, 0) == 0
    # DESIGN 0.4: W=8k window of 6,037 tokens -> quadratic in L
    assert attn_flops_prefill(6037) == 16 * 2 * 24 * 6037 * 6037 * 256


def test_cell_round_trips_arm_C_prepends_summary() -> None:
    rts = [{"idx": i, "human": "h", "events": []} for i in range(5)]
    comp = {"summary": "S", "recent_idx": [3, 4], "calls": [], "n_calls": 0}
    out = cell_round_trips("C", rts, comp)
    assert out[0]["kind"] == "summary" and out[0]["text"] == "S"
    assert [rt["idx"] for rt in out[1:]] == [3, 4]
    assert cell_round_trips("B", rts, None) == rts
    with pytest.raises(ValueError):
        cell_round_trips("C", rts, None)


def test_cli_refuses_without_cuda_before_touching_any_model(monkeypatch, tmp_path: Path) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # any attempt to import transformers/model code here would be a bug
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises((SystemExit, RuntimeError)) as exc:
        run_arms.main(["--arm", "B", "--W", "8000"] + _base(tmp_path))
    assert "CUDA" in str(exc.value)


def test_check_compaction_window_requires_every_recent_round_trip() -> None:
    comp = {"summary": "S", "recent_idx": [3, 4], "calls": [], "n_calls": 0}
    check_compaction_window({"n_recent_rts": 2}, comp)  # agrees: no error
    with pytest.raises(RuntimeError) as exc:
        check_compaction_window({"n_recent_rts": 1}, comp)
    assert "1" in str(exc.value) and "2" in str(exc.value)


# -------------------------------------------------------------------- item 9


class DistinctCharTok:
    """Vocabulary-like fake: tokens = distinct non-space characters, so a question
    made of letters the prompt already contains adds NOTHING to the prompt count."""

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        return {"input_ids": sorted(set(text.replace(" ", "")))}


def test_budget_question_measures_the_full_prompt_not_the_standalone_question() -> None:
    tok = DistinctCharTok()
    ctx = "<context>" + chr(10) + "the port is open" + chr(10) + "</context>"
    qs = [{"qid": "a", "question": "zj"}, {"qid": "b", "question": "the port"}]
    standalone = max(qs, key=lambda q: len(tok(q["question"])["input_ids"]))["question"]
    assert standalone == "the port"  # 7 distinct chars vs 2
    full = {q["question"]: len(tok(build_prompt(ctx, q["question"]))["input_ids"]) for q in qs}
    assert full["zj"] > full["the port"]  # z, j are new; t,h,e,p,o,r already occur
    assert budget_question(qs, tok, ctx) == "zj" != standalone


def test_check_all_questions_fit_names_the_offending_qid() -> None:
    tok = WsTok()
    ctx = "<context>\nfive words of context text\n</context>"
    qs = [{"qid": "short", "question": "port?"}, {"qid": "long", "question": " ".join(["w"] * 30)}]
    counts = check_all_questions_fit(ctx, qs, tok, None)
    assert counts["long"] > counts["short"]
    W = counts["long"] - 1
    with pytest.raises(SystemExit, match="long"):
        check_all_questions_fit(ctx, qs, tok, W)
    assert check_all_questions_fit(ctx, qs, tok, counts["long"]) == counts


# -------------------------------------------------------------------- items 1, 2, 13 (pure)


def test_check_cd_artifact_rules(tmp_path: Path) -> None:
    sha = "a" * 64
    with pytest.raises(RuntimeError, match="missing"):
        check_cd_artifact({"manifest": {"session_sha256": sha}}, sha, "cd.json")
    with pytest.raises(RuntimeError, match="count 0"):
        check_cd_artifact({"nodes": [], "manifest": {"session_sha256": sha}}, sha, "cd.json")
    with pytest.raises(SystemExit) as exc:
        check_cd_artifact({"nodes": CD_RECORDS, "manifest": {"session_sha256": "b" * 64}}, sha, "cd.json")
    assert "b" * 64 in str(exc.value) and sha in str(exc.value)
    with pytest.raises(SystemExit, match="None"):
        check_cd_artifact({"nodes": CD_RECORDS}, sha, "cd.json")
    check_cd_artifact({"nodes": CD_RECORDS, "manifest": {"session_sha256": sha}}, sha, "cd.json")


def test_check_compaction_artifact_rules() -> None:
    good = {"session_sha256": "s", "questions_sha256": "q", "model_id": MODEL_ID, "W": 8000,
            "budget_question": "q?"}
    kw = dict(session_sha="s", questions_sha="q", model_id=MODEL_ID, W=8000, path="c.json")
    check_compaction_artifact(good, **kw)
    for key, bad in [("session_sha256", "x"), ("questions_sha256", "y"), ("model_id", "other"), ("W", 16000)]:
        broken = {**good, key: bad}
        with pytest.raises(SystemExit) as exc:
            check_compaction_artifact(broken, **kw)
        assert repr(bad) in str(exc.value) and repr(good[key]) in str(exc.value)
    with pytest.raises(SystemExit, match="W=None"):
        check_compaction_artifact({k: v for k, v in good.items() if k != "W"}, **kw)
    with pytest.raises(SystemExit, match="budget_question"):
        check_compaction_artifact({k: v for k, v in good.items() if k != "budget_question"}, **kw)


def test_check_per_question_guards() -> None:
    pq = {"f1": {"prompt_tokens": 100, "planet_spans": 3}}
    check_per_question(pq, arm="proposed", w=1.0, inject="planet", W=100, planet_lines=3)
    check_per_question(pq, arm="B", w=0.0, inject="none", W=100, planet_lines=0)
    with pytest.raises(RuntimeError, match="prompt_tokens=100 > W=99"):
        check_per_question(pq, arm="B", w=0.0, inject="none", W=99, planet_lines=0)
    with pytest.raises(RuntimeError, match="planet_lines=4"):  # item 13 equality
        check_per_question(pq, arm="proposed", w=0.0, inject="planet", W=100, planet_lines=4)
    zero = {"f1": {"prompt_tokens": 10, "planet_spans": 0}}
    check_per_question(zero, arm="proposed", w=0.0, inject="planet", W=100, planet_lines=0)
    with pytest.raises(RuntimeError, match="silent baseline"):  # item 1
        check_per_question(zero, arm="proposed", w=1.0, inject="planet", W=100, planet_lines=0)


# -------------------------------------------------------------------- main() with fakes


def test_arm_B_end_to_end_records_hashes_timing_split_and_energy_coverage(monkeypatch, tmp_path: Path) -> None:
    llm = FakeLLM(WsTok())
    _wire_gpu(monkeypatch, llm=llm)
    base = _base(tmp_path)
    assert run_arms.main(["--arm", "B", "--W", "8000"] + base) == 0
    meta = json.loads((tmp_path / "out" / "meta.json").read_text(encoding="utf-8"))
    # item 2: binding fields (H22 (c): session_sha256 = sha of the FILTERED corpus)
    assert meta["session_sha256"] == _corpus_sha(tmp_path / "s.json")
    assert meta["session_file_sha256"] == file_sha256(tmp_path / "s.json")
    assert meta["exclude_rt"] == []
    assert meta["questions_sha256"] == file_sha256(tmp_path / "q.json")
    # arms other than proposed are not window-filtered on their own (the window's
    # oldest recent round trip is still recorded)
    assert meta["questions_used"] == ["f001", "f002"]
    assert meta["questions_dropped_in_window"] == [] and meta["first_recent_rt"] == 0
    assert meta["recent_idx"] == [0, 1]
    assert meta["questions_subset"] is None and meta["questions_subset_sha256"] is None
    assert meta["model_id"] == MODEL_ID and meta["W"] == 8000 and meta["w"] == 0.0
    assert meta["arm"] == "B" and isinstance(meta["git_sha"], str) and meta["git_sha"]
    assert meta["cd_sha256"] is None and meta["compaction_sha256"] is None
    # item 4: baselines run with inject=none
    assert meta["inject"] == "none" and meta["planet_lines"] == 0
    # item 6: E1 split from the reader's attributes
    assert meta["wall_ms_split_available"] is True
    for qid in ("f001", "f002"):
        pq = meta["per_question"][qid]
        assert pq["completion_tokens"] == 3
        assert pq["wall_ms_prefill"] == 12.5 and pq["wall_ms_decode"] == 4.5
        assert pq["wall_ms_total"] > 0
        assert pq["attn_flops_decode"] == attn_flops_decode(pq["prompt_tokens"], 3)
        assert pq["prompt_tokens"] <= 8000
        assert pq["energy_joules"] is not None and pq["energy_joules"] > 0
        assert pq["planet_lines"] == 0 and pq["planet_spans"] == 0
    # item 7: coverage flags
    assert meta["energy_coverage"] == {"first_sample_before_first_question": True,
                                       "last_sample_after_last_question": True}
    # item 9: budgeted with the longest FULL prompt
    assert meta["budget_question"] == QUESTIONS[1]["question"]
    assert meta["max_prompt_tokens"] <= 8000
    timing = [json.loads(ln) for ln in (tmp_path / "out" / "timing.jsonl").read_text().splitlines()]
    assert [t["qid"] for t in timing] == ["f001", "f002"]
    assert all(t["completion_tokens"] == 3 for t in timing)
    answers = json.loads((tmp_path / "out" / "answers.json").read_text(encoding="utf-8"))
    assert answers == {"f001": "25575 is it", "f002": "25575 is it"}


def test_sampler_without_samples_is_a_runtime_error(monkeypatch, tmp_path: Path) -> None:
    llm = FakeLLM(WsTok())
    _wire_gpu(monkeypatch, llm=llm)
    FakeSampler.empty = True
    with pytest.raises(RuntimeError, match="sampler produced no samples"):
        run_arms.main(["--arm", "B", "--W", "8000"] + _base(tmp_path))
    assert llm.generated == []  # no question was asked


def _fake_run_reader(planet_spans: int):
    def fake(llm, context_block, questions, *, w, inject, bias_cap=None, arm=None, progress=None,
             chat_template=False):
        run = ReaderRun()
        for q in questions:
            if w > 0 and planet_spans > 0:
                llm.set_mass_vector("vec")
            else:
                llm.clear_mass_vector()
            llm.generate(build_prompt(context_block, q["question"],
                                      tokenizer=llm.tokenizer if chat_template else None))
            stats = llm.mass_injection_stats()
            llm.clear_mass_vector()
            run.answers[q["qid"]] = "x"
            run.per_question[q["qid"]] = {
                "raw": "x", "positions_found": planet_spans * 4, "spans": planet_spans,
                "planet_spans": planet_spans, "satellite_spans": 0,
                "prompt_tokens": len(build_prompt(context_block, q["question"]).split()),
                **stats,
            }
            if progress:
                progress(q["qid"])
        return run

    return fake


def _corpus_sha(session_path: Path) -> str:
    from benchmark.mcbuild_bench.corpus import load_corpus

    return load_corpus(session_path, exclude_idx=()).sha256


def test_proposed_requires_nodes_and_matching_session_sha_before_model_load(monkeypatch, tmp_path: Path) -> None:
    llm = FakeLLM(WsTok())
    _wire_gpu(monkeypatch, llm=llm)
    _forbid_model_load(monkeypatch)
    base = _base(tmp_path, session=SESSION_LONG)
    sha = _corpus_sha(tmp_path / "s.json")
    cd = _cd_json(tmp_path, session_sha=sha, drop_nodes=True)
    with pytest.raises(RuntimeError, match="'nodes' is missing"):
        run_arms.main(["--arm", "proposed", "--W", "8000", "--w", "1.0", "--cd", str(cd)] + base)
    cd = _cd_json(tmp_path, session_sha=sha, nodes=[])
    with pytest.raises(RuntimeError, match="count 0"):
        run_arms.main(["--arm", "proposed", "--W", "8000", "--w", "1.0", "--cd", str(cd)] + base)
    cd = _cd_json(tmp_path, session_sha="0" * 64)
    with pytest.raises(SystemExit, match=sha):
        run_arms.main(["--arm", "proposed", "--W", "8000", "--w", "1.0", "--cd", str(cd)] + base)
    assert llm.generated == []


def test_proposed_planet_lines_and_spans_guards(monkeypatch, tmp_path: Path) -> None:
    base = _base(tmp_path, session=SESSION_LONG)
    sha = _corpus_sha(tmp_path / "s.json")
    cd = _cd_json(tmp_path, session_sha=sha)
    argv = ["--arm", "proposed", "--W", "8000", "--w", "1.0", "--cd", str(cd)] + base

    # scan found 0 planets while the window has 1 [PN line -> item 1 / 13
    llm = FakeLLM(WsTok())
    _wire_gpu(monkeypatch, llm=llm, run_reader_fn=_fake_run_reader(0))
    with pytest.raises(RuntimeError, match="planet_spans=0 but the window has planet_lines=1"):
        run_arms.main(argv)

    # agreement -> the cell is recorded with planet_lines == planet_spans == 1
    llm = FakeLLM(WsTok())
    _wire_gpu(monkeypatch, llm=llm, run_reader_fn=_fake_run_reader(1))
    assert run_arms.main(argv) == 0
    meta = json.loads((tmp_path / "out" / "meta.json").read_text(encoding="utf-8"))
    assert meta["planet_lines"] == 1 and meta["inject"] == "planet"
    assert all(pq["planet_spans"] == pq["planet_lines"] == 1 for pq in meta["per_question"].values())
    assert meta["cd_sha256"] == file_sha256(cd)
    assert "positions_found" not in {k for k in meta if k != "per_question"}

    # a CD whose window carries no planet line at all is refused before loading (item 1)
    sun_only = [dict(CD_RECORDS[0])]
    cd = _cd_json(tmp_path, session_sha=sha, nodes=sun_only)
    llm = FakeLLM(WsTok())
    _wire_gpu(monkeypatch, llm=llm, run_reader_fn=_fake_run_reader(0))
    _forbid_model_load(monkeypatch)
    with pytest.raises(RuntimeError, match="planet_lines=0"):
        run_arms.main(argv)


def test_arm_C_refuses_a_compaction_from_another_run(monkeypatch, tmp_path: Path) -> None:
    llm = FakeLLM(WsTok())
    _wire_gpu(monkeypatch, llm=llm)
    _forbid_model_load(monkeypatch)
    base = _base(tmp_path)
    comp = {
        "summary": "S", "recent_idx": [1], "calls": [], "n_calls": 0, "W": 8000,
        "model_id": MODEL_ID, "session_sha256": _corpus_sha(tmp_path / "s.json"),
        "questions_sha256": "not-this-file", "budget_question": QUESTIONS[1]["question"],
    }
    p = tmp_path / "c.json"
    p.write_text(json.dumps(comp), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        run_arms.main(["--arm", "C", "--W", "8000", "--compaction-summary", str(p)] + base)
    assert "not-this-file" in str(exc.value) and file_sha256(tmp_path / "q.json") in str(exc.value)
    comp["questions_sha256"] = file_sha256(tmp_path / "q.json")
    comp["W"] = 16000
    p.write_text(json.dumps(comp), encoding="utf-8")
    with pytest.raises(SystemExit, match="W=16000"):
        run_arms.main(["--arm", "C", "--W", "8000", "--compaction-summary", str(p)] + base)


def test_arm_C_end_to_end_binds_the_compaction_artifact(monkeypatch, tmp_path: Path) -> None:
    llm = FakeLLM(WsTok())
    _wire_gpu(monkeypatch, llm=llm)
    base = _base(tmp_path)
    comp = {
        "summary": "Paper 1.21.8 on the box", "recent_idx": [1], "calls": [], "n_calls": 1, "W": 8000,
        "model_id": MODEL_ID, "session_sha256": _corpus_sha(tmp_path / "s.json"),
        "questions_sha256": file_sha256(tmp_path / "q.json"),
        "budget_question": QUESTIONS[1]["question"],
    }
    p = tmp_path / "c.json"
    p.write_text(json.dumps(comp), encoding="utf-8")
    assert run_arms.main(["--arm", "C", "--W", "8000", "--compaction-summary", str(p)] + base) == 0
    meta = json.loads((tmp_path / "out" / "meta.json").read_text(encoding="utf-8"))
    assert meta["compaction_sha256"] == file_sha256(p)
    assert meta["compaction_n_calls"] == 1 and meta["n_recent_rts"] == 1
    assert meta["budget_question"] == QUESTIONS[1]["question"]
    assert meta["inject"] == "none"
    assert count_planet_lines(llm.generated[0]) == 0


# -------------------------------------------------------------------- 2026-09-18 items F1-F5


def test_prefill_forward_and_chunk_helpers() -> None:
    assert n_prefill_forwards(13, None) == 1
    assert n_prefill_forwards(13, 5) == 3
    assert n_prefill_forwards(8192, 8192) == 1
    # F4: a remainder of exactly 1 token bumps the chunk by one
    assert effective_prefill_chunk(13, 4) == 5
    assert effective_prefill_chunk(13, 5) == 5
    assert effective_prefill_chunk(8193, 8192) == 8193
    assert effective_prefill_chunk(13, None) is None


def test_expected_bias_counters_decode_only_and_prefill_scaled() -> None:
    # Astra round 2: the H15 counter is part of the expectation (0 when the switch is off)
    assert expected_bias_counters(16, 4, 1, 0.0) == {
        "bias_applied_calls": 48, "bias_skipped_prefill_calls": 16, "bias_skipped_sliding_calls": 0,
        "bias_applied_prefill_last_row_calls": 0}
    assert expected_bias_counters(16, 4, 3, 0.0) == {
        "bias_applied_calls": 48, "bias_skipped_prefill_calls": 48, "bias_skipped_sliding_calls": 0,
        "bias_applied_prefill_last_row_calls": 0}
    assert expected_bias_counters(16, 1, 1, 0.0)["bias_applied_calls"] == 0
    assert expected_bias_counters(16, 4, 3, 0.5) == {
        "bias_applied_calls": 16 * (3 + 3), "bias_skipped_prefill_calls": 0,
        "bias_skipped_sliding_calls": 0, "bias_applied_prefill_last_row_calls": 0}


def test_check_bias_counters_refuses_a_mismatch() -> None:
    pq = {"bias_applied_calls": 32, "bias_skipped_prefill_calls": 16, "bias_skipped_sliding_calls": 0,
          "completion_tokens": 3}
    check_bias_counters("f1", pq, n_sdpa=16, n_prefill=1, prefill_scale=0.0, decode_forwards=2)
    with pytest.raises(RuntimeError, match="bias_skipped_prefill_calls=16"):
        check_bias_counters("f1", pq, n_sdpa=16, n_prefill=2, prefill_scale=0.0, decode_forwards=2)
    with pytest.raises(RuntimeError, match="decode_forwards"):
        check_bias_counters("f1", pq, n_sdpa=16, n_prefill=1, prefill_scale=0.0, decode_forwards=1)
    sliding = dict(pq, bias_skipped_sliding_calls=1)
    with pytest.raises(RuntimeError, match="sliding"):
        check_bias_counters("f1", sliding, n_sdpa=16, n_prefill=1, prefill_scale=0.0, decode_forwards=2)


def test_proposed_end_to_end_checks_counters_and_records_loading_info(monkeypatch, tmp_path: Path) -> None:
    base = _base(tmp_path, session=SESSION_LONG)
    sha = _corpus_sha(tmp_path / "s.json")
    cd = _cd_json(tmp_path, session_sha=sha)
    argv = ["--arm", "proposed", "--W", "8000", "--w", "1.0", "--cd", str(cd)] + base
    llm = FakeLLM(WsTok())
    _wire_gpu(monkeypatch, llm=llm, run_reader_fn=_fake_run_reader(1))
    assert run_arms.main(argv) == 0
    meta = json.loads((tmp_path / "out" / "meta.json").read_text(encoding="utf-8"))
    assert meta["loaded_class_name"] == "Qwen3_5ForCausalLM"  # F5
    assert meta["loading_info"] == {"missing": 0, "unexpected": 3, "mismatched": 0}
    assert meta["allow_partial_cd"] is False
    for pq in meta["per_question"].values():
        assert pq["bias_applied_calls"] == 16 * 2 and pq["bias_skipped_prefill_calls"] == 16
        assert pq["decode_forwards"] == 2 and pq["n_prefill_forwards"] == 1
        assert pq["attn_flops_decode"] == attn_flops_decode(pq["prompt_tokens"], 3)

    # a reader whose counters disagree with 1 prefill + (n - 1) decode forwards is refused
    class WrongCounts(FakeLLM):
        def generate(self, prompt):
            out = super().generate(prompt)
            self._stats["bias_applied_calls"] = 16 * self.n_tokens  # the old 16 * n expectation
            return out

    _wire_gpu(monkeypatch, llm=WrongCounts(WsTok()), run_reader_fn=_fake_run_reader(1))
    with pytest.raises(RuntimeError, match="bias_applied_calls=48"):
        run_arms.main(argv)


def test_partial_or_stopped_cd_is_refused_unless_allowed(monkeypatch, tmp_path: Path) -> None:
    llm = FakeLLM(WsTok())
    _wire_gpu(monkeypatch, llm=llm, run_reader_fn=_fake_run_reader(1))
    base = _base(tmp_path, session=SESSION_LONG)
    sha = _corpus_sha(tmp_path / "s.json")
    # pure rule
    good = {"nodes": CD_RECORDS, "summary": {"turns": 2}, "manifest": {"session_sha256": sha}}
    check_cd_artifact(good, sha, "cd.json", n_round_trips=2)
    with pytest.raises(SystemExit, match="stopped"):
        check_cd_artifact(dict(good, stopped={"type": "JevStop"}), sha, "cd.json", n_round_trips=2)
    with pytest.raises(SystemExit, match="turns=1"):
        check_cd_artifact(dict(good, summary={"turns": 1}), sha, "cd.json", n_round_trips=2)
    check_cd_artifact(dict(good, summary={"turns": 1}), sha, "cd.json", n_round_trips=2, allow_partial=True)
    # through the CLI: partial -> refused before load; --allow-partial-cd -> runs, recorded
    payload = {"nodes": CD_RECORDS, "summary": {"turns": 1}, "manifest": {"session_sha256": sha},
               "stopped": {"type": "JevStop", "reason": "429"}}
    cd = tmp_path / "cd.json"
    cd.write_text(json.dumps(payload), encoding="utf-8")
    argv = ["--arm", "proposed", "--W", "8000", "--w", "1.0", "--cd", str(cd)] + base
    with pytest.raises(SystemExit, match="stopped"):
        run_arms.main(argv)
    assert llm.generated == []
    assert run_arms.main(argv + ["--allow-partial-cd"]) == 0
    meta = json.loads((tmp_path / "out" / "meta.json").read_text(encoding="utf-8"))
    assert meta["allow_partial_cd"] is True


def test_checkpoint_jsonl_and_resume(monkeypatch, tmp_path: Path) -> None:
    base = _base(tmp_path)
    out = tmp_path / "out"
    # first run dies on the second question: the first answer is already durable
    llm = FakeLLM(WsTok(), fail_on=2)
    _wire_gpu(monkeypatch, llm=llm)
    with pytest.raises(RuntimeError, match="generate failed"):
        run_arms.main(["--arm", "B", "--W", "8000"] + base)
    assert not (out / "answers.json").exists()
    header, done = read_checkpoint(out / "answers.jsonl")
    assert header["arm"] == "B" and header["W"] == 8000 and header["w"] == 0.0
    assert header["model_id"] == MODEL_ID and header["session_sha256"] == _corpus_sha(tmp_path / "s.json")
    assert list(done) == ["f001"]
    assert done["f001"]["answer"] == "25575 is it"
    assert done["f001"]["extra"]["energy_joules"] is not None
    assert done["f001"]["timing"]["completion_tokens"] == 3

    # resume: only f002 is generated, the final files hold both answers
    llm2 = FakeLLM(WsTok())
    _wire_gpu(monkeypatch, llm=llm2)
    assert run_arms.main(["--arm", "B", "--W", "8000", "--resume"] + base) == 0
    assert len(llm2.generated) == 1 and "hackathon box" in llm2.generated[0]
    answers = json.loads((out / "answers.json").read_text(encoding="utf-8"))
    assert answers == {"f001": "25575 is it", "f002": "25575 is it"}
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert set(meta["per_question"]) == {"f001", "f002"}
    assert meta["resumed_qids"] == ["f001"]
    timing = [json.loads(ln) for ln in (out / "timing.jsonl").read_text().splitlines()]
    assert [t["qid"] for t in timing] == ["f001", "f002"]
    _header, done = read_checkpoint(out / "answers.jsonl")
    assert list(done) == ["f001", "f002"]

    # a checkpoint from another cell is refused
    (out / "answers.jsonl").write_text(
        json.dumps(dict(header, w=0.5)) + "\n" + json.dumps({"kind": "answer", "qid": "f001"}) + "\n",
        encoding="utf-8")
    _wire_gpu(monkeypatch, llm=FakeLLM(WsTok()))
    with pytest.raises(SystemExit, match="resume refused"):
        run_arms.main(["--arm", "B", "--W", "8000", "--resume"] + base)


def test_arm_A_sets_the_effective_prefill_chunk_per_question(monkeypatch, tmp_path: Path) -> None:
    llm = FakeLLM(WsTok())
    _wire_gpu(monkeypatch, llm=llm)
    base = _base(tmp_path)
    assert run_arms.main(["--arm", "A", "--W", "full", "--prefill-chunk", "4"] + base) == 0
    meta = json.loads((tmp_path / "out" / "meta.json").read_text(encoding="utf-8"))
    assert meta["prefill_chunk_size"] == 4
    seen = []
    for pq in meta["per_question"].values():
        L = pq["prompt_tokens"]
        assert pq["prefill_chunk_effective"] == effective_prefill_chunk(L, 4)
        assert pq["n_prefill_forwards"] == n_prefill_forwards(L, pq["prefill_chunk_effective"])
        seen.append(pq["prefill_chunk_effective"])
    assert llm.chunks_seen == seen and len(seen) == 2


def test_missing_weight_bytes_is_a_system_exit_before_load(monkeypatch, tmp_path: Path) -> None:
    llm = FakeLLM(WsTok())
    _wire_gpu(monkeypatch, llm=llm)
    _forbid_model_load(monkeypatch)
    monkeypatch.setattr(run_arms, "safetensors_total_bytes", lambda mid: None)
    with pytest.raises(SystemExit, match="weight bytes"):
        run_arms.main(["--arm", "B", "--W", "8000"] + _base(tmp_path))


# -------------------------------------------------------------------- Astra round 2 (items 5, 6)


def _header_kwargs(**over) -> dict:
    base = dict(
        arm="B", W=8000, w=0.0, model_id=MODEL_ID, session_sha="s" * 64, questions_sha="q" * 64,
        cd_sha=None, compaction_sha=None, inject="none", prefill_scale=0.0, max_new_tokens=48,
        prefill_chunk=None, prefill_last_row=False, exclude_rt=(36,), questions_subset_sha=None,
        first_recent_rt=None, questions_dropped_in_window=[],
    )
    base.update(over)
    return base


def test_checkpoint_identity_covers_every_cell_parameter() -> None:
    from benchmark.mcbuild_bench.run_arms import (
        CHECKPOINT_IDENTITY,
        check_checkpoint_header,
        checkpoint_header,
    )

    for field in ("arm", "W", "w", "model_id", "session_sha256", "inject", "prefill_scale",
                  "max_new_tokens", "prefill_chunk", "questions_sha256", "cd_sha256",
                  "compaction_sha256", "prefill_last_row", "exclude_rt", "questions_subset_sha256"):
        assert field in CHECKPOINT_IDENTITY, field
    header = checkpoint_header(**_header_kwargs())
    for field in CHECKPOINT_IDENTITY:
        assert field in header, field
    assert header["exclude_rt"] == [36]  # JSON-stable list, not a tuple
    check_checkpoint_header(header, dict(header))  # same cell: accepted
    check_checkpoint_header(json.loads(json.dumps(header)), header)  # after a JSON round trip too
    changes = {
        "inject": "planet", "prefill_scale": 0.5, "max_new_tokens": 12, "prefill_chunk": 4096,
        "questions_sha": "x" * 64, "cd_sha": "c" * 64, "compaction_sha": "k" * 64,
        "prefill_last_row": True, "exclude_rt": (), "questions_subset_sha": "u" * 64,
    }
    names = {"questions_sha": "questions_sha256", "cd_sha": "cd_sha256",
             "compaction_sha": "compaction_sha256", "questions_subset_sha": "questions_subset_sha256"}
    for kw, value in changes.items():
        other = checkpoint_header(**_header_kwargs(**{kw: value}))
        with pytest.raises(SystemExit, match="resume refused: answers.jsonl %s=" % names.get(kw, kw)):
            check_checkpoint_header(header, other)


def test_resume_refuses_a_different_max_new_tokens_and_records_prefill_last_row(
    monkeypatch, tmp_path: Path
) -> None:
    from benchmark.bineval import run_reader

    load_kwargs: list[dict] = []
    llm = FakeLLM(WsTok())
    _wire_gpu(monkeypatch, llm=llm)

    def fake_load(model_id, **kw):
        load_kwargs.append(kw)
        return llm

    monkeypatch.setattr(run_reader, "load_reader", fake_load)
    base = _base(tmp_path)
    out = tmp_path / "out"
    assert run_arms.main(["--arm", "B", "--W", "8000", "--prefill-last-row"] + base) == 0
    assert load_kwargs[-1]["prefill_last_row"] is True
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["prefill_last_row"] is True
    header, _done = read_checkpoint(out / "answers.jsonl")
    assert header["prefill_last_row"] is True
    assert header["max_new_tokens"] == 48 and header["inject"] == "none"
    assert header["prefill_scale"] == 0.0 and header["prefill_chunk"] is None
    assert header["questions_sha256"] == file_sha256(tmp_path / "q.json")
    # same cell but another max_new_tokens -> refused, naming the field
    with pytest.raises(SystemExit, match="max_new_tokens"):
        run_arms.main(["--arm", "B", "--W", "8000", "--prefill-last-row", "--max-new-tokens", "12",
                       "--resume"] + base)
    # and without the switch -> refused on prefill_last_row
    with pytest.raises(SystemExit, match="prefill_last_row"):
        run_arms.main(["--arm", "B", "--W", "8000", "--resume"] + base)
    # default: off, and load_reader is told so
    assert run_arms.main(["--arm", "B", "--W", "8000"] + base) == 0
    assert load_kwargs[-1]["prefill_last_row"] is False
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["prefill_last_row"] is False


def test_expected_bias_counters_with_prefill_last_row() -> None:
    from benchmark.mcbuild_bench.run_arms import expected_bias_counters

    off = expected_bias_counters(16, 4, 1, 0.0)
    assert off["bias_applied_prefill_last_row_calls"] == 0
    on = expected_bias_counters(16, 4, 1, 0.0, prefill_last_row=True)
    assert on["bias_applied_prefill_last_row_calls"] == 16
    assert on["bias_applied_calls"] == 48 and on["bias_skipped_prefill_calls"] == 16
    pq = {"bias_applied_calls": 48, "bias_skipped_prefill_calls": 16,
          "bias_skipped_sliding_calls": 0, "bias_applied_prefill_last_row_calls": 16,
          "completion_tokens": 4}
    check_bias_counters("f1", pq, n_sdpa=16, n_prefill=1, prefill_scale=0.0, decode_forwards=3,
                        prefill_last_row=True)
    with pytest.raises(RuntimeError, match="bias_applied_prefill_last_row_calls=16"):
        check_bias_counters("f1", pq, n_sdpa=16, n_prefill=1, prefill_scale=0.0,
                            decode_forwards=3)


# -------------------------------------------------------------------- H22 (c) / (d), 2026-09-20


def test_split_questions_by_window_is_pure_and_keeps_absent_questions() -> None:
    from benchmark.mcbuild_bench.run_arms import split_questions_by_window

    qs = QUESTIONS + [ABSENT_QUESTION]
    used, dropped = split_questions_by_window(qs, first_recent_rt=1)
    assert [q["qid"] for q in used] == ["f002", "a001"] and dropped == ["f001"]
    used, dropped = split_questions_by_window(qs, first_recent_rt=0)
    assert [q["qid"] for q in used] == ["a001"] and dropped == ["f001", "f002"]
    used, dropped = split_questions_by_window(qs, first_recent_rt=None)  # no recent part
    assert [q["qid"] for q in used] == ["f001", "f002", "a001"] and dropped == []
    with pytest.raises(SystemExit, match="source_session"):
        split_questions_by_window([{"qid": "x", "kind": "value", "question": "?"}], first_recent_rt=1)


def test_exclude_rt_default_is_checked_and_recorded(monkeypatch, tmp_path: Path) -> None:
    llm = FakeLLM(WsTok())
    _wire_gpu(monkeypatch, llm=llm)
    with_36 = {"round_trips": SESSION["round_trips"] + [
        {"idx": 36, "human": "Retrospective", "events": [{"kind": "text", "text": "took 85 minutes"}]}]}
    base = _base(tmp_path, session=with_36)[:-2]  # the default --exclude-rt 36 applies
    assert run_arms.main(["--arm", "B", "--W", "8000"] + base) == 0
    meta = json.loads((tmp_path / "out" / "meta.json").read_text(encoding="utf-8"))
    assert meta["exclude_rt"] == [36] and meta["n_recent_rts"] == 2
    assert "Retrospective" not in llm.generated[0]
    from benchmark.mcbuild_bench.corpus import load_corpus

    assert meta["session_sha256"] == load_corpus(tmp_path / "s.json").sha256
    header, _ = read_checkpoint(tmp_path / "out" / "answers.jsonl")
    assert header["exclude_rt"] == [36]
    # a session without round trip 36 and no opt-out: refused before any model work
    _forbid_model_load(monkeypatch)
    base = _base(tmp_path)[:-2]
    with pytest.raises(SystemExit, match="36"):
        run_arms.main(["--arm", "B", "--W", "8000"] + base)


def test_proposed_drops_questions_inside_the_window_and_writes_the_subset(monkeypatch, tmp_path: Path) -> None:
    qs = QUESTIONS + [ABSENT_QUESTION]
    base = _base(tmp_path, questions=qs, session=SESSION_LONG)
    sha = _corpus_sha(tmp_path / "s.json")
    cd = _cd_json(tmp_path, session_sha=sha)
    llm = FakeLLM(WsTok())
    _wire_gpu(monkeypatch, llm=llm, run_reader_fn=_fake_run_reader(1))
    argv = ["--arm", "proposed", "--W", "8000", "--w", "1.0", "--cd", str(cd)] + base
    assert run_arms.main(argv) == 0
    out = tmp_path / "out"
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["n_recent_rts"] == 1 and meta["first_recent_rt"] == 1
    assert meta["questions_used"] == ["f002", "a001"]
    assert meta["questions_dropped_in_window"] == ["f001"]
    assert meta["n_questions"] == 2 and set(meta["per_question"]) == {"f002", "a001"}
    answers = json.loads((out / "answers.json").read_text(encoding="utf-8"))
    assert set(answers) == {"f002", "a001"}
    header, done = read_checkpoint(out / "answers.jsonl")
    assert header["first_recent_rt"] == 1 and header["questions_dropped_in_window"] == ["f001"]
    assert header["questions_subset_sha256"] is None
    assert list(done) == ["f002", "a001"]
    subset = json.loads((out / "questions_subset.json").read_text(encoding="utf-8"))
    assert subset["qids"] == ["f002", "a001"] and subset["dropped"] == ["f001"]
    assert subset["first_recent_rt"] == 1 and subset["arm"] == "proposed" and subset["W"] == 8000
    assert subset["session_sha256"] == sha and subset["questions_sha256"] == file_sha256(tmp_path / "q.json")
    assert subset["exclude_rt"] == []
    assert meta["questions_subset_written"] == str(out / "questions_subset.json")


def test_proposed_with_every_question_inside_the_window_is_refused(monkeypatch, tmp_path: Path) -> None:
    llm = FakeLLM(WsTok())
    _wire_gpu(monkeypatch, llm=llm, run_reader_fn=_fake_run_reader(1))
    _forbid_model_load(monkeypatch)
    base = _base(tmp_path)  # tiny SESSION: both round trips fit, first_recent_rt == 0
    cd = _cd_json(tmp_path, session_sha=_corpus_sha(tmp_path / "s.json"))
    with pytest.raises(SystemExit, match="outside"):
        run_arms.main(["--arm", "proposed", "--W", "8000", "--w", "1.0", "--cd", str(cd)] + base)
    assert llm.generated == []


def test_baseline_consumes_the_subset_written_by_the_proposed_run(monkeypatch, tmp_path: Path) -> None:
    qs = QUESTIONS + [ABSENT_QUESTION]
    base = _base(tmp_path, questions=qs, session=SESSION_LONG)
    sha = _corpus_sha(tmp_path / "s.json")
    cd = _cd_json(tmp_path, session_sha=sha)
    _wire_gpu(monkeypatch, llm=FakeLLM(WsTok()), run_reader_fn=_fake_run_reader(1))
    assert run_arms.main(["--arm", "proposed", "--W", "8000", "--w", "1.0", "--cd", str(cd)] + base) == 0
    subset_path = tmp_path / "out" / "questions_subset.json"

    # baseline (arm A, full transcript) on the SAME subset
    llm = FakeLLM(WsTok())
    _wire_gpu(monkeypatch, llm=llm)
    out_a = tmp_path / "out_a"
    argv_a = ["--arm", "A", "--W", "full", "--questions-subset", str(subset_path)] + base
    argv_a[argv_a.index("--out") + 1] = str(out_a)
    assert run_arms.main(argv_a) == 0
    meta = json.loads((out_a / "meta.json").read_text(encoding="utf-8"))
    assert meta["questions_used"] == ["f002", "a001"] and meta["n_questions"] == 2
    assert meta["questions_dropped_in_window"] == [] and meta["first_recent_rt"] == 0  # full transcript
    assert meta["questions_subset"] == str(subset_path)
    assert meta["questions_subset_sha256"] == file_sha256(subset_path)
    assert meta["questions_subset_source"]["arm"] == "proposed" and meta["questions_subset_source"]["W"] == 8000
    assert len(llm.generated) == 2
    header, done = read_checkpoint(out_a / "answers.jsonl")
    assert header["questions_subset_sha256"] == file_sha256(subset_path)
    assert list(done) == ["f002", "a001"]
    # resume with the same subset continues; another subset file is another cell
    _wire_gpu(monkeypatch, llm=FakeLLM(WsTok()))
    assert run_arms.main(argv_a + ["--resume"]) == 0
    other = tmp_path / "other_subset.json"
    other.write_text(json.dumps(dict(json.loads(subset_path.read_text(encoding="utf-8")), qids=["f002"])),
                     encoding="utf-8")
    argv_o = list(argv_a)
    argv_o[argv_o.index("--questions-subset") + 1] = str(other)
    with pytest.raises(SystemExit, match="questions_subset_sha256"):
        run_arms.main(argv_o + ["--resume"])


def test_questions_subset_binding_and_misuse_are_refused(monkeypatch, tmp_path: Path) -> None:
    qs = QUESTIONS + [ABSENT_QUESTION]
    base = _base(tmp_path, questions=qs, session=SESSION_LONG)
    _wire_gpu(monkeypatch, llm=FakeLLM(WsTok()))
    _forbid_model_load(monkeypatch)
    good = {
        "kind": "questions_subset", "arm": "proposed", "W": 8000, "w": 1.0, "first_recent_rt": 1,
        "session_sha256": _corpus_sha(tmp_path / "s.json"),
        "questions_sha256": file_sha256(tmp_path / "q.json"), "exclude_rt": [],
        "qids": ["f002", "a001"], "dropped": ["f001"],
    }
    p = tmp_path / "subset.json"
    # a subset written against other questions / another corpus
    for key in ("questions_sha256", "session_sha256"):
        p.write_text(json.dumps(dict(good, **{key: "0" * 64})), encoding="utf-8")
        with pytest.raises(SystemExit, match=key):
            run_arms.main(["--arm", "A", "--W", "full", "--questions-subset", str(p)] + base)
    # an unknown qid
    p.write_text(json.dumps(dict(good, qids=["f002", "zzz"])), encoding="utf-8")
    with pytest.raises(SystemExit, match="zzz"):
        run_arms.main(["--arm", "A", "--W", "full", "--questions-subset", str(p)] + base)
    # the proposed run PRODUCES the subset; it never consumes one
    p.write_text(json.dumps(good), encoding="utf-8")
    cd = _cd_json(tmp_path, session_sha=good["session_sha256"])
    with pytest.raises(SystemExit, match="questions-subset"):
        run_arms.main(["--arm", "proposed", "--W", "8000", "--w", "1.0", "--cd", str(cd),
                       "--questions-subset", str(p)] + base)
    # a missing file is caught by the path check
    with pytest.raises(SystemExit, match="file not found"):
        run_arms.main(["--arm", "A", "--W", "full", "--questions-subset", str(tmp_path / "nope.json")] + base)

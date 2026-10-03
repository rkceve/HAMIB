"""CPU end-to-end of the second-round cells through the real CLI on the REAL data.

Uses the published corpus, questions and correlation diagram, with the tiny 4-layer Qwen3.5
checkpoint standing in for the 27B reader. Only the GPU preflight is faked (the same way the
other run_arms tests fake it); the window composition, the marker scan with satellite
inheritance, the mass vector, the clamped bias inside the real patched attention, the counters
guard, the checkpoint, meta.json, the subset file and the scorer are all the real code.

The two cell types of pod/round2_run.sh are exercised at W = 8000 (char-level tokens) on 12
questions: the truncation control (``--arm B``) and the proposed arm with
``--inject planet+satellites --bias-cap 3.0``. Answers are meaningless with the tiny model;
what is checked is that the configuration runs, records what it should, and scores.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")

from tests.mcbuild import _tiny_qwen as tq  # noqa: E402
from tests.mcbuild._tiny_qwen import tiny_checkpoint  # noqa: E402,F401  (fixture)
from tests.mcbuild.test_run_arms import FakeSampler  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "benchmark/mcbuild_bench/data"
CD = REPO / "benchmark/mcbuild_bench/results/a100_2026-09-20/v4_36rt/cd.json"
pytestmark = pytest.mark.skipif(not CD.exists(), reason="published cd.json not present")


def _fake_gpu(monkeypatch) -> None:
    """run_arms insists on a GPU; make its checks and the reader's timing calls harmless on CPU."""
    import transformers

    from benchmark.bineval import run_reader
    from benchmark.mcbuild_bench import run_arms

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda i: type("P", (), {"total_memory": 48 * 1024 ** 3})())
    for name, val in (("synchronize", lambda *a, **k: None), ("empty_cache", lambda *a, **k: None),
                      ("memory_allocated", lambda *a, **k: 0), ("max_memory_allocated", lambda *a, **k: 0)):
        monkeypatch.setattr(torch.cuda, name, val, raising=False)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tq.CharTokenizer())
    monkeypatch.setattr(run_reader, "check_model_supported",
                        lambda cfg, allow_linear_layers: {"n_sdpa_layers": tq.N_SDPA_LAYERS})
    monkeypatch.setattr(run_reader, "check_context_fits", lambda *a, **k: {"fits": True})
    monkeypatch.setattr(run_arms, "safetensors_total_bytes", lambda mid: 55_600_000_000)
    monkeypatch.setattr(run_arms, "GpuSampler", FakeSampler)
    monkeypatch.setattr(run_arms, "SAMPLER_WARMUP_TIMEOUT_S", 0.3)
    monkeypatch.setattr(run_arms, "SAMPLER_TAIL_TIMEOUT_S", 0.3)
    monkeypatch.setattr(run_arms, "SAMPLER_POLL_S", 0.01)
    FakeSampler.empty = False


def test_round2_cells_run_end_to_end_on_cpu(tiny_checkpoint: Path, monkeypatch, tmp_path: Path) -> None:  # noqa: F811
    from benchmark.bineval import run_reader
    from benchmark.mcbuild_bench import run_arms, score_cells

    _fake_gpu(monkeypatch)
    qs = json.loads((DATA / "questions.json").read_text(encoding="utf-8"))
    picked = [q for q in qs if q["kind"] != "absent"][:11] + [q for q in qs if q["kind"] == "absent"][:1]
    qfile = tmp_path / "q12.json"
    qfile.write_text(json.dumps(picked, ensure_ascii=False), encoding="utf-8")

    loaded: list[dict] = []
    live: list = []

    def real_load(model_id, **kw):
        # the real reader on the tiny checkpoint, configured exactly as run_arms asked; each pod
        # cell is its own process, so here the previous reader hands back the attention patch first
        loaded.append(kw)
        while live:
            live.pop().restore_sdpa()
        llm = tq.make_llm(tiny_checkpoint, monkeypatch, w=kw["w"], max_new_tokens=kw["max_new_tokens"])
        llm._prefill_mass_scale = kw["prefill_scale"]
        llm._bias_cap = kw["bias_cap"]
        live.append(llm)
        return llm

    monkeypatch.setattr(run_reader, "load_reader", real_load)
    base = ["--model-id", str(tiny_checkpoint), "--questions", str(qfile),
            "--session", str(DATA / "session_redacted.json"), "--prefill-chunk", "8192",
            "--prefill-scale", "0.0", "--max-new-tokens", "4", "--quantization", "none"]
    out = tmp_path / "round2"

    def run(cell: str, *args: str) -> dict:
        argv = list(args) + base + ["--out", str(out / cell), "--gpu-csv", str(out / cell / "gpu.csv")]
        assert run_arms.main(argv) == 0, cell
        return json.loads((out / cell / "meta.json").read_text(encoding="utf-8"))

    try:
        # truncation control: raw transcript cut to W, no diagram, no bias
        m_b = run("truncB_W8000", "--arm", "B", "--W", "8000", "--w", "0", "--inject", "none")
        assert m_b["inject"] == "none" and m_b["bias_cap"] is None and loaded[-1]["bias_cap"] is None
        assert m_b["n_recent_rts"] >= 1 and len(m_b["per_question"]) == 12

        # proposed: satellites inherit their planet's mass, effective bias capped at 3.0
        m_p = run("proposed_W8000_w1.0", "--arm", "proposed", "--W", "8000", "--w", "1.0",
                  "--inject", "planet+satellites", "--bias-cap", "3.0", "--cd", str(CD))
        assert m_p["inject"] == "planet+satellites" and m_p["bias_cap"] == 3.0
        assert loaded[-1]["bias_cap"] == 3.0 and loaded[-1]["w"] == 1.0
        pq = next(iter(m_p["per_question"].values()))
        assert pq["planet_spans"] >= 1
        assert pq["positions_found"] > pq["planet_spans"], "satellite inheritance added no positions"
        # counters guard passed inside run_arms; the figures are 1 sdpa layer x (4 - 1) decode steps
        assert pq["bias_applied_calls"] == tq.N_SDPA_LAYERS * 3
        subset = json.loads((out / "proposed_W8000_w1.0" / "questions_subset.json").read_text(encoding="utf-8"))
        assert len(subset["qids"]) == 12 and subset["dropped"] == []
        header = json.loads((out / "proposed_W8000_w1.0" / "answers.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert header["bias_cap"] == 3.0 and header["inject"] == "planet+satellites"
    finally:
        while live:
            live.pop().restore_sdpa()

    # the bias the reader saw really is clamped at 3.0: rebuild the vector for this window
    from benchmark.bineval.arms import cd_from_records
    from benchmark.bineval.run_reader import build_prompt
    from benchmark.mcbuild_bench.corpus import load_corpus
    from benchmark.mcbuild_bench.windows import build_window
    from communication.cd_serializer import CDSerializer
    from server.cd_parser import find_marker_spans, marker_positions
    from server.mass_vector import positions_to_mass_vector
    from server.mass_weighted_gemma import build_mass_bias

    tok = tq.CharTokenizer()
    cd = cd_from_records(json.loads(CD.read_text(encoding="utf-8"))["nodes"])
    block = CDSerializer(level_markers=True).to_context_block(cd)
    win = build_window("proposed", 8000, block, load_corpus(DATA / "session_redacted.json").round_trips,
                       tok, picked[0]["question"], cd=cd, chat_template=True)
    ids = tok(build_prompt(win["prompt_context"], picked[0]["question"], tokenizer=tok))["input_ids"]
    pos = marker_positions(find_marker_spans(ids, tok), inject_levels={"planet"}, satellite_inherit=True)
    vec = positions_to_mass_vector(pos, len(ids), cap=float("inf"), scale=1.0)
    bias = build_mass_bias(1, len(ids), m_matrix=None, mass_vector=vec, mass_weight=1.0, prefill_mass_scale=0.0,
                           dtype=torch.float32, device=torch.device("cpu"), bias_cap=3.0, phase_is_decode=True)
    assert float(bias.max()) == pytest.approx(3.0)
    assert max(m for _, m in pos) > 3.0  # the cap did something: raw masses exceed it

    # the scorer pairs the proposed cell with the truncation control at the same W
    shutil.copytree(out / "proposed_W8000_w1.0", out / "A_full")  # stand-in baseline for the plumbing check
    assert score_cells.main(["--main", str(out), "--questions", str(qfile),
                             "--out", str(tmp_path / "scores.json"), "--md", str(tmp_path / "scores.md")]) == 0
    (cell,) = json.loads((tmp_path / "scores.json").read_text(encoding="utf-8"))["cells"]
    assert cell["b_dir"] == "truncB_W8000" and "paired_vs_B" in cell
    assert "truncation pass" in (tmp_path / "scores.md").read_text(encoding="utf-8")

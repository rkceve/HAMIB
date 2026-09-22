"""Item 7: the injection path end to end THROUGH run_reader on the tiny model.

A tiny CorrelationDiagram is serialized with ``CDSerializer(level_markers=True)``,
the prompt is built by ``run_reader.build_prompt`` and answered through
``run_reader.run_reader(..., inject="planet")`` with the char tokenizer and the
tiny Qwen3_5 checkpoint.  Assertions:

  * ``planet_spans`` == number of ``[PN`` lines in the context block;
  * the mass vector's support == the union of planet-text token positions from
    ``cd_parser.find_marker_spans``;
  * ``bias_applied_calls == n_sdpa_layers * (n - 1)``;
  * with attention recording on, the summed attention on planet positions of
    the first decode step is strictly larger for w = 2.0 than for w = 0.0 (same
    prompt, greedy).
"""

from __future__ import annotations

import time

import pytest
import torch

pytest.importorskip("transformers")

from benchmark.bineval import run_reader as rr  # noqa: E402
from communication.cd_serializer import CDSerializer  # noqa: E402
from models.correlation_diagram import CorrelationDiagram  # noqa: E402
from models.node import Node, NodeLevel  # noqa: E402
from server.cd_parser import find_marker_spans  # noqa: E402
from tests.mcbuild._tiny_qwen import N_SDPA_LAYERS  # noqa: E402

QUESTION = [{"qid": "q1", "question": "What is the RCON port?"}]


def _tiny_cd() -> CorrelationDiagram:
    cd = CorrelationDiagram()
    sun = Node(text="Hackathon build agent", level=NodeLevel.SUN, mass=1.0, node_id="s1")
    assert cd.add_sun(sun)
    p1 = Node(text="RCON port is 25575", level=NodeLevel.PLANET, mass=2.0, node_id="p1")
    p2 = Node(text="Server runs Paper", level=NodeLevel.PLANET, mass=1.0, node_id="p2")
    assert cd.add_planet(p1, "s1") and cd.add_planet(p2, "s1")
    sat = Node(text="spark is disabled", level=NodeLevel.SATELLITE, mass=0.0, node_id="r1")
    assert cd.add_satellite(sat, "p1")
    return cd


def _run(llm, block: str, w: float) -> tuple[rr.ReaderRun, torch.Tensor | None, list]:
    """One run_reader call at weight w; returns (run, mass vector set, recorded rows)."""
    llm._mass_weight = w
    seen: list = []
    real_set = llm.set_mass_vector

    def spy(vec):
        seen.append(vec.detach().clone())
        real_set(vec)

    llm.set_mass_vector = spy
    llm.start_attention_recording()
    try:
        run = rr.run_reader(llm, block, QUESTION, w=w, inject="planet", arm="cd_test")
    finally:
        rows = llm.stop_attention_recording()
        del llm.set_mass_vector
    return run, (seen[0] if seen else None), rows


def test_tiny_e2e_through_run_reader(llm) -> None:
    t0 = time.perf_counter()
    block = CDSerializer(level_markers=True).to_context_block(_tiny_cd())
    n_pn_lines = sum(1 for ln in block.splitlines() if ln.strip().startswith("[PN"))
    assert n_pn_lines == 2
    prompt = rr.build_prompt(block, QUESTION[0]["question"])
    ids = llm.tokenizer(prompt, return_tensors=None)["input_ids"]
    spans = find_marker_spans(ids, llm.tokenizer)
    planet_positions = sorted({p for lvl, _m, pos in spans if lvl == "planet" for p in pos})
    assert planet_positions, "the scan found no planet text"

    shares: dict[float, float] = {}
    for w in (0.0, 2.0):
        run, vec, rows = _run(llm, block, w)
        pq = run.per_question["q1"]
        assert pq["planet_spans"] == n_pn_lines
        assert pq["prompt_tokens"] == len(ids)
        assert vec is not None
        support = torch.nonzero(vec, as_tuple=False).flatten().tolist()
        assert support == planet_positions
        n = llm.last_generated_tokens
        assert n > 1
        assert pq["bias_applied_calls"] == N_SDPA_LAYERS * (n - 1)
        assert pq["bias_skipped_prefill_calls"] == N_SDPA_LAYERS
        assert pq["bias_skipped_sliding_calls"] == 0
        # one recorded row per sdpa layer per DECODE step
        assert len(rows) == N_SDPA_LAYERS * (n - 1)
        first = rows[0]
        assert first.shape[0] == len(ids) + 1  # the first decode step sees prompt + 1 key
        shares[w] = float(first[planet_positions].sum()) / float(first.sum())
    assert shares[2.0] > shares[0.0], shares
    print("tiny e2e through run_reader: %.1f s, shares=%r" % (time.perf_counter() - t0, shares))

"""CPU tests for experiments/gpu_preflight.py -- NO model, NO network.

Every stage of the pre-flight checklist is driven twice: once with fakes that
behave like a healthy GPU box, and once with fakes that reproduce the failure
the stage exists to catch (a tokenizer that eats a node, a model on eager
attention, a patch that never fires, a judge serving the wrong model, a context
that does not fit, a micro-run that injects nothing).

The tokenizer fake is a piece-list tokenizer in the style of
tests/reader/test_run_reader.py / tests/test_level_markers.py: it splits the
text it is handed and decodes each id back to its piece, so ``find_marker_spans``
sees exactly the merges the fake was built to produce -- including the F5
``"] word"`` merge that a real BPE performs.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from experiments import gpu_preflight as pf

# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------

_MARKER_LINE = re.compile(r"^(\s*)(\[(?:SN|RN|PN[\d.]+)\])(.*)$", re.S)
_WORDS = re.compile(r"\s*\S+|\s+$")


def _split(
    text: str,
    *,
    merged: bool = False,
    drop_node: int | None = None,
    rewrite: dict[int, str] | None = None,
) -> tuple[list[str], dict[int, str]]:
    """Text -> (pieces, decode overrides).

    ``merged=True`` glues each marker's closing ``]`` to the first word of the
    node text (the real-BPE case of F5).  ``drop_node=k`` makes the k-th marker
    decode to the empty string (a tokenizer that eats the marker).  ``rewrite``
    maps a marker index to the string its token decodes to.
    """
    pieces: list[str] = []
    overrides: dict[int, str] = {}
    node_index = 0
    for line in text.splitlines(keepends=True):
        m = _MARKER_LINE.match(line)
        if not m:
            pieces.append(line)
            continue
        indent, marker, rest = m.groups()
        if indent:
            pieces.append(indent)
        words = _WORDS.findall(rest) or [rest]
        if merged and words:
            pieces.append(marker[:-1])
            marker_pos = len(pieces) - 1
            pieces.append("]" + words[0])
            pieces.extend(words[1:])
        else:
            pieces.append(marker)
            marker_pos = len(pieces) - 1
            pieces.extend(words)
        if drop_node == node_index:
            overrides[marker_pos] = ""
        if rewrite and node_index in rewrite:
            overrides[marker_pos] = rewrite[node_index]
        node_index += 1
    return pieces, overrides


class FakeTok:
    """Stateless-looking piece tokenizer: encode splits, decode joins pieces."""

    def __init__(self, **split_kwargs: Any) -> None:
        self._kwargs = split_kwargs
        self._pieces: list[str] = []
        self._overrides: dict[int, str] = {}

    def __call__(self, text: str, **_kw: Any) -> dict:
        self._pieces, self._overrides = _split(text, **self._kwargs)
        return {"input_ids": list(range(len(self._pieces)))}

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return "".join(
            self._overrides.get(i, self._pieces[i]) for i in ids
        )


class FakeLLM:
    """A MassWeightedLLM stand-in: counters, recorder and a scripted generate."""

    def __init__(
        self,
        tokenizer: FakeTok,
        *,
        n_layers: int = 4,
        gain: float = 4.0,
        applied_layers: int | None = None,
        sliding_skips: int = 0,
        texts: list[str] | None = None,
        attn: str = "sdpa",
    ) -> None:
        self._tok = tokenizer
        self.n_layers = n_layers
        self.gain = gain
        self.applied_layers = n_layers if applied_layers is None else applied_layers
        self.sliding_skips = sliding_skips
        self._texts = list(texts) if texts else None
        self._mass_weight = 0.0
        self._max_new_tokens = 48
        self._vec: Any = None
        self._recording = False
        self.recorded_attention: list[Any] = []
        self.max_new_tokens_seen: list[int] = []
        self._model = SimpleNamespace(
            config=SimpleNamespace(
                num_hidden_layers=n_layers, _attn_implementation=attn,
            ),
            parameters=lambda: iter([]),
        )
        self._reset()

    # -- MassWeightedLLM surface --------------------------------------------
    @property
    def tokenizer(self) -> FakeTok:
        return self._tok

    def _reset(self) -> None:
        self.bias_applied_calls = 0
        self.bias_skipped_prefill_calls = 0
        self.bias_skipped_sliding_calls = 0

    def set_mass_vector(self, v: Any) -> None:
        self._vec = v
        self._reset()

    def clear_mass_vector(self) -> None:
        self._vec = None
        self._reset()

    def mass_injection_stats(self) -> dict[str, int]:
        return {
            "bias_applied_calls": self.bias_applied_calls,
            "bias_skipped_prefill_calls": self.bias_skipped_prefill_calls,
            "bias_skipped_sliding_calls": self.bias_skipped_sliding_calls,
        }

    def start_attention_recording(self) -> None:
        self.recorded_attention = []
        self._recording = True

    def stop_attention_recording(self) -> list[Any]:
        self._recording = False
        return self.recorded_attention

    def generate(self, prompt: str) -> str:
        self.max_new_tokens_seen.append(self._max_new_tokens)
        self.bias_applied_calls = self.applied_layers
        self.bias_skipped_prefill_calls = self.n_layers
        self.bias_skipped_sliding_calls = self.sliding_skips
        if self._recording:
            self.recorded_attention.extend(self._rows())
        if self._texts:
            return self._texts.pop(0)
        return "800,000 yen"

    def _rows(self) -> list[Any]:
        vec = self._vec
        length = int(vec.numel()) if vec is not None else 8
        weights = torch.ones(length, dtype=torch.float32)
        if vec is not None and self._mass_weight > 0:
            weights = weights + self.gain * float(self._mass_weight) * vec.float()
        probs = weights / weights.sum()
        return [probs.clone() for _ in range(self.n_layers)]


class FakeJudge:
    """OpenAICompatJudge stand-in: one canned reply per prompt kind."""

    def __init__(self, node_reply: str, yesno_reply: str = "yes") -> None:
        self.node_reply = node_reply
        self.yesno_reply = yesno_reply
        self._base_url = "http://127.0.0.1:8000"
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, max_tokens: int) -> str:
        self.prompts.append(prompt)
        if "JSON object" in prompt:
            return self.node_reply
        return self.yesno_reply


GOOD_NODE_REPLY = (
    '{"summary": "The rent deposit is 4,800,000 yen.", "comprehensiveness": 20, '
    '"independence": 40, "detail": 90}'
)


def fake_torch(*, cuda: bool = True, name: str = "NVIDIA H200", gb: float = 141.0) -> Any:
    return SimpleNamespace(
        __version__="2.11.0+cu128",
        version=SimpleNamespace(cuda="12.8"),
        cuda=SimpleNamespace(
            is_available=lambda: cuda,
            device_count=lambda: 1 if cuda else 0,
            get_device_properties=lambda _i: SimpleNamespace(
                name=name, total_memory=int(gb * (1024 ** 3))
            ),
        ),
    )


FAKE_TRANSFORMERS = SimpleNamespace(__version__="5.8.0")

QWEN_CONFIG = SimpleNamespace(
    model_type="qwen3",
    layer_types=["full_attention"] * 4,
    use_sliding_window=False,
    sliding_window=None,
    sliding_window_pattern=None,
    max_position_embeddings=262144,
    num_hidden_layers=4,
    num_key_value_heads=4,
    head_dim=128,
)
GEMMA_CONFIG = SimpleNamespace(
    model_type="gemma3",
    layer_types=["sliding_attention", "full_attention"],
    use_sliding_window=True,
    sliding_window=1024,
    sliding_window_pattern=6,
    max_position_embeddings=8192,
    num_hidden_layers=2,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def test_preflight_cd_is_three_level_with_distinct_planet_masses() -> None:
    block = pf.preflight_context_block()
    lines = pf.parse_block_lines(block)
    assert [lvl for lvl, _m, _t in lines] == [
        "sun", "planet", "satellite", "satellite", "planet", "satellite"
    ]
    assert [m for lvl, m, _t in lines if lvl == "planet"] == [2.0, 1.0]


def test_normalize_text_strips_the_merged_marker_residue() -> None:
    assert pf.normalize_text("]  Budget ") == "Budget"
    assert pf.normalize_text("[PN2.0] Budget") == "Budget"
    assert pf.normalize_text("Budget\n") == "Budget"
    # a genuinely different text still differs
    assert pf.normalize_text("] Budgets") != "Budget"


def test_judge_base_url_from_a_real_openai_compat_judge() -> None:
    from management.harness.backends import OpenAICompatJudge

    judge = OpenAICompatJudge("http://127.0.0.1:8000/", "Qwen/Qwen3.8-27B")
    assert pf.judge_base_url(judge) == "http://127.0.0.1:8000"


def test_attn_implementation_of_model_and_wrapper() -> None:
    raw = SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa"))
    assert pf.attn_implementation_of(raw) == "sdpa"
    wrapper = SimpleNamespace(attn_implementation="eager")
    assert pf.attn_implementation_of(wrapper) == "eager"
    # a wrapper with no property of its own: read the wrapped model's config
    wrapped = SimpleNamespace(
        _model=SimpleNamespace(config=SimpleNamespace(_attn_implementation="eager"))
    )
    assert pf.attn_implementation_of(wrapped) == "eager"
    assert pf.attn_implementation_of(SimpleNamespace()) is None


# --------------------------------------------------------------------------
# S0 env
# --------------------------------------------------------------------------

def test_env_records_versions_and_gpu() -> None:
    res = pf.stage_env(torch_mod=fake_torch(), transformers_mod=FAKE_TRANSFORMERS)
    assert res.ok and res.name == "env"
    assert res.detail["torch"] == "2.11.0+cu128"
    assert res.detail["transformers"] == "5.8.0"
    assert res.detail["cuda_available"] is True
    assert res.detail["gpu_name"] == "NVIDIA H200"
    assert res.detail["gpu_total_memory_gb"] == pytest.approx(141.0, abs=0.01)


def test_env_fails_without_cuda() -> None:
    res = pf.stage_env(
        torch_mod=fake_torch(cuda=False), transformers_mod=FAKE_TRANSFORMERS
    )
    assert not res.ok
    assert "CUDA is not available" in res.message
    assert res.detail["gpu_name"] is None


def test_env_can_be_run_on_cpu_for_diagnostics() -> None:
    res = pf.stage_env(
        torch_mod=fake_torch(cuda=False),
        transformers_mod=FAKE_TRANSFORMERS,
        require_cuda=False,
    )
    assert res.ok


# --------------------------------------------------------------------------
# S1 tokenizer markers
# --------------------------------------------------------------------------

def test_tokenizer_stage_passes_on_a_clean_tokenizer() -> None:
    res = pf.stage_tokenizer(FakeTok())
    assert res.ok, res.message
    assert res.detail["spans"] == res.detail["block_lines"] == 6
    assert res.detail["span_levels"] == [
        "sun", "planet", "satellite", "satellite", "planet", "satellite"
    ]
    assert res.detail["span_masses"] == [0.0, 2.0, 0.0, 0.0, 1.0, 0.0]
    assert res.detail["empty_spans"] == []


def test_tokenizer_stage_passes_on_the_merged_bracket_tokenizer() -> None:
    """F5: a BPE that glues '] word' into one token must still yield spans."""
    res = pf.stage_tokenizer(FakeTok(merged=True))
    assert res.ok, res.message
    assert res.detail["empty_spans"] == []
    assert res.detail["span_masses"] == [0.0, 2.0, 0.0, 0.0, 1.0, 0.0]


def test_tokenizer_stage_fails_and_names_the_node_a_tokenizer_dropped() -> None:
    # the 6th node line (the last satellite) loses its marker entirely
    res = pf.stage_tokenizer(FakeTok(drop_node=5))
    assert not res.ok
    assert "The site is in Nakameguro" in res.message
    assert res.detail["missing_nodes"] == ["The site is in Nakameguro"]
    assert res.detail["spans"] == 5 and res.detail["block_lines"] == 6


def test_tokenizer_stage_fails_when_a_planet_mass_is_mangled() -> None:
    res = pf.stage_tokenizer(FakeTok(rewrite={1: "[PN9.0]"}))
    assert not res.ok
    assert "mass 9" in res.message and "expected 2" in res.message
    assert res.detail["span_masses"][1] == 9.0


# --------------------------------------------------------------------------
# S2 model config
# --------------------------------------------------------------------------

def test_config_stage_accepts_a_qwen_like_config_without_a_model() -> None:
    res = pf.stage_config(QWEN_CONFIG)
    assert res.ok, res.message
    assert res.detail["num_hidden_layers"] == 4
    assert res.detail["max_position_embeddings"] == 262144
    assert res.detail["attn_implementation"] is None


def test_config_stage_descends_into_text_config() -> None:
    wrapper = SimpleNamespace(model_type="qwen3_vl", text_config=QWEN_CONFIG)
    res = pf.stage_config(wrapper)
    assert res.ok and res.detail["num_hidden_layers"] == 4


def test_config_stage_asserts_sdpa_on_the_loaded_model() -> None:
    model = SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa"))
    assert pf.stage_config(QWEN_CONFIG, model=model).ok
    eager = SimpleNamespace(config=SimpleNamespace(_attn_implementation="eager"))
    res = pf.stage_config(QWEN_CONFIG, model=eager)
    assert not res.ok
    assert "eager" in res.message and "sdpa" in res.message


def test_config_stage_rejects_a_sliding_window_model() -> None:
    res = pf.stage_config(GEMMA_CONFIG)
    assert not res.ok
    assert "sliding_attention" in res.message


# --------------------------------------------------------------------------
# A3 (2026-09-07): the hybrid is the DEFAULT reader, so its layer counts are
# asserted from the model id, and the linear-attention kernels are reported.
# --------------------------------------------------------------------------

HYBRID_TEXT_CONFIG = SimpleNamespace(
    model_type="qwen3_5_text",
    num_hidden_layers=64,
    layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 16,
    num_key_value_heads=4,
    head_dim=256,
    max_position_embeddings=262144,
)
HYBRID_CONFIG = SimpleNamespace(model_type="qwen3_5", text_config=HYBRID_TEXT_CONFIG)
HYBRID_ID = "Qwen/Qwen3.8-27B"


def test_config_stage_accepts_the_hybrid_with_exactly_16_sdpa_layers() -> None:
    res = pf.stage_config(
        HYBRID_CONFIG, allow_linear_layers=True, model_id=HYBRID_ID
    )
    assert res.ok, res.message
    assert res.detail["n_sdpa_layers"] == 16
    assert res.detail["n_linear_layers"] == 48
    assert res.detail["n_sdpa_layers_expected"] == 16
    assert res.detail["model_id"] == HYBRID_ID


def test_config_stage_fails_when_the_hybrid_is_not_16_of_64() -> None:
    """A wrong revision / re-uploaded config must not run silently."""
    off = SimpleNamespace(
        model_type="qwen3_5",
        text_config=SimpleNamespace(
            model_type="qwen3_5_text",
            num_hidden_layers=64,
            layer_types=(["linear_attention"] * 7 + ["full_attention"]) * 8,
            max_position_embeddings=262144,
        ),
    )
    res = pf.stage_config(off, allow_linear_layers=True, model_id=HYBRID_ID)
    assert not res.ok
    assert "exactly 16 full-attention layers" in res.message
    assert "n_sdpa_layers=8" in res.message


def test_config_stage_has_no_layer_expectation_for_a_dense_model() -> None:
    res = pf.stage_config(QWEN_CONFIG, model_id="Qwen/Qwen3-32B")
    assert res.ok and res.detail["n_sdpa_layers_expected"] is None


def test_config_stage_reports_the_linear_attention_kernels(monkeypatch) -> None:
    """Reported, never asserted: the fallback is correct, only slower."""
    fake = {
        "fla_importable": False, "fla_version": None,
        "causal_conv1d_importable": False, "torch_cuda_available": False,
        "fast_path_available": False, "would_use_torch_fallback": True,
        "error": None,
    }
    monkeypatch.setattr(pf, "linear_attention_kernel_report", lambda: fake)
    res = pf.stage_config(
        HYBRID_CONFIG, allow_linear_layers=True, model_id=HYBRID_ID
    )
    assert res.ok
    assert res.detail["linear_attention_kernels"] == fake


def test_the_kernel_report_never_raises_on_this_cpu_box() -> None:
    """It runs inside a pre-flight stage; a missing package is data, not a crash."""
    from benchmark.bineval.run_reader import linear_attention_kernel_report

    rep = linear_attention_kernel_report()
    assert set(rep) == {
        "fla_importable", "fla_version", "causal_conv1d_importable",
        "torch_cuda_available", "fast_path_available",
        "would_use_torch_fallback", "error",
    }
    if rep["error"] is None:
        # CPU-only dev box: transformers gates BOTH on torch CUDA
        # (utils/import_utils.py:810-817), so the torch fallback is used.
        assert rep["fast_path_available"] is False
        assert rep["would_use_torch_fallback"] is True


# --------------------------------------------------------------------------
# S3 single-decode injection
# --------------------------------------------------------------------------

def _inject_llm(**kwargs: Any) -> FakeLLM:
    return FakeLLM(FakeTok(), **kwargs)


def test_inject_stage_passes_and_reports_a_growing_share() -> None:
    llm = _inject_llm()
    res = pf.stage_inject(llm)
    assert res.ok, res.message
    shares = res.detail["mass_share"]
    assert shares["w=0.5"] > shares["w=0"] > 0
    assert res.detail["stats_high"]["bias_applied_calls"] == 4
    assert res.detail["stats_high"]["bias_skipped_prefill_calls"] == 4
    assert res.detail["positions_found"] > 0
    # max_new_tokens=2, not 1: the first new token comes out of the prefill, so
    # 1 would run no decode step at all.
    assert llm.max_new_tokens_seen == [pf.INJECT_MAX_NEW_TOKENS] * 3
    assert llm._max_new_tokens == 48  # restored


def test_inject_stage_fails_when_the_patch_never_fires() -> None:
    res = pf.stage_inject(_inject_llm(applied_layers=0))
    assert not res.ok
    assert res.detail["checks"]["bias_applied_calls_match"] is False
    assert "bias_applied_calls_match" in res.message


def test_inject_stage_fails_when_a_sliding_layer_skipped_the_bias() -> None:
    res = pf.stage_inject(_inject_llm(sliding_skips=2))
    assert not res.ok
    assert res.detail["checks"]["no_sliding_skips"] is False


def test_inject_stage_fails_when_the_share_does_not_grow() -> None:
    res = pf.stage_inject(_inject_llm(gain=0.0))
    assert not res.ok
    assert res.detail["checks"]["share_increases"] is False
    assert res.detail["checks"]["deterministic_at_w0"] is True


def test_inject_stage_fails_on_non_deterministic_decoding() -> None:
    res = pf.stage_inject(_inject_llm(texts=["yen", "YEN", "yen"]))
    assert not res.ok
    assert res.detail["checks"]["deterministic_at_w0"] is False


def test_inject_stage_fails_on_an_empty_generation() -> None:
    res = pf.stage_inject(_inject_llm(texts=["", "", ""]))
    assert not res.ok
    assert res.detail["checks"]["text_nonempty_at_w0"] is False


def test_inject_stage_fails_when_the_prompt_carries_no_planet() -> None:
    llm = _inject_llm()
    res = pf.stage_inject(llm, prompt="<CONTEXT>\n[SN] no planet here\n</CONTEXT>\n")
    assert not res.ok
    assert "0 planet positions" in res.message


# --------------------------------------------------------------------------
# S4 judge endpoint
# --------------------------------------------------------------------------

def test_judge_stage_passes_and_reports_latency() -> None:
    judge = FakeJudge(GOOD_NODE_REPLY)
    res = pf.stage_judge(
        judge,
        model_id="Qwen/Qwen3.8-27B",
        list_models_fn=lambda _url: ["Qwen/Qwen3.8-27B"],
    )
    assert res.ok, res.message
    assert res.detail["node_summary"].startswith("The rent deposit")
    assert res.detail["node_scores"] == {
        "comprehensiveness": 20, "independence": 40, "detail": 90
    }
    assert res.detail["yesno"] is True
    assert res.detail["node_latency_s"] >= 0
    assert res.detail["yesno_latency_s"] >= 0
    assert res.detail["base_url"] == "http://127.0.0.1:8000"
    # the fragment reaches the model verbatim inside the fence
    assert pf.JUDGE_FRAGMENT in judge.prompts[0]


def test_judge_stage_accepts_a_stripped_think_block() -> None:
    reply = "<think>the deposit is six months of rent</think>" + GOOD_NODE_REPLY
    res = pf.stage_judge(FakeJudge(reply), model_id=None)
    assert res.ok, res.message


def test_judge_stage_fails_on_prose_and_logs_the_raw_reply() -> None:
    res = pf.stage_judge(FakeJudge("Sure! The deposit is 4.8M yen."), model_id=None)
    assert not res.ok
    assert "did not parse" in res.message
    assert res.detail["node_reply"] == "Sure! The deposit is 4.8M yen."


def test_judge_stage_fails_on_unparsable_yes_no() -> None:
    res = pf.stage_judge(
        FakeJudge(GOOD_NODE_REPLY, yesno_reply="yes and no"), model_id=None
    )
    assert not res.ok
    assert "yes/no" in res.message
    assert res.detail["yesno_reply"] == "yes and no"


def test_judge_stage_fails_on_surviving_think_markup() -> None:
    res = pf.stage_judge(
        FakeJudge("<think>still reasoning" + GOOD_NODE_REPLY), model_id=None
    )
    assert not res.ok
    assert "reasoning markup" in res.message
    assert res.detail["think_leaked"] == ["node"]


def test_judge_stage_fails_when_the_endpoint_serves_another_model() -> None:
    res = pf.stage_judge(
        FakeJudge(GOOD_NODE_REPLY),
        model_id="Qwen/Qwen3.8-27B",
        list_models_fn=lambda _url: ["meta-llama/Llama-3-8B"],
    )
    assert not res.ok
    assert "Qwen/Qwen3.8-27B" in res.message and "Llama-3-8B" in res.message


def test_judge_stage_fails_when_the_models_endpoint_is_unreachable() -> None:
    def boom(_url: str) -> list[str]:
        raise OSError("connection refused")

    res = pf.stage_judge(
        FakeJudge(GOOD_NODE_REPLY), model_id="m", list_models_fn=boom
    )
    assert not res.ok
    assert "/v1/models failed" in res.message


# --------------------------------------------------------------------------
# S5 context / memory
# --------------------------------------------------------------------------

def test_context_stage_picks_the_largest_arm_and_reports_the_numbers() -> None:
    res = pf.stage_context(
        QWEN_CONFIG,
        gpu_mem_gb=141.0, weight_bytes=55_600_000_000,
        arm_tokens={"floor": 0, "trunc_6x": 14000, "full": 86000, "summary_9x": None},
    )
    assert res.ok, res.message
    assert res.detail["arm"] == "full"
    check = res.detail["context_check"]
    # A4: 86000 tiktoken tokens are budgeted with a 15% margin.
    assert check["effective_prompt_tokens"] == 98900
    assert check["total_tokens"] == 98900 + pf.CONTEXT_MAX_NEW_TOKENS
    assert check["kv_cache_bytes"] > 0
    assert "86000 tokens" in res.message


def test_context_stage_fails_when_the_arm_exceeds_the_position_table() -> None:
    small = SimpleNamespace(
        model_type="qwen3", layer_types=["full_attention"],
        max_position_embeddings=4096, num_hidden_layers=4,
        num_key_value_heads=4, head_dim=128,
    )
    res = pf.stage_context(small, gpu_mem_gb=141.0, weight_bytes=1, arm_tokens={"full": 86000})
    assert not res.ok
    assert "'full'" in res.message and "max_position_embeddings" in res.message


def test_context_stage_fails_when_the_kv_cache_does_not_fit() -> None:
    # 86k tokens of this config is ~0.66 GB of KV, and the budget is 90% of the
    # card (GPU_MEM_HEADROOM), so 0.5 GB is the first size that cannot hold it.
    res = pf.stage_context(QWEN_CONFIG, gpu_mem_gb=0.5, weight_bytes=1, arm_tokens={"full": 86000})
    assert not res.ok
    assert "does not fit" in res.message


def test_context_stage_fails_when_no_arm_has_a_size() -> None:
    res = pf.stage_context(
        QWEN_CONFIG, gpu_mem_gb=141.0, weight_bytes=1, arm_tokens={"summary_9x": None}
    )
    assert not res.ok
    assert "no arm reported a token count" in res.message


# --------------------------------------------------------------------------
# S6 end-to-end micro-run
# --------------------------------------------------------------------------

def test_micro_stage_answers_one_question_at_both_weights() -> None:
    llm = FakeLLM(FakeTok())
    res = pf.stage_micro(llm)
    assert res.ok, res.message
    runs = res.detail["runs"]
    assert set(runs) == {"w=0", "w=0.1"}
    assert runs["w=0"]["answer"] == "800,000 yen"
    assert runs["w=0.1"]["positions_found"] > 0
    assert runs["w=0.1"]["planet_spans"] == 2


def test_micro_stage_fails_when_the_context_carries_no_planet() -> None:
    llm = FakeLLM(FakeTok())
    res = pf.stage_micro(
        llm, context="<CONTEXT>\n[SN] Restaurant plan\n</CONTEXT>"
    )
    assert not res.ok
    assert "run_reader raised at w=0.1" in res.message
    assert "0 planet spans" in res.message


def test_micro_stage_works_with_the_merged_bracket_tokenizer() -> None:
    res = pf.stage_micro(FakeLLM(FakeTok(merged=True)))
    assert res.ok, res.message
    assert res.detail["runs"]["w=0.1"]["positions_found"] > 0


# --------------------------------------------------------------------------
# runner + CLI
# --------------------------------------------------------------------------

class FakeResources:
    """Everything the stage adapters ask for, with no model and no network."""

    def __init__(self, *, cuda: bool = True, arm_tokens: dict | None = None) -> None:
        self.model_id = "fake/model"
        self._tok = FakeTok()
        self._llm = FakeLLM(self._tok)
        self._cuda = cuda
        self._arm_tokens = arm_tokens or {"full": 20000, "floor": 0}
        self.loaded = False

    def tokenizer(self) -> FakeTok:
        return self._tok

    def config(self) -> Any:
        return QWEN_CONFIG

    def llm(self) -> FakeLLM:
        self.loaded = True
        return self._llm

    def peek_llm(self) -> FakeLLM | None:
        return self._llm if self.loaded else None

    def judge(self) -> FakeJudge:
        return FakeJudge(GOOD_NODE_REPLY)

    def gpu_mem_gb(self) -> float:
        return 141.0

    def weight_bytes(self) -> int:
        return 55_600_000_000  # item K: the runner passes the checkpoint's weight size

    def arm_tokens(self) -> dict:
        return dict(self._arm_tokens)


def test_resolve_stages() -> None:
    assert pf.resolve_stages("all") == list(pf.STAGE_ORDER)
    assert pf.resolve_stages("micro") == ["micro"]
    with pytest.raises(SystemExit):
        pf.resolve_stages("nope")


def test_runner_writes_preflight_json_and_returns_zero(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        pf, "stage_env", lambda **_kw: pf._ok("env", "fake env", cuda_available=True)
    )
    monkeypatch.setattr(
        pf, "stage_judge",
        lambda judge, **_kw: pf._ok("judge", "fake judge"),
    )
    lines: list[str] = []
    out = tmp_path / "preflight.json"
    results, code = pf.run_stages(
        list(pf.STAGE_ORDER), FakeResources(), out_path=out, printer=lines.append
    )
    assert code == 0, [r.message for r in results if not r.ok]
    assert [r.name for r in results] == list(pf.STAGE_ORDER)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["ok"] is True and payload["stopped_at"] is None
    assert [s["stage"] for s in payload["stages"]] == list(pf.STAGE_ORDER)
    assert payload["model_id"] == "fake/model"
    assert all(line.startswith(("PASS", "wrote")) for line in lines)


def test_runner_stops_at_the_first_failure(tmp_path) -> None:
    lines: list[str] = []
    out = tmp_path / "preflight.json"
    results, code = pf.run_stages(
        ["env", "tokenizer", "micro"],
        FakeResources(cuda=False),
        out_path=out,
        printer=lines.append,
        opts={"require_cuda": True},
    )
    assert code == 1
    assert [r.name for r in results] == ["env"]  # tokenizer/micro never ran
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["ok"] is False and payload["stopped_at"] == "env"
    assert lines[0].startswith("FAIL env")


def test_runner_turns_a_raising_stage_into_a_failed_result() -> None:
    class Boom(FakeResources):
        def tokenizer(self):
            raise RuntimeError("tokenizer files are missing")

    results, code = pf.run_stages(
        ["tokenizer"], Boom(), printer=lambda _s: None
    )
    assert code == 1 and not results[0].ok
    assert "tokenizer files are missing" in results[0].message
    assert results[0].detail["error_type"] == "RuntimeError"


def test_config_stage_only_loads_the_model_when_a_later_stage_needs_it() -> None:
    # config alone: no load, and the sdpa assertion is reported as unchecked
    res_only = FakeResources()
    results, _code = pf.run_stages(["config"], res_only, printer=lambda _s: None)
    assert res_only.loaded is False
    assert results[0].detail["attn_implementation"] is None
    # config + inject: the model is loaded once, and sdpa is asserted on it
    res_both = FakeResources()
    res_both._llm._model.config._attn_implementation = "sdpa"
    results, _code = pf.run_stages(["config", "inject"], res_both, printer=lambda _s: None)
    assert res_both.loaded is True
    assert results[0].detail["attn_implementation"] == "sdpa"


def test_cli_parser_defaults() -> None:
    args = pf.build_parser().parse_args(["--model-id", "m"])
    assert args.stage == "all" and args.out == "preflight.json"
    args = pf.build_parser().parse_args(
        ["--model-id", "m", "--stage", "judge", "--judge-base-url", "http://h:8000"]
    )
    assert args.stage == "judge" and args.judge_base_url == "http://h:8000"


# --------------------------------------------------------------------------
# preflight_or_exit + the modal_spec_run hook
# --------------------------------------------------------------------------

class ReleasingResources(FakeResources):
    """FakeResources that records the memory hand-back the phases depend on."""

    released = 0

    def release(self) -> None:
        self.released += 1
        self.loaded = False


def test_preflight_or_exit_returns_the_results_and_releases_the_model(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        pf, "stage_env", lambda **_kw: pf._ok("env", "fake env")
    )
    res = ReleasingResources()
    out = tmp_path / "preflight.json"
    results = pf.preflight_or_exit(
        ("env", "tokenizer", "config"), "fake/model",
        out_path=out, printer=lambda _s: None, resources=res,
    )
    assert [r.name for r in results] == ["env", "tokenizer", "config"]
    assert all(r.ok for r in results)
    assert res.released == 1
    assert json.loads(out.read_text(encoding="utf-8"))["ok"] is True


def test_preflight_or_exit_raises_system_exit_naming_the_failed_stage(
    tmp_path,
) -> None:
    res = ReleasingResources(cuda=False)
    with pytest.raises(SystemExit) as excinfo:
        pf.preflight_or_exit(
            ("env", "micro"), "fake/model",
            out_path=tmp_path / "preflight.json",
            printer=lambda _s: None,
            resources=res,
        )
    assert "'env'" in str(excinfo.value)
    assert "CUDA is not available" in str(excinfo.value)
    # the model is released even on the failure path, and micro never ran
    assert res.released == 1
    payload = json.loads((tmp_path / "preflight.json").read_text(encoding="utf-8"))
    assert payload["stopped_at"] == "env"
    assert [s["stage"] for s in payload["stages"]] == ["env"]


def _spy_preflight(monkeypatch, *, fail: bool = False) -> list[tuple]:
    """Replace ``preflight_or_exit`` and record how each phase calls it."""
    calls: list[tuple] = []

    def fake(stages, model_id, **kwargs):
        calls.append((tuple(stages), model_id, kwargs))
        if fail:
            raise SystemExit("pre-flight FAILED at stage %r: boom" % stages[0])
        return []

    monkeypatch.setattr(pf, "preflight_or_exit", fake)
    return calls


def test_reader_phase_preflights_before_it_touches_the_model(
    tmp_path, monkeypatch
) -> None:
    from experiments import modal_spec_run as msr

    calls = _spy_preflight(monkeypatch, fail=True)
    with pytest.raises(SystemExit):
        msr._phase_reader_impl("fake/model", run_id="rid1", vol=str(tmp_path))
    assert calls[0][0] == ("env", "tokenizer", "config", "context", "micro")
    assert calls[0][1] == "fake/model"
    # F3: under the run id, not at the volume root
    assert calls[0][2]["out_path"] == tmp_path / "rid1" / "preflight_reader.json"
    assert calls[0][2]["allow_linear_layers"] is False
    # SystemExit came out of the checklist, so no weight was ever fetched
    assert not list((tmp_path / "rid1").glob("answers/*.json"))
    # ...but the run identity was recorded before anything was spent
    assert (tmp_path / "rid1" / "run_info.json").exists()


def test_instrument_phase_preflights_before_it_touches_the_model(
    tmp_path, monkeypatch
) -> None:
    from experiments import modal_spec_run as msr

    calls = _spy_preflight(monkeypatch, fail=True)
    with pytest.raises(SystemExit):
        msr._phase_instrument_impl(
            "fake/model", run_id="rid1", vol=str(tmp_path),
            allow_linear_layers=True,
        )
    assert calls[0][0] == ("env", "tokenizer", "config", "inject")
    assert calls[0][2]["out_path"] == tmp_path / "rid1" / "preflight_instrument.json"
    assert calls[0][2]["allow_linear_layers"] is True
    assert not (tmp_path / "rid1" / "instrument.json").exists()


def test_manager_phase_preflights_the_judge_and_stops_the_server(
    tmp_path, monkeypatch
) -> None:
    import subprocess

    from experiments import modal_spec_run as msr

    calls = _spy_preflight(monkeypatch, fail=True)
    proc = SimpleNamespace(
        returncode=0,
        poll=lambda: 0,
        terminate=lambda: None,
        wait=lambda timeout=None: 0,
        kill=lambda: None,
    )
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_kw: proc)
    monkeypatch.setattr(msr, "_wait_for_vllm", lambda *_a, **_kw: None)
    ran: list[str] = []
    monkeypatch.setattr(
        msr, "_run_and_commit", lambda *_a, **_kw: ran.append("build")
    )

    with pytest.raises(SystemExit):
        msr._phase_manager_impl(
            "fake/model", run_id="rid1", vol=str(tmp_path), port=8123
        )
    assert calls[0][0] == ("judge",)
    assert calls[0][2]["judge_base_url_"] == "http://127.0.0.1:8123"
    assert calls[0][2]["out_path"] == tmp_path / "rid1" / "preflight_manager.json"
    assert ran == []  # the 664-turn build never started


def test_dry_run_is_unaffected_by_the_preflight_hook(tmp_path) -> None:
    """--dry-run runs no phase, so it must still write all 38 cells on CPU."""
    from experiments.modal_spec_run import build_cells, dry_run

    res = dry_run(tmp_path, max_questions=1, run_id="rid1")
    root = tmp_path / "rid1"
    assert res["cells"] == len(build_cells()) == 38
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["cells"]) == 38
    assert manifest["run_id"] == "rid1"
    assert len(list((root / "answers").glob("*.meta.json"))) == 38
    # no phase ran, so no checklist artifact was written
    assert not list(root.glob("preflight_*.json"))


# --------------------------------------------------------------------------
# item C (2026-09-18): span decode must use the token IDS at the positions
# --------------------------------------------------------------------------


class OffsetTok(FakeTok):
    """Ids are NOT positions: id = position + 1000.  Decoding a position as if
    it were an id raises, so the stage can only pass when it decodes
    ``[ids[p] for p in pos]``."""

    OFFSET = 1000

    def __call__(self, text: str, **_kw: Any) -> dict:
        ids = super().__call__(text, **_kw)["input_ids"]
        return {"input_ids": [i + self.OFFSET for i in ids]}

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        out = []
        for i in ids:
            if i < self.OFFSET:
                raise KeyError("position %d decoded as a token id" % i)
            out.append(self._overrides.get(i - self.OFFSET, self._pieces[i - self.OFFSET]))
        return "".join(out)


def test_tokenizer_stage_decodes_ids_not_positions() -> None:
    res = pf.stage_tokenizer(OffsetTok())
    assert res.ok, res.message


def test_context_stage_refuses_unknown_weight_bytes() -> None:
    """item K: the stage cannot budget without the weight term."""
    res = pf.stage_context(QWEN_CONFIG, gpu_mem_gb=141.0, arm_tokens={"full": 86000})
    assert not res.ok and "weight bytes unknown" in res.message

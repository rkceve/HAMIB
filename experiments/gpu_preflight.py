"""gpu_preflight.py -- cheap fail-fast checklist, run ON THE GPU BOX.

WHY THIS EXISTS
---------------
Every expensive phase of ``experiments/modal_spec_run.py`` (a 664-turn manager
build, a ~40-cell reader grid) starts by paying for a 27B model download and a
model load.  The bugs that historically killed those runs were NOT in the
experiment logic: a tokenizer whose BPE merged ``"] "`` into the following word
and lost a node, a model that quietly loaded with ``eager`` attention so the
mass patch never ran, a judge endpoint serving a different model id than the one
asked for, a context that did not fit the card.  Each of those is observable in
MINUTES with a 300-token prompt -- but only after the model is on the box, so it
cannot be checked from the CPU dev machine (D-11).

This module is that checklist.  Each stage is a pure function that takes the
REAL objects as arguments (tokenizer, config, loaded model, judge) and returns a
:class:`PreflightResult`; nothing here loads a model by itself except the
:class:`GpuResources` helper the CLI uses.  That split is what makes every stage
unit-testable on CPU with fakes, including its failure mode.

STAGES
------
S0 ``env``       python / torch / transformers versions, CUDA, GPU name + memory.
S1 ``tokenizer`` the REAL tokenizer over a serialized 3-level CD: one span per
                 node line, levels and planet masses preserved, every span's
                 decoded text equal to the node text modulo whitespace, every
                 span non-empty (the ``[PN``-merged-token case).
S2 ``config``    ``check_model_supported`` + ``_attn_implementation == "sdpa"``.
S3 ``inject``    one decode step on a ~300-token single-needle prompt: the
                 per-layer bias counters, a strictly larger planet attention
                 share at w=0.5 than at w=0, non-empty and deterministic text.
S4 ``judge``     one Q_NODE round trip (parsed), one yes/no round trip, no
                 ``<think>`` residue, latency, and the served model id.
S5 ``context``   ``check_context_fits`` for the largest arm on the real card.
S6 ``micro``     one question end to end through ``run_reader`` at w=0 and w=0.1.

CLI (GPU host)::

    python experiments/gpu_preflight.py --stage all \\
        --model-id Qwen/Qwen3.8-27B \\
        --judge-base-url http://127.0.0.1:8000 \\
        --out preflight.json

Exit code is 0 only when every requested stage passed; the runner STOPS at the
first failure (the point is to spend seconds, not hours, on a broken box).

NEVER run this on the CPU dev machine: ``--stage`` anything but ``env`` will try
to load the model.  The tests in ``tests/reader/test_gpu_preflight.py`` drive
every stage with fakes instead.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow `python experiments/gpu_preflight.py`
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.bineval.run_reader import (  # noqa: E402
    HYBRID_N_LINEAR_LAYERS,
    build_reader_mass_vector,
    check_context_fits,
    check_model_supported,
    default_allow_linear_layers,
    expected_n_sdpa_layers,
    linear_attention_kernel_report,
)
from communication.cd_serializer import CDSerializer  # noqa: E402
from experiments.modal_spec_run import (  # noqa: E402
    DEFAULT_ARMS,
    INSTRUMENT_PLANET_MASS,
    build_instrument_prompt,
    mass_share,
)
from management.harness.backends import strip_think_blocks  # noqa: E402
from management.harness.judge import parse_yes_no  # noqa: E402
from management.harness.prompts import Q_BELONGS, Q_NODE  # noqa: E402
from management.harness.spec_manager import parse_node_object  # noqa: E402
from models.correlation_diagram import CorrelationDiagram  # noqa: E402
from models.node import Node, NodeLevel  # noqa: E402
from server.cd_parser import find_marker_spans, marker_positions  # noqa: E402

# S3: two new tokens, not one.  HF's ``generate`` emits the FIRST new token out
# of the prefill forward, so max_new_tokens=1 runs no ``seq_q == 1`` call at all
# and ``bias_applied_calls`` would be 0 (same reasoning as _phase_instrument_impl).
INJECT_MAX_NEW_TOKENS = 2
INJECT_W_GRID: tuple[float, float] = (0.0, 0.5)
# S4: the node text cap the spec manager uses (SpecConfig.max_node_chars).
JUDGE_MAX_NODE_CHARS = 120
JUDGE_NODE_MAX_TOKENS = 200
JUDGE_YESNO_MAX_TOKENS = 16
JUDGE_FRAGMENT = (
    "User: how much is the rent deposit for the Nakameguro place? "
    "Assistant: the deposit is 4,800,000 yen, six months of the 800,000 yen rent."
)
JUDGE_YESNO_A = "The rent deposit is 4,800,000 yen."
JUDGE_YESNO_B = "Budget of the restaurant plan"
# S5 / S6 defaults.
CONTEXT_MAX_NEW_TOKENS = 48
MICRO_W_GRID: tuple[float, float] = (0.0, 0.1)
MICRO_QID = "preflight-micro-1"
MICRO_QUESTION = "How much is the monthly rent?"
MICRO_ARM = "cd_mass_6x"

STAGE_ORDER: tuple[str, ...] = (
    "env",
    "tokenizer",
    "config",
    "inject",
    "judge",
    "context",
    "micro",
)


# --------------------------------------------------------------------------
# result type
# --------------------------------------------------------------------------

@dataclass
class PreflightResult:
    """One stage's verdict. ``detail`` always carries a short ``message``."""

    name: str
    ok: bool
    detail: dict = field(default_factory=dict)

    @property
    def message(self) -> str:
        return str(self.detail.get("message", ""))

    def to_dict(self) -> dict:
        return {"stage": self.name, "ok": self.ok, "detail": self.detail}


def _ok(name: str, message: str, **detail: Any) -> PreflightResult:
    return PreflightResult(name, True, dict(detail, message=message))


def _fail(name: str, message: str, **detail: Any) -> PreflightResult:
    return PreflightResult(name, False, dict(detail, message=message))


def require_ok(result: PreflightResult) -> PreflightResult:
    """Abort the caller when a stage failed. Used by the Modal phases."""
    if not result.ok:
        raise RuntimeError(
            "pre-flight stage %r failed: %s | detail=%s"
            % (result.name, result.message, json.dumps(result.detail, default=str))
        )
    return result


# --------------------------------------------------------------------------
# small shared helpers (pure)
# --------------------------------------------------------------------------

def encode(tokenizer: Any, text: str) -> list[int]:
    """``tokenizer(text)["input_ids"]`` flattened to a plain list of ints."""
    ids = tokenizer(text)["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


_MARKER_LINE_RE = re.compile(r"^(\s*)\[(SN|RN|PN([\d.]+))\]\s?(.*)$")
_MARKER_HEAD_RE = re.compile(r"^\[(?:SN|RN|PN[\d.]+)\]\s*")
_LEVEL_BY_TAG = {"S": "sun", "P": "planet", "R": "satellite"}


def parse_block_lines(block: str) -> list[tuple[str, float, str]]:
    """``<CONTEXT>`` block -> [(level, mass, text)] in serialized order.

    This is the SOURCE OF TRUTH for what the tokenizer is supposed to preserve:
    it reads the same characters the model will see, not the in-memory CD.
    """
    out: list[tuple[str, float, str]] = []
    for line in block.splitlines():
        m = _MARKER_LINE_RE.match(line)
        if not m:
            continue
        tag = m.group(2)
        mass = float(m.group(3)) if m.group(3) else 0.0
        out.append((_LEVEL_BY_TAG[tag[0]], mass, m.group(4).strip()))
    return out


def normalize_text(text: str) -> str:
    """Whitespace-insensitive comparison key, with a marker head removed.

    The ``[PN``-merged-token case (F5) puts the marker's own ``]`` -- and
    sometimes the whole marker -- inside the FIRST concept token, so the decoded
    span legitimately starts with ``"] "``.  That residue is stripped here; a
    genuinely different text still compares unequal.
    """
    stripped = text.strip()
    head = _MARKER_HEAD_RE.match(stripped)
    if head:
        stripped = stripped[head.end():]
    elif stripped.startswith("]"):
        stripped = stripped[1:]
    return " ".join(stripped.split())


def preflight_cd() -> CorrelationDiagram:
    """A small 3-level CD: 1 sun, 2 planets, 3 satellites.

    ``normalize()`` sets each planet's mass to its satellite count, so the two
    planets carry DIFFERENT masses (2.0 and 1.0) and a stage that mixed the two
    up cannot pass by accident.
    """
    cd = CorrelationDiagram()
    sun = Node(text="Restaurant plan", level=NodeLevel.SUN, mass=1.0)
    cd.add_sun(sun)
    budget = Node(text="Budget", level=NodeLevel.PLANET, mass=1.0)
    cd.add_planet(budget, sun.node_id)
    cd.add_satellite(
        Node(text="Monthly rent is 800,000 yen", level=NodeLevel.SATELLITE, mass=0.1),
        budget.node_id,
    )
    cd.add_satellite(
        Node(text="The loan is 30,000,000 yen", level=NodeLevel.SATELLITE, mass=0.1),
        budget.node_id,
    )
    location = Node(text="Location", level=NodeLevel.PLANET, mass=1.0)
    cd.add_planet(location, sun.node_id)
    cd.add_satellite(
        Node(text="The site is in Nakameguro", level=NodeLevel.SATELLITE, mass=0.1),
        location.node_id,
    )
    cd.normalize()
    return cd


def preflight_context_block(cd: CorrelationDiagram | None = None) -> str:
    """The spec-faithful serialization of :func:`preflight_cd`."""
    return CDSerializer(level_markers=True).to_context_block(cd or preflight_cd())


def attn_implementation_of(model: Any) -> str | None:
    """``_attn_implementation`` of a raw HF model OR of a MassWeightedLLM.

    Three shapes, in order: the wrapper's own ``attn_implementation`` property
    (MassWeightedGemma exposes one), a raw model's ``config``, and -- last --
    the wrapped ``_model.config``, which is the shape a wrapper that has NOT
    been given the property still exposes.  Anything else is unknown, and the
    caller treats unknown as a failure rather than as sdpa.
    """
    impl = getattr(model, "attn_implementation", None)
    if isinstance(impl, str):
        return impl
    for holder in (model, getattr(model, "_model", None)):
        cfg = getattr(holder, "config", None)
        if cfg is None:
            continue
        value = getattr(cfg, "_attn_implementation", None)
        if isinstance(value, str):
            return value
    return None


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _text_config(cfg: Any) -> Any:
    """Descend into ``text_config`` (vision-language wrappers) when present."""
    sub = _cfg_get(cfg, "text_config", None)
    if sub is not None and _cfg_get(sub, "num_hidden_layers", None) is not None:
        return sub
    return cfg


def _model_device(llm: Any) -> Any:
    try:
        return next(llm._model.parameters()).device
    except (StopIteration, AttributeError):
        return None


def n_layers_of(llm: Any) -> int | None:
    """Layer count of the model behind a MassWeightedLLM (None if unknown)."""
    cfg = getattr(getattr(llm, "_model", None), "config", None)
    if cfg is None:
        return None
    value = _cfg_get(_text_config(cfg), "num_hidden_layers", None)
    return int(value) if value is not None else None


def n_sdpa_layers_of(llm: Any) -> int | None:
    """Layers of that model the sdpa mass patch can actually bias (A3).

    A hybrid model (Qwen3.8-27B: 48 ``linear_attention`` + 16
    ``full_attention``) calls the patched kernel on its full-attention layers
    only, so ``bias_applied_calls`` per decode step equals 16, not 64. Comparing
    against ``num_hidden_layers`` would fail every injected cell of a model the
    run deliberately chose.
    """
    cfg = getattr(getattr(llm, "_model", None), "config", None)
    if cfg is None:
        return None
    text_cfg = _text_config(cfg)
    layer_types = _cfg_get(text_cfg, "layer_types", None)
    if layer_types:
        return sum(1 for t in layer_types if str(t) == "full_attention")
    return n_layers_of(llm)


def list_models(base_url: str, *, timeout: float = 10.0) -> list[str]:
    """GET ``{base_url}/v1/models`` -> the served model ids (urllib only)."""
    url = base_url.rstrip("/") + "/v1/models"
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
        data = json.loads(resp.read().decode("utf-8"))
    return [str(item["id"]) for item in data.get("data", []) if item.get("id")]


def judge_base_url(judge: Any) -> str | None:
    """Recover the base url from an OpenAICompatJudge (or a fake)."""
    for attr in ("base_url", "_base_url"):
        value = getattr(judge, attr, None)
        if isinstance(value, str) and value:
            return value.rstrip("/")
    url = getattr(judge, "_url", None)
    if isinstance(url, str) and url:
        return url.split("/v1/")[0].rstrip("/")
    return None


# --------------------------------------------------------------------------
# S0 environment
# --------------------------------------------------------------------------

def stage_env(
    *,
    torch_mod: Any = None,
    transformers_mod: Any = None,
    require_cuda: bool = True,
) -> PreflightResult:
    """Record the versions and the card. Fails when CUDA is not there.

    Modules are injected so the CPU tests can drive both branches; the CLI
    imports them locally (never at module import time, D-11).
    """
    detail: dict[str, Any] = {"python": platform.python_version()}
    if torch_mod is None:
        try:
            import torch as torch_mod  # type: ignore[no-redef]
        except Exception as exc:  # pragma: no cover - torch is installed here
            return _fail("env", "torch is not importable: %s" % exc)
    if transformers_mod is None:
        try:
            import transformers as transformers_mod  # type: ignore[no-redef]
        except Exception as exc:  # pragma: no cover - installed here
            return _fail("env", "transformers is not importable: %s" % exc)

    detail["torch"] = getattr(torch_mod, "__version__", None)
    detail["transformers"] = getattr(transformers_mod, "__version__", None)
    cuda = getattr(torch_mod, "cuda", None)
    available = bool(cuda.is_available()) if cuda is not None else False
    detail["cuda_available"] = available
    detail["cuda_version"] = getattr(getattr(torch_mod, "version", None), "cuda", None)
    if cuda is not None and available:
        detail["device_count"] = int(cuda.device_count())
        props = cuda.get_device_properties(0)
        detail["gpu_name"] = getattr(props, "name", None)
        total = getattr(props, "total_memory", None)
        detail["gpu_total_memory_gb"] = (
            round(int(total) / (1024 ** 3), 2) if total else None
        )
    else:
        detail["device_count"] = 0
        detail["gpu_name"] = None
        detail["gpu_total_memory_gb"] = None

    if require_cuda and not available:
        return _fail(
            "env",
            "CUDA is not available (torch %s): this checklist only means "
            "anything on the GPU box" % detail["torch"],
            **detail,
        )
    return _ok(
        "env",
        "torch %s / transformers %s / %s (%.1f GB)"
        % (
            detail["torch"],
            detail["transformers"],
            detail["gpu_name"] or "no GPU",
            detail["gpu_total_memory_gb"] or 0.0,
        ),
        **detail,
    )


# --------------------------------------------------------------------------
# S1 tokenizer markers
# --------------------------------------------------------------------------

def stage_tokenizer(
    tokenizer: Any, *, cd: CorrelationDiagram | None = None
) -> PreflightResult:
    """The real BPE must not lose, merge or relabel a single node line.

    Checks, on a serialized 3-level CD:
      1. one span per node line;
      2. the level sequence equals the CD's;
      3. the planet masses equal the serialized ones;
      4. every span decodes back to its node text (whitespace-insensitive);
      5. every span has at least one position (the ``[PN``-merged-token case,
         where BPE glues ``"] "`` to the following word -- F5).
    """
    block = preflight_context_block(cd)
    expected = parse_block_lines(block)
    ids = encode(tokenizer, block)
    spans = find_marker_spans(ids, tokenizer)

    decoded = [
        # ``pos`` are token POSITIONS; decode the ids found there (2026-09-18 fix:
        # decoding positions as ids only worked for id == position fakes).
        normalize_text(tokenizer.decode([ids[p] for p in pos], skip_special_tokens=False)) if pos else ""
        for _lvl, _m, pos in spans
    ]
    detail: dict[str, Any] = {
        "block_lines": len(expected),
        "spans": len(spans),
        "prompt_tokens": len(ids),
        "expected_levels": [lvl for lvl, _m, _t in expected],
        "span_levels": [lvl for lvl, _m, _p in spans],
        "expected_masses": [m for _lvl, m, _t in expected],
        "span_masses": [m for _lvl, m, _p in spans],
        "empty_spans": [i for i, (_lvl, _m, pos) in enumerate(spans) if not pos],
    }

    if len(spans) != len(expected):
        seen = set(decoded)
        missing = [t for _lvl, _m, t in expected if normalize_text(t) not in seen]
        return _fail(
            "tokenizer",
            "the tokenizer lost %d node line(s): %d spans for %d lines%s"
            % (
                len(expected) - len(spans),
                len(spans),
                len(expected),
                ("; missing: " + ", ".join(repr(t) for t in missing)) if missing else "",
            ),
            missing_nodes=missing,
            **detail,
        )

    problems: list[str] = []
    for i, ((exp_lvl, exp_mass, exp_text), (lvl, mass, pos)) in enumerate(
        zip(expected, spans)
    ):
        if lvl != exp_lvl:
            problems.append(
                "node %d %r: level %r, expected %r" % (i, exp_text, lvl, exp_lvl)
            )
        if abs(mass - exp_mass) > 1e-9:
            problems.append(
                "node %d %r: mass %g, expected %g" % (i, exp_text, mass, exp_mass)
            )
        if not pos:
            problems.append("node %d %r: no token positions" % (i, exp_text))
            continue
        if decoded[i] != normalize_text(exp_text):
            problems.append(
                "node %d %r: decodes to %r" % (i, exp_text, decoded[i])
            )
    if problems:
        return _fail(
            "tokenizer",
            "%d marker span problem(s): %s" % (len(problems), "; ".join(problems)),
            problems=problems,
            **detail,
        )
    return _ok(
        "tokenizer",
        "%d/%d node lines survive tokenization (%d prompt tokens)"
        % (len(spans), len(expected), len(ids)),
        **detail,
    )


# --------------------------------------------------------------------------
# S2 model config
# --------------------------------------------------------------------------

def stage_config(
    config: Any,
    *,
    model: Any = None,
    allow_linear_layers: bool = False,
    model_id: str | None = None,
) -> PreflightResult:
    """``check_model_supported`` + (when a loaded model is given) sdpa.

    When ``model_id`` is the hybrid reader, the layer counts are ASSERTED, not
    just reported: a Qwen/Qwen3.8-27B whose config does not read 16 full + 48
    linear is not the model this run was designed around (a re-uploaded config,
    a wrong revision), and every ``bias_applied_calls`` assertion downstream
    would compare against the wrong number.
    """
    try:
        info = check_model_supported(config, allow_linear_layers=allow_linear_layers)
    except ValueError as exc:
        return _fail("config", "model config is not supported: %s" % exc)

    detail: dict[str, Any] = {
        "model_type": info.get("model_type"),
        "layer_types": info.get("layer_types"),
        "layer_types_summary": info.get("layer_types_summary"),
        "num_hidden_layers": info.get("num_hidden_layers"),
        # A3: the layer count every injection assertion must compare against.
        "n_sdpa_layers": info.get("n_sdpa_layers"),
        "n_linear_layers": info.get("n_linear_layers"),
        "allow_linear_layers": bool(allow_linear_layers),
        "max_position_embeddings": info.get("max_position_embeddings"),
        "use_sliding_window": info.get("use_sliding_window"),
        "sliding_window": info.get("sliding_window"),
        "attn_implementation": None,
        "model_id": model_id,
        "n_sdpa_layers_expected": expected_n_sdpa_layers(model_id),
        # Reported, never asserted: which linear-attention kernels this
        # container actually has, and whether modeling_qwen3_5 would therefore
        # take its pure-torch fallback. Affects the 48 LINEAR layers only.
        "linear_attention_kernels": linear_attention_kernel_report(),
    }
    expected = detail["n_sdpa_layers_expected"]
    if expected is not None and info.get("n_sdpa_layers") != expected:
        return _fail(
            "config",
            "%s must inject on exactly %d full-attention layers, but its config "
            "reports n_sdpa_layers=%r (layer_types_summary=%r): this is not the "
            "hybrid the run was designed around"
            % (model_id, expected, info.get("n_sdpa_layers"),
               info.get("layer_types_summary")),
            **detail,
        )
    if expected is not None and info.get("n_linear_layers") != HYBRID_N_LINEAR_LAYERS:
        return _fail(
            "config",
            "%s must have exactly %d linear_attention layers, but its config "
            "reports n_linear_layers=%r (layer_types_summary=%r)"
            % (model_id, HYBRID_N_LINEAR_LAYERS, info.get("n_linear_layers"),
               info.get("layer_types_summary")),
            **detail,
        )
    if model is not None:
        impl = attn_implementation_of(model)
        detail["attn_implementation"] = impl
        if impl != "sdpa":
            return _fail(
                "config",
                "model loaded with attn_implementation=%r, expected 'sdpa': the "
                "mass patch only intervenes on the sdpa path, so every injected "
                "cell would silently run as the baseline" % (impl,),
                **detail,
            )
    return _ok(
        "config",
        "%s: %s of %s layers reachable by the sdpa patch, "
        "max_position_embeddings=%s, attn=%s"
        % (
            detail["model_type"],
            detail["n_sdpa_layers"],
            detail["num_hidden_layers"],
            detail["max_position_embeddings"],
            detail["attn_implementation"] or "not checked (model not loaded)",
        ),
        **detail,
    )


# --------------------------------------------------------------------------
# S3 single-decode injection
# --------------------------------------------------------------------------

def _injection_run(
    llm: Any,
    prompt: str,
    ids: Sequence[int],
    tokenizer: Any,
    positions: Sequence[int],
    w: float,
    n_layers: int,
) -> dict:
    """One generate() with the recorder on. Returns text + stats + mass share."""
    llm._mass_weight = float(w)
    vec, info = build_reader_mass_vector(
        ids, tokenizer, "planet", None, w=w, prompt_text=prompt,
        device=_model_device(llm),
    )
    if vec is None:
        raise RuntimeError("no mass vector was built at w=%g" % w)
    llm.set_mass_vector(vec)
    llm.start_attention_recording()
    try:
        text = llm.generate(prompt)
    finally:
        recorded = llm.stop_attention_recording()
        stats = dict(llm.mass_injection_stats())
        llm.clear_mass_vector()
    if len(recorded) < n_layers:
        raise RuntimeError(
            "recorded %d attention rows at w=%g, expected at least one decode "
            "step of %d layers: the sdpa patch never saw a seq_q==1 call"
            % (len(recorded), w, n_layers)
        )
    # One entry per layer per decode step in call order -> index n_layers-1 is
    # the LAST layer of the FIRST decode step.
    return {
        "text": text,
        "stats": stats,
        "share": mass_share(recorded[n_layers - 1], positions),
        "recorded_vectors": len(recorded),
        "positions_found": info.positions_found,
    }


def stage_inject(
    llm: Any,
    *,
    tokenizer: Any = None,
    prompt: str | None = None,
    w_grid: Sequence[float] = INJECT_W_GRID,
    n_layers: int | None = None,
) -> PreflightResult:
    """Prove the bias reaches the tokens on the PATCHED path, in one decode step.

    ``w_grid`` is (low, high); the low value is run TWICE so the greedy decode's
    determinism is checked with the same objects that produce the measurement.
    """
    tok = tokenizer if tokenizer is not None else llm.tokenizer
    text_prompt = prompt if prompt is not None else build_instrument_prompt()
    ids = encode(tok, text_prompt)
    spans = find_marker_spans(ids, tok)
    positions = [p for p, _m in marker_positions(spans, inject_levels={"planet"})]
    # A3: the SDPA-reachable layer count, not the model depth.
    layers = n_layers if n_layers is not None else n_sdpa_layers_of(llm)

    detail: dict[str, Any] = {
        "prompt_tokens": len(ids),
        "planet_spans": sum(1 for lvl, _m, _p in spans if lvl == "planet"),
        "positions_found": len(positions),
        "n_layers": layers,
        "n_sdpa_layers": layers,
        "num_hidden_layers": n_layers_of(llm),
        "w_grid": list(w_grid),
        "planet_mass": INSTRUMENT_PLANET_MASS,
    }
    if not positions:
        return _fail(
            "inject",
            "the marker scan found 0 planet positions in the single-needle "
            "prompt (%d spans): injection cannot be measured" % len(spans),
            **detail,
        )
    if not layers:
        return _fail(
            "inject",
            "cannot determine how many layers the sdpa patch reaches "
            "(n_sdpa_layers)",
            **detail,
        )

    w_low, w_high = float(w_grid[0]), float(w_grid[-1])
    saved_max_new = getattr(llm, "_max_new_tokens", None)
    try:
        if saved_max_new is not None:
            llm._max_new_tokens = INJECT_MAX_NEW_TOKENS
        try:
            low_a = _injection_run(llm, text_prompt, ids, tok, positions, w_low, layers)
            low_b = _injection_run(llm, text_prompt, ids, tok, positions, w_low, layers)
            high = _injection_run(llm, text_prompt, ids, tok, positions, w_high, layers)
        except RuntimeError as exc:
            return _fail("inject", str(exc), **detail)
    finally:
        if saved_max_new is not None:
            llm._max_new_tokens = saved_max_new
        llm.clear_mass_vector()

    stats = high["stats"]
    detail.update(
        {
            "mass_share": {"w=%g" % w_low: low_a["share"], "w=%g" % w_high: high["share"]},
            "stats_high": stats,
            "text_low": low_a["text"],
            "text_low_repeat": low_b["text"],
            "text_high": high["text"],
            "recorded_vectors": high["recorded_vectors"],
        }
    )
    checks = {
        "bias_applied_calls_match": stats.get("bias_applied_calls") == layers,
        "bias_skipped_prefill_calls_match": (
            stats.get("bias_skipped_prefill_calls") == layers
        ),
        "no_sliding_skips": stats.get("bias_skipped_sliding_calls") == 0,
        "share_increases": high["share"] > low_a["share"],
        "text_nonempty_at_w0": bool(low_a["text"].strip()),
        "deterministic_at_w0": low_a["text"] == low_b["text"],
    }
    detail["checks"] = checks
    failed = [k for k, v in checks.items() if not v]
    if failed:
        return _fail(
            "inject",
            "injection checks failed %r (applied=%s skipped_prefill=%s "
            "skipped_sliding=%s n_layers=%d, share %g -> %g)"
            % (
                failed,
                stats.get("bias_applied_calls"),
                stats.get("bias_skipped_prefill_calls"),
                stats.get("bias_skipped_sliding_calls"),
                layers,
                low_a["share"],
                high["share"],
            ),
            **detail,
        )
    return _ok(
        "inject",
        "planet attention share %.4f -> %.4f (w %g -> %g), %d layers biased "
        "on the decode step" % (low_a["share"], high["share"], w_low, w_high, layers),
        **detail,
    )


# --------------------------------------------------------------------------
# S4 judge endpoint
# --------------------------------------------------------------------------

def stage_judge(
    judge: Any,
    *,
    model_id: str | None = None,
    base_url: str | None = None,
    max_chars: int = JUDGE_MAX_NODE_CHARS,
    fragment: str = JUDGE_FRAGMENT,
    list_models_fn: Callable[[str], list[str]] | None = None,
) -> PreflightResult:
    """One Q_NODE round trip and one yes/no round trip against the real server.

    The raw reply is ALWAYS put in the detail on a parse failure -- a judge that
    answers in prose instead of JSON is the single most common cause of a
    100%-defaulted manager build, and the reply is the only evidence of it.
    """
    detail: dict[str, Any] = {"model_id": model_id}
    node_prompt = Q_NODE.format(max_chars=max_chars, text=fragment)
    t0 = time.perf_counter()
    node_reply = judge.complete(node_prompt, max_tokens=JUDGE_NODE_MAX_TOKENS)
    detail["node_latency_s"] = round(time.perf_counter() - t0, 3)
    detail["node_reply"] = node_reply
    parsed = parse_node_object(node_reply, max_chars)
    if parsed is None:
        return _fail(
            "judge",
            "the judge's Q_NODE reply did not parse into a node object "
            "(raw reply in detail.node_reply)",
            **detail,
        )
    detail["node_summary"], detail["node_scores"] = parsed

    yn_prompt = Q_BELONGS.format(a=JUDGE_YESNO_A, b=JUDGE_YESNO_B)
    t1 = time.perf_counter()
    yn_reply = judge.complete(yn_prompt, max_tokens=JUDGE_YESNO_MAX_TOKENS)
    detail["yesno_latency_s"] = round(time.perf_counter() - t1, 3)
    detail["yesno_reply"] = yn_reply
    verdict = parse_yes_no(yn_reply)
    if verdict is None:
        return _fail(
            "judge",
            "the judge's yes/no reply did not parse "
            "(raw reply in detail.yesno_reply)",
            **detail,
        )
    detail["yesno"] = verdict

    # Two ways reasoning markup breaks a judge, and BOTH have to be caught here.
    # ``strip_think_blocks`` can never leave a literal ``<think>`` behind -- it
    # returns "" instead (H5) -- so testing only for the tag would be a check
    # that can never fire.  A reply that strips to nothing is the real symptom:
    # the server spent every one of its max_tokens inside an unterminated
    # reasoning block and never emitted the answer.
    leaked = [
        label
        for label, raw in (("node", node_reply), ("yesno", yn_reply))
        if "<think>" in strip_think_blocks(raw)
        or (raw.strip() and not strip_think_blocks(raw).strip())
    ]
    if leaked:
        return _fail(
            "judge",
            "reasoning markup swallowed the %s reply: stripping <think> blocks "
            "leaves nothing behind, so every judge call would be scored as "
            "unparsed" % ", ".join(leaked),
            think_leaked=leaked,
            **detail,
        )

    url = base_url or judge_base_url(judge)
    detail["base_url"] = url
    if model_id and url:
        lister = list_models_fn or list_models
        try:
            served = lister(url)
        except Exception as exc:  # network / server error is a real failure here
            return _fail(
                "judge", "GET %s/v1/models failed: %s" % (url, exc), **detail
            )
        detail["served_models"] = served
        if model_id not in served:
            return _fail(
                "judge",
                "the endpoint serves %r but the run asks for %r: every judge "
                "call would go to a different model than the reader"
                % (served, model_id),
                **detail,
            )
    return _ok(
        "judge",
        "Q_NODE parsed in %.2fs, yes/no in %.2fs, model %s"
        % (detail["node_latency_s"], detail["yesno_latency_s"], model_id or "unchecked"),
        **detail,
    )


# --------------------------------------------------------------------------
# S5 context / memory
# --------------------------------------------------------------------------

def stage_context(
    config: Any,
    *,
    gpu_mem_gb: float,
    arm_tokens: Mapping[str, int | None],
    max_new_tokens: int = CONTEXT_MAX_NEW_TOKENS,
    tokenizer: Any = None,
    arm_text: Callable[[str], str] | None = None,
    weight_bytes: int | None = None,
) -> PreflightResult:
    """``check_context_fits`` for the LARGEST arm, on the real card's memory.

    ``weight_bytes`` (item K): the checkpoint's weight size from
    ``run_reader.safetensors_total_bytes``; None fails the stage unless the
    config states ``num_parameters`` (the check never budgets 0 weight bytes).

    A4: ``arm_tokens`` are tiktoken ``cl100k_base`` counts, but the reader
    tokenizes with the MODEL's tokenizer (Qwen3.8 has a 248k vocabulary), so the
    two disagree by a few percent in either direction.  When a real tokenizer
    (and a way to get the arm's text) is available the largest arm is ALSO
    counted with it; both counts and their ratio go into the detail, and a real
    count over ``max_position_embeddings`` fails the stage outright.
    """
    sized = {a: t for a, t in arm_tokens.items() if t is not None}
    if not sized:
        return _fail(
            "context",
            "no arm reported a token count: nothing to check against %.1f GB"
            % gpu_mem_gb,
            arm_tokens=dict(arm_tokens),
        )
    arm = max(sized, key=lambda a: int(sized[a]))
    tokens = int(sized[arm])
    detail: dict[str, Any] = {
        "arm": arm,
        "arm_tokens": {a: t for a, t in arm_tokens.items()},
        "gpu_mem_gb": gpu_mem_gb,
        "max_new_tokens": max_new_tokens,
        "weight_bytes": weight_bytes,
    }
    try:
        info = check_context_fits(
            config, tokens, max_new_tokens, gpu_mem_gb, weight_bytes=weight_bytes
        )
    except ValueError as exc:
        return _fail(
            "context",
            "largest arm %r (%d tokens) does not fit: %s" % (arm, tokens, exc),
            **detail,
        )
    detail["context_check"] = info

    # A4: the same arm, counted with the tokenizer that will actually run.
    if tokenizer is not None and arm_text is not None:
        try:
            real_tokens = len(encode(tokenizer, arm_text(arm)))
        except Exception as exc:  # a missing artifact must not mask the stage
            detail["real_tokenizer_error"] = "%s: %s" % (type(exc).__name__, exc)
        else:
            detail["tiktoken_tokens"] = tokens
            detail["real_tokenizer_tokens"] = real_tokens
            detail["real_over_tiktoken"] = (real_tokens / tokens) if tokens else None
            max_pos = info["max_position_embeddings"]
            if max_pos is not None and real_tokens + max_new_tokens > int(max_pos):
                return _fail(
                    "context",
                    "largest arm %r is %d tokens under the MODEL tokenizer "
                    "(%d under tiktoken, ratio %.3f): %d + %d generated > "
                    "max_position_embeddings=%s"
                    % (arm, real_tokens, tokens, real_tokens / tokens,
                       real_tokens, max_new_tokens, max_pos),
                    **detail,
                )

    kv_gb = (info["kv_cache_bytes"] or 0) / (1024 ** 3)
    return _ok(
        "context",
        "largest arm %r: %d tokens (tiktoken)%s, KV %.2f GB of %.1f GB "
        "(max_position_embeddings=%s)"
        % (
            arm,
            tokens,
            (" / %d real" % detail["real_tokenizer_tokens"])
            if "real_tokenizer_tokens" in detail else "",
            kv_gb,
            gpu_mem_gb,
            info["max_position_embeddings"],
        ),
        **detail,
    )


# --------------------------------------------------------------------------
# S6 end-to-end micro-run
# --------------------------------------------------------------------------

def stage_micro(
    llm: Any,
    *,
    context: str | None = None,
    question: dict | None = None,
    w_grid: Sequence[float] = MICRO_W_GRID,
    arm: str = MICRO_ARM,
) -> PreflightResult:
    """ONE question through ``run_reader`` at w=0 and w=0.1.

    This is the only stage that exercises the exact call the grid makes, so it
    also catches the guards inside ``run_reader`` (the M2 planet-span guard
    fires here rather than on cell 1 of 40).
    """
    from benchmark.bineval import run_reader as rr

    ctx = context if context is not None else preflight_context_block()
    q = question or {"qid": MICRO_QID, "question": MICRO_QUESTION}
    qid = q["qid"]
    detail: dict[str, Any] = {"arm": arm, "qid": qid, "w_grid": list(w_grid), "runs": {}}

    for w in w_grid:
        try:
            run = rr.run_reader(llm, ctx, [q], w=float(w), inject="planet", arm=arm)
        except RuntimeError as exc:
            return _fail(
                "micro", "run_reader raised at w=%g: %s" % (w, exc), **detail
            )
        meta = run.per_question.get(qid, {})
        detail["runs"]["w=%g" % w] = {
            "answer": run.answers.get(qid),
            "positions_found": meta.get("positions_found"),
            "planet_spans": meta.get("planet_spans"),
            "bias_applied_calls": meta.get("bias_applied_calls"),
        }
        if qid not in run.answers:
            return _fail(
                "micro",
                "run_reader produced no answer for %r at w=%g" % (qid, w),
                **detail,
            )
        if w > 0 and not meta.get("positions_found"):
            return _fail(
                "micro",
                "w=%g found 0 injectable positions in the micro context: the "
                "cell would be the text-only baseline under an injected name"
                % w,
                **detail,
            )
    return _ok(
        "micro",
        "one question answered at w=%s (positions_found=%s at w=%g)"
        % (
            list(w_grid),
            detail["runs"]["w=%g" % w_grid[-1]]["positions_found"],
            w_grid[-1],
        ),
        **detail,
    )


# --------------------------------------------------------------------------
# resources: the ONLY place that loads anything (GPU host)
# --------------------------------------------------------------------------

class GpuResources:
    """Lazily built real objects for the CLI. GPU ONLY. UNVERIFIED (no GPU here).

    Everything is cached, so ``--stage all`` loads exactly one model copy and
    the config stage can assert sdpa on the model the injection stage uses.
    """

    def __init__(
        self,
        model_id: str,
        *,
        judge_base_url_: str | None = None,
        chat: str | None = None,
        arms: Sequence[str] = DEFAULT_ARMS,
        gpu_mem_gb: float | None = None,
        allow_linear_layers: bool = False,
    ) -> None:
        self.model_id = model_id
        self.judge_base_url = judge_base_url_
        self.chat = chat
        self.arms = list(arms)
        self.allow_linear_layers = bool(allow_linear_layers)
        self._gpu_mem_gb = gpu_mem_gb
        self._config: Any = None
        self._tokenizer: Any = None
        self._llm: Any = None
        self._judge: Any = None

    def config(self) -> Any:
        if self._config is None:
            from transformers import AutoConfig

            self._config = AutoConfig.from_pretrained(self.model_id)
        return self._config

    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            if self._llm is not None:
                self._tokenizer = self._llm.tokenizer
            else:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        return self._tokenizer

    def llm(self) -> Any:
        if self._llm is None:
            from benchmark.bineval import run_reader as rr

            self._llm = rr.load_reader(
                self.model_id, max_new_tokens=CONTEXT_MAX_NEW_TOKENS, w=0.0
            )
        return self._llm

    def peek_llm(self) -> Any:
        """The loaded model, or None -- never triggers a load."""
        return self._llm

    def judge(self) -> Any:
        if self._judge is None:
            from management.harness.backends import OpenAICompatJudge

            if not self.judge_base_url:
                raise SystemExit("--judge-base-url is required for the judge stage")
            self._judge = OpenAICompatJudge(
                self.judge_base_url,
                self.model_id,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        return self._judge

    def release(self) -> None:
        """Drop the loaded model and hand the card's memory back.

        The phases in ``modal_spec_run`` load their OWN model right after the
        checklist, so a preflight copy left alive would double the resident
        weights and OOM the very run it is meant to protect.
        """
        if self._llm is None:
            return
        self._llm = None
        self._tokenizer = None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - torch is present on the GPU box
            pass

    def weight_bytes(self) -> int | None:
        """item K: weight size from safetensors metadata (no weight download)."""
        from benchmark.bineval.run_reader import safetensors_total_bytes

        return safetensors_total_bytes(self.model_id)

    def gpu_mem_gb(self) -> float:
        if self._gpu_mem_gb is None:
            import torch

            props = torch.cuda.get_device_properties(0)
            self._gpu_mem_gb = float(props.total_memory) / (1024 ** 3)
        return float(self._gpu_mem_gb)

    def arm_tokens(self) -> dict[str, int | None]:
        from benchmark.bineval import arms as arms_mod

        chat = arms_mod.load_chat(self.chat) if self.chat else arms_mod.load_chat()
        out: dict[str, int | None] = {}
        for arm in self.arms:
            try:
                out[arm] = arms_mod.build_arm_context(arm, chat=chat, cd_json=None).tokens
            except (FileNotFoundError, ValueError):
                # An arm whose stored artifact is absent says nothing about
                # whether the GRID fits; the sized arms still bound it.
                out[arm] = None
        return out

    def arm_text(self, arm: str) -> str:
        """The arm's context text (A4: re-counted with the real tokenizer)."""
        from benchmark.bineval import arms as arms_mod

        chat = arms_mod.load_chat(self.chat) if self.chat else arms_mod.load_chat()
        return arms_mod.build_arm_context(arm, chat=chat, cd_json=None).text


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

def _run_env(res: Any, opts: dict) -> PreflightResult:
    return stage_env(require_cuda=bool(opts.get("require_cuda", True)))


def _run_tokenizer(res: Any, opts: dict) -> PreflightResult:
    return stage_tokenizer(res.tokenizer())


def _run_config(res: Any, opts: dict) -> PreflightResult:
    # The sdpa assertion needs a loaded model. Loading one just for this stage
    # would cost the download this checklist exists to protect, so the model is
    # only pulled in when a later requested stage needs it anyway.
    model = res.llm() if opts.get("needs_model") else res.peek_llm()
    return stage_config(
        res.config(), model=model,
        allow_linear_layers=bool(getattr(res, "allow_linear_layers", False)),
        model_id=getattr(res, "model_id", None),
    )


def _run_inject(res: Any, opts: dict) -> PreflightResult:
    return stage_inject(res.llm())


def _run_judge(res: Any, opts: dict) -> PreflightResult:
    return stage_judge(res.judge(), model_id=res.model_id)


def _run_context(res: Any, opts: dict) -> PreflightResult:
    # A4: pass the real tokenizer ONLY when one already exists -- fetching one
    # just for this stage would cost the download the checklist protects.
    tokenizer = None
    arm_text = getattr(res, "arm_text", None)
    if opts.get("needs_model") or getattr(res, "peek_llm", lambda: None)() is not None:
        try:
            tokenizer = res.tokenizer()
        except Exception:  # a tokenizer we cannot build is simply not used
            tokenizer = None
    weight_fn = getattr(res, "weight_bytes", None)
    return stage_context(
        res.config(), gpu_mem_gb=res.gpu_mem_gb(), arm_tokens=res.arm_tokens(),
        tokenizer=tokenizer, arm_text=arm_text if callable(arm_text) else None,
        weight_bytes=weight_fn() if callable(weight_fn) else None,
    )


def _run_micro(res: Any, opts: dict) -> PreflightResult:
    return stage_micro(res.llm())


STAGE_RUNNERS: dict[str, Callable[[Any, dict], PreflightResult]] = {
    "env": _run_env,
    "tokenizer": _run_tokenizer,
    "config": _run_config,
    "inject": _run_inject,
    "judge": _run_judge,
    "context": _run_context,
    "micro": _run_micro,
}


def resolve_stages(stage: str) -> list[str]:
    if stage == "all":
        return list(STAGE_ORDER)
    if stage not in STAGE_RUNNERS:
        raise SystemExit("unknown stage %r (known: %s, all)" % (stage, ", ".join(STAGE_ORDER)))
    return [stage]


def run_stages(
    names: Sequence[str],
    resources: Any,
    *,
    out_path: str | Path | None = None,
    printer: Callable[[str], None] = print,
    opts: dict | None = None,
) -> tuple[list[PreflightResult], int]:
    """Run the stages in order, STOPPING at the first failure.

    Returns ``(results, exit_code)`` and writes ``preflight.json`` (when
    ``out_path`` is given) whatever happened -- a failed checklist is exactly
    the artifact worth keeping.
    """
    options = dict(opts or {})
    options.setdefault("needs_model", bool({"inject", "micro"} & set(names)))
    results: list[PreflightResult] = []
    stopped_at: str | None = None
    for name in names:
        runner = STAGE_RUNNERS[name]
        try:
            result = runner(resources, options)
        except Exception as exc:  # a stage that RAISES is a failed stage
            result = _fail(
                name,
                "%s: %s" % (type(exc).__name__, exc),
                error_type=type(exc).__name__,
            )
        results.append(result)
        printer("%-4s %-9s %s" % ("PASS" if result.ok else "FAIL", name, result.message))
        if not result.ok:
            stopped_at = name
            break

    exit_code = 0 if stopped_at is None else 1
    if out_path is not None:
        payload = {
            "ok": stopped_at is None,
            "requested": list(names),
            "stopped_at": stopped_at,
            "model_id": getattr(resources, "model_id", None),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "stages": [r.to_dict() for r in results],
        }
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        printer("wrote %s" % path)
    return results, exit_code


def preflight_or_exit(
    stages: Sequence[str],
    model_id: str,
    *,
    out_path: str | Path | None = None,
    printer: Callable[[str], None] = print,
    resources: Any = None,
    **resource_kwargs: Any,
) -> list[PreflightResult]:
    """Run ``stages`` and ``SystemExit`` on the first failure. GPU ONLY.

    This is the entry point the phases of ``experiments/modal_spec_run.py``
    call before they spend anything.  ``SystemExit`` rather than ``RuntimeError``
    so a Modal function stops with a non-zero status instead of being retried.
    """
    res = resources if resources is not None else GpuResources(
        model_id, **resource_kwargs
    )
    try:
        results, code = run_stages(
            list(stages), res, out_path=out_path, printer=printer
        )
    finally:
        release = getattr(res, "release", None)
        if callable(release):
            release()
    if code:
        bad = next((r for r in results if not r.ok), None)
        raise SystemExit(
            "pre-flight FAILED at stage %r: %s"
            % (bad.name if bad else "?", bad.message if bad else "unknown")
        )
    return results


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="fail-fast GPU pre-flight checklist for the spec run"
    )
    p.add_argument(
        "--stage",
        default="all",
        choices=("all",) + STAGE_ORDER,
        help="which stage to run (default: all, in order, stopping at the first failure)",
    )
    p.add_argument("--model-id", required=True)
    p.add_argument("--judge-base-url", default=None, help="e.g. http://127.0.0.1:8000")
    p.add_argument("--chat", default=None, help="chat json for the context stage")
    p.add_argument(
        "--gpu-mem-gb", type=float, default=None,
        help="override the card's memory (default: read from torch.cuda)",
    )
    p.add_argument("--out", default="preflight.json")
    p.add_argument(
        "--allow-cpu", action="store_true",
        help="do not fail the env stage when CUDA is absent (diagnostics only)",
    )
    p.add_argument(
        "--allow-linear-layers", action=argparse.BooleanOptionalAction,
        default=None,
        help="accept a HYBRID model (linear + full attention): the injection "
             "then reaches the full-attention layers ONLY. DEFAULT: on for the "
             "decided hybrid reader, off for every other model id; "
             "--no-allow-linear-layers forces the dense-only rule",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    resources = GpuResources(
        args.model_id,
        judge_base_url_=args.judge_base_url,
        chat=args.chat,
        gpu_mem_gb=args.gpu_mem_gb,
        allow_linear_layers=(
            bool(args.allow_linear_layers)
            if args.allow_linear_layers is not None
            else default_allow_linear_layers(args.model_id)
        ),
    )
    _results, code = run_stages(
        resolve_stages(args.stage),
        resources,
        out_path=args.out,
        opts={"require_cuda": not args.allow_cpu},
    )
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

"""modal_spec_run.py -- Modal orchestration for the spec-faithful run (S2.3).

Phases (each a separate Modal function so at most ONE model copy is resident):

1. ``phase_manager``   vLLM serves the judge on 127.0.0.1:8000; a 1-session
                       PILOT build runs first and writes ``pilot_summary.json``;
                       ``should_abort_pilot`` decides whether the full
                       ``build_cd_offline --extractor spec --judge local`` runs.
                       Output: ``/vol/<run_id>/cd_spec.json``.
2. ``phase_instrument`` HF model; proves the injection actually reaches the
                       tokens ON THE PATCHED PATH (see B2 below).
                       Output: ``/vol/<run_id>/instrument.json``.  Any failed
                       check fails the run.
3. ``phase_reader``    the (arm x w x inject) grid through
                       ``benchmark.bineval.run_reader``. Output:
                       ``/vol/<run_id>/answers/<cell>.json`` (+ ``.meta.json``)
                       and ``/vol/<run_id>/run_manifest.json``.
4. download            local; pulls ``/vol/<run_id>/`` into
                       ``benchmark/bineval/results/spec_run/``.

RUN ISOLATION (F3)
------------------
The Modal volume PERSISTS between runs, and both the manager (checkpoint
resume) and the reader (``answers/<cell>.json`` exists -> "already_present")
treat whatever they find on it as their own.  Two runs a week apart with a
different grid, a different model or a different pairing mode therefore used to
merge into one directory that describes neither.  Every output now lives under
``/vol/<run_id>/``; ``run_id`` defaults to ``<UTC timestamp>-<short git sha>``
and is PRINTED, and ``/vol/<run_id>/run_info.json`` records what produced it.
Pass the SAME ``--run-id`` to the later phases and to ``--download``.

MODEL CHOICE (A3) -- read this before spending a GPU hour
---------------------------------------------------------
The mass patch replaces ``F.scaled_dot_product_attention``.  It reaches a layer
only if that layer calls sdpa.

``Qwen/Qwen3.8-27B`` is a HYBRID-attention model: its config (model_type
``qwen3_5``) puts everything under ``text_config``, whose ``layer_types`` repeat
3 x ``linear_attention`` + 1 x ``full_attention`` (``full_attention_interval``
4) -- 48 linear and 16 full of 64 layers, ``num_key_value_heads`` 4,
``head_dim`` 256, ``max_position_embeddings`` 262144.  The injection would
reach 16 of 64 layers, and the linear layers would carry the context
un-biased.

Two runnable configurations:

1. DEFAULT (2026-09-07 owner decision) -- ``Qwen/Qwen3.8-27B`` for BOTH the
   judge and the reader (``DEFAULT_MODEL_ID`` = ``DEFAULT_READER_MODEL_ID`` =
   ``HYBRID_MODEL_ID``).  This is the MEASURED CONFIGURATION: the injection
   reaches the 16 full-attention layers and the 48 linear layers carry the
   context un-biased.  Every number the default run produces is a 16-of-64-layer
   intervention and must be reported as such.  ``--allow-linear-layers``
   therefore defaults to ON for this model id, and ``run_info.json`` records
   ``n_sdpa_layers_expected`` = 16 so a later reader cannot mistake the run for
   a full-depth one.  ``--no-allow-linear-layers`` forces the dense-only rule
   back on and makes the default model fail the config check on purpose.
2. The DENSE alternative -- ``--reader-model-id Qwen/Qwen3-32B
   --judge-model-id Qwen/Qwen3-32B`` (``DENSE_MODEL_ID``; ``Qwen/Qwen3-14B`` is
   the smaller one).  Every layer is ``full_attention``, so the bias reaches all
   64 and ``n_sdpa_layers_expected`` is left unset.  For any OTHER hybrid model
   id ``--allow-linear-layers`` still defaults to off and
   ``check_model_supported`` refuses the config with the arithmetic in the
   message -- a measured choice, never a silent partial-depth run.

LINEAR-ATTENTION KERNELS (2026-09-07)
-------------------------------------
transformers 5.8 ``modeling_qwen3_5.py`` imports the linear-attention fast path
behind two guards -- ``is_flash_linear_attention_available()`` (``fla`` >= 0.2.2
AND CUDA) and ``is_causal_conv1d_available()`` (``causal_conv1d`` AND CUDA), at
``modeling_qwen3_5.py:49-64``.  When either is missing,
``is_fast_path_available`` (:205-207) is False, the module logs a warning and
uses its PURE-TORCH implementations (``torch_causal_conv1d_update``,
``torch_chunk_gated_delta_rule``, ``torch_recurrent_gated_delta_rule``, wired at
:407-418) plus ``Qwen3_5RMSNormGated`` in place of fla's ``FusedRMSNormGated``
(:393-397).  The model therefore RUNS without either package: no compiled kernel
is demanded, so the image stays on ``debian_slim`` and is NOT rebased on a CUDA
devel image.

DECISION: install ``flash-linear-attention`` (FLA_PIN) -- it is Triton-based and
needs no nvcc -- and do NOT install ``causal-conv1d``, whose PyPI sdist compiles
CUDA sources and would need ``nvidia/cuda:12.x-devel`` plus a torch-visible
build environment.  The conv path then takes the ``F.silu(self.conv1d(...))``
branch at :483.  Both facts are recorded per run: ``preflight_reader.json``'s
config stage carries ``linear_attention_kernels`` (which of the two imported,
and whether the fallback would be used).  This only affects the 48 LINEAR
layers, which the mass patch never touches.

WHAT THE INSTRUMENT MEASURES (B2, 2026-09-07 review)
----------------------------------------------------
The previous version loaded the model with ``attn_implementation="eager"`` and
read ``output_attentions``.  Eager attention never calls
``F.scaled_dot_product_attention``, which is the ONLY thing the mass patch
replaces, so that check measured the UNPATCHED model -- and it read the
attention at PREFILL, where the 1D mass bias is deliberately None
(``prefill_mass_scale = 0``).  It could not have failed, and it could not have
passed for the right reason.

The instrument now runs under ``sdpa`` and reads
``MassWeightedLLM.recorded_attention``: the patched sdpa recomputes
``softmax(q @ k^T * scale + attn_mask)`` for the last query row of every DECODE
call, AFTER the mass bias has been folded into ``attn_mask``.  That is the
distribution the model actually used.

Naive expectation vs what is asserted: the bias is additive pre-softmax, so a
planet of mass 4.0 at w = 1.0 multiplies the unnormalised weight of its tokens
by ``exp(w * mass) = exp(4) ~ 54.6``.  Softmax then RENORMALISES, so the
observed share goes from ``s`` to ``s*e^4 / (s*e^4 + (1-s))``: that ratio equals
``e^4`` only while ``s`` is negligible and saturates at ``1/s`` as ``s`` grows.
The assertion is therefore the loose-but-real ``share(1.0) / share(0) >= 2``
plus strict monotonicity in w, NOT ``>= exp(4)``.

EXACT COMMANDS
--------------
Deploy the app (creates/updates the Modal app and its volumes)::

    modal token new                      # once, if not authenticated
    python experiments/modal_spec_run.py --deploy

Run the phases DETACHED (F5).  ``--spawn`` needs the deployed app; the call
survives the client disconnecting, which a 12-hour function will otherwise not::

    python experiments/modal_spec_run.py --spawn manager    --run-id RID
    python experiments/modal_spec_run.py --wait  manager    --run-id RID
    python experiments/modal_spec_run.py --spawn instrument --run-id RID
    python experiments/modal_spec_run.py --spawn reader     --run-id RID

``--run <phase>`` still exists and is fine for a short phase, but it is TIED TO
THE CLIENT SESSION: closing the laptop kills a 12-hour function.  It prints a
warning saying so.

Download the results (``--run-id`` is required: the volume holds every run)::

    python experiments/modal_spec_run.py --download --run-id RID \
        --dest benchmark/bineval/results/spec_run
    # equivalent raw CLI:
    modal volume get cms-spec-results /RID benchmark/bineval/results/spec_run

Score LOCALLY (no GPU, no network).  ``--subset generated`` is MANDATORY (F4):
the reader answers the 173 GENERATED non-excluded questions, so scoring with
the scorer's default ``--subset all`` puts 189 in the denominator and reports
every arm ~8 points low::

    python experiments/modal_spec_run.py --score \
        benchmark/bineval/results/spec_run/RID
    # writes scores.csv next to answers/; per cell, equivalent to:
    python -m benchmark.bineval.score_binary \
        --answers <answers/<cell>.json> \
        --questions benchmark/bineval/questions_restaurant.json \
        --out <scores/<cell>.json> --subset generated --max-words 32

Dry run (CPU, no Modal, no model, no network) -- what the tests exercise::

    python experiments/modal_spec_run.py --dry-run          # writes to a temp dir
    python experiments/modal_spec_run.py --dry-run --out <dir>

``modal`` is imported ONCE at module import inside a try/except (M5) so the
three Modal functions can be defined at MODULE level -- Modal re-imports this
module inside the container and would not see functions defined in a closure.
On a machine without Modal the import fails quietly and only the dry run and
the pure helpers are available.  torch/transformers are likewise imported inside
the functions that need them, never at module level.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow `python experiments/modal_spec_run.py`
    sys.path.insert(0, str(REPO_ROOT))

APP_NAME = "cms-spec-run"
VOLUME_NAME = "cms-spec-results"
VOL_MOUNT = "/vol"
# F6: the weights are ~56 GB and were re-downloaded by EVERY phase, because a
# Modal container starts with an empty HF cache. One shared volume mounted at
# HF_HOME on all three functions turns that into one download per model.
HF_CACHE_VOLUME_NAME = "cms-hf-cache"
HF_CACHE_MOUNT = "/root/.cache/huggingface"
REMOTE_REPO = "/root/cms-prototype"
GPU_SPEC = "H200"
# Directive 5: judge and reader are the SAME open-weight model on the same GPU.
# A3: Qwen/Qwen3.8-27B is hybrid-attention (48 linear + 16 full of 64 layers,
# config.json verified 2026-09-08), so the sdpa mass patch reaches a quarter of
# the stack -- accepted deliberately. To run the DENSE alternative instead:
# --judge-model-id Qwen/Qwen3-32B
# --reader-model-id Qwen/Qwen3-32B.
#
# 2026-09-07 owner decision (supersedes the dense default above): BOTH roles run
# Qwen/Qwen3.8-27B and the injection deliberately reaches its 16 full-attention
# layers only. The hybrid arithmetic is unchanged and still reported --
# n_sdpa_layers_expected = 16 in run_info.json, n_sdpa_layers = 16 in every meta.
HYBRID_MODEL_ID = "Qwen/Qwen3.8-27B"
# The layer counts of HYBRID_MODEL_ID's config.json, duplicated here as literals
# because this module cannot import the repo at MODULE level (see _enter_repo:
# inside the Modal container REPO_ROOT is "/" until the function body runs).
# tests/reader/test_prerun_fixes.py asserts they equal run_reader's constants.
HYBRID_N_SDPA_LAYERS = 16
HYBRID_N_LINEAR_LAYERS = 48
DEFAULT_MODEL_ID = HYBRID_MODEL_ID
# The dense full-attention alternative: every layer calls sdpa, so the bias
# reaches all 64. Ask for it with --reader-model-id / --judge-model-id.
DENSE_MODEL_ID = "Qwen/Qwen3-32B"
DEFAULT_READER_MODEL_ID = DEFAULT_MODEL_ID
DEFAULT_CHAT = "benchmark/longchat/restaurant_chat_v2.json"
DEFAULT_QUESTIONS = "benchmark/bineval/questions_restaurant.json"

# B5: a 664-turn manager build plus a ~40-cell reader grid does not fit in 6h.
FUNCTION_TIMEOUT_S = 12 * 60 * 60
# B5: checkpoint often enough that a preemption costs minutes, not hours.
MANAGER_CKPT_INTERVAL = 25
# B5: the Q_NODE calls of one turn go to vLLM in parallel.
MANAGER_WORKERS = 8
# F2: management.harness.manager / spec_manager default. > 0 means the manager
# ranks candidates with an SBERT embedding shortlist, which needs
# sentence-transformers INSIDE the vLLM image.
DEFAULT_SHORTLIST_K = 5

# M4: pinned images.  Two of them, because vLLM pins its own (older)
# transformers and the reader needs transformers 5.8 for the sdpa mask path.
VLLM_PIN = "vllm==0.28.0"  # 2026-08-26 stable; Qwen3.8 day-0 support (vllm.ai/blog/2026-08-12-qwen3.8)
TORCH_PIN = "torch==2.11.*"
TRANSFORMERS_PIN = "transformers==5.8.*"
# The qwen3_5 linear-attention fast path (modeling_qwen3_5.py:59-61) imports
# ``fla.modules`` / ``fla.ops.gated_delta_rule``; transformers gates it on
# fla >= 0.2.2 (utils/import_utils.py:810-812). 0.5.2 is the current release
# (2026-07-27) and pulls fla-core==0.5.2 + transformers>=4.45 -- deliberately
# WITHOUT a backend extra, because ``[cuda]`` would pull its own torch and
# resolve TORCH_PIN away (the same trap documented for image_vllm below).
# Triton kernels, compiled at run time: no nvcc, so debian_slim is enough.
FLA_PIN = "flash-linear-attention==0.5.2"
# F2: `--judge local` runs the manager with shortlist_k=5, which imports
# sentence_transformers INSIDE the manager -- after vLLM is up and the pilot has
# started, i.e. after the expensive part. It was missing from the vLLM image.
# NOTE: no torch pin is added alongside it. sentence-transformers depends on
# torch, and the vLLM image already carries the exact torch vLLM was built
# against; pinning a second torch here would let pip resolve one of the two away
# and break the server this image exists to run.
SENTENCE_TRANSFORMERS_PIN = "sentence-transformers"

# B3: what NOT to ship to the container.  ``models/`` is this repo's own source
# package (models/node.py, models/correlation_diagram.py) -- excluding it, as an
# earlier version of this file did, would have made every remote import fail.
# F7: ``benchmark/bineval/results/**`` ALSO excluded the two source artifacts the
# grid reads -- ``results/pilot/summary_6x.txt`` (the ``summary_9x`` arm) and
# ``results/pilot/oracle_cd_full.txt`` -- so those arms would have been reported
# as "skipped, optional artifact missing" on every remote run. The ignore is now
# per-subdirectory and ``results/pilot/`` (1.5 MB) ships.
IMAGE_IGNORE = [
    "benchmark/bineval/results/spec_run*/**",
    "benchmark/bineval/results/cd/**",
    "benchmark/bineval/results/audit/**",
    "benchmark/bineval/results/personamem_probe/**",
    "**/*.pdf",
    "**/*.html",
    "**/.git/**",
    "**/__pycache__/**",
    "wandb/**",
    "outputs/**",
]

# The two files the ignore list must NOT match (asserted by the tests).
IMAGE_REQUIRED_PATHS = (
    "benchmark/bineval/results/pilot/summary_6x.txt",
    "benchmark/bineval/results/pilot/oracle_cd_full.txt",
)

# B4: usable HBM per GPU class, for the pre-flight and for --include-long-arms.
GPU_MEM_GB: dict[str, float] = {
    "A100-40GB": 40.0,
    "A100-80GB": 80.0,
    "H100": 80.0,
    "H200": 141.0,
    "B200": 180.0,
}
# The `full` (86k tokens) and `trunc_2x` arms need a KV cache that an 80GB card
# cannot hold next to a 27B bf16 model.
LONG_ARM_MIN_GPU_GB = 141.0
LONG_ARMS = ("full", "trunc_2x")

# ---- default grid (S2.3 phase 3) -----------------------------------------
# B4: `full` and `trunc_2x` are NOT default arms -- they need an H200.  Add them
# with --include-long-arms (which requires an explicit --gpu of >= 141GB).
DEFAULT_ARMS = [
    "trunc_6x",
    "trunc_10x",
    "summary_9x",
    "cd_mass_6x",
    "cd_mass_10x",
    "cd_random_6x",
    "floor",
]
# M6: w alone says nothing; the pre-softmax bias is w * mass, and the observed
# planet masses run into the dozens, so the old grid (up to 1.0) explored
# effective biases of e^40 and larger.  This grid stays inside a sane band, and
# every cell's meta records w * max_planet_mass so the real knob is visible.
DEFAULT_W_GRID = (0.0, 0.02, 0.05, 0.1, 0.2, 0.5)
DEFAULT_INJECT_MODES = ["planet", "planet+satellites"]
# one exploratory prefill cell
EXPLORATORY_CELL = {
    "arm": "cd_mass_6x",
    "w": 0.1,
    "inject": "planet",
    "prefill_scale": 1.0,
    "max_questions": 20,
}

PILOT_DEFAULTED_MAX = 0.2
PILOT_NODE_FALLBACK_MAX = 0.2

# Arms whose source artifact is optional: a missing file marks the cell
# "skipped" instead of killing the run.  Everything else must exist.
OPTIONAL_ARTIFACT_ARMS = ("summary_9x", "oracle_cd_full")

# ``oracle_cd_full`` is stored in the LEGACY [PN{mass}]-on-every-line format;
# find_marker_spans reads the level from the marker, so every line would look
# like a planet.  It is a text-only arm.
LEGACY_TEXT_ONLY_ARMS = ("oracle_cd_full",)

_PN_MASS_RE = re.compile(r"\[PN(\d+(?:\.\d+)?)\]")


# --------------------------------------------------------------------------
# modal (M5): imported once, at module level, guarded
# --------------------------------------------------------------------------

try:  # pragma: no cover - exercised only where modal is installed
    modal: Any = importlib.import_module("modal")
except Exception:  # ImportError, and anything modal raises at import time
    modal = None


def modal_available() -> bool:
    return modal is not None


# --------------------------------------------------------------------------
# A1: where the repo is, HERE and in the container
# --------------------------------------------------------------------------

def repo_path(rel: str | Path) -> Path:
    """A repo-relative path, resolved against the CURRENT ``REPO_ROOT``.

    A1: when Modal runs this file as ``__main__`` it mounts it at
    ``/root/modal_spec_run.py``, so the module-level
    ``Path(__file__).resolve().parents[1]`` evaluates to ``/`` inside the
    container and every ``REPO_ROOT / "benchmark/..."`` became ``/benchmark/...``
    -- a FileNotFoundError one GPU-hour into the reader phase.  ``_enter_repo``
    rebinds the global to ``REMOTE_REPO``; going through this helper is what
    makes that rebinding visible to code that already imported the module.
    """
    return REPO_ROOT / str(rel)


# --------------------------------------------------------------------------
# F7: which repo paths the image ignores
# --------------------------------------------------------------------------

def _ignore_pattern_to_regex(pattern: str) -> "re.Pattern[str]":
    """Glob -> regex with the semantics this module relies on.

    ``**/`` matches zero or more leading directories, ``**`` matches anything
    including ``/``, ``*`` and ``?`` never cross a ``/``.  Written out rather
    than delegated to ``fnmatch`` because ``fnmatch`` lets ``*`` cross ``/``,
    which would make ``benchmark/*/results`` match three levels down.
    """
    out: list[str] = []
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if pattern.startswith("**/", i):
            out.append("(?:[^/]+/)*")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif ch == "*":
            out.append("[^/]*")
            i += 1
        elif ch == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(ch))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def is_ignored(path: str, patterns: Iterable[str] = IMAGE_IGNORE) -> bool:
    """True when a repo-relative POSIX path is excluded from the image."""
    rel = str(path).replace("\\", "/").lstrip("./")
    return any(_ignore_pattern_to_regex(p).match(rel) for p in patterns)


# --------------------------------------------------------------------------
# F3: run identity
# --------------------------------------------------------------------------

def git_sha(default: str = "unknown") -> str:
    """Short git sha of this checkout, or ``default``. Never raises."""
    import subprocess

    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=10,
        )
    except Exception:  # noqa: BLE001 -- 'never raises' means never
        return default
    sha = proc.stdout.strip()
    return sha if proc.returncode == 0 and sha else default


def default_run_id(*, now: Any = None, sha: str | None = None) -> str:
    """``<UTC timestamp>-<short git sha>``; the sha is dropped when unknown."""
    import time

    stamp = time.strftime("%Y%m%dT%H%M%SZ", now if now is not None else time.gmtime())
    code = sha if sha is not None else git_sha()
    return "%s-%s" % (stamp, code) if code and code != "unknown" else stamp


def run_root(vol: str | Path, run_id: str) -> Path:
    """``/vol/<run_id>``. Every output of every phase lives under this."""
    if not run_id or "/" in run_id or "\\" in run_id or run_id in (".", ".."):
        raise ValueError("run_id must be a single path segment, got %r" % (run_id,))
    return Path(vol) / run_id


def write_run_info(root: str | Path, **fields: Any) -> dict:
    """Merge ``fields`` into ``<root>/run_info.json`` and return the result.

    Merged rather than overwritten because the three phases each know a
    different part of it and run in separate containers.
    """
    path = Path(root) / "run_info.json"
    payload: dict = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
    payload.update({k: v for k, v in fields.items() if v is not None})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return payload


def build_run_info(
    run_id: str,
    *,
    model_id: str | None = None,
    reader_model_id: str | None = None,
    arms: Iterable[str] | None = None,
    w_grid: Iterable[float] | None = None,
    allow_linear_layers: bool | None = None,
    n_sdpa_layers_expected: int | None = None,
    versions: dict | None = None,
    pairing: str = "round_trip",
) -> dict:
    """The run_info.json payload. Pure, so the tests can read it without Modal."""
    return {
        "run_id": run_id,
        "git_sha": git_sha(),
        # build_cd_offline drives the spec manager once per user->assistant
        # round trip (0036); the checkpoint carries the same word (F3).
        "pairing": pairing,
        "model_id": model_id,
        "reader_model_id": reader_model_id,
        "arms": list(arms) if arms is not None else None,
        "w_grid": list(w_grid) if w_grid is not None else None,
        "allow_linear_layers": allow_linear_layers,
        # A3: how many layers the sdpa patch MUST reach for this reader (16 for
        # the hybrid default). Written next to the model id so a later reader of
        # the directory cannot mistake a 16/64 run for a full-depth one.
        "n_sdpa_layers_expected": (
            n_sdpa_layers_expected
            if n_sdpa_layers_expected is not None
            else expected_n_sdpa_layers(reader_model_id)
        ),
        "versions": versions or {},
    }


# --------------------------------------------------------------------------
# F2: the manager's embedding shortlist needs sentence-transformers
# --------------------------------------------------------------------------

def needs_sentence_transformers(judge: str, shortlist_k: int) -> bool:
    """Will the manager import ``sentence_transformers``?

    ``build_cd_offline`` forces ``shortlist_k = 0`` for ``--judge fake`` (a fake
    judge cannot rank anything), so only a REAL judge with a positive shortlist
    loads an SBERT model.  Pure, so the decision is testable without vLLM.
    """
    return judge != "fake" and int(shortlist_k) > 0


def require_sentence_transformers(
    judge: str, shortlist_k: int, *, import_fn: Any = None
) -> bool:
    """Fail BEFORE vLLM starts when the shortlist import would fail later.

    Without this the missing package surfaces inside the manager, minutes into
    the pilot -- after the 56 GB model download and the server start-up.
    """
    if not needs_sentence_transformers(judge, shortlist_k):
        return False
    importer = import_fn if import_fn is not None else importlib.import_module
    try:
        importer("sentence_transformers")
    except Exception as exc:
        raise SystemExit(
            "the manager will run with judge=%r and shortlist_k=%d, which imports "
            "sentence_transformers for the embedding shortlist, but it is not "
            "importable here: %s. Add it to the image (SENTENCE_TRANSFORMERS_PIN) "
            "or run with --shortlist-k 0" % (judge, shortlist_k, exc)
        ) from exc
    return True


# --------------------------------------------------------------------------
# pure helpers (CPU, tested)
# --------------------------------------------------------------------------

def flatten_pilot_summary(summary: dict) -> dict:
    """B1: build_cd_offline's summary -> the four numbers the abort rule reads.

    ``build_cd_offline`` writes ``{"harness_calls": {kind: n}, "total": n,
    "harness_quality": {"defaulted": n, "node_fallback": n, ...}}``.  The abort
    rule reads flat ``calls`` / ``defaulted`` / ``nodes`` / ``node_fallback``
    keys, which that file has NEVER written: before this flattening every lookup
    defaulted to 0, so ``should_abort_pilot`` could only ever fire its "0 nodes"
    branch and a pilot with 100% defaulted judge calls sailed through.
    """
    calls_by_kind = summary.get("harness_calls") or {}
    quality = summary.get("harness_quality") or {}
    return {
        "calls": int(sum(calls_by_kind.values())),
        "defaulted": int(quality.get("defaulted", 0) or 0),
        "nodes": int(summary.get("total", 0) or 0),
        "node_fallback": int(quality.get("node_fallback", 0) or 0),
    }


def should_abort_pilot(
    pilot: dict,
    *,
    defaulted_max: float = PILOT_DEFAULTED_MAX,
    node_fallback_max: float = PILOT_NODE_FALLBACK_MAX,
) -> tuple[bool, str]:
    """Abort rule on pilot quality (S2.3 phase 1).

    Abort when ``defaulted / calls > defaulted_max`` or
    ``node_fallback / nodes > node_fallback_max``. Zero denominators never
    abort on that term (there is nothing to judge yet); a pilot that produced
    no nodes at all aborts with its own reason.

    The input is the FLAT shape of :func:`flatten_pilot_summary`.
    """
    calls = float(pilot.get("calls", 0) or 0)
    defaulted = float(pilot.get("defaulted", 0) or 0)
    nodes = float(pilot.get("nodes", 0) or 0)
    fallback = float(pilot.get("node_fallback", 0) or 0)

    reasons: list[str] = []
    if calls > 0 and defaulted / calls > defaulted_max:
        reasons.append(
            "defaulted/calls = %d/%d = %.3f > %.2f"
            % (defaulted, calls, defaulted / calls, defaulted_max)
        )
    if nodes > 0 and fallback / nodes > node_fallback_max:
        reasons.append(
            "node_fallback/nodes = %d/%d = %.3f > %.2f"
            % (fallback, nodes, fallback / nodes, node_fallback_max)
        )
    if nodes == 0:
        reasons.append("pilot produced 0 nodes")
    if reasons:
        return True, "; ".join(reasons)
    return False, "ok"


def max_planet_mass(context_text: str) -> float:
    """Largest ``[PN{mass}]`` value in a serialized context block (0.0 if none).

    M6: the pre-softmax bias is ``w * mass``, so the w grid is only
    interpretable next to the biggest mass the arm actually carries.
    """
    masses = [float(m) for m in _PN_MASS_RE.findall(context_text)]
    return max(masses) if masses else 0.0


def cell_name(arm: str, w: float, inject: str, prefill_scale: float = 0.0) -> str:
    """Filesystem-safe cell id. Deterministic and injective over the grid."""
    parts = [arm, "w%g" % w]
    if w > 0:
        parts.append(inject.replace("+", "-"))
    if prefill_scale > 0:
        parts.append("pf%g" % prefill_scale)
    return "__".join(parts)


def resolve_arms(
    arms: Iterable[str] | None, *, include_long: bool, gpu: str | None
) -> list[str]:
    """B4: the arm list, with the long arms gated behind an explicit big GPU."""
    chosen = list(arms) if arms is not None else list(DEFAULT_ARMS)
    if not include_long:
        return chosen
    if gpu is None:
        raise SystemExit(
            "--include-long-arms requires an explicit --gpu of at least %.0fGB "
            "(e.g. --gpu H200): %s need a KV cache an 80GB card cannot hold "
            "next to a 27B bf16 model" % (LONG_ARM_MIN_GPU_GB, ", ".join(LONG_ARMS))
        )
    mem = GPU_MEM_GB.get(gpu)
    if mem is None:
        raise SystemExit(
            "unknown --gpu %r (known: %s)" % (gpu, ", ".join(sorted(GPU_MEM_GB)))
        )
    if mem < LONG_ARM_MIN_GPU_GB:
        raise SystemExit(
            "--include-long-arms needs >= %.0fGB of GPU memory, --gpu %s has %.0fGB"
            % (LONG_ARM_MIN_GPU_GB, gpu, mem)
        )
    return [a for a in LONG_ARMS if a not in chosen] + chosen


def validate_cell(cell: dict) -> dict:
    """Refuse a cell that cannot mean what its name says.

    Today that is exactly one rule: ``oracle_cd_full`` is stored in the LEGACY
    ``[PN{mass}]``-on-every-line format, where every line looks like a planet to
    ``find_marker_spans``, so an injected cell on it would spray the mass over
    suns and satellites alike.  It is a text-only arm.
    """
    if cell["w"] > 0 and cell["arm"] in LEGACY_TEXT_ONLY_ARMS:
        raise ValueError(
            "arm %r is stored in the legacy [PN{mass}]-on-every-line format "
            "and cannot be used with w=%g: the level is read from the marker, "
            "so every line would look like a planet and the mass would land on "
            "suns and satellites alike" % (cell["arm"], cell["w"])
        )
    return cell


def build_cells(
    arms: Iterable[str] = DEFAULT_ARMS,
    w_grid: Iterable[float] = DEFAULT_W_GRID,
    inject_modes: Iterable[str] = DEFAULT_INJECT_MODES,
    *,
    exploratory: dict | None = EXPLORATORY_CELL,
    max_questions: int | None = None,
) -> list[dict]:
    """The (arm x w x inject) grid.

    Only ``cd_*`` arms sweep w (the others carry no ``[PN`` markers, so w would
    be a no-op); ``inject`` only varies for w > 0.

    M7: a ``cd_*`` cell at w = 0 keeps ``inject`` at its REAL value.  The bias is
    ``0 * mass``, i.e. an all-zero tensor, which is still built, still combined
    with the attention mask and still added -- so the w=0 control runs through
    exactly the same kernel path as the injected cells and the only difference
    between them is one scalar.  Setting ``inject="none"`` there (the old
    behaviour) skipped ``set_mass_vector`` entirely and made the control a
    DIFFERENT code path, which is the one thing a control must not be.
    """
    cells: list[dict] = []
    seen: set[str] = set()
    inject_list = list(inject_modes)
    default_inject = inject_list[0] if inject_list else "planet"

    def add(arm: str, w: float, inject: str, prefill: float, maxq: int | None) -> None:
        name = cell_name(arm, w, inject, prefill)
        if name in seen:
            return
        seen.add(name)
        cells.append(
            validate_cell(
                {
                    "cell": name,
                    "arm": arm,
                    "w": w,
                    # M7: the real inject value, even at w=0 for a cd arm.
                    "inject": inject,
                    "prefill_scale": prefill,
                    "max_questions": maxq,
                }
            )
        )

    for arm in arms:
        # A cd arm sweeps w; oracle_cd_full is asked to sweep only so that
        # validate_cell can refuse it loudly instead of silently producing a
        # single w=0 cell for an arm the caller clearly meant to inject into.
        is_cd = arm.startswith("cd_")
        sweeps = is_cd or arm in LEGACY_TEXT_ONLY_ARMS
        for w in (w_grid if sweeps else [0.0]):
            if w == 0:
                add(arm, 0.0, default_inject if is_cd else "none", 0.0, max_questions)
            else:
                for inject in inject_list:
                    add(arm, w, inject, 0.0, max_questions)
    if exploratory:
        add(
            exploratory["arm"],
            exploratory["w"],
            exploratory["inject"],
            exploratory.get("prefill_scale", 0.0),
            exploratory.get("max_questions"),
        )
    return cells


def parse_w_grid(raw: str) -> tuple[float, ...]:
    """``--w-grid "0,0.05,0.2"`` -> (0.0, 0.05, 0.2). Sorted, deduplicated."""
    values = sorted({float(part) for part in raw.split(",") if part.strip()})
    if not values:
        raise SystemExit("--w-grid needs at least one value")
    if any(v < 0 for v in values):
        raise SystemExit("--w-grid values must be >= 0")
    return tuple(values)


def _arm_tokens(arm: str, chat: dict | None, cd_json: Path | None) -> int | None:
    """tiktoken size of an arm's context.

    Returns None ONLY when an OPTIONAL stored artifact is missing
    (``summary_9x`` / ``oracle_cd_full``); every other failure propagates.
    Swallowing ``FileNotFoundError`` and ``ValueError`` for every arm -- the old
    behaviour -- turned "the chat file moved" and "this arm needs a policy" into
    a silent ``tokens: null`` in the manifest.
    """
    from benchmark.bineval import arms as arms_mod

    try:
        ctx = arms_mod.build_arm_context(
            arm,
            chat=chat,
            cd_json=str(cd_json) if cd_json and Path(cd_json).exists() else None,
        )
    except FileNotFoundError:
        if arm in OPTIONAL_ARTIFACT_ARMS:
            return None
        raise
    return ctx.tokens


# --------------------------------------------------------------------------
# GPU-side implementations (NEVER executed locally / on --dry-run)
# --------------------------------------------------------------------------

def vllm_command(model_id: str, port: int) -> list[str]:
    """M3: the vLLM serve command.

    ``--reasoning-parser none`` was removed: ``none`` is not a registered parser
    name, so vLLM exits at startup with "invalid choice" and the phase died
    before the first judge call.  Thinking is disabled where it belongs, in the
    judge's request body (``build_cd_offline.DEFAULT_JUDGE_EXTRA_BODY`` sends
    ``chat_template_kwargs.enable_thinking = false`` by default).
    """
    return [
        "vllm", "serve", model_id,
        "--host", "127.0.0.1", "--port", str(port),
        "--max-model-len", "16384",
        # Leave room for the manager process next to the server.
        "--gpu-memory-utilization", "0.85",
        # The manager asks at most MANAGER_WORKERS questions concurrently.
        "--max-num-seqs", "16",
    ]


def _wait_for_vllm(server: Any, port: int, deadline_s: int) -> None:
    """Poll ``/v1/models`` until it answers, the server dies, or time runs out."""
    import time
    import urllib.request

    deadline = time.time() + deadline_s
    while True:
        if server.poll() is not None:
            raise RuntimeError("vllm serve exited with code %s" % server.returncode)
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:%d/v1/models" % port, timeout=5
            ):
                return
        except OSError:
            if time.time() > deadline:
                raise RuntimeError(
                    "vllm did not become ready in %ds" % deadline_s
                ) from None
            time.sleep(5)


def _run_and_commit(cmd: list[str], watch: Path, commit_fn: Any) -> None:
    """Run ``cmd``, committing the volume every time ``watch`` is rewritten.

    B5: the checkpoint is written by the CHILD process, so the parent cannot
    commit "inside" it; watching the file's mtime is the same event observed
    from outside.  A preemption then costs at most MANAGER_CKPT_INTERVAL turns.
    """
    import subprocess
    import time

    proc = subprocess.Popen(cmd)
    last_seen: float | None = None
    try:
        while proc.poll() is None:
            time.sleep(10)
            try:
                mtime = watch.stat().st_mtime
            except OSError:
                continue
            if mtime != last_seen:
                last_seen = mtime
                if commit_fn is not None:
                    commit_fn()
    finally:
        if proc.poll() is None:  # pragma: no cover - only on an exception
            proc.terminate()
    if proc.returncode != 0:
        raise RuntimeError("%s exited with %s" % (cmd[:4], proc.returncode))


def _phase_manager_impl(
    model_id: str,
    *,
    run_id: str,
    chat: str = DEFAULT_CHAT,
    vol: str = VOL_MOUNT,
    port: int = 8000,
    judge_max_wait_s: int = 900,
    workers: int = MANAGER_WORKERS,
    shortlist_k: int = DEFAULT_SHORTLIST_K,
    commit_fn: Any = None,
    hf_commit_fn: Any = None,
) -> dict:
    """vLLM judge + spec CD build. GPU ONLY. UNVERIFIED (no GPU here)."""
    import subprocess

    # F2: FIRST. The shortlist import happens inside the manager, minutes into
    # the pilot -- after the 56 GB download and the server start-up. Failing
    # here costs seconds instead.
    require_sentence_transformers("local", shortlist_k)

    vol_path = run_root(vol, run_id)
    vol_path.mkdir(parents=True, exist_ok=True)
    write_run_info(
        vol_path,
        **build_run_info(run_id, model_id=model_id, pairing="round_trip"),
        phase_manager={"shortlist_k": shortlist_k, "workers": workers, "chat": chat},
    )
    server = subprocess.Popen(vllm_command(model_id, port))
    try:
        _wait_for_vllm(server, port, judge_max_wait_s)
        # Fail fast BEFORE the 664-turn build: one Q_NODE + one yes/no round
        # trip proves the endpoint parses and serves the model asked for.
        from experiments.gpu_preflight import preflight_or_exit

        preflight_or_exit(
            ("judge",), model_id,
            out_path=vol_path / "preflight_manager.json",
            judge_base_url_="http://127.0.0.1:%d" % port,
        )

        base = [
            sys.executable, "-m", "benchmark.bineval.build_cd_offline",
            "--chat", chat, "--extractor", "spec",
            "--judge", "local",
            "--judge-base-url", "http://127.0.0.1:%d" % port,
            "--judge-model", model_id,
            "--manager-workers", str(workers),
            "--shortlist-k", str(shortlist_k),
            "--ckpt-interval", str(MANAGER_CKPT_INTERVAL),
        ]

        # -- pilot: one session, quality gate ------------------------------
        pilot_out = vol_path / "cd_spec_pilot.json"
        subprocess.run(base + ["--max-sessions", "1", "--out", str(pilot_out)], check=True)
        raw_summary = json.loads(pilot_out.read_text(encoding="utf-8")).get("summary", {})
        pilot_summary = dict(raw_summary)
        pilot_summary.update(flatten_pilot_summary(raw_summary))
        abort, reason = should_abort_pilot(pilot_summary)
        pilot_summary["abort"] = abort
        pilot_summary["abort_reason"] = reason
        if hf_commit_fn is not None:
            # F6: vLLM has loaded the weights by now; persist the cache even if
            # the pilot gate is about to abort the run.
            hf_commit_fn()
        (vol_path / "pilot_summary.json").write_text(
            json.dumps(pilot_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if commit_fn is not None:
            commit_fn()  # B5: the pilot verdict survives whatever happens next
        if abort:
            raise RuntimeError("pilot quality gate failed: %s" % reason)

        # -- full build, resuming from the checkpoint when there is one ----
        out = vol_path / "cd_spec.json"
        ckpt = out.with_name(out.name + ".ckpt")
        cmd = base + ["--out", str(out)]
        if ckpt.exists():
            cmd += ["--resume-from", str(ckpt)]
            print("[manager] resuming from %s" % ckpt)
        _run_and_commit(cmd, ckpt, commit_fn)
        if commit_fn is not None:
            commit_fn()
        if hf_commit_fn is not None:
            hf_commit_fn()  # F6: the weights are now in the shared HF cache
        return {"pilot": pilot_summary, "aborted": False, "run_id": run_id}
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()


INSTRUMENT_PROMPT_FILLER = "the quick brown fox jumps over the lazy dog"
INSTRUMENT_W_GRID = (0.0, 0.25, 1.0)
INSTRUMENT_MIN_RATIO = 2.0
# The planet mass written into the instrument prompt.
INSTRUMENT_PLANET_MASS = 4.0


def build_instrument_prompt(filler_repeats: int = 30) -> str:
    """A ~300-token single-needle prompt with exactly ONE ``[PN4.0]`` line."""
    filler = " ".join([INSTRUMENT_PROMPT_FILLER] * filler_repeats)
    return (
        "<CONTEXT>\n"
        "[SN] Restaurant plan\n"
        "  [PN%.1f] The rent deposit is 4,800,000 yen\n"
        "    [RN] %s\n"
        "</CONTEXT>\n\n"
        "Question: What is the rent deposit?\nAnswer:"
        % (INSTRUMENT_PLANET_MASS, filler)
    )


def mass_share(probabilities: Any, positions: Iterable[int]) -> float:
    """Fraction of the attention probability mass sitting on ``positions``."""
    idx = [p for p in positions if 0 <= p < int(probabilities.shape[0])]
    if not idx:
        return 0.0
    total = float(probabilities.sum())
    if total <= 0:
        return 0.0
    return float(probabilities[idx].sum()) / total


def check_ratio_grows(
    shares: dict[float, float], *, min_ratio: float = INSTRUMENT_MIN_RATIO
) -> dict:
    """B2 verdict on the per-w mass shares: strictly increasing, and >= ratio.

    Pure, so it can be unit-tested on synthetic vectors without a GPU.
    """
    ws = sorted(shares)
    values = [shares[w] for w in ws]
    monotone = all(a < b for a, b in zip(values, values[1:]))
    base = shares[ws[0]]
    top = shares[ws[-1]]
    ratio = (top / base) if base > 0 else float("inf")
    return {
        "w_grid": ws,
        "mass_share": {("w=%g" % w): shares[w] for w in ws},
        "ratio": ratio,
        "min_ratio": min_ratio,
        "ratio_monotone": monotone,
        "ratio_at_least_min": ratio >= min_ratio,
    }


def _phase_instrument_impl(
    model_id: str,
    *,
    run_id: str,
    vol: str = VOL_MOUNT,
    allow_linear_layers: bool = False,
    hf_commit_fn: Any = None,
) -> dict:
    """Single-needle injection instrumentation. GPU ONLY. UNVERIFIED.

    DEVIATION from the brief, deliberate and load-bearing: ``generate`` is
    called with ``max_new_tokens=2``, not 1.  HF's ``generate`` produces the
    FIRST new token out of the prefill forward, so ``max_new_tokens=1`` runs no
    ``seq_q == 1`` call at all -- there would be no decode step to record and
    ``bias_applied_calls`` would be 0.  With 2 the run is exactly one prefill
    (n_layers skipped) plus exactly one decode step (n_layers applied), which is
    what the three assertions below describe.
    """
    from transformers import AutoConfig, AutoTokenizer

    from benchmark.bineval.run_reader import build_reader_mass_vector, check_model_supported
    from server.cd_parser import find_marker_spans, marker_positions
    from server.mass_weighted_gemma import MassWeightedLLM

    from experiments.gpu_preflight import preflight_or_exit

    vol_path = run_root(vol, run_id)
    vol_path.mkdir(parents=True, exist_ok=True)
    write_run_info(
        vol_path,
        run_id=run_id,
        reader_model_id=model_id,
        allow_linear_layers=allow_linear_layers,
        n_sdpa_layers_expected=expected_n_sdpa_layers(model_id),
    )
    preflight_or_exit(
        ("env", "tokenizer", "config", "inject"), model_id,
        out_path=vol_path / "preflight_instrument.json",
        allow_linear_layers=allow_linear_layers,
    )

    config = AutoConfig.from_pretrained(model_id)
    layer_info = check_model_supported(config, allow_linear_layers=allow_linear_layers)
    # F1 + A3: NOT ``config.num_hidden_layers``. A vision-language wrapper has no
    # such attribute at the top level (it lives under text_config) and would
    # crash here; a hybrid model has one, but it counts layers the sdpa patch
    # can never reach. ``n_sdpa_layers`` is the number of layers that actually
    # call the patched kernel, which is what every assertion below compares to.
    n_layers = int(layer_info["n_sdpa_layers"])

    prompt = build_instrument_prompt()
    tok = AutoTokenizer.from_pretrained(model_id)
    ids = tok(prompt)["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]

    n_pn_lines = sum(1 for ln in prompt.splitlines() if "[PN" in ln)
    spans = find_marker_spans(ids, tok)
    planet_positions = [p for p, _m in marker_positions(spans, inject_levels={"planet"})]

    checks: dict[str, Any] = {
        "prompt_tokens": len(ids),
        "pn_lines": n_pn_lines,
        "planet_spans": sum(1 for lvl, _m, _p in spans if lvl == "planet"),
        "positions_found": len(planet_positions),
        "n_layers": n_layers,
        "n_sdpa_layers": layer_info["n_sdpa_layers"],
        "n_linear_layers": layer_info["n_linear_layers"],
        "num_hidden_layers": layer_info["num_hidden_layers"],
        "allow_linear_layers": allow_linear_layers,
    }
    checks["positions_found_nonzero"] = len(planet_positions) > 0
    checks["planet_spans_match_pn_lines"] = checks["planet_spans"] == n_pn_lines

    # ONE model copy, loaded through the same path the reader uses (sdpa, bf16,
    # the mass patch); the only difference is the recorder.
    llm = MassWeightedLLM(
        model_id=model_id, max_new_tokens=2, do_sample=False, quantization="none"
    )
    llm._prefill_mass_scale = 0.0
    llm._bias_cap = None
    llm.load()
    if hf_commit_fn is not None:
        hf_commit_fn()  # F6: persist the download for the reader phase
    if llm.attn_implementation != "sdpa":
        raise RuntimeError(
            "instrument needs attn_implementation='sdpa', got %r"
            % (llm.attn_implementation,)
        )
    device = next(llm._model.parameters()).device

    shares: dict[float, float] = {}
    stats_by_w: dict[str, dict] = {}
    try:
        for w in INSTRUMENT_W_GRID:
            llm._mass_weight = float(w)
            vec, info = build_reader_mass_vector(
                ids, tok, "planet", None, w=w, prompt_text=prompt, device=device
            )
            if vec is None:
                raise RuntimeError("instrument built no mass vector (w=%g)" % w)
            llm.set_mass_vector(vec)
            llm.start_attention_recording()
            try:
                llm.generate(prompt)
            finally:
                recorded = llm.stop_attention_recording()
                stats = llm.mass_injection_stats()
                llm.clear_mass_vector()
            if not recorded:
                raise RuntimeError(
                    "no decode-step attention was recorded (w=%g): the sdpa "
                    "patch never saw a seq_q==1 call" % w
                )
            # One entry per layer per decode step, in call order -> entry
            # n_layers-1 is the LAST layer of the FIRST decode step.
            shares[float(w)] = mass_share(recorded[n_layers - 1], planet_positions)
            stats_by_w["w=%g" % w] = dict(stats, recorded_vectors=len(recorded))
            checks["positions_found"] = info.positions_found
    finally:
        llm.clear_mass_vector()

    checks.update(check_ratio_grows(shares))
    checks["mass_injection_stats"] = stats_by_w
    # Exactly one prefill and exactly one decode step, every SDPA layer each
    # time (A3: a linear-attention layer never calls the patched kernel, so it
    # is counted in neither number).
    last = stats_by_w["w=%g" % INSTRUMENT_W_GRID[-1]]
    checks["bias_applied_calls_match"] = last["bias_applied_calls"] == n_layers
    checks["bias_skipped_prefill_calls_match"] = (
        last["bias_skipped_prefill_calls"] == n_layers
    )

    payload = {
        "model_id": model_id,
        "layer_info": layer_info,
        "note": (
            "mass_share is measured on the PATCHED sdpa path at the first "
            "decode step, last layer. exp(w*mass) = exp(%.1f) is the naive "
            "expectation at w=1.0; softmax renormalisation makes the observed "
            "ratio smaller, so the assertion is ratio >= %.1f. The bias reaches "
            "%s of %s layers (A3)."
            % (
                INSTRUMENT_PLANET_MASS, INSTRUMENT_MIN_RATIO,
                layer_info["n_sdpa_layers"], layer_info["num_hidden_layers"],
            )
        ),
        "checks": checks,
    }
    failed = [
        key
        for key in (
            "positions_found_nonzero",
            "planet_spans_match_pn_lines",
            "ratio_monotone",
            "ratio_at_least_min",
            "bias_applied_calls_match",
            "bias_skipped_prefill_calls_match",
        )
        if not checks[key]
    ]
    payload["failed"] = failed
    vol_path.mkdir(parents=True, exist_ok=True)
    (vol_path / "instrument.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if failed:
        raise RuntimeError("instrumentation checks failed: %r" % failed)
    return payload


def _phase_reader_impl(
    model_id: str,
    *,
    run_id: str,
    cells: list[dict] | None = None,
    vol: str = VOL_MOUNT,
    chat: str = DEFAULT_CHAT,
    questions: str = DEFAULT_QUESTIONS,
    bias_cap: float | None = None,
    gpu_mem_gb: float | None = None,
    allow_linear_layers: bool = False,
    commit_fn: Any = None,
    hf_commit_fn: Any = None,
) -> dict:
    """Run every grid cell. GPU ONLY. UNVERIFIED."""
    from transformers import AutoConfig

    from benchmark.bineval import arms as arms_mod
    from benchmark.bineval import run_reader as rr

    from experiments.gpu_preflight import preflight_or_exit

    cells = cells or build_cells()
    vol_path = run_root(vol, run_id)
    (vol_path / "answers").mkdir(parents=True, exist_ok=True)
    write_run_info(
        vol_path,
        run_id=run_id,
        reader_model_id=model_id,
        allow_linear_layers=allow_linear_layers,
        n_sdpa_layers_expected=expected_n_sdpa_layers(model_id),
        arms=sorted({c["arm"] for c in cells}),
        w_grid=sorted({c["w"] for c in cells}),
    )
    preflight_or_exit(
        ("env", "tokenizer", "config", "context", "micro"), model_id,
        out_path=vol_path / "preflight_reader.json",
        chat=str(repo_path(chat)),
        gpu_mem_gb=gpu_mem_gb,
        allow_linear_layers=allow_linear_layers,
    )
    config = AutoConfig.from_pretrained(model_id)
    layer_info = rr.check_model_supported(
        config, allow_linear_layers=allow_linear_layers
    )
    chat_obj = arms_mod.load_chat(repo_path(chat))
    cd_json = vol_path / "cd_spec.json"

    manifest: list[dict] = []
    llm = None
    current_key: tuple | None = None
    for cell in cells:
        validate_cell(cell)  # cells may arrive hand-built over the wire
        answers_path = vol_path / "answers" / ("%s.json" % cell["cell"])
        if answers_path.exists():  # B5: resume a preempted grid
            manifest.append(dict(cell, status="already_present"))
            continue
        try:
            ctx = arms_mod.build_arm_context(
                cell["arm"],
                chat=chat_obj,
                cd_json=str(cd_json) if cd_json.exists() else None,
            )
        except FileNotFoundError as exc:
            if cell["arm"] not in OPTIONAL_ARTIFACT_ARMS:
                raise
            manifest.append(dict(cell, status="skipped", reason=str(exc)))
            continue

        if cell["w"] > 0 and cell["arm"].startswith("cd_") and "[PN" not in ctx.text:
            # M2: a CD that serialized without a single planet line would run
            # the injected cell as a plain text baseline under an injected name.
            raise RuntimeError(
                "cell %s has w=%g but its context carries no '[PN' line "
                "(%d tokens): refusing to run a silent baseline"
                % (cell["cell"], cell["w"], ctx.tokens)
            )

        qs = rr.load_questions(
            repo_path(questions), max_questions=cell.get("max_questions")
        )
        context_check = None
        if gpu_mem_gb is not None:
            context_check = rr.check_context_fits(
                config, ctx.tokens, 48, gpu_mem_gb,
                weight_bytes=rr.safetensors_total_bytes(model_id),
            )

        key = (cell["w"], cell["prefill_scale"])
        if llm is None or key != current_key:
            # the mass weight / prefill scale are read at sdpa time from the
            # instance, so one load per (w, prefill) pair is enough.
            if llm is None:
                llm = rr.load_reader(
                    model_id, max_new_tokens=48, w=cell["w"],
                    prefill_scale=cell["prefill_scale"], bias_cap=bias_cap,
                )
                if hf_commit_fn is not None:
                    hf_commit_fn()  # F6: one download, then the cache is shared
            else:
                llm._mass_weight = float(cell["w"])
                llm._prefill_mass_scale = float(cell["prefill_scale"])
            current_key = key
        run = rr.run_reader(
            llm, ctx.text, qs, w=cell["w"], inject=cell["inject"],
            bias_cap=bias_cap, arm=cell["arm"],
        )
        top_mass = max_planet_mass(ctx.text)
        run.run_meta = rr.run_meta(
            model_id=model_id, layer_info=layer_info, w=cell["w"], inject=cell["inject"],
            prefill_scale=cell["prefill_scale"], bias_cap=bias_cap,
            context_tokens=ctx.tokens, arm=cell["arm"], context_check=context_check,
            extra={
                "cell": cell["cell"],
                "max_planet_mass": top_mass,
                "w_times_max_mass": cell["w"] * top_mass,
                "arm_meta": ctx.meta,
            },
        )
        rr.write_answers(answers_path, run)
        manifest.append(
            dict(
                cell,
                status="ok",
                tokens=ctx.tokens,
                n_questions=len(qs),
                max_planet_mass=top_mass,
                w_times_max_mass=cell["w"] * top_mass,
            )
        )
        if commit_fn is not None:
            commit_fn()  # B5: every finished cell survives a preemption

    (vol_path / "run_manifest.json").write_text(
        json.dumps(
            {"run_id": run_id, "model_id": model_id, "cells": manifest},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    if commit_fn is not None:
        commit_fn()
    return {"cells": len(manifest), "run_id": run_id}


# --------------------------------------------------------------------------
# Modal app -- MODULE LEVEL (M5), so a remote import sees the functions
# --------------------------------------------------------------------------

def _build_images() -> tuple[Any, Any]:
    """M4: two pinned images.

    ``image_vllm`` (phase_manager) carries vLLM, which pins its own transformers
    and torch; ``image_hf`` (phase_instrument / phase_reader) carries the exact
    transformers the sdpa mask path was verified against.  Putting both in one
    image means pip resolves one of the two pins away, silently.

    ``image_hf`` also carries FLA_PIN: the reader is now the hybrid
    Qwen/Qwen3.8-27B, whose 48 linear layers use fla's Triton kernels when they
    are importable and a pure-torch fallback when they are not (see the module
    docstring, LINEAR-ATTENTION KERNELS).  ``causal-conv1d`` is deliberately NOT
    installed -- it compiles CUDA sources and would force a
    ``modal.Image.from_registry("nvidia/cuda:12.x-devel-ubuntu22.04",
    add_python="3.12")`` base, which transformers 5.8 does not require.
    """
    common: dict[str, Any] = {"remote_path": REMOTE_REPO, "ignore": IMAGE_IGNORE}
    # F6: HF_HOME points at the shared cache volume, so the 56 GB download
    # happens once for the whole app instead of once per phase.
    env = {"HF_HOME": HF_CACHE_MOUNT}
    image_vllm_ = (
        modal.Image.debian_slim(python_version="3.12")
        # F2: sentence-transformers, WITHOUT a second torch pin -- vLLM's own
        # torch satisfies it, and pinning torch twice lets pip resolve away the
        # build vLLM needs.
        .pip_install(
            VLLM_PIN, SENTENCE_TRANSFORMERS_PIN, "tiktoken", "jsonschema",
            "pyyaml", "numpy",
        )
        .env(env)
        .add_local_dir(str(REPO_ROOT), **common)
    )
    image_hf_ = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            TORCH_PIN, TRANSFORMERS_PIN, FLA_PIN, "accelerate", "tiktoken",
            "jsonschema", "pyyaml", "numpy",
        )
        .env(env)
        .add_local_dir(str(REPO_ROOT), **common)
    )
    return image_vllm_, image_hf_


def _enter_repo(remote_repo: str = REMOTE_REPO) -> None:
    """chdir into the mounted repo AND rebind ``REPO_ROOT`` to it (A1).

    Modal runs this file as ``__main__`` from ``/root/modal_spec_run.py``, where
    ``Path(__file__).resolve().parents[1]`` is ``/``. Without the rebinding
    every ``repo_path("benchmark/...")`` resolved to ``/benchmark/...`` and the
    reader phase died on a FileNotFoundError after the model was already loaded.
    """
    global REPO_ROOT

    os.chdir(remote_repo)
    if remote_repo not in sys.path:
        sys.path.insert(0, remote_repo)
    REPO_ROOT = Path(remote_repo)


app: Any = None
volume: Any = None
image_vllm: Any = None
image_hf: Any = None
phase_manager: Any = None
phase_instrument: Any = None
phase_reader: Any = None

hf_cache: Any = None

if modal_available():  # pragma: no cover - requires modal installed
    image_vllm, image_hf = _build_images()
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    # F6: shared across all three functions, mounted at HF_HOME.
    hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)
    app = modal.App(APP_NAME)
    _VOLUMES = {VOL_MOUNT: volume, HF_CACHE_MOUNT: hf_cache}

    @app.function(
        image=image_vllm, gpu=GPU_SPEC, volumes=_VOLUMES,
        timeout=FUNCTION_TIMEOUT_S,
    )
    def phase_manager(  # noqa: F811
        model_id: str = DEFAULT_MODEL_ID,
        run_id: str = "",
        shortlist_k: int = DEFAULT_SHORTLIST_K,
    ) -> dict:
        _enter_repo()
        return _phase_manager_impl(
            model_id, run_id=run_id or default_run_id(), shortlist_k=shortlist_k,
            commit_fn=volume.commit, hf_commit_fn=hf_cache.commit,
        )

    @app.function(
        image=image_hf, gpu=GPU_SPEC, volumes=_VOLUMES,
        timeout=FUNCTION_TIMEOUT_S,
    )
    def phase_instrument(  # noqa: F811
        model_id: str = DEFAULT_READER_MODEL_ID,
        run_id: str = "",
        allow_linear_layers: bool = False,
    ) -> dict:
        _enter_repo()
        out = _phase_instrument_impl(
            model_id, run_id=run_id or default_run_id(),
            allow_linear_layers=allow_linear_layers, hf_commit_fn=hf_cache.commit,
        )
        volume.commit()
        return out

    @app.function(
        image=image_hf, gpu=GPU_SPEC, volumes=_VOLUMES,
        timeout=FUNCTION_TIMEOUT_S,
    )
    def phase_reader(  # noqa: F811
        model_id: str = DEFAULT_READER_MODEL_ID,
        cells: list[dict] | None = None,
        gpu_mem_gb: float | None = None,
        run_id: str = "",
        allow_linear_layers: bool = False,
    ) -> dict:
        _enter_repo()
        return _phase_reader_impl(
            model_id, run_id=run_id or default_run_id(), cells=cells,
            gpu_mem_gb=gpu_mem_gb, allow_linear_layers=allow_linear_layers,
            commit_fn=volume.commit, hf_commit_fn=hf_cache.commit,
        )


def require_modal() -> None:
    if not modal_available():
        raise SystemExit(
            "modal is not installed in this environment; --deploy / --run / "
            "--spawn / --wait / --download need it (pip install modal). "
            "--dry-run and --score do not."
        )


# --------------------------------------------------------------------------
# dry run (CPU, no Modal, no model)
# --------------------------------------------------------------------------

FAKE_ANSWER = "unknown"


def dry_run(
    out_dir: str | Path,
    *,
    chat_path: str | Path | None = None,
    questions_path: str | Path | None = None,
    max_questions: int | None = None,
    cells: list[dict] | None = None,
    with_tokens: bool = True,
    run_id: str | None = None,
    model_id: str | None = None,
    reader_model_id: str | None = None,
) -> dict:
    """Produce the real file layout with fakes. No Modal, no model, no network.

    Writes, under ``<out_dir>/<run_id>/`` -- the SAME layout the volume gets
    (F3): run_info.json, pilot_summary.json, cd_spec.json (placeholder),
    instrument.json (placeholder marked dry_run), answers/<cell>.json for every
    cell (all answers "unknown") + .meta.json sidecars, run_manifest.json.
    """
    from benchmark.bineval import arms as arms_mod
    from benchmark.bineval import run_reader as rr

    chat_path = chat_path if chat_path is not None else repo_path(DEFAULT_CHAT)
    questions_path = (
        questions_path if questions_path is not None else repo_path(DEFAULT_QUESTIONS)
    )
    rid = run_id or default_run_id()
    out = run_root(out_dir, rid)
    (out / "answers").mkdir(parents=True, exist_ok=True)

    cells = cells if cells is not None else build_cells(max_questions=max_questions)

    write_run_info(
        out,
        **build_run_info(
            rid,
            model_id=model_id,
            reader_model_id=reader_model_id,
            arms=sorted({c["arm"] for c in cells}),
            w_grid=sorted({c["w"] for c in cells}),
        ),
        dry_run=True,
    )

    # 1. pilot: the REAL build_cd_offline summary shape, flattened exactly as
    #    _phase_manager_impl flattens it (B1).
    raw_summary = {
        "sun": 1, "planet": 3, "satellite": 9, "total": 13, "turns": 14,
        "harness_calls": {"node": 22, "belongs": 12, "same": 6},
        "harness_cache_hits": 3,
        "harness_quality": {
            "unparsed": 1, "defaulted": 0, "node_fallback": 0,
            "vanished": 2, "attached": 4,
        },
    }
    pilot = dict(raw_summary)
    pilot.update(flatten_pilot_summary(raw_summary))
    abort, reason = should_abort_pilot(pilot)
    pilot["abort"] = abort
    pilot["abort_reason"] = reason
    pilot["dry_run"] = True
    (out / "pilot_summary.json").write_text(
        json.dumps(pilot, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 2. CD placeholder (empty node list: the cd_* arms have no real CD here)
    (out / "cd_spec.json").write_text(
        json.dumps({"nodes": [], "summary": {"dry_run": True}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 3. instrument placeholder
    (out / "instrument.json").write_text(
        json.dumps(
            {"dry_run": True, "checks": None,
             "note": "instrumentation requires a GPU; not run in --dry-run"},
            ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 4. answers, one file per cell, every answer "unknown"
    chat_obj = arms_mod.load_chat(chat_path) if with_tokens else None
    cd_json = out / "cd_spec.json"
    questions = rr.load_questions(questions_path, max_questions=max_questions)
    manifest: list[dict] = []
    token_cache: dict[str, int | None] = {}
    for cell in cells:
        qs = questions[: cell["max_questions"]] if cell.get("max_questions") else questions
        answers = {q["qid"]: FAKE_ANSWER for q in qs}
        run = rr.ReaderRun(
            answers=answers,
            per_question={
                q["qid"]: {"raw": FAKE_ANSWER, "positions_found": 0, "spans": 0,
                           "planet_spans": 0, "satellite_spans": 0,
                           "prompt_tokens": None, "bias_applied_calls": 0,
                           "bias_skipped_prefill_calls": 0,
                           "bias_skipped_sliding_calls": 0}
                for q in qs
            },
        )
        arm = cell["arm"]
        if arm not in token_cache:
            token_cache[arm] = _arm_tokens(arm, chat_obj, cd_json) if with_tokens else None
        tokens = token_cache[arm]
        status = "skipped" if (with_tokens and tokens is None) else "ok"
        run.run_meta = {
            "dry_run": True, "model_id": None, "transformers_version": None,
            "torch_version": None, "layer_types": None,
            "max_position_embeddings": None, "context_check": None,
            "cell": cell["cell"], "arm": arm,
            "w": cell["w"], "inject": cell["inject"],
            "prefill_scale": cell["prefill_scale"], "bias_cap": None,
            "context_tokens": tokens,
            # M6: the real knob is w * mass; the dry run has no CD, so 0.0.
            "max_planet_mass": 0.0,
            "w_times_max_mass": 0.0,
        }
        rr.write_answers(out / "answers" / ("%s.json" % cell["cell"]), run)
        entry = dict(
            cell, status=status, tokens=tokens, n_questions=len(qs),
            max_planet_mass=0.0, w_times_max_mass=0.0,
        )
        if status == "skipped":
            entry["reason"] = "optional artifact for arm %r is not present" % arm
        manifest.append(entry)

    (out / "run_manifest.json").write_text(
        json.dumps({"dry_run": True, "run_id": rid, "model_id": None,
                    "cells": manifest},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "cells": len(manifest),
        "out": str(out),
        "run_id": rid,
        "run_root": str(out),
        "pilot": pilot,
    }


# --------------------------------------------------------------------------
# F5: detached execution (spawn / wait)
# --------------------------------------------------------------------------

PHASES = ("manager", "instrument", "reader")


def _output_ctx(modal_mod: Any) -> Any:
    """A2: ``modal.enable_output()`` when the installed modal has it.

    Without it a 12-hour function prints NOTHING to the client -- no image
    build log, no container stdout -- so the only way to tell a stuck run from a
    slow one is the Modal dashboard.  Older clients (and the test fake) do not
    have it; ``nullcontext`` keeps those working.
    """
    import contextlib

    enable = getattr(modal_mod, "enable_output", None)
    return enable() if callable(enable) else contextlib.nullcontext()


def phase_call_kwargs(phase: str, args: argparse.Namespace) -> dict:
    """The keyword arguments one phase is called with. Pure, so it is tested."""
    if phase == "manager":
        return {
            "model_id": resolve_judge_model_id(args),
            "run_id": args.run_id,
            "shortlist_k": args.shortlist_k,
        }
    if phase == "instrument":
        return {
            "model_id": resolve_reader_model_id(args),
            "run_id": args.run_id,
            "allow_linear_layers": resolve_allow_linear_layers(args),
        }
    if phase == "reader":
        return {
            "model_id": resolve_reader_model_id(args),
            "cells": cells_from_args(args),
            # B4: without --gpu the pre-flight still runs, against the memory of
            # the card the function actually asks for.
            "gpu_mem_gb": GPU_MEM_GB.get(args.gpu or GPU_SPEC),
            "run_id": args.run_id,
            "allow_linear_layers": resolve_allow_linear_layers(args),
        }
    raise SystemExit("unknown phase %r (known: %s)" % (phase, ", ".join(PHASES)))


def call_record_path(phase: str, call_dir: str | Path) -> Path:
    return Path(call_dir) / ("spawned_%s.json" % phase)


def spawn_phase(
    phase: str, args: argparse.Namespace, *, modal_mod: Any = None
) -> dict:
    """``.spawn()`` one phase of the DEPLOYED app and record the call id.

    ``app.run()`` (``--run``) keeps the function tied to this client: the
    ephemeral app is torn down when the connection drops, which for a 12-hour
    phase means a closed laptop kills the run.  The documented pattern for a
    long job is ``modal run --detach`` or -- as here -- deploy once and
    ``.spawn()`` against the deployed function, which returns immediately with a
    call id that ``--wait`` can attach to later, from anywhere.
    """
    m = modal_mod if modal_mod is not None else modal
    fn = m.Function.from_name(APP_NAME, "phase_" + phase)
    kwargs = phase_call_kwargs(phase, args)
    with _output_ctx(m):
        call = fn.spawn(**kwargs)
    record = {
        "phase": phase,
        "call_id": getattr(call, "object_id", None) or str(call),
        "run_id": args.run_id,
        "app": APP_NAME,
        "kwargs": {k: v for k, v in kwargs.items() if k != "cells"},
        "n_cells": len(kwargs["cells"]) if "cells" in kwargs else None,
    }
    path = call_record_path(phase, args.call_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return record


def wait_phase(
    phase: str, args: argparse.Namespace, *, modal_mod: Any = None
) -> Any:
    """Attach to a spawned call and block on its result."""
    m = modal_mod if modal_mod is not None else modal
    path = call_record_path(phase, args.call_dir)
    if not path.exists():
        raise SystemExit(
            "no spawn record at %s: run --spawn %s first (or pass --call-dir)"
            % (path, phase)
        )
    record = json.loads(path.read_text(encoding="utf-8"))
    call = m.FunctionCall.from_id(record["call_id"])
    with _output_ctx(m):
        return call.get()


# --------------------------------------------------------------------------
# model ids (A3)
# --------------------------------------------------------------------------

def resolve_judge_model_id(args: argparse.Namespace) -> str:
    """The manager/judge model: --judge-model-id, else --model-id."""
    return getattr(args, "judge_model_id", None) or args.model_id


def resolve_reader_model_id(args: argparse.Namespace) -> str:
    """The reader/instrument model.

    Still does NOT fall back to ``--model-id``: the two roles stay configured
    independently, so overriding the judge alone can never silently change what
    the injection ran on.  Since the 2026-09-07 decision both defaults are the
    SAME hybrid ``Qwen/Qwen3.8-27B``, and that IS the measured configuration --
    16 of 64 layers biased, recorded as ``n_sdpa_layers_expected``.
    ``--reader-model-id`` (e.g. ``DENSE_MODEL_ID``) asks for the dense run.
    """
    return getattr(args, "reader_model_id", None) or DEFAULT_READER_MODEL_ID


def is_hybrid_model_id(model_id: str | None) -> bool:
    """Is ``model_id`` the hybrid reader (case- and revision-insensitive)?

    Duplicated from ``benchmark.bineval.run_reader.is_hybrid_model_id`` because
    this module has no repo import at MODULE level (see ``_enter_repo``); a test
    pins the two implementations and their layer constants together.
    """
    if not model_id:
        return False
    head = str(model_id).split("@", 1)[0].rstrip("/")
    return head.lower() == HYBRID_MODEL_ID.lower()


def resolve_allow_linear_layers(args: Any) -> bool:
    """``--allow-linear-layers`` / ``--no-allow-linear-layers`` / the default.

    ``None`` (neither flag given) means "decide from the RESOLVED READER model
    id": True for the hybrid, which is now the default reader, and False for
    anything else, so an unvetted hybrid still fails ``check_model_supported``
    loudly instead of quietly running at partial depth.
    """
    explicit = getattr(args, "allow_linear_layers", None)
    if explicit is not None:
        return bool(explicit)
    return is_hybrid_model_id(resolve_reader_model_id(args))


def expected_n_sdpa_layers(model_id: str | None) -> int | None:
    """How many layers the injection MUST reach, from the model id alone.

    16 for the hybrid; ``None`` (no expectation to record or assert) for a dense
    or unknown model.
    """
    return HYBRID_N_SDPA_LAYERS if is_hybrid_model_id(model_id) else None


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="spec-faithful run orchestration (Modal)")
    p.add_argument("--dry-run", action="store_true", help="CPU rehearsal; no Modal, no model")
    p.add_argument("--deploy", action="store_true", help="modal deploy the app")
    p.add_argument("--run", choices=PHASES, default=None,
                   help="run one phase on Modal (TIED TO THIS CLIENT: see --spawn)")
    p.add_argument("--spawn", choices=PHASES, default=None,
                   help="spawn one phase of the DEPLOYED app and return at once")
    p.add_argument("--wait", choices=PHASES, default=None,
                   help="block on the call id recorded by a previous --spawn")
    p.add_argument("--call-dir", default=".",
                   help="where spawned_<phase>.json is written/read (default: cwd)")
    p.add_argument("--download", action="store_true", help="pull the volume locally")
    p.add_argument("--score", default=None, metavar="DIR",
                   help="score a downloaded run directory into scores.csv (local, no GPU)")
    p.add_argument(
        "--run-id", default=None,
        help="output namespace on the volume (default: <UTC timestamp>-<git sha>); "
             "pass the SAME id to every phase and to --download",
    )
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                   help="manager/judge model (default: %s; %s is the dense "
                        "alternative)" % (DEFAULT_MODEL_ID, DENSE_MODEL_ID))
    p.add_argument("--judge-model-id", default=None,
                   help="override the manager/judge model (default: --model-id)")
    p.add_argument(
        "--reader-model-id", default=None,
        help="reader/instrument model (default: %s -- HYBRID, so the injection "
             "reaches 16 of 64 layers by design; %s is the dense alternative)"
             % (DEFAULT_READER_MODEL_ID, DENSE_MODEL_ID),
    )
    p.add_argument(
        "--allow-linear-layers", action=argparse.BooleanOptionalAction,
        default=None,
        help="accept a HYBRID reader (linear + full attention): the injection "
             "then reaches the full-attention layers ONLY. DEFAULT: on when the "
             "resolved reader model id is %s, off otherwise; "
             "--no-allow-linear-layers forces the dense-only rule"
             % HYBRID_MODEL_ID,
    )
    p.add_argument("--shortlist-k", type=int, default=DEFAULT_SHORTLIST_K,
                   help="manager embedding shortlist size (0 disables SBERT)")
    p.add_argument(
        "--out", default=None,
        help="--dry-run output directory (default: a fresh temp dir, printed)",
    )
    p.add_argument(
        "--dest",
        default=str(REPO_ROOT / "benchmark" / "bineval" / "results" / "spec_run"),
    )
    p.add_argument("--max-questions", type=int, default=None)
    p.add_argument(
        "--w-grid", default=None,
        help="comma-separated w values overriding %s" % (list(DEFAULT_W_GRID),),
    )
    p.add_argument("--gpu", default=None, choices=sorted(GPU_MEM_GB),
                   help="target GPU class; required by --include-long-arms")
    p.add_argument(
        "--include-long-arms", action="store_true",
        help="add %s to the grid (needs --gpu with >= %.0fGB)"
             % (", ".join(LONG_ARMS), LONG_ARM_MIN_GPU_GB),
    )
    p.add_argument("--no-tokens", action="store_true",
                   help="dry-run only: skip tiktoken context sizing (faster)")
    return p


def cells_from_args(args: argparse.Namespace) -> list[dict]:
    arms = resolve_arms(None, include_long=args.include_long_arms, gpu=args.gpu)
    w_grid = parse_w_grid(args.w_grid) if args.w_grid else DEFAULT_W_GRID
    return build_cells(arms, w_grid, max_questions=args.max_questions)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.score:
        from experiments.score_spec_run import score_run_dir

        res = score_run_dir(
            args.score, questions=str(repo_path(DEFAULT_QUESTIONS))
        )
        print("scored %d cells -> %s" % (len(res["rows"]), res["csv"]))
        return 0

    if args.dry_run:
        out_dir = args.out or tempfile.mkdtemp(prefix="spec_run_dry_")
        args.run_id = args.run_id or default_run_id()
        res = dry_run(
            out_dir, max_questions=args.max_questions,
            cells=cells_from_args(args), with_tokens=not args.no_tokens,
            run_id=args.run_id, model_id=resolve_judge_model_id(args),
            reader_model_id=resolve_reader_model_id(args),
        )
        print("run_id = %s" % res["run_id"])
        print("dry-run wrote %d cells to %s" % (res["cells"], res["out"]))
        return 0

    if args.deploy:
        require_modal()
        with _output_ctx(modal):
            app.deploy()
        print("deployed %s" % APP_NAME)
        return 0

    if args.spawn or args.wait or args.run:
        require_modal()
        args.run_id = args.run_id or default_run_id()
        print("run_id = %s  (pass --run-id %s to the other phases and --download)"
              % (args.run_id, args.run_id))

    if args.spawn:
        record = spawn_phase(args.spawn, args)
        print(json.dumps(record, ensure_ascii=False, indent=2))
        print("attach later with: --wait %s --call-dir %s"
              % (args.spawn, args.call_dir))
        return 0

    if args.wait:
        result = wait_phase(args.wait, args)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0

    if args.run:
        print(
            "[warn] --run uses an EPHEMERAL app tied to this client session: if "
            "the connection drops the %ds function is killed with it. Use "
            "--deploy once and then --spawn/--wait for anything long."
            % FUNCTION_TIMEOUT_S
        )
        fn = {"manager": phase_manager, "instrument": phase_instrument,
              "reader": phase_reader}[args.run]
        kwargs = phase_call_kwargs(args.run, args)
        with _output_ctx(modal), app.run():
            result = fn.remote(**kwargs)
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0

    if args.download:
        require_modal()
        import subprocess

        if not args.run_id:
            raise SystemExit(
                "--download needs --run-id: the volume holds every run, and "
                "pulling '/' would merge them. The id was printed when the "
                "phase was launched and is also in <run>/run_info.json"
            )
        dest = Path(args.dest) / args.run_id
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["modal", "volume", "get", VOLUME_NAME, "/" + args.run_id, str(dest)],
            check=True,
        )
        print("downloaded %s:/%s -> %s" % (VOLUME_NAME, args.run_id, dest))
        return 0

    build_parser().print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

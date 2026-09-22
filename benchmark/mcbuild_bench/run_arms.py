"""run_arms.py — one (arm, W, w) cell of mcbuild-bench through the existing reader (DESIGN §6).

Wraps ``benchmark.bineval.run_reader`` (load_reader / run_reader / run_meta /
write_answers) — it does not copy them.  Per cell:

0. validate EVERY CLI value before torch / transformers are imported (item 12):
   ``--arm`` choices, ``--W`` in {8000, 16000, 32000} or ``full`` (arm A only),
   ``--w >= 0`` and ``== 0`` for the baselines, ``--prefill-chunk > 0``,
   ``--max-new-tokens > 0``, every input path exists, ``--inject`` is ``none``
   for arms A/B/C (H11: no marker scan on baselines);
1. load questions (``score_binary.load_questions`` + ``select_questions(qs, "all",
   False)``) and the redacted session (``round_trips``); sha256 of both files;
2. artifact binding (item 2): arm proposed requires ``cd.json`` with non-empty
   ``nodes`` and ``manifest.session_sha256`` equal to the session's; arm C requires
   the compaction artifact's ``session_sha256`` / ``questions_sha256`` /
   ``model_id`` / ``W`` to equal this run's — mismatch is a SystemExit quoting
   both values;
3. build ONE window with ``windows.build_window`` budgeted with the question
   whose FULL reader prompt is longest (``budget_question``; arm C reuses the one
   recorded by compaction_c), then assert for EVERY question that
   ``count_tokens(build_prompt(prompt_context, q)) <= W`` (item 9);
4. arm proposed with ``w > 0``: the window must contain at least one ``[PN`` line
   (``planet_lines``; item 1);
5. pre-flight: ``check_model_supported(AutoConfig, allow_linear_layers=True)`` and
   ``check_context_fits(config, max prompt tokens, max_new_tokens, gpu_mem_gb)``
   with ``gpu_mem_gb`` from ``torch.cuda.get_device_properties(0).total_memory`` —
   there is NO CPU path, the CLI refuses to run without CUDA;
6. ``load_reader`` (arm A additionally sets
   ``llm._model.generation_config.prefill_chunk_size`` — honoured because
   ``MassWeightedGemma.generate`` calls ``model.generate(**inputs, max_new_tokens,
   do_sample)`` without ``generation_config=``, so transformers merges
   ``model.generation_config`` for every other key);
7. ``GpuSampler.start()`` and wait for >= 2 numeric samples (item 7);
   ``run_reader.run_reader(llm, prompt_context, ...)`` with a ``progress`` callback
   that timestamps each question and reads ``llm.last_generated_tokens`` /
   ``last_prefill_ms`` / ``last_decode_ms`` (item 6); then wait for a sample at or
   after the last question's end before ``stop()``;
8. post-run guards: ``planet_spans == planet_lines`` for the injected arm and
   ``> 0`` when ``w > 0`` (items 1, 13); ``prompt_tokens <= W`` for every
   question (item 9);
9. per question (E1): ``completion_tokens`` (= ``last_generated_tokens``),
   ``wall_ms_total`` / ``wall_ms_prefill`` / ``wall_ms_decode``,
   ``attn_flops_prefill``, ``attn_flops_decode``, ``energy_joules`` (null only
   when the question is shorter than the sampling gap), plus the window fields;
10. write ``answers.json`` (+ ``answers.meta.json`` from write_answers),
    ``meta.json`` (run_meta + extra + merged per_question, incl. all hashes,
    ``git_sha`` and ``energy_coverage``) and ``timing.jsonl``.

FLOPs model (E1, Qwen3.8-27B: 16 sdpa layers x 24 heads x head_dim 256).
Generating n tokens costs 1 prefill forward (chunked prefill: ceil(L/chunk) of
them) plus n-1 DECODE forwards -- the first token comes out of the prefill
forward (F1, 2026-09-18)::

    attn_flops_prefill = 16 * 2 * 24 * L * L * 256
    attn_flops_decode  = sum_{t=1..n-1} 16 * 2 * 24 * (L + t) * 256

Per-question guards added 2026-09-18:
  F1  injected questions (arm proposed, w > 0): ``bias_applied_calls ==
      n_sdpa * (n - 1)``, ``bias_skipped_prefill_calls == n_sdpa *
      n_prefill_forwards`` (prefill_scale 0), ``bias_skipped_sliding_calls == 0``,
      and ``llm.last_decode_forwards == n - 1``;
  F2  a partial / stopped CD is refused unless ``--allow-partial-cd``;
  F3  ``<out>/answers.jsonl`` receives every finished question immediately
      (header line + one ``answer`` line each, flushed + fsynced); ``--resume``
      skips the qids already there when the header matches this cell;
  F4  arm A: ``prompt_tokens % prefill_chunk == 1`` bumps the chunk by one
      (``prefill_chunk_effective``), so no 1-token final chunk is ever run;
  F5  ``loaded_class_name`` / ``loading_info`` from the reader go into meta.json;
  K   the pre-flight receives the checkpoint's weight bytes from the safetensors
      metadata (``run_reader.safetensors_total_bytes``); unobtainable -> SystemExit
      before any weight is fetched.

Astra round 2 (2026-09-18): the checkpoint header carries the FULL cell identity
(``CHECKPOINT_IDENTITY``: arm, W, w, model, session/questions/cd/compaction
hashes, inject, prefill_scale, max_new_tokens, prefill_chunk, prefill_last_row)
and ``--resume`` refuses on any mismatch; ``--prefill-last-row`` (H15 option (b),
off by default) is passed to ``load_reader`` and recorded in meta, and its
counter ``bias_applied_prefill_last_row_calls`` must equal ``n_sdpa`` per question
when on (0 when off).

Field names (H11 / item 13): ``planet_lines`` = count of ``[PN`` lines in the
window text (this module); ``planet_spans`` = markers located by run_reader's scan.
They must be equal.  ``positions_found`` (token positions) is run_reader's own
field and is recorded only.

Ryosuke's decisions of 2026-09-20 (DECISIONS H22):
  (c) the corpus is ``corpus.load_corpus(session, --exclude-rt)`` (default 36, the
      retrospective; checked filter); ``session_sha256`` everywhere is the sha of
      the FILTERED content, ``exclude_rt`` is part of the checkpoint identity and
      recorded in meta;
  (d) arm ``proposed``: after ``build_window`` every question whose fact has
      ``source_session >= first_recent_rt`` (``min(window["recent_idx"])``, the
      oldest round trip inside the window's recent part) is DROPPED — its answer
      would be in the context verbatim; ``kind == "absent"`` questions are kept.
      ``questions_used`` / ``questions_dropped_in_window`` / ``first_recent_rt`` go
      into meta.json and the answers.jsonl header, and the used qids are written to
      ``<out>/questions_subset.json`` (bound to session / questions sha).  A cell
      with no question left is refused.  The baseline is scored on the SAME list:
      ``--questions-subset <that file>`` (baseline arms only; sha in the checkpoint
      identity as ``questions_subset_sha256``; binding hashes and qids checked).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

from benchmark.bineval.build_cd_offline import git_sha
from benchmark.bineval.run_reader import (
    INJECT_MODES,
    ReaderRun,
    build_prompt,
    safetensors_total_bytes,
)
from benchmark.mcbuild_bench.corpus import DEFAULT_EXCLUDE_RT_CLI, load_corpus, parse_exclude_rt
from benchmark.mcbuild_bench.gpu_sampler import (
    GpuSampler,
    energy_joules,
    parse_samples,
    wait_for_samples,
)
from benchmark.mcbuild_bench.windows import (
    ARMS,
    assemble_context,
    build_window,
    count_planet_lines,
    count_tokens,
    summary_block,
)

# E1 attention-FLOPs constants for Qwen3.8-27B (DESIGN §2.1, §6).
N_SDPA_LAYERS = 16
N_HEADS = 24
HEAD_DIM = 256
W_FULL = "full"
W_CHOICES = (8000, 16000, 32000)  # C6
BASELINE_ARMS = ("A", "B", "C")

# Item 7: sampler coverage waits.
SAMPLER_WARMUP_SAMPLES = 2
SAMPLER_WARMUP_TIMEOUT_S = 15.0
SAMPLER_TAIL_TIMEOUT_S = 10.0
SAMPLER_POLL_S = 0.25


# --------------------------------------------------------------------------
# pure helpers (CPU-testable)
# --------------------------------------------------------------------------

def attn_flops_prefill(prompt_tokens: int) -> int:
    L = int(prompt_tokens)
    return N_SDPA_LAYERS * 2 * N_HEADS * L * L * HEAD_DIM


def attn_flops_decode(prompt_tokens: int, completion_tokens: int) -> int:
    """Decode forwards only: t = 1 .. n-1 (F1; the first token is the prefill's)."""
    L = int(prompt_tokens)
    return sum(
        N_SDPA_LAYERS * 2 * N_HEADS * (L + t) * HEAD_DIM
        for t in range(1, int(completion_tokens))
    )


def n_prefill_forwards(prompt_tokens: int, chunk: int | None) -> int:
    """1 without chunked prefill, else ceil(L / chunk)."""
    if chunk is None:
        return 1
    return max(1, math.ceil(int(prompt_tokens) / int(chunk)))


def effective_prefill_chunk(prompt_tokens: int, chunk: int | None) -> int | None:
    """F4: a final chunk of exactly ONE token is avoided by bumping the chunk by one."""
    if chunk is None:
        return None
    chunk = int(chunk)
    return chunk + 1 if int(prompt_tokens) % chunk == 1 else chunk


def expected_bias_counters(
    n_sdpa: int, completion_tokens: int, n_prefill: int, prefill_scale: float,
    prefill_last_row: bool = False,
) -> dict[str, int]:
    """F1: what ``MassWeightedGemma.mass_injection_stats()`` must show after
    generating ``completion_tokens`` tokens with a mass vector set.

    H15 (b): with ``prefill_last_row`` the LAST prefill forward additionally
    biases its final row once per sdpa layer (``bias_applied_prefill_last_row_calls
    == n_sdpa``); the prefill skips and decode applications are unchanged."""
    decode = max(int(completion_tokens) - 1, 0)
    if float(prefill_scale) > 0.0:
        applied = int(n_sdpa) * (int(n_prefill) + decode)
        skipped = 0
    else:
        applied = int(n_sdpa) * decode
        skipped = int(n_sdpa) * int(n_prefill)
    return {
        "bias_applied_calls": applied,
        "bias_skipped_prefill_calls": skipped,
        "bias_skipped_sliding_calls": 0,
        "bias_applied_prefill_last_row_calls": int(n_sdpa) if prefill_last_row else 0,
    }


def check_bias_counters(
    qid: str, pq: dict, *, n_sdpa: int, n_prefill: int, prefill_scale: float,
    decode_forwards: int | None, prefill_last_row: bool = False,
) -> None:
    """F1 guard for an injected question; RuntimeError names the first mismatch."""
    n = int(pq["completion_tokens"])
    if decode_forwards is None or int(decode_forwards) != max(n - 1, 0):
        raise RuntimeError(
            "%s: reader reports decode_forwards=%r for completion_tokens=%d; expected "
            "%d (1 prefill + n-1 decode forwards)" % (qid, decode_forwards, n, max(n - 1, 0))
        )
    want = expected_bias_counters(n_sdpa, n, n_prefill, prefill_scale, prefill_last_row)
    for key, expected in want.items():
        # the H15 counter is absent on a reader without the switch: treated as 0
        have = int(pq.get(key, 0))
        if have != expected:
            raise RuntimeError(
                "%s: %s=%d, expected %d (n_sdpa_layers=%d, prefill forwards=%d, decode "
                "forwards=%d, prefill_scale=%g): the injection did not run as accounted"
                % (qid, key, have, expected, n_sdpa, n_prefill, max(n - 1, 0), prefill_scale)
            )


# --------------------------------------------------------------------------
# F3: durable per-question checkpoint (<out>/answers.jsonl)
# --------------------------------------------------------------------------

CHECKPOINT_NAME = "answers.jsonl"
# Astra round 2, item 5: EVERY parameter that changes an answer is part of the
# cell identity, so --resume can never splice answers from two different cells.
CHECKPOINT_IDENTITY = (
    "arm", "W", "w", "model_id", "session_sha256", "questions_sha256", "cd_sha256",
    "compaction_sha256", "inject", "prefill_scale", "max_new_tokens", "prefill_chunk",
    "prefill_last_row", "exclude_rt", "questions_subset_sha256", "quantization",
    "chat_template",
)
QUESTIONS_SUBSET_NAME = "questions_subset.json"


def checkpoint_header(
    *, arm: str, W: int | None, w: float, model_id: str, session_sha: str,
    questions_sha: str, cd_sha: str | None, compaction_sha: str | None,
    inject: str, prefill_scale: float, max_new_tokens: int, prefill_chunk: int | None,
    prefill_last_row: bool, exclude_rt=(), questions_subset_sha: str | None = None,
    first_recent_rt: int | None = None, questions_dropped_in_window=(),
    quantization: str = "none", chat_template: bool = False,
) -> dict:
    return {
        "kind": "header", "arm": arm, "W": W if W is not None else W_FULL, "w": float(w),
        "model_id": model_id, "session_sha256": session_sha, "questions_sha256": questions_sha,
        "cd_sha256": cd_sha, "compaction_sha256": compaction_sha,
        "inject": inject, "prefill_scale": float(prefill_scale),
        "max_new_tokens": int(max_new_tokens),
        # arm A only (the chunk is not used by the other arms; mirrors meta's
        # prefill_chunk_size)
        "prefill_chunk": None if prefill_chunk is None else int(prefill_chunk),
        "prefill_last_row": bool(prefill_last_row),
        "quantization": quantization,
        # H30: the H6 alternative (chat template, thinking off). Identity, because it
        # changes the prompt string and therefore every token count.
        "chat_template": bool(chat_template),
        # H22 (c) / (d): a JSON-stable list (the header is compared after a round trip)
        "exclude_rt": [int(i) for i in exclude_rt],
        "questions_subset_sha256": questions_subset_sha,
        # recorded, not identity (derived from the window of this very cell)
        "first_recent_rt": None if first_recent_rt is None else int(first_recent_rt),
        "questions_dropped_in_window": list(questions_dropped_in_window),
        "git_sha": git_sha(),
    }


def read_checkpoint(path: str | Path) -> tuple[dict | None, dict[str, dict]]:
    """(header, {qid: answer record}) of an answers.jsonl; (None, {}) when absent."""
    path = Path(path)
    if not path.exists():
        return None, {}
    header: dict | None = None
    done: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("kind") == "header":
            header = rec
        elif rec.get("kind") == "answer":
            done[rec["qid"]] = rec
    return header, done


def check_checkpoint_header(header: dict | None, expected: dict) -> None:
    """--resume only continues a checkpoint of the SAME cell (every
    ``CHECKPOINT_IDENTITY`` field); the refusal names the mismatching field."""
    if header is None:
        raise SystemExit("resume refused: answers.jsonl has no header line")
    for key in CHECKPOINT_IDENTITY:
        if header.get(key) != expected.get(key):
            raise SystemExit(
                "resume refused: answers.jsonl %s=%r does not match this cell's %r"
                % (key, header.get(key), expected.get(key))
            )


def append_checkpoint(path: str | Path, record: dict) -> None:
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def file_sha256(path: str | Path) -> str:
    """sha256 of the file bytes (artifact binding, item 2)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_W(text: str) -> int | None:
    """``--W``: an integer budget or the literal ``full`` (None)."""
    if text == W_FULL:
        return None
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "--W must be an integer or 'full', got %r" % (text,)
        ) from None
    if value <= 0:
        raise argparse.ArgumentTypeError("--W must be positive, got %d" % value)
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="mcbuild-bench: run one arm cell")
    p.add_argument("--arm", choices=list(ARMS), required=True)
    p.add_argument("--W", type=parse_W, required=True,
                   help="window budget in reader tokens (%s), or 'full' (arm A)"
                        % "/".join(str(w) for w in W_CHOICES))
    p.add_argument("--w", type=float, default=0.0)
    p.add_argument("--model-id", required=True)
    p.add_argument("--questions", required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--cd", default=None, help="CD JSON (required for --arm proposed)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--gpu-csv", required=True)
    p.add_argument("--prefill-chunk", type=int, default=8192,
                   help="arm A only: generation_config.prefill_chunk_size")
    p.add_argument("--inject", default=None, choices=list(INJECT_MODES),
                   help="proposed: default planet; arms A/B/C: forced to none (H11)")
    p.add_argument("--prefill-scale", type=float, default=0.0)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--compaction-summary", default=None,
                   help="compaction_c output JSON (required for --arm C)")
    p.add_argument("--allow-partial-cd", action="store_true",
                   help="F2: run arm proposed on a stopped / partial CD (recorded in meta)")
    p.add_argument("--resume", action="store_true",
                   help="F3: skip the qids already in <out>/answers.jsonl (same cell only)")
    p.add_argument("--quantization", choices=("none", "nf4", "fp4"), default="none",
                   help="bitsandbytes 4-bit on an unquantized checkpoint (46 GB-card fallback, H24); none = bf16")
    p.add_argument("--no-chat-template", dest="chat_template", action="store_false",
                   help="H30 / H6 alternative: restore the raw completion prompt. Qwen3.8 then "
                        "answers inside a <think> block and smoke criterion F4 fails")
    p.set_defaults(chat_template=True)
    p.add_argument("--prefill-last-row", action="store_true",
                   help="H15 (b): also add w*mass to the FINAL prefill row (the row that "
                        "yields the first answer token); default off (DECISIONS H18)")
    p.add_argument("--exclude-rt", default=DEFAULT_EXCLUDE_RT_CLI,
                   help="H22 (c): comma-separated round-trip indices dropped from the corpus "
                        "(checked: they must exist); 'none' disables (default: %(default)s)")
    p.add_argument("--questions-subset", default=None,
                   help="H22 (d), baseline arms only: the questions_subset.json written by the "
                        "proposed cell this baseline is compared with; only its qids are asked")
    return p


def resolve_inject(arm: str, inject: str | None) -> str:
    """H11: baselines never scan for markers; ``proposed`` defaults to ``planet``."""
    if arm in BASELINE_ARMS:
        if inject not in (None, "none"):
            raise SystemExit(
                "--inject %r is not allowed for baseline arm %s: baselines run "
                "with inject=none (H11, no marker scan)" % (inject, arm)
            )
        return "none"
    return inject if inject is not None else "planet"


def validate_args(args: argparse.Namespace) -> None:
    """Item 12: every cross-rule and path check, BEFORE torch/transformers load.
    Sets ``args.inject`` to its resolved value."""
    if args.arm == "A" and args.W is not None:
        raise SystemExit("--arm A is the full-context reference: use --W full")
    if args.arm != "A" and args.W is None:
        raise SystemExit("--W full is only valid with --arm A")
    if args.W is not None and args.W not in W_CHOICES:
        raise SystemExit("--W must be one of %r or 'full', got %d" % (W_CHOICES, args.W))
    if args.w < 0:
        raise SystemExit("--w must be >= 0, got %g" % args.w)
    if args.arm in BASELINE_ARMS and args.w != 0:
        raise SystemExit("--w must be 0 for baseline arm %s, got %g" % (args.arm, args.w))
    if args.prefill_chunk <= 0:
        raise SystemExit("--prefill-chunk must be > 0, got %d" % args.prefill_chunk)
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be > 0, got %d" % args.max_new_tokens)
    if args.arm == "proposed" and not args.cd:
        raise SystemExit("--cd is required with --arm proposed")
    if args.arm == "C" and not args.compaction_summary:
        raise SystemExit("--compaction-summary is required with --arm C")
    required = [("--questions", args.questions), ("--session", args.session)]
    if args.arm == "proposed":
        required.append(("--cd", args.cd))
    if args.arm == "C":
        required.append(("--compaction-summary", args.compaction_summary))
    if args.questions_subset is not None:
        if args.arm == "proposed":
            raise SystemExit(
                "--questions-subset is for the baseline arms: the proposed cell WRITES "
                "<out>/%s after filtering its own window (H22 (d))" % QUESTIONS_SUBSET_NAME
            )
        required.append(("--questions-subset", args.questions_subset))
    for label, path in required:
        if not Path(path).is_file():
            raise SystemExit("%s: file not found: %s" % (label, path))
    try:
        args.exclude_rt_parsed = parse_exclude_rt(args.exclude_rt)
    except ValueError as e:
        raise SystemExit("--exclude-rt: %s" % e) from None
    args.inject = resolve_inject(args.arm, args.inject)


# --------------------------------------------------------------------------
# H22 (d): questions outside the proposed method's window
# --------------------------------------------------------------------------

def split_questions_by_window(
    questions: list[dict], first_recent_rt: int | None,
) -> tuple[list[dict], list[str]]:
    """(used questions, dropped qids): a question whose fact's round trip
    (``source_session``) is ``>= first_recent_rt`` lies INSIDE the window's
    recent part and is dropped; ``kind == "absent"`` questions are kept;
    ``first_recent_rt is None`` (no recent round trip) drops nothing."""
    used: list[dict] = []
    dropped: list[str] = []
    for q in questions:
        if q.get("kind") == "absent" or first_recent_rt is None:
            used.append(q)
            continue
        rt = q.get("source_session")
        if not isinstance(rt, int) or isinstance(rt, bool):
            raise SystemExit(
                "question %s has no integer source_session (round trip of its fact); the "
                "window filter cannot place it" % q.get("qid")
            )
        if rt >= int(first_recent_rt):
            dropped.append(q["qid"])
        else:
            used.append(q)
    return used, dropped


def questions_subset_payload(
    *, arm: str, W: int | None, w: float, first_recent_rt: int | None, session_sha: str,
    questions_sha: str, exclude_rt, used: list[dict], dropped: list[str],
) -> dict:
    return {
        "kind": "questions_subset", "arm": arm, "W": W if W is not None else W_FULL,
        "w": float(w), "first_recent_rt": first_recent_rt,
        "session_sha256": session_sha, "questions_sha256": questions_sha,
        "exclude_rt": [int(i) for i in exclude_rt],
        "qids": [q["qid"] for q in used], "dropped": list(dropped),
    }


def load_questions_subset(
    path: str | Path, *, session_sha: str, questions_sha: str, questions: list[dict],
) -> tuple[dict, list[dict]]:
    """Read a proposed cell's subset file and select those qids (question-file
    order).  The subset must be bound to THIS corpus and question file, and every
    qid must exist; a mismatch is a SystemExit naming the field."""
    subset = json.loads(Path(path).read_text(encoding="utf-8"))
    for key, want in (("session_sha256", session_sha), ("questions_sha256", questions_sha)):
        if subset.get(key) != want:
            raise SystemExit(
                "%s: %s=%r does not match this run's %r (the subset was written for another "
                "corpus / question file)" % (path, key, subset.get(key), want)
            )
    qids = subset.get("qids")
    if not isinstance(qids, list) or not qids:
        raise SystemExit("%s: 'qids' missing or empty" % path)
    known = {q["qid"] for q in questions}
    unknown = [qid for qid in qids if qid not in known]
    if unknown:
        raise SystemExit("%s: qids not in --questions: %r" % (path, unknown))
    wanted = set(qids)
    return subset, [q for q in questions if q["qid"] in wanted]


def budget_question(questions: list[dict], tokenizer: Any, context_block: str = "",
                    *, chat_template: bool = False) -> str:
    """The question whose FULL reader prompt around ``context_block`` tokenizes
    longest (item 9).  Token counts are not additive, so the question is measured
    inside the prompt, never standalone; ``check_all_questions_fit`` then verifies
    every question against the final window."""
    return max(
        (q["question"] for q in questions),
        key=lambda t: count_tokens(tokenizer, build_prompt(
            context_block, t, tokenizer=tokenizer if chat_template else None)),
    )


def check_all_questions_fit(
    prompt_context: str, questions: list[dict], tokenizer: Any, W: int | None,
    *, chat_template: bool = False
) -> dict[str, int]:
    """Item 9: ``count_tokens(build_prompt(prompt_context, q)) <= W`` for EVERY
    question; SystemExit naming the offending qid.  Returns qid -> prompt tokens."""
    counts: dict[str, int] = {}
    for q in questions:
        n = count_tokens(tokenizer, build_prompt(
            prompt_context, q["question"], tokenizer=tokenizer if chat_template else None))
        counts[q["qid"]] = n
        if W is not None and n > W:
            raise SystemExit(
                "question %s: the full reader prompt is %d tokens > W=%d; the window "
                "was budgeted with a shorter question (non-additive tokenization)"
                % (q["qid"], n, W)
            )
    return counts


def check_cd_artifact(
    cd_json: dict, session_sha: str, cd_path: str, *,
    n_round_trips: int | None = None, allow_partial: bool = False,
) -> None:
    """Item 1 / 2: ``nodes`` present and non-empty (``load_cd`` would default a
    missing key to []); ``manifest.session_sha256`` equals the session's.
    F2: with ``n_round_trips`` given, a CD carrying ``stopped`` or whose
    ``summary.turns`` differs from the session's round-trip count is refused
    unless ``allow_partial``."""
    nodes = cd_json.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise RuntimeError(
            "%s: 'nodes' is %s (count %d): refusing to run arm proposed on an "
            "empty CD" % (cd_path, "missing" if nodes is None else "present",
                          len(nodes) if isinstance(nodes, list) else 0)
        )
    manifest = cd_json.get("manifest") if isinstance(cd_json.get("manifest"), dict) else {}
    cd_sha = manifest.get("session_sha256")
    if cd_sha != session_sha:
        raise SystemExit(
            "%s was built from a different session: manifest.session_sha256=%r, "
            "current --session sha256=%r" % (cd_path, cd_sha, session_sha)
        )
    if n_round_trips is not None and not allow_partial:
        if "stopped" in cd_json:
            raise SystemExit(
                "%s is a PARTIAL CD (stopped=%r); pass --allow-partial-cd to run on it "
                "anyway" % (cd_path, cd_json.get("stopped"))
            )
        summary = cd_json.get("summary") if isinstance(cd_json.get("summary"), dict) else {}
        turns = summary.get("turns")
        if turns != n_round_trips:
            raise SystemExit(
                "%s covers summary.turns=%r round trips but the session has %d; pass "
                "--allow-partial-cd to run on it anyway" % (cd_path, turns, n_round_trips)
            )


def check_compaction_artifact(
    compaction: dict, *, session_sha: str, questions_sha: str, model_id: str, W: int,
    path: str,
) -> None:
    """Item 2: the compaction artifact must belong to this session / questions /
    model / W; a missing key is a mismatch."""
    expected = {
        "session_sha256": session_sha,
        "questions_sha256": questions_sha,
        "model_id": model_id,
        "W": W,
    }
    for key, want in expected.items():
        have = compaction.get(key)
        if have != want:
            raise SystemExit(
                "%s: compaction %s=%r does not match this run's %r"
                % (path, key, have, want)
            )
    if not isinstance(compaction.get("budget_question"), str):
        raise SystemExit("%s: compaction lacks 'budget_question'; rerun compaction_c" % path)


def cell_round_trips(arm: str, session_rts: list[dict], compaction: dict | None) -> list[dict]:
    """Round-trip list handed to build_window; baseline C prepends its summary."""
    if arm != "C":
        return list(session_rts)
    if compaction is None:
        raise ValueError("arm C needs the compaction summary")
    recent = set(compaction["recent_idx"])
    kept = [rt for rt in session_rts if rt["idx"] in recent]
    return [summary_block(compaction["summary"])] + kept


def check_compaction_window(window: dict, compaction: dict) -> None:
    """F5: build_window must keep EVERY round trip compaction_c declared recent;
    otherwise the two budgets disagree and the cell is wrong."""
    n_window = int(window["n_recent_rts"])
    n_compaction = len(compaction["recent_idx"])
    if n_window != n_compaction:
        raise RuntimeError(
            "arm C: build_window kept %d recent round trips but compaction_c "
            "declared %d recent (recent_idx=%r); rerun compaction_c with the same "
            "tokenizer and budget question" % (n_window, n_compaction, compaction["recent_idx"])
        )


def check_per_question(
    per_question: dict[str, dict], *, arm: str, w: float, inject: str, W: int | None,
    planet_lines: int,
) -> None:
    """Post-run guards (items 1, 9, 13)."""
    for qid, pq in per_question.items():
        if W is not None and int(pq["prompt_tokens"]) > W:
            raise RuntimeError(
                "%s: prompt_tokens=%d > W=%d (the reader tokenized more than the "
                "window budget)" % (qid, pq["prompt_tokens"], W)
            )
        if arm == "proposed" and inject != "none":
            if pq["planet_spans"] != planet_lines:
                raise RuntimeError(
                    "arm proposed, %s: planet_spans=%d but the window has planet_lines=%d "
                    "[PN lines: the marker scan missed planets, refusing to record "
                    "this cell" % (qid, pq["planet_spans"], planet_lines)
                )
            if w > 0 and pq["planet_spans"] <= 0:
                raise RuntimeError(
                    "arm proposed, %s: planet_spans=%d with w=%g: no planet to inject "
                    "into, this cell would be a silent baseline" % (qid, pq["planet_spans"], w)
                )


def energy_coverage(csv_path: str | Path, first_t_start: float, last_t_end: float) -> dict:
    """Item 7: did the sampler bracket the question span?"""
    samples = parse_samples(csv_path) if Path(csv_path).exists() else []
    if not samples:
        return {"first_sample_before_first_question": False,
                "last_sample_after_last_question": False}
    ts = [t for t, _p in samples]
    return {
        "first_sample_before_first_question": min(ts) <= first_t_start,
        "last_sample_after_last_question": max(ts) >= last_t_end,
    }


def _reader_timing(llm: Any) -> tuple[int, float | None, float | None, int | None]:
    """Item 6 / F1: the attributes ``MassWeightedGemma.generate`` records."""
    n = getattr(llm, "last_generated_tokens", None)
    if n is None:
        raise RuntimeError(
            "reader exposes no last_generated_tokens after generate(): the E1 "
            "completion_tokens / wall-ms split cannot be recorded"
        )
    decode_forwards = getattr(llm, "last_decode_forwards", None)
    return (
        int(n), getattr(llm, "last_prefill_ms", None), getattr(llm, "last_decode_ms", None),
        None if decode_forwards is None else int(decode_forwards),
    )


# --------------------------------------------------------------------------
# main (GPU host only)
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)  # item 12: before torch / transformers

    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "run_arms needs a CUDA GPU: no CPU path exists (D-11) and the "
            "pre-flight reads torch.cuda.get_device_properties(0)"
        )
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    from transformers import AutoConfig, AutoTokenizer

    from benchmark.bineval import run_reader
    from benchmark.bineval.arms import load_cd
    from benchmark.bineval.score_binary import load_questions, select_questions
    from communication.cd_serializer import CDSerializer

    try:
        corpus = load_corpus(args.session, args.exclude_rt_parsed)  # H22 (c): checked filter
    except ValueError as e:
        raise SystemExit(str(e)) from None
    session_sha = corpus.sha256  # sha of the FILTERED content
    questions_sha = file_sha256(args.questions)
    questions = select_questions(load_questions(Path(args.questions)), "all", False)
    session_rts = corpus.round_trips
    print("[corpus] %s: %d of %d round trips (exclude_rt=%s) sha256=%s" % (
        args.session, len(session_rts), corpus.n_session_round_trips,
        json.dumps(list(corpus.exclude_rt)), session_sha))

    # Window (before any weight is fetched): the pre-flight needs its token count.
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    cd = cd_block = None
    cd_sha = None
    if args.arm == "proposed":
        cd_json = json.loads(Path(args.cd).read_text(encoding="utf-8"))
        check_cd_artifact(  # items 1, 2, F2
            cd_json, session_sha, args.cd,
            n_round_trips=len(session_rts), allow_partial=bool(args.allow_partial_cd),
        )
        cd_sha = file_sha256(args.cd)
        cd = load_cd(args.cd)
        cd_block = CDSerializer(level_markers=True).to_context_block(cd)
    compaction = None
    compaction_sha = None
    if args.arm == "C":
        compaction = json.loads(Path(args.compaction_summary).read_text(encoding="utf-8"))
        check_compaction_artifact(
            compaction, session_sha=session_sha, questions_sha=questions_sha,
            model_id=args.model_id, W=args.W, path=args.compaction_summary,
        )
        compaction_sha = file_sha256(args.compaction_summary)
    round_trips = cell_round_trips(args.arm, session_rts, compaction)
    if compaction is not None:
        # The very question compaction_c budgeted with (F5 agreement).
        question = compaction["budget_question"]
    else:
        question = budget_question(questions, tokenizer, assemble_context(cd_block, []),
                                   chat_template=args.chat_template)
    window = build_window(args.arm, args.W, cd_block, round_trips, tokenizer, question, cd=cd,
                          chat_template=args.chat_template)
    if compaction is not None and compaction["summary"] not in window["prompt_context"]:
        raise RuntimeError(
            "arm C: the compaction summary did not survive window composition "
            "(W=%d); rerun compaction_c" % args.W
        )
    if compaction is not None:
        check_compaction_window(window, compaction)
    planet_lines = count_planet_lines(window["prompt_context"]) if cd_block else 0
    if args.arm == "proposed" and args.w > 0 and planet_lines <= 0:
        raise RuntimeError(
            "arm proposed with w=%g: the serialized window holds planet_lines=%d "
            "[PN lines (CD nodes=%d): nothing to inject into"
            % (args.w, planet_lines, len(cd_json["nodes"]))
        )
    # H22 (d): the proposed cell answers only questions whose facts lie OUTSIDE its
    # window; a baseline answers the subset the proposed cell wrote.
    first_recent_rt = min(window["recent_idx"]) if window["recent_idx"] else None
    subset: dict | None = None
    subset_sha: str | None = None
    if args.arm == "proposed":
        used, dropped = split_questions_by_window(questions, first_recent_rt)
        if not used:
            raise SystemExit(
                "arm proposed, W=%s: no question lies outside the window (first_recent_rt=%r, "
                "%d questions all inside): this cell would only re-read its own context"
                % (args.W, first_recent_rt, len(dropped))
            )
        print("[run_arms] window recent round trips %s; %d questions used, %d dropped inside "
              "the window: %s" % (window["recent_idx"], len(used), len(dropped), dropped))
    elif args.questions_subset is not None:
        subset, used = load_questions_subset(
            args.questions_subset, session_sha=session_sha, questions_sha=questions_sha,
            questions=questions,
        )
        subset_sha = file_sha256(args.questions_subset)
        dropped = []
    else:
        used, dropped = list(questions), []
    prompt_tokens_by_qid = check_all_questions_fit(
        window["prompt_context"], used, tokenizer, args.W,
        chat_template=args.chat_template,
    )  # item 9
    max_prompt_tokens = max(prompt_tokens_by_qid.values())

    config = AutoConfig.from_pretrained(args.model_id)
    layer_info = run_reader.check_model_supported(config, allow_linear_layers=True)
    n_sdpa = int(layer_info["n_sdpa_layers"])
    # K: the weight term of the pre-flight comes from the safetensors metadata.
    weight_bytes = safetensors_total_bytes(args.model_id)
    if weight_bytes is None:
        raise SystemExit(
            "weight bytes unknown for %r (no model.safetensors.index.json total_size and "
            "no readable model.safetensors header): refusing to pre-flight with 0 weight "
            "bytes" % args.model_id
        )
    context_check = run_reader.check_context_fits(
        config, max_prompt_tokens, args.max_new_tokens, gpu_mem_gb, weight_bytes=weight_bytes
    )

    # F3: checkpoint identity and resume set, BEFORE the weights are fetched.
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / CHECKPOINT_NAME
    header = checkpoint_header(
        arm=args.arm, W=args.W, w=args.w, model_id=args.model_id, session_sha=session_sha,
        questions_sha=questions_sha, cd_sha=cd_sha, compaction_sha=compaction_sha,
        inject=args.inject, prefill_scale=args.prefill_scale,
        max_new_tokens=args.max_new_tokens,
        prefill_chunk=int(args.prefill_chunk) if args.arm == "A" else None,
        prefill_last_row=bool(args.prefill_last_row), quantization=args.quantization,
        chat_template=bool(args.chat_template),
        exclude_rt=corpus.exclude_rt, questions_subset_sha=subset_sha,
        first_recent_rt=first_recent_rt, questions_dropped_in_window=dropped,
    )
    done: dict[str, dict] = {}
    if args.resume and ckpt_path.exists():
        prev_header, done = read_checkpoint(ckpt_path)
        check_checkpoint_header(prev_header, header)
    else:
        ckpt_path.write_text("", encoding="utf-8")
        append_checkpoint(ckpt_path, header)
    subset_written: Path | None = None
    if args.arm == "proposed":
        # H22 (d): the list the baseline must be scored on (bound to this corpus / questions).
        subset_written = out_dir / QUESTIONS_SUBSET_NAME
        subset_written.write_text(json.dumps(questions_subset_payload(
            arm=args.arm, W=args.W, w=args.w, first_recent_rt=first_recent_rt,
            session_sha=session_sha, questions_sha=questions_sha, exclude_rt=corpus.exclude_rt,
            used=used, dropped=dropped,
        ), ensure_ascii=False, indent=2), encoding="utf-8")
    questions = used
    resumed_qids = [q["qid"] for q in questions if q["qid"] in done]

    llm = run_reader.load_reader(
        args.model_id,
        max_new_tokens=args.max_new_tokens,
        prefill_scale=args.prefill_scale,
        bias_cap=None,
        w=args.w,
        prefill_last_row=bool(args.prefill_last_row),
        quantization=args.quantization,
    )

    sampler = GpuSampler(args.gpu_csv)
    run = ReaderRun()
    timing: list[dict] = []
    new_timing: list[dict] = []
    per_question_extra: dict[str, dict] = {}
    clock: dict[str, float] = {}

    def progress(qid: str) -> None:
        # Called by run_reader right after each question's generate().
        now = time.time()
        n_tokens, prefill_ms, decode_ms, decode_forwards = _reader_timing(llm)
        new_timing.append({
            "qid": qid, "t_start": clock["prev"], "t_end": now,
            "wall_ms_total": (now - clock["prev"]) * 1000.0,
            "wall_ms_prefill": prefill_ms, "wall_ms_decode": decode_ms,
            "completion_tokens": n_tokens, "decode_forwards": decode_forwards,
        })
        clock["prev"] = now
        print("[run_arms] %s done (%.0f ms, %d tokens)" % (qid, new_timing[-1]["wall_ms_total"], n_tokens))

    sampler.start()
    tail_ok = not bool([q for q in questions if q["qid"] not in done])  # nothing new -> vacuous
    try:
        if not wait_for_samples(args.gpu_csv, SAMPLER_WARMUP_SAMPLES,
                                SAMPLER_WARMUP_TIMEOUT_S, SAMPLER_POLL_S):
            raise RuntimeError("sampler produced no samples")
        for q in questions:
            qid = q["qid"]
            if qid in done:
                rec = done[qid]
                run.answers[qid] = rec["answer"]
                run.per_question[qid] = rec["per_question"]
                timing.append(rec["timing"])
                per_question_extra[qid] = rec["extra"]
                continue
            # F4: chunk size per question (arm A only), from the window's own count.
            chunk_eff = (
                effective_prefill_chunk(prompt_tokens_by_qid[qid], args.prefill_chunk)
                if args.arm == "A" else None
            )
            if args.arm == "A":
                llm._model.generation_config.prefill_chunk_size = chunk_eff
            clock["prev"] = time.time()
            part = run_reader.run_reader(
                llm, window["prompt_context"], [q],
                w=args.w, inject=args.inject, bias_cap=None, arm=args.arm, progress=progress,
                chat_template=args.chat_template,
            )
            pq = part.per_question[qid]
            t = new_timing[-1] if new_timing else None
            if t is None or t["qid"] != qid:
                raise RuntimeError("no timing record for %s: the progress hook was not called" % qid)
            # Energy right away (F3): wait for one sample past the question's end.
            tail_ok = wait_for_samples(
                args.gpu_csv, 1, SAMPLER_TAIL_TIMEOUT_S, SAMPLER_POLL_S, after_ts=t["t_end"],
            )
            try:
                energy = energy_joules(args.gpu_csv, t["t_start"], t["t_end"])
            except ValueError:
                energy = None  # the question was shorter than the sampling gap
            L = int(pq["prompt_tokens"])
            n = int(t["completion_tokens"])
            n_prefill = n_prefill_forwards(L, chunk_eff)
            extra = {
                "completion_tokens": n,
                "decode_forwards": t["decode_forwards"],
                "n_prefill_forwards": n_prefill,
                "prefill_chunk_effective": chunk_eff,
                "wall_ms_total": t["wall_ms_total"],
                "wall_ms_prefill": t["wall_ms_prefill"],
                "wall_ms_decode": t["wall_ms_decode"],
                "attn_flops_prefill": attn_flops_prefill(L),
                "attn_flops_decode": attn_flops_decode(L, n),
                "energy_joules": energy,
                "energy_tail_sampled": bool(tail_ok),
                "window_tokens": window["window_tokens"],
                "n_recent_rts": window["n_recent_rts"],
                "cd_tokens": window["cd_tokens"],
                "planet_lines": planet_lines,
            }
            if args.arm == "proposed" and args.w > 0 and int(pq["planet_spans"]) > 0:
                check_bias_counters(  # F1
                    qid, {**pq, **extra}, n_sdpa=n_sdpa, n_prefill=n_prefill,
                    prefill_scale=args.prefill_scale, decode_forwards=t["decode_forwards"],
                    prefill_last_row=bool(args.prefill_last_row),
                )
            run.answers[qid] = part.answers[qid]
            run.per_question[qid] = pq
            timing.append(t)
            per_question_extra[qid] = extra
            append_checkpoint(ckpt_path, {
                "kind": "answer", "qid": qid, "answer": part.answers[qid],
                "per_question": pq, "timing": t, "extra": extra,
            })
    finally:
        sampler.stop()

    check_per_question(
        run.per_question, arm=args.arm, w=args.w, inject=args.inject, W=args.W,
        planet_lines=planet_lines,
    )

    coverage = energy_coverage(args.gpu_csv, new_timing[0]["t_start"], new_timing[-1]["t_end"]) if new_timing else {
        "first_sample_before_first_question": False, "last_sample_after_last_question": False}
    coverage["last_sample_after_last_question"] = bool(
        coverage["last_sample_after_last_question"] and tail_ok
    )
    extra = {
        "loaded_class_name": getattr(llm, "loaded_class_name", None),  # F5
        "loading_info": getattr(llm, "loading_info", None),
        "n_sdpa_layers": n_sdpa,
        "prefill_scale": args.prefill_scale,
        "chat_template": bool(args.chat_template),  # H30 / H6 alternative
        "prefill_last_row": bool(args.prefill_last_row),  # H15 (b) switch
        "quantization": args.quantization,
        "weight_bytes": weight_bytes,
        "allow_partial_cd": bool(args.allow_partial_cd),
        "resume": bool(args.resume),
        "resumed_qids": resumed_qids,
        "checkpoint": str(ckpt_path),
        "arm": args.arm,
        "model_id": args.model_id,
        "W": args.W if args.W is not None else W_FULL,
        "w": args.w,
        "git_sha": git_sha(),
        "session_sha256": session_sha,  # corpus hash (filtered content, H22 (c))
        "session_file_sha256": corpus.session_file_sha256,
        "exclude_rt": list(corpus.exclude_rt),
        "n_round_trips": len(session_rts),
        "questions_sha256": questions_sha,
        # H22 (d)
        "first_recent_rt": first_recent_rt,
        "recent_idx": window["recent_idx"],
        "questions_used": [q["qid"] for q in questions],
        "questions_dropped_in_window": dropped,
        "questions_subset": args.questions_subset,
        "questions_subset_sha256": subset_sha,
        "questions_subset_source": (
            {k: subset[k] for k in ("arm", "W", "w", "first_recent_rt") if k in subset}
            if subset is not None else None
        ),
        "questions_subset_written": None if subset_written is None else str(subset_written),
        "cd_sha256": cd_sha,
        "compaction_sha256": compaction_sha,
        "budget_question": question,
        "window_tokens": window["window_tokens"],
        "max_prompt_tokens": max_prompt_tokens,
        "n_recent_rts": window["n_recent_rts"],
        "cd_tokens": window["cd_tokens"],
        "evicted_planets": window["evicted_planets"],
        "planet_lines": planet_lines,
        "prefill_chunk_size": int(args.prefill_chunk) if args.arm == "A" else None,
        "max_new_tokens": args.max_new_tokens,
        "gpu_mem_gb": gpu_mem_gb,
        "gpu_csv": str(args.gpu_csv),
        "n_questions": len(questions),
        "compaction_summary": args.compaction_summary,
        "compaction_n_calls": compaction["n_calls"] if compaction else None,
        "wall_ms_split_available": True,
        "energy_coverage": coverage,
        "per_question_extra": per_question_extra,
    }
    run.run_meta = run_reader.run_meta(
        model_id=args.model_id,
        layer_info=layer_info,
        w=args.w,
        inject=args.inject,
        prefill_scale=args.prefill_scale,
        bias_cap=None,
        context_tokens=window["window_tokens"],
        arm=args.arm,
        context_check=context_check,
        extra=extra,
    )
    run_reader.write_answers(out_dir / "answers.json", run)

    meta = dict(run.run_meta)
    meta["per_question"] = {
        qid: {**pq, **per_question_extra.get(qid, {})} for qid, pq in run.per_question.items()
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out_dir / "timing.jsonl").open("w", encoding="utf-8") as f:
        for t in timing:
            f.write(json.dumps({**t, **per_question_extra.get(t["qid"], {})}) + "\n")
    print("wrote %s (%d answers, %d resumed, window_tokens=%d, n_recent_rts=%d)"
          % (out_dir, len(run.answers), len(resumed_qids), window["window_tokens"],
             window["n_recent_rts"]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

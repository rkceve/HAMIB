"""compaction_c.py — baseline C, product-style compaction (DECISIONS C4, DESIGN §6/§8).

Stream the round trips chronologically.  Keep a ``summary`` (initially empty)
and a list of ``recent`` round trips.  Whenever the FULL reader prompt
``_full_prompt(None, [render_round_trip(summary_block(summary))] + recent +
[next], question)`` — the exact string ``windows.build_window`` counts for arm C
— would exceed ``W`` tokens, re-summarize ``summary + recent`` with the reader
model (fixed §8 instruction, cap ``W // 4`` tokens), replace ``summary``, clear
``recent``, then admit ``next``.  Every summarization call is counted as
compute.  There is NO additive approximation (item 5): token counts are not
additive across block boundaries, so the fit test tokenizes the final string.

Decision surfaced for review (not in DECISIONS.md): DESIGN §0.1 measures the
largest round trip at 23,670 tokens, so at W=8k a single round trip can exceed
``W - reserve`` on its own.  ``compact`` handles that the way a product
compactor does: the oversized round trip is admitted and, since the window still
overflows, immediately folded into the summary (one more summarization call,
``recent`` becomes empty).  It is never dropped silently.  If even
``summary`` alone does not fit, the summarizer violated its cap and the run
stops with RuntimeError.

``compact`` is pure (tests use a fake ``summarize_fn``); the CLI wires the real
reader through ``run_reader.load_reader`` + ``llm.generate`` (GPU only).

Item J (2026-09-18): the GPU sampler (``--gpu-csv``, required) runs during the
whole compaction; every summarization call records ``prompt_tokens`` (reader
tokenizer), ``completion_tokens`` (``llm.last_generated_tokens``, the tokens
actually generated), ``wall_ms``, ``t_start`` / ``t_end`` and ``energy_joules``
(trapezoid over the CSV; null only when the call was shorter than the sampling
gap).  The artifact carries ``gpu_csv`` and ``energy_joules_total``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

from benchmark.mcbuild_bench.corpus import DEFAULT_EXCLUDE_RT_CLI, load_corpus, parse_exclude_rt
from benchmark.mcbuild_bench.gpu_sampler import energy_joules, wait_for_samples

# ``_full_prompt`` is private to windows.py; it is imported on purpose (F5 / item
# 5) so the fit test tokenizes the very string build_window measures against.
from benchmark.mcbuild_bench.windows import (
    BLOCK_SEPARATOR,
    _full_prompt,
    assemble_context,
    count_tokens,
    render_round_trip,
    summary_block,
)

# DESIGN §8, baseline C summarizer instruction (reader model).
SUMMARIZE_INSTRUCTION = (
    "Summarize the following conversation log for later reference. Keep every "
    "concrete value, name, path, decision and instruction. Plain text, at most "
    "{cap} tokens.\n\n{older_log}"
)

SummarizeFn = Callable[..., dict]  # summarize_fn(text, cap_tokens=int) -> dict
RenderFn = Callable[[dict], str]

# Item J: sampler waits (same values as run_arms).
SAMPLER_WARMUP_SAMPLES = 2
SAMPLER_WARMUP_TIMEOUT_S = 15.0
SAMPLER_TAIL_TIMEOUT_S = 10.0
SAMPLER_POLL_S = 0.25
# Optional per-call fields a summarize_fn may return; copied into the call record.
CALL_OPTIONAL_KEYS = ("energy_joules", "t_start", "t_end")


def energy_total(calls: list[dict]) -> float:
    """Sum of the recorded ``energy_joules`` (null entries skipped)."""
    return float(sum(float(c["energy_joules"]) for c in calls if c.get("energy_joules") is not None))


def summary_cap_tokens(W: int) -> int:
    """C4: summary cap = W/4 tokens."""
    return W // 4


def _join(parts: list[str]) -> str:
    return BLOCK_SEPARATOR.join(p for p in parts if p)


def scaffold_reserve(tokenizer: Any, question: str) -> int:
    """Tokens of the FULL reader prompt around an EMPTY summary block and
    ``question``.  RECORDED in the compaction artifact as ``reserve`` for the
    report; it is NOT used for budgeting any more (item 5: ``compact`` counts
    the exact final string instead of ``W - reserve``).
    """
    return count_tokens(
        tokenizer, _full_prompt(None, [render_round_trip(summary_block(""))], question)
    )


def window_prompt(summary: str, recent_bodies: list[str], question: str) -> str:
    """The exact string ``windows.build_window`` tokenizes for arm C: the reader
    template around ``<context>`` + ``### Summary`` block + recent round trips."""
    return _full_prompt(
        None, [render_round_trip(summary_block(summary))] + recent_bodies, question
    )


def compact(
    round_trips: list[dict],
    W: int,
    tokenizer: Any,
    summarize_fn: SummarizeFn,
    render_rt: RenderFn = render_round_trip,
    *,
    question: str,
) -> dict:
    """Stream ``round_trips`` so that the FULL reader prompt for ``question``
    never exceeds ``W`` tokens; see module docstring.

    ``question`` must be the one run_arms budgets the window with
    (``run_arms.budget_question``); ``render_rt`` must stay
    ``windows.render_round_trip`` for the declared ``recent_idx`` to equal
    ``build_window(...)["n_recent_rts"]``.

    ``summarize_fn(text, cap_tokens=...)`` must return a dict with ``text``,
    ``prompt_tokens``, ``completion_tokens`` and ``wall_ms``.

    Returns ``{"summary": str, "recent_idx": [idx...], "calls": [...],
    "n_calls": int}``.
    """
    if W <= 0:
        raise ValueError("W must be positive, got %r" % (W,))
    if not isinstance(question, str):
        raise ValueError("question must be a string")
    cap = summary_cap_tokens(W)

    summary = ""
    recent: list[dict] = []
    calls: list[dict] = []

    def fits(recent_rts: list[dict]) -> bool:
        prompt = window_prompt(summary, [render_rt(rt) for rt in recent_rts], question)
        return count_tokens(tokenizer, prompt) <= W

    def summarize_older() -> None:
        nonlocal summary, recent
        older = _join([summary] + [render_rt(rt) for rt in recent])
        if not older:
            return
        res = summarize_fn(older, cap_tokens=cap)
        summary = str(res["text"])
        call = {
            "prompt_tokens": int(res["prompt_tokens"]),
            "completion_tokens": int(res["completion_tokens"]),
            "wall_ms": float(res["wall_ms"]),
            "older_round_trips": [rt["idx"] for rt in recent],
        }
        for key in CALL_OPTIONAL_KEYS:  # item J
            if key in res:
                call[key] = res[key]
        calls.append(call)
        recent = []

    for rt in round_trips:
        if fits(recent + [rt]):
            recent.append(rt)
            continue
        if recent:  # F6: never re-summarize the summary alone
            summarize_older()
        recent.append(rt)
        if not fits(recent):
            # The single round trip overflows on its own: fold it in as well.
            summarize_older()
            if not fits(recent):
                raise RuntimeError(
                    "the reader prompt with the summary alone (%d tokens) exceeds "
                    "W=%d: the summarizer violated its %d-token cap"
                    % (count_tokens(tokenizer, window_prompt(summary, [], question)), W, cap)
                )

    return {
        "summary": summary,
        "recent_idx": [rt["idx"] for rt in recent],
        "calls": calls,
        "n_calls": len(calls),
    }


# --------------------------------------------------------------------------
# CLI (GPU host only; untested locally by design)
# --------------------------------------------------------------------------

def make_reader_summarize_fn(
    llm: Any,
    *,
    gpu_csv: str | Path | None = None,
    tail_timeout_s: float = SAMPLER_TAIL_TIMEOUT_S,
    poll_s: float = SAMPLER_POLL_S,
) -> SummarizeFn:
    """§8 baseline-C instruction through ``llm.generate``.

    Item J: ``prompt_tokens`` via ``llm.tokenizer`` (the reader's own count),
    ``completion_tokens`` = ``llm.last_generated_tokens`` (what was generated,
    not a re-tokenization of the text), ``wall_ms``, and with ``gpu_csv`` the
    call's ``energy_joules`` after waiting for one sample past its end.
    """

    def summarize(text: str, *, cap_tokens: int) -> dict:
        prompt = SUMMARIZE_INSTRUCTION.format(cap=cap_tokens, older_log=text)
        t_start = time.time()
        t0 = time.perf_counter()
        out = llm.generate(prompt)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        t_end = time.time()
        n = getattr(llm, "last_generated_tokens", None)
        if n is None:
            raise RuntimeError(
                "reader exposes no last_generated_tokens after generate(): the "
                "summarization call's completion_tokens cannot be recorded (J)"
            )
        energy = None
        if gpu_csv is not None:
            wait_for_samples(gpu_csv, 1, tail_timeout_s, poll_s, after_ts=t_end)
            try:
                energy = energy_joules(gpu_csv, t_start, t_end)
            except ValueError:
                energy = None  # shorter than the sampling gap / not bracketed
        return {
            "text": out.strip(),
            "prompt_tokens": count_tokens(llm.tokenizer, prompt),
            "completion_tokens": int(n),
            "wall_ms": wall_ms,
            "t_start": t_start,
            "t_end": t_end,
            "energy_joules": energy,
        }

    return summarize


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="mcbuild-bench baseline C compaction")
    p.add_argument("--session", required=True)
    p.add_argument("--W", type=int, required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--quantization", choices=("none", "nf4", "fp4"), default="none")
    p.add_argument("--out", required=True, help="output JSON (compaction_calls.jsonl written next to it)")
    p.add_argument("--questions", required=True,
                   help="data/questions.json: the window is budgeted with the question "
                        "whose FULL reader prompt is longest (run_arms.budget_question)")
    p.add_argument("--gpu-csv", required=True,
                   help="item J: nvidia-smi CSV written by GpuSampler during the compaction")
    p.add_argument("--exclude-rt", default=DEFAULT_EXCLUDE_RT_CLI,
                   help="H22 (c): comma-separated round-trip indices dropped from the corpus "
                        "(checked: they must exist); 'none' disables (default: %(default)s)")
    return p


def validate_args(args: argparse.Namespace) -> None:
    """Item 12: CLI values checked BEFORE torch / transformers are imported.
    Sets ``args.exclude_rt_parsed``."""
    from benchmark.mcbuild_bench.run_arms import W_CHOICES  # torch-free at import

    if args.W not in W_CHOICES:
        raise SystemExit("--W must be one of %r, got %r" % (W_CHOICES, args.W))
    try:
        args.exclude_rt_parsed = parse_exclude_rt(args.exclude_rt)
    except ValueError as e:
        raise SystemExit("--exclude-rt: %s" % e) from None
    for label, path in (("--session", args.session), ("--questions", args.questions)):
        if not Path(path).is_file():
            raise SystemExit("%s: file not found: %s" % (label, path))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("compaction_c needs a CUDA GPU (no CPU path, D-11)")

    from benchmark.bineval import run_reader
    from benchmark.bineval.score_binary import load_questions, select_questions
    from benchmark.mcbuild_bench.gpu_sampler import GpuSampler
    from benchmark.mcbuild_bench.run_arms import budget_question, file_sha256

    try:
        corpus = load_corpus(args.session, args.exclude_rt_parsed)  # H22 (c)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    round_trips = corpus.round_trips
    questions = select_questions(load_questions(Path(args.questions)), "all", False)
    cap = summary_cap_tokens(args.W)
    llm = run_reader.load_reader(args.model_id, max_new_tokens=cap, prefill_scale=0.0,
                                 bias_cap=None, w=0.0, quantization=args.quantization)
    # Item 9: the question whose FULL reader prompt (around an empty summary
    # block) is longest; run_arms reuses it from the artifact (budget_question).
    question = budget_question(
        questions, llm.tokenizer, assemble_context(None, [render_round_trip(summary_block(""))])
    )
    reserve = scaffold_reserve(llm.tokenizer, question)
    sampler = GpuSampler(args.gpu_csv)  # item J: the manager-side energy of baseline C
    sampler.start()
    t0 = time.perf_counter()
    try:
        if not wait_for_samples(args.gpu_csv, SAMPLER_WARMUP_SAMPLES,
                                SAMPLER_WARMUP_TIMEOUT_S, SAMPLER_POLL_S):
            raise RuntimeError("sampler produced no samples")
        result = compact(
            round_trips, args.W, llm.tokenizer,
            make_reader_summarize_fn(llm, gpu_csv=args.gpu_csv),
            render_round_trip, question=question,
        )
    finally:
        sampler.stop()
    # Item 2: artifact binding read back by run_arms.
    result["W"] = args.W
    result["reserve"] = reserve
    result["budget_question"] = question
    result["cap_tokens"] = cap
    result["model_id"] = args.model_id
    result["session_sha256"] = corpus.sha256  # corpus hash (filtered content, H22 (c))
    result["session_file_sha256"] = corpus.session_file_sha256
    result["exclude_rt"] = list(corpus.exclude_rt)
    result["questions_sha256"] = file_sha256(args.questions)
    result["wall_s"] = time.perf_counter() - t0
    result["gpu_csv"] = str(args.gpu_csv)
    result["energy_joules_total"] = energy_total(result["calls"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out.parent / "compaction_calls.jsonl").open("w", encoding="utf-8") as f:
        for call in result["calls"]:
            f.write(json.dumps(call, ensure_ascii=False) + "\n")
    print("wrote %s (%d summarization calls, %d recent round trips)"
          % (out, result["n_calls"], len(result["recent_idx"])))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

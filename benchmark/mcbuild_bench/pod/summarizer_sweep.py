"""Pod-only diagnostic: how often does the local summarizer break the 120-char node rule (D2)?

V4 stopped at chunk 52 with SummarizerStop on a PATH listing (both attempts > 120 chars). Before
changing anything, measure on the real corpus chunks (same chunker as build_cd) with the current
prompt (variant "v0", DESIGN §8 verbatim) and candidate variants. No Jev call is made; only the
local summarizer server is used. Output: JSON with per-variant counts and failure examples.

Run (reader venv, summarizer server up on 8123):
    python -m benchmark.mcbuild_bench.pod.summarizer_sweep --sample 300 --out /workspace/mcb/runs/sweep.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
from pathlib import Path

from benchmark.mcbuild_bench.build_cd import make_manager, round_trip_texts
from benchmark.mcbuild_bench.corpus import load_corpus
from benchmark.mcbuild_bench.summarizer_client import (
    INSTRUCTION, MAX_TOKENS, RETRY_SUFFIX, node_text_problem,
)

SYSTEM_V1 = (
    "You compress text. Reply with exactly one line of at most 120 characters, in the excerpt's "
    "language, keeping the concrete values that matter. Never copy lists, paths or code verbatim: "
    "describe what they are."
)


def call(url: str, messages: list[dict], max_tokens: int = MAX_TOKENS) -> tuple[str, str, float]:
    body = json.dumps({"model": "summarizer", "messages": messages, "temperature": 0,
                       "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        env = json.loads(r.read().decode())
    ch = env["choices"][0]
    return ch["message"]["content"].strip(), ch["finish_reason"], time.perf_counter() - t0


def variant_v0(url: str, excerpt: str) -> list[tuple[str, str]]:
    """Current client behaviour: instruction+excerpt, retry = same + RETRY_SUFFIX."""
    out = []
    for retried in (False, True):
        content = INSTRUCTION + "\n\n" + excerpt + (RETRY_SUFFIX if retried else "")
        text, finish, _ = call(url, [{"role": "user", "content": content}])
        out.append((text, finish))
        if node_text_problem(text, finish) is None:
            break
    return out


def variant_v1(url: str, excerpt: str) -> list[tuple[str, str]]:
    """System instruction; retry = compress the model's own previous answer."""
    out = []
    msgs = [{"role": "system", "content": SYSTEM_V1},
            {"role": "user", "content": INSTRUCTION + "\n\n" + excerpt}]
    text, finish, _ = call(url, msgs)
    out.append((text, finish))
    if node_text_problem(text, finish) is None:
        return out
    msgs += [{"role": "assistant", "content": text},
             {"role": "user", "content": f"That was {len(text)} characters. Rewrite it as one line of at most "
                                         "120 characters (about 20 words). Output the statement only."}]
    text, finish, _ = call(url, msgs)
    out.append((text, finish))
    return out


VARIANTS = {"v0": variant_v0, "v1": variant_v1}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="benchmark/mcbuild_bench/data/session_redacted.json")
    ap.add_argument("--url", default="http://127.0.0.1:8123/v1/chat/completions")
    ap.add_argument("--sample", type=int, default=300, help="0 = every chunk")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", default="v0,v1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    corpus = load_corpus(args.session)
    manager = make_manager(None, None)
    chunks: list[str] = []
    for rt in corpus.round_trips:
        u, a = round_trip_texts(rt)
        chunks += [c.text for c in manager.chunk(u, a, turn=int(rt["idx"]))]
    total = len(chunks)
    if args.sample and args.sample < total:
        chunks = random.Random(args.seed).sample(chunks, args.sample)
    print(f"chunks total={total} sampled={len(chunks)}", flush=True)

    result: dict = {"chunks_total": total, "sampled": len(chunks), "variants": {}}
    for name in args.variants.split(","):
        fn = VARIANTS[name]
        counts = {"ok_first": 0, "ok_retry": 0, "fail": 0}
        fails: list[dict] = []
        t0 = time.perf_counter()
        for i, ex in enumerate(chunks):
            attempts = fn(args.url, ex)
            last_text, last_finish = attempts[-1]
            if node_text_problem(last_text, last_finish) is None:
                counts["ok_first" if len(attempts) == 1 else "ok_retry"] += 1
            else:
                counts["fail"] += 1
                fails.append({"excerpt": ex[:300], "attempts": [(t[:200], f, node_text_problem(t, f)) for t, f in attempts]})
            if (i + 1) % 25 == 0:
                print(f"[{name}] {i + 1}/{len(chunks)} {counts} {time.perf_counter() - t0:.0f}s", flush=True)
        result["variants"][name] = {"counts": counts, "seconds": time.perf_counter() - t0, "failures": fails}
        print(f"[{name}] DONE {counts}", flush=True)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print("SWEEP_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
LongMemEval benchmark — baseline (full context) vs hamib_sbert (CD-only).

Each question has a "haystack" of past chat sessions. For each question we:
  - baseline: concatenate all sessions as chat history, then ask the question
  - hamib_sbert: replay each turn through SBERTExtractor to build CD (no generation
    per turn — extract only), then ask the question using CD context

Scoring: case-insensitive substring match of gold answer in the response.

Usage:
    python -m experiments.bench_longmemeval \
        --model-id meta-llama/Llama-3.3-70B-Instruct \
        --n-questions 200 \
        --output ./results/longmemeval_llama33

Environment variables:
    HAMIB_LONGMEMEVAL_PATH : path to a downloaded LongMemEval snapshot
    HF_HOME              : Hugging Face cache root (optional; defaults to ~/.cache/huggingface)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import unicodedata
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from experiments.bench_energy_monitor import EnergyMonitor

LONGMEMEVAL_PATH = os.environ.get(
    "HAMIB_LONGMEMEVAL_PATH",
    str(_ROOT / "data" / "longmemeval_s"),
)


def _load_model(model_id: str, max_new_tokens: int):
    mid = model_id.lower()
    if "gpt-oss" in mid or "gptoss" in mid:
        from server.mass_weighted_gptoss import MassWeightedGPTOSS
        return MassWeightedGPTOSS(model_id=model_id, max_new_tokens=max_new_tokens, do_sample=False)
    elif "llama" in mid:
        from server.mass_weighted_llama import MassWeightedLlama
        return MassWeightedLlama(model_id=model_id, max_new_tokens=max_new_tokens, do_sample=False)
    elif "qwen" in mid:
        from server.mass_weighted_qwen import MassWeightedQwen
        return MassWeightedQwen(model_id=model_id, max_new_tokens=max_new_tokens, do_sample=False)
    elif "gemma-3n" in mid or "gemma3n" in mid:
        from server.mass_weighted_gemma3n import MassWeightedGemma3n
        return MassWeightedGemma3n(model_id=model_id, max_new_tokens=max_new_tokens, do_sample=False)
    else:
        from server.mass_weighted_gemma import MassWeightedGemma
        return MassWeightedGemma(model_id=model_id, max_new_tokens=max_new_tokens, do_sample=False)


import re as _re
_LME_PUNCT_RE = _re.compile(r"[,\.\;\:\!\?\(\)\[\]\{\}\"'`]+")
_LME_WS_RE = _re.compile(r"\s+")


def _normalize(s) -> str:
    s = str(s)
    s = unicodedata.normalize("NFKC", s.lower())
    for ch in ("‐", "‑", "–", "—", "－"):
        s = s.replace(ch, "-")
    s = _LME_PUNCT_RE.sub(" ", s)
    s = _LME_WS_RE.sub(" ", s).strip()
    return s


def _score(response, gold) -> bool:
    """Case-insensitive substring match after punctuation/whitespace normalize.
    Gold may be any scalar (str / int / float) — coerced to str."""
    g = _normalize(gold).strip()
    if not g:
        return False
    return g in _normalize(response)


def _load_questions(n: int):
    with open(LONGMEMEVAL_PATH) as f:
        data = json.load(f)
    return data[:n]


def _format_history_chat(sessions: list, max_chars: int = 60000) -> str:
    """Concatenate sessions as a chat history string.
    Limit to max_chars to avoid blowing the context (truncate from the start
    so the most recent sessions are preserved)."""
    parts = []
    for sess in sessions:
        for turn in sess:
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "").strip()
            parts.append(f"{role}: {content}")
    full = "\n".join(parts)
    if len(full) > max_chars:
        full = "...\n" + full[-max_chars:]
    return full


def _flat_pairs(sessions: list) -> list[tuple[str, str]]:
    """Extract (user, assistant) pairs across all sessions.

    Each turn's text is prefixed with its role (``User: ...`` / ``Assistant:
    ...``). Without this prefix the dialogue extractor cannot tell who is
    speaking, so fact-bearing sentences end up orphaned in the CD (no
    speaker entity to attach to). This impairs paired HAMIB-vs-baseline
    comparisons: baseline sees verbatim role-attributed chat history while
    HAMIB sees only the extracted CD, which without the speaker prefix has lost
    the attribution.
    """
    out: list[tuple[str, str]] = []
    for sess in sessions:
        pending_user = None
        for turn in sess:
            role = turn.get("role")
            content = (turn.get("content") or "").strip()
            tx_attrib = f"{role.capitalize()}: {content}" if (role and content) else content
            if role == "user":
                if pending_user is not None:
                    out.append((pending_user, ""))
                pending_user = tx_attrib
            elif role == "assistant":
                u = pending_user or ""
                out.append((u, tx_attrib))
                pending_user = None
        if pending_user is not None:
            out.append((pending_user, ""))
    return out


def run_baseline(gemma, questions: list, output: Path) -> dict:
    print(f"\n=== longmemeval baseline ({len(questions)} items) ===", flush=True)
    em = EnergyMonitor()
    em.start()
    t_start = time.perf_counter()
    correct = 0
    ms_list = []
    item_records = []
    for i, q in enumerate(questions, 1):
        history = _format_history_chat(q["haystack_sessions"])
        prompt = (
            f"The following is a long chat history between User and Assistant.\n"
            f"{history}\n\n"
            f"Based on the chat history above, answer the following question concisely.\n"
            f"Question: {q['question']}\n"
            f"Answer:"
        )
        t0 = time.perf_counter()
        try:
            resp = gemma.generate(prompt)
        except Exception as e:
            resp = f"[ERR: {e}]"
        ms = (time.perf_counter() - t0) * 1000.0
        ms_list.append(ms)
        ok = _score(resp, q["answer"])
        if ok:
            correct += 1
        item_records.append({
            "i": i, "qid": q["question_id"], "qtype": q["question_type"],
            "question": q["question"][:200], "gold": q["answer"],
            "correct": bool(ok), "ms": round(ms, 1),
            "response": resp[:400],
            "prompt_chars": len(prompt),
        })
        if i in (1,2,3,10,25,50,75,100,150,200) or i == len(questions):
            print(f"  [lme base] {i}/{len(questions)}  acc={correct}/{i}={correct/i:.2f}  "
                  f"{ms:.0f}ms  prompt_chars={len(prompt)}", flush=True)
    em.stop()
    energy = em.summary()
    total = time.perf_counter() - t_start
    ms_list.sort()
    p50 = ms_list[len(ms_list)//2] if ms_list else 0
    p95 = ms_list[int(len(ms_list)*0.95)] if ms_list else 0
    summary = {
        "mode": "baseline", "n": len(questions),
        "correct": correct, "acc": correct / len(questions),
        "total_s": round(total, 1), "p50_ms": round(p50, 0), "p95_ms": round(p95, 0),
        "gpu_J": energy.get("gpu_energy_J", 0.0),
        "cpu_J": energy.get("cpu_energy_J_est", 0.0),
    }
    out = output / "longmemeval_baseline.json"
    json.dump({"summary": summary, "items": item_records}, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"  -> {out}  acc={correct}/{len(questions)} = {correct/len(questions):.3f}  "
          f"total={total:.0f}s  gpu_J={summary['gpu_J']:.0f}", flush=True)
    return summary


def run_hamib(gemma, extractor_fn, questions: list, output: Path,
            *, cap_sun_mass: float = 1.0, cap_planet_mass: float = 0.5,
            cap_sat_mass: float = 0.1, start_from_index: int = 0,
            output_filename: str = "longmemeval_hamib_sbert.json") -> dict:
    """
    Args:
      start_from_index: 0-based question index to start from (for resume).
                        Output filename will get a suffix if start>0.
      output_filename: output JSON name (use a custom name for partial/resume runs).
    """
    if start_from_index > 0:
        print(f"\n=== longmemeval hamib_sbert RESUME from i={start_from_index} "
              f"({len(questions) - start_from_index} items) ===", flush=True)
        questions = questions[start_from_index:]
    else:
        print(f"\n=== longmemeval hamib_sbert ({len(questions)} items) ===", flush=True)
    from store.cd_store import CDStore
    from server.hamib_session import HAMIBSession
    em = EnergyMonitor()
    em.start()
    t_start = time.perf_counter()
    correct = 0
    ms_list = []
    cd_max = 0
    item_records = []
    EN_TEMPLATE = (
        "The following are facts extracted from past chat sessions. Use them "
        "to answer the user's question. Quote names, dates, and numbers "
        "exactly as they appear in the facts.\n\n"
        "{context_block}\n\n"
        "Question: {user_text}\nAnswer:"
    )
    for local_i, q in enumerate(questions, 1):
        i = local_i + start_from_index  # global question index (1-based)
        store = CDStore()
        session = HAMIBSession(
            store, _preloaded_gemma=gemma,
            use_mass=True, speculative_update=False,
            disable_eval2=True, extractor_fn=extractor_fn,
            prompt_template=EN_TEMPLATE,
        )
        # ===== build CD by replaying (user, assistant) pairs through HAMIB
        # extractor + merger pipeline. No generation per turn — we just
        # populate the CD so the final query can be answered with CD context. =====
        cd = store.get_current()
        for user_text, assistant_text in _flat_pairs(q["haystack_sessions"]):
            try:
                session._update_cd(cd, user_text, assistant_text)
            except Exception:
                pass
        # mass-cap fix: GraphMerger accumulates mass on every duplicate merge,
        # which makes exp(mass_weight * accumulated_mass) blow up attention
        # even at modest mass_weight. Cap per-level masses to the config defaults.
        cd = store.get_current()
        for se in cd.suns:
            if se.sun.mass > cap_sun_mass:
                se.sun.mass = cap_sun_mass
            for pe in se.planets:
                if pe.planet.mass > cap_planet_mass:
                    pe.planet.mass = cap_planet_mass
                for sat in pe.satellites:
                    if sat.mass > cap_sat_mass:
                        sat.mass = cap_sat_mass
        cd_now = len(list(store.get_current().all_nodes()))
        if cd_now > cd_max:
            cd_max = cd_now
        # ===== ask question using the built CD =====
        t0 = time.perf_counter()
        try:
            resp = session.chat(q["question"])
        except Exception as e:
            resp = f"[ERR: {e}]"
        ms = (time.perf_counter() - t0) * 1000.0
        ms_list.append(ms)
        ok = _score(resp, q["answer"])
        if ok:
            correct += 1
        item_records.append({
            "i": i, "qid": q["question_id"], "qtype": q["question_type"],
            "question": q["question"][:200], "gold": q["answer"],
            "correct": bool(ok), "ms": round(ms, 1), "cd": cd_now,
            "response": resp[:400],
        })
        total_q = len(questions) + start_from_index
        if i in (1,2,3,10,25,50,75,100,125,150,175,200) or local_i == len(questions):
            print(f"  [lme hamib ] {i}/{total_q}  acc={correct}/{local_i}={correct/local_i:.2f}  "
                  f"{ms:.0f}ms cd={cd_now}", flush=True)
        del session
        if local_i % 10 == 0:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # incremental write: every 25 Q flush JSON so a mid-run kill preserves
        # partial data (avoids all-or-nothing loss).
        if local_i % 25 == 0:
            try:
                partial_summary = {
                    "mode": "hamib_sbert", "in_progress": True,
                    "n_done_local": local_i, "n_target": len(questions),
                    "correct": correct, "acc": correct / local_i if local_i else 0.0,
                    "cd_max": cd_max, "start_from_index": start_from_index,
                }
                partial_out = output / output_filename
                json.dump({"summary": partial_summary, "items": item_records},
                          open(partial_out, "w"), ensure_ascii=False, indent=2)
            except Exception as _e:
                print(f"  [WARN] incremental write failed: {_e}", flush=True)
    em.stop()
    energy = em.summary()
    total = time.perf_counter() - t_start
    ms_list.sort()
    p50 = ms_list[len(ms_list)//2] if ms_list else 0
    p95 = ms_list[int(len(ms_list)*0.95)] if ms_list else 0
    summary = {
        "mode": "hamib_sbert", "n": len(questions),
        "correct": correct, "acc": correct / len(questions),
        "total_s": round(total, 1), "p50_ms": round(p50, 0), "p95_ms": round(p95, 0),
        "gpu_J": energy.get("gpu_energy_J", 0.0),
        "cpu_J": energy.get("cpu_energy_J_est", 0.0),
        "cd_max": cd_max,
    }
    out = output / output_filename
    summary["start_from_index"] = start_from_index
    json.dump({"summary": summary, "items": item_records}, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"  -> {out}  acc={correct}/{len(questions)} = {correct/len(questions):.3f}  "
          f"total={total:.0f}s  gpu_J={summary['gpu_J']:.0f}  cd_max={cd_max}"
          + (f"  (resume from i={start_from_index})" if start_from_index else ""),
          flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--n-questions", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--extractor", choices=["dialogue", "regex_only"], default="dialogue",
                    help="dialogue = NER-like regex for natural dialogue (default for "
                         "LongMemEval); regex_only = NAME+CODE pair only "
                         "(for bench_scaler-style benchmarks)")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--skip-hamib", action="store_true")
    ap.add_argument("--mass-weight", type=float, default=0.3)
    ap.add_argument("--max-entities", type=int, default=1)
    ap.add_argument("--max-satellites", type=int, default=2)
    ap.add_argument("--cap-sun-mass", type=float, default=1.0)
    ap.add_argument("--cap-planet-mass", type=float, default=0.5)
    ap.add_argument("--cap-sat-mass", type=float, default=0.1)
    ap.add_argument("--use-3level-hierarchy", action="store_true",
                    help="emit the 3-level hierarchy (sun 'Conversation' "
                         "-> planet entity -> satellite fact); default off uses "
                         "flat extraction")
    ap.add_argument("--start-from-index", type=int, default=0,
                    help="Skip first N HAMIB questions (for resume after timeout).")
    ap.add_argument("--hamib-output-name", type=str, default="longmemeval_hamib_sbert.json",
                    help="Output filename for HAMIB run (use a suffix when resuming).")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(f"longmemeval bench: model={args.model_id}  n={args.n_questions}  "
          f"extractor={args.extractor}  output={args.output}")
    print("=" * 70, flush=True)

    questions = _load_questions(args.n_questions)
    print(f"loaded {len(questions)} questions", flush=True)

    gemma = _load_model(args.model_id, args.max_new_tokens)
    gemma.load()
    if args.mass_weight != 1.0:
        gemma._mass_weight = args.mass_weight
        print(f"[bench] mass_weight overridden -> {args.mass_weight}", flush=True)

    if args.extractor == "dialogue":
        from experiments.dialogue_extractor import make_dialogue_extractor_fn
        extractor_fn = make_dialogue_extractor_fn(
            max_entities=args.max_entities,
            max_satellites=args.max_satellites,
            use_3level_hierarchy=args.use_3level_hierarchy,
        )
        print(f"[bench] dialogue extractor: use_3level_hierarchy={args.use_3level_hierarchy}",
              flush=True)
    else:
        from server.sbert_extractor import make_extractor_fn
        extractor_fn = make_extractor_fn(regex_only=True)

    summaries = {}
    if not args.skip_baseline:
        summaries["baseline"] = run_baseline(gemma, questions, args.output)
    if not args.skip_hamib:
        summaries["hamib_sbert"] = run_hamib(
            gemma, extractor_fn, questions, args.output,
            cap_sun_mass=args.cap_sun_mass,
            cap_planet_mass=args.cap_planet_mass,
            cap_sat_mass=args.cap_sat_mass,
            start_from_index=args.start_from_index,
            output_filename=args.hamib_output_name,
        )

    json.dump(summaries, open(args.output/"longmemeval_summary.json","w"), ensure_ascii=False, indent=2)
    md = ["# LongMemEval Summary", "", f"Model: `{args.model_id}`  N={args.n_questions}", "",
          "| mode | acc | total_s | p50_ms | p95_ms | gpu_J | cd_max |",
          "|---|---|---|---|---|---|---|"]
    for s in summaries.values():
        md.append(f"| {s['mode']} | {s['correct']}/{s['n']} ({s['acc']:.3f}) "
                  f"| {s['total_s']:.0f} | {s['p50_ms']:.0f} | {s['p95_ms']:.0f} "
                  f"| {s['gpu_J']:.0f} | {s.get('cd_max', 0)} |")
    (args.output/"longmemeval_summary.md").write_text("\n".join(md) + "\n")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()

"""
GPT-OSS-20b 3-way benchmark (GPT-OSS variant of bench_gemma3n_3way.py)

modes:
  1. vanilla_gptoss      : plain GPT-OSS + chat_template + raw history (no HAMIB)
  2. hamib_gptoss_sbert    : GPT-OSS + SBERT extractor (HAMIB pipeline, recommended)
  3. hamib_gptoss_gptoss   : GPT-OSS + GPT-OSS extractor (HAMIB pipeline, heavy, optional)

Scenario: L1 (N=10, 30 turn) / L2 (N=25, 75 turn) / L3 (N=50, 150 turn)
Metrics: recall / inference_ms / GPU energy / RAM peak / max CD nodes
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from experiments.bench_energy_monitor import EnergyMonitor
from experiments.bench_scaler import build_scenario, _phase


@dataclass
class TurnRec:
    mode: str
    turn: int
    phase: str
    user_text: str
    response: str
    inference_ms: float
    prompt_chars: int = 0
    cd_node_count: int = 0


def run_vanilla(model_wrapper, all_msgs, n) -> list[TurnRec]:
    """vanilla GPT-OSS: keep sending raw history via chat template"""
    print(f"\n=== vanilla_gptoss (N={n}) ===", flush=True)
    tok = model_wrapper.tokenizer
    records = []
    chat_history = []
    for i, msg in enumerate(all_msgs, start=1):
        chat_history.append({"role": "user", "content": msg})
        try:
            # Pass reasoning_effort='low' to match the HAMIB path; fall back if unsupported.
            try:
                prompt = tok.apply_chat_template(
                    chat_history, tokenize=False, add_generation_prompt=True,
                    reasoning_effort="low",
                )
            except TypeError:
                prompt = tok.apply_chat_template(
                    chat_history, tokenize=False, add_generation_prompt=True,
                )
        except Exception:
            prompt = "\n".join(
                f"User: {h['content']}" if h['role']=='user' else f"Assistant: {h['content']}"
                for h in chat_history
            ) + "\nAssistant:"
        t0 = time.perf_counter()
        try:
            # vanilla wants raw generation without the patch, but the loader is
            # already patched, so clear mass_vec/M so no mass injection happens
            model_wrapper.clear_mass_vector()
            model_wrapper.clear_m_matrix()
            # generate() internally calls _to_chat_template, but the prompt above
            # is already in Harmony form so it won't match User: ... Assistant: -> pass through
            resp = model_wrapper.generate(prompt)
        except Exception as e:
            resp = f"[ERR: {e}]"
        ms = (time.perf_counter() - t0) * 1000
        chat_history.append({"role": "assistant", "content": resp})
        records.append(TurnRec("vanilla_gptoss", i, _phase(i, n), msg[:80], resp,
                                round(ms, 1), len(prompt), 0))
        if i % 5 == 0 or i <= 3 or _phase(i, n) == "phase3":
            try:
                print(f"  T{i:>3} [{_phase(i, n)[:6]}] {ms:>6.0f}ms tok={len(prompt):>5} resp={resp[:60]!r}", flush=True)
            except UnicodeEncodeError:
                pass
    return records


def run_hamib(model_wrapper, mode, all_msgs, n, extractor_fn=None) -> list[TurnRec]:
    """HAMIB mode: inject mass into MassWeightedGPTOSS via HAMIBSession"""
    from store.cd_store import CDStore
    from server.hamib_session import HAMIBSession
    print(f"\n=== {mode} (N={n}) ===", flush=True)
    store = CDStore()
    session = HAMIBSession(
        store, _preloaded_gemma=model_wrapper,
        use_mass=True, speculative_update=False,
        disable_eval2=True,
        extractor_fn=extractor_fn,
    )
    records = []
    for i, msg in enumerate(all_msgs, start=1):
        try:
            resp = session.chat(msg)
            t = session.last_timing
        except Exception as e:
            resp = f"[ERR: {e}]"
            t = {"total_ms": 0}
        cd_n = len(list(store.get_current().all_nodes()))
        records.append(TurnRec(mode, i, _phase(i, n), msg[:80], resp,
                                round(t.get("total_ms", 0), 1),
                                getattr(session, 'last_context_tokens', 0) or 0,
                                cd_n))
        if i % 5 == 0 or i <= 3 or _phase(i, n) == "phase3":
            try:
                print(f"  T{i:>3} [{_phase(i, n)[:6]}] {t.get('total_ms', 0):>6.0f}ms cd={cd_n} resp={resp[:60]!r}", flush=True)
            except UnicodeEncodeError:
                pass
    del session
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(2)
    return records


def _normalize_for_score(s: str) -> str:
    """Normalize Unicode dash variants (en-dash, em-dash, non-breaking-hyphen,
    fullwidth) to ASCII hyphen. GPT-OSS sometimes rewrites 'CRANE-27' to
    'CRANE‑27' (U+2011) during markdown post-processing.
    """
    import unicodedata
    s = unicodedata.normalize("NFKC", s)
    for ch in ("‐", "‑", "–", "—", "－"):
        s = s.replace(ch, "-")
    return s


def score(records, facts):
    p3 = [r for r in records if r.phase == "phase3"]
    hits = {}
    for (name, code) in facts:
        n_name = _normalize_for_score(name)
        n_code = _normalize_for_score(code)
        rel = [r for r in p3 if n_name in _normalize_for_score(r.user_text)]
        hit = any(n_code in _normalize_for_score(r.response) for r in rel)
        hits[f"{name}->{code}"] = hit
    n = sum(hits.values())
    return n, len(facts), hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-facts", type=int, default=10, help="L1=10, L2=25, L3=50")
    ap.add_argument("--modes", nargs="+",
                    default=["vanilla_gptoss", "hamib_gptoss_sbert"])
    ap.add_argument("--model-id", default="openai/gpt-oss-20b")
    ap.add_argument("--output", type=Path, default=_ROOT / "experiments" / "results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--regex-only", action="store_true",
                    help="disable SBERT sliding window; build the CD from NAME+CODE pairs only")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"GPT-OSS 3-way bench: N={args.n_facts} facts, {3*args.n_facts} turns")
    print(f"  modes: {args.modes}, model: {args.model_id}")
    print(f"  max_new_tokens: {args.max_new_tokens}")
    print("=" * 70, flush=True)

    all_msgs, facts = build_scenario(args.n_facts, args.seed)
    print(f"scenario built: {len(all_msgs)} turns", flush=True)

    print("\n[Load] GPT-OSS-20b ...", flush=True)
    from server.mass_weighted_gptoss import MassWeightedGPTOSS
    mw = MassWeightedGPTOSS(
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    mw.load()

    sbert_fn = None
    if "hamib_gptoss_sbert" in args.modes:
        from server.sbert_extractor import make_extractor_fn
        sbert_fn = make_extractor_fn(
            threshold=0.2,
            enable_regex_hybrid=True,
            regex_only=args.regex_only,
        )
        mode_label = "regex_only" if args.regex_only else "regex+SBERT span"
        print(f"[Load] SBERT extractor ready ({mode_label})", flush=True)

    all_records = []
    summaries = {}
    for mode in args.modes:
        with EnergyMonitor() as em:
            if mode == "vanilla_gptoss":
                rs = run_vanilla(mw, all_msgs, args.n_facts)
            elif mode == "hamib_gptoss_gptoss":
                rs = run_hamib(mw, mode, all_msgs, args.n_facts)
            elif mode == "hamib_gptoss_sbert":
                rs = run_hamib(mw, mode, all_msgs, args.n_facts, extractor_fn=sbert_fn)
            else:
                print(f"[skip] {mode}")
                continue
        em_summary = em.summary()
        all_records.extend(rs)
        n_hit, n_tot, hits = score(rs, facts)
        ms = sorted([r.inference_ms for r in rs])
        total_s = sum(r.inference_ms for r in rs) / 1000
        max_cd = max(r.cd_node_count for r in rs) if rs else 0
        summaries[mode] = {
            "n_facts": args.n_facts,
            "recall": n_hit / n_tot, "n_hit": n_hit, "n_total": n_tot,
            "total_s": total_s,
            "p50_ms": ms[len(ms) // 2] if ms else 0,
            "p95_ms": ms[int(len(ms) * 0.95)] if ms else 0,
            "phase3_s": sum(r.inference_ms for r in rs if r.phase == "phase3") / 1000,
            "max_cd_nodes": max_cd,
            "max_prompt_chars": max(r.prompt_chars for r in rs) if rs else 0,
            **em_summary,
            "energy_J_per_correct": em_summary["gpu_energy_J"] / max(n_hit, 1),
            "time_s_per_correct": total_s / max(n_hit, 1),
            "hits": hits,
        }
        try:
            print(
                f"\n[{mode}] N={args.n_facts} recall={n_hit}/{n_tot} "
                f"time={total_s:.1f}s gpu={em_summary['gpu_energy_J']:.0f}J cd={max_cd}",
                flush=True,
            )
        except UnicodeEncodeError:
            pass
        _save(all_records, summaries, args.output, args.n_facts)

    _save(all_records, summaries, args.output, args.n_facts)
    _summary(summaries)


def _save(records, summaries, out, n):
    csv_path = out / f"exp_gptoss_3way_N{n}.csv"
    json_path = out / f"exp_gptoss_3way_N{n}.json"
    if records:
        fields = list(asdict(records[0]).keys())
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows([asdict(r) for r in records])
        json_path.write_text(json.dumps({
            "n_facts": n, "summaries": summaries,
            "records": [asdict(r) for r in records],
        }, ensure_ascii=False, indent=2), encoding="utf-8")


def _summary(summaries):
    print("\n" + "=" * 70 + "\nFINAL GPT-OSS 3-WAY SUMMARY\n" + "=" * 70)
    for mode, s in summaries.items():
        print(f"\n[{mode}]")
        print(f"  recall={s['n_hit']}/{s['n_total']} ({s['recall']*100:.0f}%)")
        print(f"  total_time={s['total_s']:.1f}s p50={s['p50_ms']:.0f}ms")
        print(f"  gpu_J={s['gpu_energy_J']:.0f} cpu_J={s['cpu_energy_J_est']:.0f}")
        print(f"  ram_peak={s['ram_peak_mb']:.0f}MB cd_max={s['max_cd_nodes']}")
        print(f"  s/correct={s['time_s_per_correct']:.1f} J/correct={s['energy_J_per_correct']:.1f}")


if __name__ == "__main__":
    main()

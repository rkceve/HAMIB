"""
Difficulty-scaled benchmark: L2 (25 fact) / L3 (50 fact) / L4 (100 fact)

Purpose:
  Probe the limits of baseline / cms_gemma / cms_sbert + measure energy efficiency.

Scenario:
  Phase 1: N fact storage (1 fact / turn)
  Phase 2: N chitchat turn (with distractors)
  Phase 3: N question (recall each fact)
  total: 3N turn

Choice of N:
  L1 (10), L2 (25), L3 (50), L4 (100)

Metrics:
  per turn: inference_ms, prompt_chars, cd_node_count
  per mode: total_time, gpu_J, cpu_J, ram_peak, GPU peak W, CPU peak %
  efficiency: s / correct_recall, J / correct_recall, MB / correct_recall
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

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# HF_HOME can be overridden via environment variable
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

import torch
from experiments.bench_energy_monitor import EnergyMonitor


# fact name pool (100+ entries)
NAME_POOL = [
    # Greek 24 entries
    "Alpha","Beta","Gamma","Delta","Epsilon","Zeta","Eta","Theta",
    "Iota","Kappa","Lambda","Mu","Nu","Xi","Omicron","Pi",
    "Rho","Sigma","Tau","Upsilon","Phi","Chi","Psi","Omega",
    # Roman / Greek mythology 28 entries
    "Ares","Athena","Apollo","Hermes","Helios","Hera","Hades","Demeter",
    "Dionysus","Eros","Hephaestus","Hestia","Poseidon","Selene","Titan","Atlas",
    "Boreas","Chronos","Daphne","Echo","Faune","Galene","Hyperion","Iris",
    "Janus","Knossos","Leander","Medea",
    # supplement: constellations / mythology 50 entries
    "Andromeda","Auriga","Cassiopeia","Cygnus","Draco","Lyra","Orion","Perseus",
    "Pegasus","Phoenix","Vega","Sirius","Polaris","Capella","Arcturus","Antares",
    "Rigel","Spica","Aldebaran","Altair","Deneb","Procyon","Betelgeuse","Castor",
    "Pollux","Regulus","Mintaka","Bellatrix","Mira","Almach","Mizar","Alcor",
    "Caelum","Volans","Lepus","Carina","Vela","Puppis","Pyxis","Crux",
    "Tucana","Pavo","Phoenix2","Indus","Grus","Pisces","Aquila","Cetus",
    "Hydra","Lupus",
]
assert len(set(NAME_POOL)) >= 100, f"name pool needs ≥100 unique, got {len(set(NAME_POOL))}"

# code pool: CRANE-N (1..200 random, no duplicates)
import random
random.seed(42)
CODE_POOL = [f"CRANE-{i}" for i in random.sample(range(1, 201), 200)]


def generate_facts(n: int) -> list[tuple[str, str]]:
    """Generate N (name, code) pairs"""
    names = NAME_POOL[:n]
    codes = CODE_POOL[:n]
    return list(zip(names, codes))


# chitchat templates
CHITCHAT_TEMPLATES = [
    "今日は天気がいいね。",
    "週末はどこか出かけたい気分。",
    "最近ハマっている趣味は?",
    "推理小説とかどう?",
    "おすすめのカフェとかある?",
    "本を読むのが好きだけど時間がない。",
    "AI に興味あるんだ。",
    "Python 使ったことある?",
    "Docker は便利だよね。",
    "今夜は寒くなるって。",
    "近所のバス番号 CR-9 が変わるらしい。",
    "BUS-12 の路線図が更新された。",
    "整理券 LB-50 まで配布。",
    "工事看板の番号 D-5 を見かけた。",
    "店のスタンプカードは CN-44 が今期。",
    "明日は雨らしいよ。",
    "暖かい鍋でも食べたい。",
    "京都の紅葉が見たいな。",
    "クラシックも悪くない。",
    "リファクタリングのタイミングって難しい。",
]

# fact phrasing templates
FACT_TEMPLATES = [
    "{name} プロジェクトの担当コードは「{code}」だよ。覚えておいてね。",
    "あと重要、{name} プロジェクトのコードは「{code}」です。",
    "それから {name} は {code} ね。",
    "{name} の担当コードは {code} だよ。",
    "{name} については {code} でお願いします。",
    "{name} プロジェクトのコードは「{code}」って覚えておいて。",
    "ところで {name} は {code} が割り当てられてる。",
    "{name} の担当は {code}。",
    "あと {name} プロジェクトは「{code}」。前回も忘れないでね。",
    "{name} のコードは {code}。",
]

QUESTION_TEMPLATES = [
    "そういえば、{name} プロジェクトの担当コードって何だっけ?",
    "{name} プロジェクトのコードも教えて。",
    "{name} のコードは?",
    "{name} は CRANE-いくつだったかな?",
    "{name} の担当コードを思い出させて。",
    "{name} のコードって何でしたっけ?",
    "{name} プロジェクトのコードは?",
    "{name} のコードを再確認させて。",
    "{name} の担当コード何だっけ?",
    "{name} のコードを教えて。",
]


def _load_model(model_id: str, max_new_tokens: int = 32, do_sample: bool = False):
    """Auto-select and return the appropriate MassWeighted* class from model_id"""
    mid = model_id.lower()
    if any(k in mid for k in ("llama", "gpt-oss", "mistral", "falcon")):
        from server.mass_weighted_llama import MassWeightedLlama
        return MassWeightedLlama(model_id=model_id, max_new_tokens=max_new_tokens, do_sample=do_sample)
    elif "qwen" in mid:
        from server.mass_weighted_qwen import MassWeightedQwen
        return MassWeightedQwen(model_id=model_id, max_new_tokens=max_new_tokens, do_sample=do_sample)
    elif any(k in mid for k in ("gemma-3n", "gemma3n", "gemma_3n")):
        from server.mass_weighted_gemma3n import MassWeightedGemma3n
        return MassWeightedGemma3n(model_id=model_id, max_new_tokens=max_new_tokens, do_sample=do_sample)
    else:
        from server.mass_weighted_gemma import MassWeightedGemma
        return MassWeightedGemma(model_id=model_id, max_new_tokens=max_new_tokens, do_sample=do_sample)


def build_scenario(n: int, seed: int = 42) -> tuple[list[str], list[tuple[str,str]]]:
    """Build a scenario of N fact / N chitchat / N question = 3N turns (L1-L4)"""
    rng = random.Random(seed)
    facts = generate_facts(n)

    # Phase 1: N fact (1 turn per fact)
    p1 = []
    for i, (name, code) in enumerate(facts):
        tmpl = FACT_TEMPLATES[i % len(FACT_TEMPLATES)]
        p1.append(tmpl.format(name=name, code=code))

    # Phase 2: N chitchat (chosen from templates via rng, duplicates allowed)
    p2 = [rng.choice(CHITCHAT_TEMPLATES) for _ in range(n)]

    # Phase 3: N question (shuffled order)
    indices = list(range(len(facts)))
    rng.shuffle(indices)
    p3 = []
    for idx in indices:
        name, code = facts[idx]
        tmpl = QUESTION_TEMPLATES[idx % len(QUESTION_TEMPLATES)]
        p3.append(tmpl.format(name=name))

    return p1 + p2 + p3, facts


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


def _make_phase_fn(n_facts: int, total_turns: int):
    """Build a function that returns phase labels based on the actual turn count.
    l5_noise makes total_turns > 3*n_facts, so a fixed 2N boundary misclassifies.
    phase1: turn 1..n_facts, phase2: n_facts+1..total_turns-n_facts, phase3: last n_facts turns.
    """
    p2_start = n_facts + 1
    p3_start = total_turns - n_facts + 1
    def _phase(t: int) -> str:
        if t < p3_start:
            if t < p2_start:
                return "phase1"
            return "phase2"
        return "phase3"
    return _phase


def run_baseline(gemma, all_msgs: list[str], n: int) -> list[TurnRec]:
    print(f"\n=== baseline (N={n}) ===", flush=True)
    phase_of = _make_phase_fn(n, len(all_msgs))
    history = []
    records = []
    for i, msg in enumerate(all_msgs, start=1):
        prompt = "\n".join(history + [f"User: {msg}", "Assistant:"])
        t0 = time.perf_counter()
        try:
            resp = gemma.generate(prompt)
        except Exception as e:
            resp = f"[ERR: {e}]"
        ms = (time.perf_counter() - t0) * 1000
        history.append(f"User: {msg}")
        history.append(f"Assistant: {resp}")
        ph = phase_of(i)
        records.append(TurnRec("baseline", i, ph, msg[:80], resp,
                                round(ms,1), len(prompt), 0))
        if i % 5 == 0 or i <= 3 or ph == "phase3":
            try:
                print(f"  T{i:>3} [{ph[:6]}] {ms:>6.0f}ms tok={len(prompt):>5}", flush=True)
            except UnicodeEncodeError:
                pass
    return records


def run_cms(gemma, mode: str, all_msgs: list[str], n: int, extractor_fn=None) -> list[TurnRec]:
    from store.cd_store import CDStore
    from server.hamib_session import HAMIBSession
    print(f"\n=== {mode} (N={n}) ===", flush=True)
    phase_of = _make_phase_fn(n, len(all_msgs))
    store = CDStore()
    session = HAMIBSession(
        store, _preloaded_gemma=gemma,
        use_mass=True, speculative_update=False,
        disable_eval2=True, extractor_fn=extractor_fn,
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
        ph = phase_of(i)
        records.append(TurnRec(mode, i, ph, msg[:80], resp,
                                round(t.get("total_ms",0),1),
                                getattr(session,'last_context_tokens',0) or 0,
                                cd_n))
        if i % 5 == 0 or i <= 3 or ph == "phase3":
            try:
                print(f"  T{i:>3} [{ph[:6]}] {t.get('total_ms',0):>6.0f}ms cd={cd_n}", flush=True)
            except UnicodeEncodeError:
                pass
    del session
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(2)
    return records


def _normalize(s: str) -> str:
    """Normalize Unicode dash variants to ASCII hyphen (handles GPT-OSS/Llama/Qwen post-processing)."""
    import unicodedata
    s = unicodedata.normalize("NFKC", s)
    for ch in ("‐", "‑", "–", "—", "－"):
        s = s.replace(ch, "-")
    return s


def score(records: list[TurnRec], facts: list[tuple[str,str]]) -> tuple[int,int,dict]:
    p3 = [r for r in records if r.phase == "phase3"]
    hits = {}
    for (name, code) in facts:
        n_name = _normalize(name)
        n_code = _normalize(code)
        rel = [r for r in p3 if n_name in _normalize(r.user_text)]
        hit = any(n_code in _normalize(r.response) for r in rel)
        hits[f"{name}->{code}"] = hit
    n = sum(hits.values())
    return n, len(facts), hits


def build_scenario_l5_noise(n: int, seed: int = 42, noise_mul: int = 3) -> tuple[list[str], list[tuple[str,str]]]:
    """L5-noise: extend Phase 2 by a factor of noise_mul. {noise_mul}N chitchat turns stress memory.
    total turns = N + noise_mul*N + N = (2 + noise_mul)*N
    """
    rng = random.Random(seed)
    facts = generate_facts(n)
    p1 = [FACT_TEMPLATES[i % len(FACT_TEMPLATES)].format(name=nm, code=cd)
          for i, (nm, cd) in enumerate(facts)]
    p2 = [rng.choice(CHITCHAT_TEMPLATES) for _ in range(n * noise_mul)]
    indices = list(range(len(facts)))
    rng.shuffle(indices)
    p3 = [QUESTION_TEMPLATES[idx % len(QUESTION_TEMPLATES)].format(name=facts[idx][0])
          for idx in indices]
    return p1 + p2 + p3, facts


def build_scenario_l5_adv(n: int, seed: int = 42) -> tuple[list[str], list[tuple[str,str]]]:
    """L5-adv: inject similarly-named distractors into Phase 2.
    e.g. mix the fake "Alpha2 code is CRANE-199" with the real Alpha.
    """
    rng = random.Random(seed)
    facts = generate_facts(n)
    p1 = [FACT_TEMPLATES[i % len(FACT_TEMPLATES)].format(name=nm, code=cd)
          for i, (nm, cd) in enumerate(facts)]
    # distractors: assign fake codes to fake names (e.g. "Alpha2")
    distractor_codes = [f"CRANE-{i}" for i in range(201, 201 + n)]
    p2 = []
    for i in range(n):
        real_name = facts[i % len(facts)][0]
        fake_name = f"{real_name}2"
        fake_code = distractor_codes[i % len(distractor_codes)]
        p2.append(f"{fake_name} プロジェクトのコードは「{fake_code}」だよ。")
    rng.shuffle(p2)
    indices = list(range(len(facts)))
    rng.shuffle(indices)
    p3 = [QUESTION_TEMPLATES[idx % len(QUESTION_TEMPLATES)].format(name=facts[idx][0])
          for idx in indices]
    return p1 + p2 + p3, facts


def build_scenario_l5_multi(n: int, seed: int = 42) -> tuple[list[str], list[tuple[str,str]]]:
    """L5-multi: Phase 3 queries request multiple facts at once.
    Questions of the form "tell me the codes of Alpha and Beta each".
    """
    rng = random.Random(seed)
    facts = generate_facts(n)
    p1 = [FACT_TEMPLATES[i % len(FACT_TEMPLATES)].format(name=nm, code=cd)
          for i, (nm, cd) in enumerate(facts)]
    p2 = [rng.choice(CHITCHAT_TEMPLATES) for _ in range(n)]
    # Phase 3: ask in pairs
    pairs = [(facts[i], facts[(i+1) % n]) for i in range(n)]
    rng.shuffle(pairs)
    p3 = [f"{a[0]} と {b[0]} のコードをそれぞれ教えて。" for a, b in pairs]
    return p1 + p2 + p3, facts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-facts", type=int, required=True, help="L1=10, L2=25, L3=50, L4=100")
    ap.add_argument("--modes", nargs="+", default=["baseline","cms_gemma","cms_sbert"])
    ap.add_argument("--model-id", default="google/gemma-3-4b-it")
    ap.add_argument("--output", type=Path, default=_ROOT/"experiments"/"results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scenario", default="standard",
                    choices=["standard","l5_noise","l5_adv","l5_multi"],
                    help="standard=L1-L4 / l5_noise=long distraction / l5_adv=decoy names / l5_multi=multi-fact queries")
    ap.add_argument("--noise-mul", type=int, default=3,
                    help="l5_noise Phase 2 multiplier (default=3)")
    ap.add_argument("--regex-only", action="store_true",
                    help="disable SBERT sliding window; build the CD from NAME+CODE pairs only")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.scenario == "standard":
        all_msgs, facts = build_scenario(args.n_facts, args.seed)
        tag = f"N{args.n_facts}"
    elif args.scenario == "l5_noise":
        all_msgs, facts = build_scenario_l5_noise(args.n_facts, args.seed, args.noise_mul)
        tag = f"N{args.n_facts}_l5noise_x{args.noise_mul}"
    elif args.scenario == "l5_adv":
        all_msgs, facts = build_scenario_l5_adv(args.n_facts, args.seed)
        tag = f"N{args.n_facts}_l5adv"
    elif args.scenario == "l5_multi":
        all_msgs, facts = build_scenario_l5_multi(args.n_facts, args.seed)
        tag = f"N{args.n_facts}_l5multi"
    else:
        raise ValueError(args.scenario)

    print("="*70)
    print(f"Scaler bench: scenario={args.scenario} N={args.n_facts} turns={len(all_msgs)}")
    print(f"  model: {args.model_id}")
    print(f"  modes: {args.modes}")
    print("="*70, flush=True)

    print(f"\nScenario built: {len(all_msgs)} turns, {len(facts)} facts")

    gemma = _load_model(args.model_id, max_new_tokens=32, do_sample=False)
    gemma.load()

    sbert_fn = None
    if "cms_sbert" in args.modes:
        from server.sbert_extractor import make_extractor_fn
        sbert_fn = make_extractor_fn(
            threshold=0.2,
            enable_regex_hybrid=True,
            regex_only=args.regex_only,
        )
        mode_label = "regex_only" if args.regex_only else "regex+SBERT"
        print(f"[Load] SBERT extractor ready ({mode_label})", flush=True)

    all_records = []
    summaries = {}
    for mode in args.modes:
        with EnergyMonitor() as em:
            if mode == "baseline":
                rs = run_baseline(gemma, all_msgs, args.n_facts)
            elif mode == "cms_gemma":
                rs = run_cms(gemma, "cms_gemma", all_msgs, args.n_facts)
            elif mode == "cms_sbert":
                rs = run_cms(gemma, "cms_sbert", all_msgs, args.n_facts, extractor_fn=sbert_fn)
            else:
                print(f"[skip] {mode}")
                continue
        em_summary = em.summary()
        all_records.extend(rs)
        n_hit, n_tot, hits = score(rs, facts)
        # latency stats
        ms = sorted([r.inference_ms for r in rs])
        p50 = ms[len(ms)//2] if ms else 0
        p95 = ms[int(len(ms)*0.95)] if ms else 0
        p99 = ms[int(len(ms)*0.99)] if ms else 0
        # efficiency
        total_s = sum(r.inference_ms for r in rs)/1000
        max_cd = max(r.cd_node_count for r in rs) if rs else 0
        summaries[mode] = {
            "n_facts": args.n_facts,
            "recall": n_hit/n_tot, "n_hit": n_hit, "n_total": n_tot,
            "total_s": total_s,
            "p50_ms": p50, "p95_ms": p95, "p99_ms": p99,
            "phase3_s": sum(r.inference_ms for r in rs if r.phase=="phase3")/1000,
            "max_cd_nodes": max_cd,
            "max_prompt_chars": max(r.prompt_chars for r in rs) if rs else 0,
            **em_summary,
            "energy_J_per_correct": em_summary["gpu_energy_J"]/max(n_hit,1),
            "time_s_per_correct": total_s/max(n_hit,1),
            "hits": hits,
        }
        try:
            print(f"\n[{mode}] N={args.n_facts} recall={n_hit}/{n_tot} time={total_s:.1f}s "
                  f"gpu={em_summary['gpu_energy_J']:.0f}J cd={max_cd}", flush=True)
        except UnicodeEncodeError:
            pass
        _save(all_records, summaries, args.output, tag)

    _save(all_records, summaries, args.output, tag)
    _summary(summaries)


def _save(records, summaries, out, tag):
    csv_path = out / f"exp_scaler_{tag}.csv"
    json_path = out / f"exp_scaler_{tag}.json"
    if records:
        fields = list(asdict(records[0]).keys())
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows([asdict(r) for r in records])
        json_path.write_text(json.dumps({
            "tag": tag, "summaries": summaries,
            "records": [asdict(r) for r in records],
        }, ensure_ascii=False, indent=2), encoding="utf-8")


def _summary(summaries):
    print("\n"+"="*70+"\nFINAL SCALER SUMMARY\n"+"="*70)
    cols = ["mode","recall","total_s","gpu_J","cpu_J","p50","p95","s/corr","J/corr","cd_max"]
    print(" ".join(f"{c:<10}" for c in cols))
    print("-"*100)
    for mode, s in summaries.items():
        print(" ".join([
            f"{mode:<10}",
            f"{s['n_hit']}/{s['n_total']:<6}",
            f"{s['total_s']:>8.0f}s",
            f"{s['gpu_energy_J']:>8.0f}J",
            f"{s['cpu_energy_J_est']:>8.0f}J",
            f"{s['p50_ms']:>8.0f}ms",
            f"{s['p95_ms']:>8.0f}ms",
            f"{s['time_s_per_correct']:>6.1f}",
            f"{s['energy_J_per_correct']:>6.1f}",
            f"{s['max_cd_nodes']:>4}",
        ]))


if __name__ == "__main__":
    main()

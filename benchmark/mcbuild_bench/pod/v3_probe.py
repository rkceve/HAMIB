"""V3 (DESIGN.md §10, F4): injection probe on a ~300-token block with three planet lines.

Run inside venv_reader from the repository root:
    python -m benchmark.mcbuild_bench.pod.v3_probe --model-id RedHatAI/Qwen3.8-27B-INT4 \
        --w-grid 0,0.1,0.3,1,3 --out /workspace/mcb/runs/v3.json

For each w: the mass vector is built from the serialized CD exactly as run_reader does (planet lines
only), attention rows of the first decode step are recorded, and the share of attention that lands
on planet-text tokens is reported. Pass rules: planet_spans == 3; bias_applied_calls == 16*(n-1);
bias_skipped_sliding_calls == 0; planet share strictly increases with w; >= 3 w values give a
non-empty, non-degenerate answer. Nothing here touches Jev.
"""
from __future__ import annotations

import argparse
import json

import torch

from benchmark.bineval.arms import cd_from_records
from benchmark.bineval.run_reader import build_prompt, build_reader_mass_vector, load_reader
from communication.cd_serializer import CDSerializer

RECORDS = [
    {"node_id": "s1", "text": "Demo Minecraft server for the hackathon build agent", "level": "sun", "mass": 0.0, "parent_id": None, "created_turn": 0},
    {"node_id": "p1", "text": "The demo server runs Paper 1.21.8 on Java 21 with whitelist enabled", "level": "planet", "mass": 3.0, "parent_id": "s1", "created_turn": 0},
    {"node_id": "p2", "text": "RCON listens on port 25576 and is reachable only inside the VPN", "level": "planet", "mass": 2.0, "parent_id": "s1", "created_turn": 0},
    {"node_id": "p3", "text": "The ground surface of the flat demo world is at y=0", "level": "planet", "mass": 1.0, "parent_id": "s1", "created_turn": 0},
    {"node_id": "r1", "text": "Java 21 was chosen because Paper 1.21.8 requires it", "level": "satellite", "mass": 0.0, "parent_id": "p1", "created_turn": 0},
]
FILLER = ("### Human\nLet us continue the build. " + "The weather in the plain is calm and nothing else happened. " * 18)
QUESTION = "Which port does RCON listen on?"


def degenerate(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    words = t.split()
    return len(words) >= 6 and len(set(words)) <= 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--w-grid", default="0,0.1,0.3,1,3")
    ap.add_argument("--max-new-tokens", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quantization", choices=("none", "nf4", "fp4"), default="none")
    args = ap.parse_args()
    grid = [float(x) for x in args.w_grid.split(",")]

    cd = cd_from_records(RECORDS)
    block = CDSerializer(level_markers=True).to_context_block(cd)
    context = "<context>\n" + block + "\n" + FILLER + "\n</context>"
    prompt = build_prompt(context, QUESTION)
    llm = load_reader(args.model_id, max_new_tokens=args.max_new_tokens, w=0.0, quantization=args.quantization)
    tok = llm.tokenizer
    ids = tok(prompt)["input_ids"]
    n_sdpa = 16
    print(f"prompt tokens: {len(ids)}; planet lines in block: {block.count('[PN')}")

    rows: list[dict] = []
    for w in grid:
        llm._mass_weight = float(w)
        device = next(llm._model.parameters()).device
        vec, info = build_reader_mass_vector(ids, tok, "planet", None, w=max(w, 1e-9), prompt_text=prompt, device=device)
        planet_pos = [i for i, v in enumerate(vec.tolist()) if v > 0] if vec is not None else []
        if vec is not None:
            llm.set_mass_vector(vec if w > 0 else torch.zeros_like(vec))
        llm.start_attention_recording()
        out = llm.generate(prompt)
        recorded = llm.stop_attention_recording()
        stats = llm.mass_injection_stats()
        llm.clear_mass_vector()
        n = llm.last_generated_tokens or 0
        first_step = recorded[:n_sdpa]  # one row per sdpa layer at decode step 1
        share = None
        if first_step and planet_pos:
            shares = []
            for row in first_step:
                r = row.detach().float().cpu()
                pos = [p for p in planet_pos if p < r.shape[-1]]
                shares.append(float(r[pos].sum() / r.sum()))
            share = sum(shares) / len(shares)
        rows.append({
            "w": w, "answer": out.strip()[:60], "generated_tokens": n, "planet_spans": info.planet_spans,
            "planet_token_positions": len(planet_pos), "planet_share_step1": share,
            "bias_applied_calls": stats["bias_applied_calls"],
            "bias_skipped_prefill_calls": stats["bias_skipped_prefill_calls"],
            "bias_skipped_sliding_calls": stats["bias_skipped_sliding_calls"],
            "expected_applied": n_sdpa * max(0, n - 1) if w > 0 else 0,
            "degenerate": degenerate(out),
            "prefill_ms": llm.last_prefill_ms, "decode_ms": llm.last_decode_ms,
        })
        print(json.dumps(rows[-1], ensure_ascii=False))

    shares = [r["planet_share_step1"] for r in rows if r["planet_share_step1"] is not None]
    checks = {
        "planet_spans_3": all(r["planet_spans"] == 3 for r in rows),
        "counters_exact": all(r["bias_applied_calls"] == r["expected_applied"] for r in rows if r["w"] > 0),
        "no_sliding_skips": all(r["bias_skipped_sliding_calls"] == 0 for r in rows),
        "share_monotone": all(b > a for a, b in zip(shares, shares[1:])),
        "non_degenerate_ge3": sum(1 for r in rows if not r["degenerate"]) >= 3,
    }
    result = {"model_id": args.model_id, "prompt_tokens": len(ids), "rows": rows, "checks": checks, "pass": all(checks.values())}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, ensure_ascii=False)
    print(json.dumps(checks))
    print("V3 PASS" if result["pass"] else "V3 FAIL")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

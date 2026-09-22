"""
Analyze existing LongMemEval benchmark results (baseline + HAMIB).

Reads the already-collected JSON result files and computes:
  - CD size distribution (min, max, mean, median, percentiles)
  - Per-qtype CD size and accuracy breakdown
  - Point-biserial correlation: cd_nodes vs correct
  - Pearson correlation: cd_nodes vs latency (ms)
  - Baseline prompt_chars distribution
  - Rough compression ratio estimate

No external dependencies — stdlib only.

Usage:
    python -m experiments.analyze_existing_results \
        --hamib results/longmemeval/longmemeval_hamib_sbert.json \
        --baseline results/longmemeval/longmemeval_baseline.json
"""
from __future__ import annotations
import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def _dist_stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "p25": round(_percentile(values, 25), 2),
        "p75": round(_percentile(values, 75), 2),
        "std": round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
    }


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - 1)
    sx = statistics.stdev(xs)
    sy = statistics.stdev(ys)
    if sx == 0 or sy == 0:
        return float("nan")
    return cov / (sx * sy)


def _point_biserial(binary: list[bool], continuous: list[float]) -> float:
    n = len(binary)
    if n < 3:
        return float("nan")
    g1 = [c for b, c in zip(binary, continuous) if b]
    g0 = [c for b, c in zip(binary, continuous) if not b]
    if not g1 or not g0:
        return float("nan")
    m1 = statistics.mean(g1)
    m0 = statistics.mean(g0)
    n1 = len(g1)
    n0 = len(g0)
    sd = statistics.stdev(continuous)
    if sd == 0:
        return float("nan")
    return (m1 - m0) / sd * math.sqrt(n1 * n0 / (n * n))


def analyze(hamib_path: Path, baseline_path: Path) -> dict:
    with open(hamib_path, encoding="utf-8") as f:
        hamib_data = json.load(f)
    with open(baseline_path, encoding="utf-8") as f:
        baseline_data = json.load(f)

    h_items = hamib_data["items"]
    b_items = baseline_data["items"]

    # --- CD size distribution ---
    cd_sizes = [it["cd"] for it in h_items]
    cd_dist = _dist_stats([float(x) for x in cd_sizes])

    # --- Per-qtype breakdown ---
    by_qtype_h: dict[str, list] = defaultdict(list)
    by_qtype_b: dict[str, list] = defaultdict(list)
    for it in h_items:
        by_qtype_h[it["qtype"]].append(it)
    for it in b_items:
        by_qtype_b[it["qtype"]].append(it)

    qtype_table = {}
    for qt in sorted(set(list(by_qtype_h.keys()) + list(by_qtype_b.keys()))):
        h_group = by_qtype_h.get(qt, [])
        b_group = by_qtype_b.get(qt, [])
        h_acc = sum(1 for x in h_group if x["correct"]) / max(len(h_group), 1)
        b_acc = sum(1 for x in b_group if x["correct"]) / max(len(b_group), 1)
        h_cd_vals = [float(x["cd"]) for x in h_group]
        qtype_table[qt] = {
            "n": len(h_group),
            "hamib_acc": round(h_acc, 3),
            "baseline_acc": round(b_acc, 3),
            "ratio": round(h_acc / b_acc, 2) if b_acc > 0 else None,
            "cd_mean": round(statistics.mean(h_cd_vals), 1) if h_cd_vals else 0,
            "cd_median": round(statistics.median(h_cd_vals), 1) if h_cd_vals else 0,
        }

    # --- Correlations ---
    cd_f = [float(it["cd"]) for it in h_items]
    correct_b = [it["correct"] for it in h_items]
    ms_f = [float(it["ms"]) for it in h_items]

    r_cd_correct = _point_biserial(correct_b, cd_f)
    r_cd_ms = _pearson(cd_f, ms_f)

    # --- Baseline prompt_chars ---
    b_chars = [float(it.get("prompt_chars", 0)) for it in b_items if it.get("prompt_chars")]
    baseline_chars_dist = _dist_stats(b_chars)

    # --- Rough compression ratio estimate ---
    avg_chars_per_node = 40.0
    h_est_chars = [it["cd"] * avg_chars_per_node for it in h_items]
    h_est_chars_mean = statistics.mean(h_est_chars) if h_est_chars else 0
    b_chars_mean = statistics.mean(b_chars) if b_chars else 0
    compression_ratio_est = round(b_chars_mean / h_est_chars_mean, 2) if h_est_chars_mean > 0 else None

    # --- Aggregate accuracy check ---
    h_correct = sum(1 for it in h_items if it["correct"])
    b_correct = sum(1 for it in b_items if it["correct"])

    result = {
        "hamib_summary": {
            "n": len(h_items),
            "correct": h_correct,
            "acc": round(h_correct / max(len(h_items), 1), 3),
        },
        "baseline_summary": {
            "n": len(b_items),
            "correct": b_correct,
            "acc": round(b_correct / max(len(b_items), 1), 3),
        },
        "cd_size_distribution": cd_dist,
        "baseline_prompt_chars_distribution": baseline_chars_dist,
        "compression_ratio_estimate": {
            "assumed_chars_per_node": avg_chars_per_node,
            "hamib_est_prompt_chars_mean": round(h_est_chars_mean, 0),
            "baseline_prompt_chars_mean": round(b_chars_mean, 0),
            "ratio": compression_ratio_est,
            "note": "rough estimate — A1 enables exact measurement in future runs",
        },
        "correlations": {
            "cd_vs_correct_point_biserial_r": round(r_cd_correct, 4) if not math.isnan(r_cd_correct) else None,
            "cd_vs_ms_pearson_r": round(r_cd_ms, 4) if not math.isnan(r_cd_ms) else None,
        },
        "per_qtype": qtype_table,
    }
    return result


def _print_report(result: dict) -> None:
    print("=" * 60)
    print("LongMemEval Existing Results Analysis")
    print("=" * 60)

    hs = result["hamib_summary"]
    bs = result["baseline_summary"]
    print(f"\nHAMIB:    {hs['correct']}/{hs['n']} = {hs['acc']:.3f}")
    print(f"Baseline: {bs['correct']}/{bs['n']} = {bs['acc']:.3f}")

    cd = result["cd_size_distribution"]
    print(f"\nCD size:  min={cd['min']}  p25={cd['p25']}  median={cd['median']}"
          f"  p75={cd['p75']}  max={cd['max']}  mean={cd['mean']}  std={cd['std']}")

    bc = result["baseline_prompt_chars_distribution"]
    print(f"Baseline: min={bc['min']}  median={bc['median']}  max={bc['max']}  mean={bc['mean']} chars")

    cr = result["compression_ratio_estimate"]
    print(f"\nCompression ratio (est): {cr['ratio']}x"
          f"  (baseline {cr['baseline_prompt_chars_mean']:.0f} chars"
          f" / hamib est {cr['hamib_est_prompt_chars_mean']:.0f} chars)")

    corr = result["correlations"]
    print("\nCorrelations:")
    print(f"  cd vs correct (point-biserial r): {corr['cd_vs_correct_point_biserial_r']}")
    print(f"  cd vs ms      (Pearson r):        {corr['cd_vs_ms_pearson_r']}")

    print("\nPer-qtype breakdown:")
    print(f"  {'qtype':<30s} {'n':>4s} {'h_acc':>6s} {'b_acc':>6s} {'ratio':>6s} {'cd_med':>7s}")
    print(f"  {'-'*30} {'-'*4} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")
    for qt, v in sorted(result["per_qtype"].items()):
        ratio_str = f"{v['ratio']:.2f}" if v["ratio"] is not None else "N/A"
        print(f"  {qt:<30s} {v['n']:>4d} {v['hamib_acc']:>6.3f} {v['baseline_acc']:>6.3f}"
              f" {ratio_str:>6s} {v['cd_median']:>7.1f}")


def main():
    ap = argparse.ArgumentParser(description="Analyze existing LongMemEval results")
    ap.add_argument("--hamib", type=Path, required=True, help="Path to HAMIB result JSON")
    ap.add_argument("--baseline", type=Path, required=True, help="Path to baseline result JSON")
    ap.add_argument("--output", type=Path, default=None, help="Output JSON path (optional)")
    args = ap.parse_args()

    result = analyze(args.hamib, args.baseline)
    _print_report(result)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n-> {args.output}")


if __name__ == "__main__":
    main()

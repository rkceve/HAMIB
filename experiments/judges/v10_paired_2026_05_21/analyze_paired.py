"""
De-anonymize judge output, then compute paired McNemar test + bootstrap 95% CI
for HAMIB vs baseline accuracy on a blinded LongMemEval LLM-judge run.

Protocol:
  - paired McNemar one-sided test (HAMIB > baseline)
  - bootstrap 10000 iter for ratio + diff 95% CI
  - per-qtype breakdown by the 6 LongMemEval qtypes

Usage:
    python -m experiments.judges.v10_paired_2026_05_21.analyze_paired
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
# PRIVATE points to the de-anonymization MAPPING file used at evaluation time.
# This directory is intentionally NOT included in the public repository
# (see ../README.md "MAPPING_*.json de-anonymization keys are kept private"),
# so this script will not run unchanged on a fresh clone — it is published as a
# reference implementation of the aggregation step. To reproduce, supply the
# MAPPING file from the private source.
PRIVATE = HERE / "_private"

BENCH_CONFIG = {
    "lme": {
        "judge_out": HERE / "judge_output_lme.json",
        "mapping": PRIVATE / "MAPPING_lme_v10_paired.json",
        "report": HERE / "report_lme_v10_paired.md",
    },
}


def mcnemar_one_sided(b: int, c: int) -> float:
    """Exact one-sided McNemar test (HAMIB > baseline). b = baseline-only, c = cms-only.

    H0: P(cms_only) <= P(baseline_only).
    """
    n = b + c
    if n == 0:
        return 1.0
    # P(X >= c | n, p=0.5)
    return float(stats.binomtest(c, n, 0.5, alternative="greater").pvalue)


def bootstrap_ratio_ci(
    cms_correct: list[int],
    base_correct: list[int],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 47,
) -> tuple[float, float, float, float]:
    """Bootstrap 95% CI for both ratio (cms_acc / base_acc) and diff (cms - base)."""
    rng = np.random.default_rng(seed)
    n = len(cms_correct)
    arr_cms = np.asarray(cms_correct)
    arr_base = np.asarray(base_correct)
    ratios = np.empty(n_boot)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        c = arr_cms[idx].mean()
        b = arr_base[idx].mean()
        ratios[i] = (c / b) if b > 0 else np.nan
        diffs[i] = c - b
    lo_r = float(np.nanpercentile(ratios, 100 * alpha / 2))
    hi_r = float(np.nanpercentile(ratios, 100 * (1 - alpha / 2)))
    lo_d = float(np.nanpercentile(diffs, 100 * alpha / 2))
    hi_d = float(np.nanpercentile(diffs, 100 * (1 - alpha / 2)))
    return lo_r, hi_r, lo_d, hi_d


def analyze(bench: str) -> None:
    cfg = BENCH_CONFIG[bench]
    judge = json.load(open(cfg["judge_out"], encoding="utf-8"))
    mapping = json.load(open(cfg["mapping"], encoding="utf-8"))["mapping"]

    anon_to_label = {j["anonymous_id"]: bool(j["label"]) for j in judge["judgments"]}

    # Group by (mode, item_key) where item_key = (orig_qid, qtype) for LongMemEval
    def key(m: dict) -> tuple:
        return (m["orig_qid"], m["qtype"])

    cms_label_by_key: dict[tuple, bool] = {}
    base_label_by_key: dict[tuple, bool] = {}
    qtype_by_key: dict[tuple, str] = {}
    for m in mapping:
        k = key(m)
        lbl = anon_to_label.get(m["anonymous_id"])
        if lbl is None:
            continue
        if m["mode"] == "cms":
            cms_label_by_key[k] = lbl
        else:
            base_label_by_key[k] = lbl
        qtype_by_key[k] = m.get("qtype", "multi-session")

    paired_keys = sorted(cms_label_by_key.keys() & base_label_by_key.keys())
    cms_correct = [int(cms_label_by_key[k]) for k in paired_keys]
    base_correct = [int(base_label_by_key[k]) for k in paired_keys]
    n = len(paired_keys)

    both = sum(1 for a, b in zip(cms_correct, base_correct, strict=True) if a and b)
    cms_only = sum(1 for a, b in zip(cms_correct, base_correct, strict=True) if a and not b)
    base_only = sum(1 for a, b in zip(cms_correct, base_correct, strict=True) if b and not a)
    neither = sum(1 for a, b in zip(cms_correct, base_correct, strict=True) if not a and not b)

    cms_acc = sum(cms_correct) / n if n else 0.0
    base_acc = sum(base_correct) / n if n else 0.0
    ratio = (cms_acc / base_acc) if base_acc > 0 else float("inf")

    p_one_sided = mcnemar_one_sided(base_only, cms_only)
    lo_r, hi_r, lo_d, hi_d = bootstrap_ratio_ci(cms_correct, base_correct)

    # Per-qtype breakdown
    qtype_breakdown = defaultdict(lambda: {"n": 0, "cms": 0, "base": 0})
    for k in paired_keys:
        qt = qtype_by_key.get(k, "?")
        qtype_breakdown[qt]["n"] += 1
        if cms_label_by_key[k]:
            qtype_breakdown[qt]["cms"] += 1
        if base_label_by_key[k]:
            qtype_breakdown[qt]["base"] += 1

    # Substring labels comparison (sanity check)
    cms_substring = sum(1 for m in mapping if m["mode"] == "cms" and m.get("substring_label"))
    base_substring = sum(1 for m in mapping if m["mode"] == "baseline" and m.get("substring_label"))
    n_cms = sum(1 for m in mapping if m["mode"] == "cms")
    n_base = sum(1 for m in mapping if m["mode"] == "baseline")

    # Build report
    lines = []
    lines.append(f"# Paired LLM-judge analysis — {bench.upper()} v10 (HAMIB) vs v9 (baseline)")
    lines.append("")
    lines.append(f"Judge model: `{judge.get('judge_model', '?')}`")
    lines.append(f"Paired N: {n} (HAMIB items: {n_cms}, baseline items: {n_base})")
    lines.append("")
    lines.append("## Aggregate (LLM-judge)")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| HAMIB accuracy | {cms_acc:.3f} ({sum(cms_correct)}/{n}) |")
    lines.append(f"| baseline accuracy | {base_acc:.3f} ({sum(base_correct)}/{n}) |")
    lines.append(f"| **ratio (HAMIB / baseline)** | **{ratio:.3f}×** |")
    lines.append(f"| diff (HAMIB - baseline) | {cms_acc - base_acc:+.3f} |")
    lines.append("")
    lines.append("## Paired contingency")
    lines.append("")
    lines.append("| | baseline correct | baseline wrong |")
    lines.append("|---|---|---|")
    lines.append(f"| **HAMIB correct** | {both} | {cms_only} |")
    lines.append(f"| **HAMIB wrong** | {base_only} | {neither} |")
    lines.append("")
    lines.append("## Statistical tests")
    lines.append("")
    lines.append("| test | value |")
    lines.append("|---|---|")
    lines.append(f"| McNemar one-sided p (HAMIB > baseline) | {p_one_sided:.4f} |")
    lines.append(f"| Bootstrap 95% CI of ratio | [{lo_r:.3f}, {hi_r:.3f}] |")
    lines.append(f"| Bootstrap 95% CI of diff | [{lo_d:+.3f}, {hi_d:+.3f}] |")
    sig = "★ significant (p<0.05)" if p_one_sided < 0.05 else "not significant (p>=0.05)"
    contains_one = "contains 1.0" if lo_r <= 1.0 <= hi_r else "does NOT contain 1.0"
    lines.append(f"| interpretation | {sig}; ratio CI {contains_one} |")
    lines.append("")
    lines.append("## Per-qtype breakdown")
    lines.append("")
    lines.append("| qtype | n | base correct | cms correct | base acc | cms acc | ratio |")
    lines.append("|---|---|---|---|---|---|---|")
    for qt in sorted(qtype_breakdown):
        b = qtype_breakdown[qt]
        b_acc = b["base"] / b["n"] if b["n"] else 0.0
        c_acc = b["cms"] / b["n"] if b["n"] else 0.0
        rr = (c_acc / b_acc) if b_acc > 0 else float("inf")
        lines.append(f"| {qt} | {b['n']} | {b['base']} | {b['cms']} | {b_acc:.3f} | {c_acc:.3f} | {rr:.2f}× |")
    lines.append("")
    lines.append("## Substring (sanity check, from bench's own scoring)")
    lines.append("")
    lines.append("| mode | substring correct | total | acc |")
    lines.append("|---|---|---|---|")
    lines.append(f"| HAMIB | {cms_substring} | {n_cms} | {cms_substring/n_cms:.3f} |")
    lines.append(f"| baseline | {base_substring} | {n_base} | {base_substring/n_base:.3f} |")
    sub_ratio = (cms_substring / n_cms) / (base_substring / n_base) if base_substring > 0 else float("inf")
    lines.append(f"| substring ratio (HAMIB / baseline) | {sub_ratio:.3f}× | | |")
    lines.append("")
    lines.append("## Comparison: substring vs LLM-judge")
    lines.append("")
    lines.append("| metric | substring | LLM-judge |")
    lines.append("|---|---|---|")
    lines.append(f"| HAMIB acc | {cms_substring/n_cms:.3f} | {cms_acc:.3f} |")
    lines.append(f"| baseline acc | {base_substring/n_base:.3f} | {base_acc:.3f} |")
    lines.append(f"| ratio | {sub_ratio:.3f}× | {ratio:.3f}× |")
    delta = ratio - sub_ratio
    if abs(delta) >= 0.1:
        direction = "LLM-judge above substring (paraphrase credit)" if delta > 0 else "LLM-judge below substring"
        lines.append(f"| delta | | {delta:+.3f}× {direction} |")
    lines.append("")

    report = "\n".join(lines)
    open(cfg["report"], "w", encoding="utf-8").write(report)
    print(report)
    print(f"\n→ Written to {cfg['report'].relative_to(HERE.parent.parent.parent)}")


def main() -> None:
    bench = sys.argv[1] if len(sys.argv) >= 2 else "lme"
    if bench not in BENCH_CONFIG:
        print(f"Usage: python -m {Path(__file__).stem} [lme]", file=sys.stderr)
        sys.exit(1)
    analyze(bench)


if __name__ == "__main__":
    main()

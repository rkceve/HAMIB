"""V7: score the main run locally (DESIGN.md 10, DECISIONS B5 / H26 / H29).

For every proposed cell directory ``<main>/proposed_W<W>_w<w>`` the questions listed in its
``questions_subset.json`` (H22 d) are scored for the cell AND for arm A (``<main>/A_full``, the
single full-transcript run -- H29) with ``score_binary.score_condition`` (tier 1 only, strict
short-needle rule, ``judge_none``).  Paired statistics per cell (B5): exact one-sided McNemar
(proposed > A and A > proposed) and a 10000-iteration paired bootstrap (seed 47) of the pass-rate
difference and ratio.  Reader compute per question from ``meta.json`` (E1):
``attn_flops_prefill + attn_flops_decode``, ``wall_ms_total``, ``energy_joules``, prompt tokens --
means over the same subset for both arms, plus the ratio proposed / A.

Usage:
    python -m benchmark.mcbuild_bench.score_cells --main <dir> --questions data/questions.json \
        --out results/<tag>/scores.json --md results/<tag>/scores.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

from benchmark.bineval.score_binary import judge_none, load_questions, score_condition

CELL_RE = re.compile(r"^proposed_W(?P<W>\d+)_w(?P<w>[0-9.]+)$")
COMPUTE_KEYS = ("attn_flops_total", "wall_ms_total", "energy_joules", "prompt_tokens")


def mcnemar_one_sided(b: int, c: int) -> float:
    """Exact one-sided McNemar, H0: P(c) <= P(b); b = A-only passes, c = proposed-only passes
    (same as experiments/judges/v10_paired_2026_05_21/analyze_paired.py)."""
    n = b + c
    if n == 0:
        return 1.0
    return float(stats.binomtest(c, n, 0.5, alternative="greater").pvalue)


def bootstrap_ci(prop: list[int], base: list[int], n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 47) -> dict:
    rng = np.random.default_rng(seed)
    p, b = np.asarray(prop, dtype=float), np.asarray(base, dtype=float)
    n = len(p)
    idx = rng.integers(0, n, (n_boot, n))
    pm, bm = p[idx].mean(axis=1), b[idx].mean(axis=1)
    diffs = pm - bm
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(bm > 0, pm / bm, np.nan)
    q = (100 * alpha / 2, 100 * (1 - alpha / 2))
    return {
        "diff_ci95": [float(np.percentile(diffs, q[0])), float(np.percentile(diffs, q[1]))],
        "ratio_ci95": [float(np.nanpercentile(ratios, q[0])), float(np.nanpercentile(ratios, q[1]))],
        "n_boot": n_boot, "seed": seed,
    }


def load_cell(cell_dir: Path) -> tuple[dict[str, str], dict[str, dict], dict]:
    """(answers by qid, per-question compute by qid, meta) of one run_arms output dir."""
    answers = json.loads((cell_dir / "answers.json").read_text(encoding="utf-8"))
    meta = json.loads((cell_dir / "meta.json").read_text(encoding="utf-8"))
    compute: dict[str, dict] = {}
    for qid, pq in meta["per_question"].items():
        compute[qid] = {
            "attn_flops_total": int(pq["attn_flops_prefill"]) + int(pq["attn_flops_decode"]),
            "wall_ms_total": float(pq["wall_ms_total"]),
            "energy_joules": None if pq.get("energy_joules") is None else float(pq["energy_joules"]),
            "prompt_tokens": int(pq["prompt_tokens"]),
        }
    return answers, compute, meta


def mean_or_none(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def compute_summary(compute: dict[str, dict], qids: list[str]) -> dict:
    out = {}
    for k in COMPUTE_KEYS:
        out[k + "_mean"] = mean_or_none([compute[q][k] for q in qids if q in compute])
        out[k + "_missing"] = sum(1 for q in qids if q not in compute or compute[q][k] is None)
    return out


def pass_vector(scored: dict, qids: list[str]) -> list[int]:
    verdict = {it["qid"]: it["verdict"] for it in scored["items"]}
    return [1 if verdict.get(q) == "pass" else 0 for q in qids]


def score_cell(cell_dir: Path, a_dir: Path, questions: list[dict]) -> dict:
    subset = json.loads((cell_dir / "questions_subset.json").read_text(encoding="utf-8"))
    qids = list(subset["qids"])
    by_qid = {q["qid"]: q for q in questions}
    missing = [q for q in qids if q not in by_qid]
    if missing:
        raise ValueError(f"{cell_dir.name}: subset qids not in the questions file: {missing[:5]}")
    scored_qs = [by_qid[q] for q in qids]
    p_ans, p_comp, p_meta = load_cell(cell_dir)
    a_ans, a_comp, a_meta = load_cell(a_dir)
    for name, ans in (("proposed", p_ans), ("A", a_ans)):
        lacking = [q for q in qids if q not in ans]
        if lacking:
            raise ValueError(f"{cell_dir.name}: arm {name} has no answer for {lacking[:5]}")
    p_sc = score_condition(scored_qs, {}, p_ans, judge_none, strict_short=True)
    a_sc = score_condition(scored_qs, {}, a_ans, judge_none, strict_short=True)
    pv, av = pass_vector(p_sc, qids), pass_vector(a_sc, qids)
    b = sum(1 for x, y in zip(pv, av) if y == 1 and x == 0)  # A only
    c = sum(1 for x, y in zip(pv, av) if x == 1 and y == 0)  # proposed only
    pc, ac = compute_summary(p_comp, qids), compute_summary(a_comp, qids)
    ratios = {}
    for k in COMPUTE_KEYS:
        pm, am = pc[k + "_mean"], ac[k + "_mean"]
        ratios[k] = None if pm is None or am is None or am == 0 else pm / am
    m = CELL_RE.match(cell_dir.name)
    assert m is not None
    return {
        "cell": cell_dir.name, "W": int(m["W"]), "w": float(m["w"]),
        "n_questions": len(qids), "dropped_in_window": list(subset.get("dropped", [])),
        "first_recent_rt": subset.get("first_recent_rt"),
        "questions_subset_sha256": hashlib.sha256(
            (cell_dir / "questions_subset.json").read_bytes()).hexdigest(),
        "proposed": {"pass": p_sc["aggregate"]["pass"], "fail": p_sc["aggregate"]["fail"],
                     "indeterminate": p_sc["aggregate"]["indeterminate"],
                     "pass_rate": p_sc["aggregate"]["pass_rate_tier1"], "compute": pc,
                     "items": p_sc["items"], "window_tokens": p_meta.get("context_tokens")},
        "A": {"pass": a_sc["aggregate"]["pass"], "fail": a_sc["aggregate"]["fail"],
              "indeterminate": a_sc["aggregate"]["indeterminate"],
              "pass_rate": a_sc["aggregate"]["pass_rate_tier1"], "compute": ac,
              "items": a_sc["items"], "window_tokens": a_meta.get("context_tokens")},
        "paired": {
            "both_pass": sum(1 for x, y in zip(pv, av) if x == 1 and y == 1),
            "both_fail": sum(1 for x, y in zip(pv, av) if x == 0 and y == 0),
            "A_only": b, "proposed_only": c,
            "mcnemar_p_proposed_gt_A": mcnemar_one_sided(b, c),
            "mcnemar_p_A_gt_proposed": mcnemar_one_sided(c, b),
            "bootstrap": bootstrap_ci(pv, av),
        },
        "compute_ratio_proposed_over_A": ratios,
        "pending_tier2": {"proposed": p_sc["pending_tier2"], "A": a_sc["pending_tier2"]},
    }


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def markdown(cells: list[dict]) -> str:
    lines = ["| W | w | n | proposed pass | A pass | diff (95% CI) | McNemar p (prop>A / A>prop) "
             "| attn FLOPs ratio | wall ratio | energy ratio | prompt tokens prop / A |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in cells:
        pr, ar = r["proposed"], r["A"]
        bs = r["paired"]["bootstrap"]["diff_ci95"]
        cr = r["compute_ratio_proposed_over_A"]
        lines.append(
            f"| {r['W']} | {r['w']} | {r['n_questions']} | {pr['pass']} ({pr['pass_rate']:.3f}) "
            f"| {ar['pass']} ({ar['pass_rate']:.3f}) | {pr['pass_rate'] - ar['pass_rate']:+.3f} "
            f"({bs[0]:+.3f}, {bs[1]:+.3f}) | {r['paired']['mcnemar_p_proposed_gt_A']:.3f} / "
            f"{r['paired']['mcnemar_p_A_gt_proposed']:.3f} | {_fmt(cr['attn_flops_total'])} "
            f"| {_fmt(cr['wall_ms_total'])} | {_fmt(cr['energy_joules'])} "
            f"| {pr['compute']['prompt_tokens_mean']:.0f} / {ar['compute']['prompt_tokens_mean']:.0f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True, help="directory holding proposed_W*_w* and A_full")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--a-dir", default="A_full")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default=None)
    args = ap.parse_args(argv)
    main_dir = Path(args.main)
    questions = load_questions(Path(args.questions))

    def key(d: Path) -> tuple[int, float]:
        m = CELL_RE.match(d.name)
        assert m is not None
        return int(m["W"]), float(m["w"])

    cell_dirs = sorted((d for d in main_dir.iterdir() if d.is_dir() and CELL_RE.match(d.name)), key=key)
    if not cell_dirs:
        raise SystemExit(f"no proposed_W*_w* directories under {main_dir}")
    cells = [score_cell(d, main_dir / args.a_dir, questions) for d in cell_dirs]
    payload = {"kind": "mcbuild_bench_scores", "main_dir": str(main_dir), "a_dir": args.a_dir,
               "questions_sha256": hashlib.sha256(Path(args.questions).read_bytes()).hexdigest(),
               "cells": cells}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    md = markdown(cells)
    if args.md:
        Path(args.md).write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

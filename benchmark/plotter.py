"""
Plotter: outputs benchmark results as graphs and CSV.

Output graphs:
  1. Server input tokens vs turn number (HAMIB vs Baseline)
  2. Server inference time vs turn number (HAMIB vs Baseline)
  3. Recall consistency score (HAMIB vs Baseline, recall turns only)
  4. Client-side CD build time vs turn number (HAMIB only)
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from benchmark.runner import BenchmarkResult


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "benchmark_results"


def save_all(cms: BenchmarkResult, baseline: BenchmarkResult) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _save_csv(cms, OUTPUT_DIR / "cms_results.csv")
    _save_csv(baseline, OUTPUT_DIR / "baseline_results.csv")
    _save_json(cms, baseline, OUTPUT_DIR / "summary.json")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("HAMIB vs Baseline benchmark results", fontsize=14)

    _plot_tokens(axes[0][0], cms, baseline)
    _plot_inference_time(axes[0][1], cms, baseline)
    _plot_recall(axes[1][0], cms, baseline)
    _plot_cd_build(axes[1][1], cms)

    plt.tight_layout()
    out = OUTPUT_DIR / "benchmark_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Plotter] saved graph: {out}")
    print(f"[Plotter] saved CSV:   {OUTPUT_DIR}/{{cms,baseline}}_results.csv")
    print(f"[Plotter] saved JSON:  {OUTPUT_DIR}/summary.json")


# -- Individual graphs ------------------------------------------------

def _plot_tokens(ax, cms: BenchmarkResult, baseline: BenchmarkResult):
    ax.set_title("Server input tokens")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Tokens")
    _plot_line(ax, cms, "input_tokens", "HAMIB", "tab:blue")
    _plot_line(ax, baseline, "input_tokens", "Baseline", "tab:orange")
    _mark_recall_turns(ax, cms)
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))


def _plot_inference_time(ax, cms: BenchmarkResult, baseline: BenchmarkResult):
    ax.set_title("Server inference time (ms)")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Inference time (ms)")
    _plot_line(ax, cms, "inference_ms", "HAMIB", "tab:blue")
    _plot_line(ax, baseline, "inference_ms", "Baseline", "tab:orange")
    _mark_recall_turns(ax, cms)
    ax.legend()


def _plot_recall(ax, cms: BenchmarkResult, baseline: BenchmarkResult):
    ax.set_title("Recall consistency (recall turns only)")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Correct (1=hit / 0=miss)")

    cms_recall = [(t.turn_id, t.recall_hit) for t in cms.turns if t.turn_type == "recall"]
    base_recall = [(t.turn_id, t.recall_hit) for t in baseline.turns if t.turn_type == "recall"]

    if cms_recall:
        xs, ys = zip(*cms_recall)
        ax.scatter(xs, ys, label="HAMIB", color="tab:blue", s=120, zorder=5)
        ax.plot(xs, ys, color="tab:blue", alpha=0.5)
    if base_recall:
        xs, ys = zip(*base_recall)
        ax.scatter(xs, [y - 0.05 for y in ys], label="Baseline", color="tab:orange",
                   s=120, marker="^", zorder=5)
        ax.plot(xs, ys, color="tab:orange", alpha=0.5)

    # Text annotation of the accuracy summary
    if cms_recall:
        acc = sum(y for _, y in cms_recall) / len(cms_recall)
        ax.text(0.02, 0.95, f"HAMIB accuracy: {acc:.0%}", transform=ax.transAxes,
                color="tab:blue", va="top")
    if base_recall:
        acc = sum(y for _, y in base_recall) / len(base_recall)
        ax.text(0.02, 0.85, f"Baseline accuracy: {acc:.0%}", transform=ax.transAxes,
                color="tab:orange", va="top")

    ax.set_ylim(-0.2, 1.3)
    ax.set_yticks([0, 1])
    ax.legend()


def _plot_cd_build(ax, cms: BenchmarkResult):
    ax.set_title("Client CD build time (ms)  (HAMIB only)")
    ax.set_xlabel("Turn")
    ax.set_ylabel("CD build time (ms)")
    xs = [t.turn_id for t in cms.turns]
    ys = [t.cd_build_ms for t in cms.turns]
    ax.bar(xs, ys, color="tab:green", alpha=0.7, label="CD build (client)")
    ax.legend()
    note = "This time is client-side load only; it does not use the server"
    ax.text(0.02, 0.95, note, transform=ax.transAxes, fontsize=7, va="top", color="gray")


# -- Utilities --------------------------------------------------------

def _plot_line(ax, result: BenchmarkResult, field: str, label: str, color: str):
    xs = [t.turn_id for t in result.turns]
    ys = [getattr(t, field) for t in result.turns]
    ax.plot(xs, ys, label=label, color=color, marker="o", markersize=3)


def _mark_recall_turns(ax, result: BenchmarkResult):
    for t in result.turns:
        if t.turn_type == "recall":
            ax.axvline(t.turn_id, color="red", linestyle="--", alpha=0.3, linewidth=0.8)


def _save_csv(result: BenchmarkResult, path: Path):
    fields = [
        "turn_id", "turn_type", "input_tokens", "inference_ms",
        "peak_memory_mb", "cd_build_ms", "recall_hit", "fact_value",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t in result.turns:
            w.writerow({k: getattr(t, k) for k in fields})


def _save_json(cms: BenchmarkResult, baseline: BenchmarkResult, path: Path):
    def recall_acc(r: BenchmarkResult) -> float:
        hits = [t.recall_hit for t in r.turns if t.turn_type == "recall"]
        return round(sum(hits) / len(hits), 3) if hits else 0.0

    summary = {
        "cms": {
            "recall_accuracy": recall_acc(cms),
            "avg_input_tokens": round(np.mean([t.input_tokens for t in cms.turns]), 1),
            "avg_inference_ms": round(np.mean([t.inference_ms for t in cms.turns]), 1),
            "avg_cd_build_ms":  round(np.mean([t.cd_build_ms  for t in cms.turns]), 1),
        },
        "baseline": {
            "recall_accuracy": recall_acc(baseline),
            "avg_input_tokens": round(np.mean([t.input_tokens for t in baseline.turns]), 1),
            "avg_inference_ms": round(np.mean([t.inference_ms for t in baseline.turns]), 1),
            "avg_cd_build_ms":  0.0,
        },
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

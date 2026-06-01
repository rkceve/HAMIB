"""
Benchmark execution entry point.

Usage (from the hamib_prototype/ directory):
  python -m benchmark.run_benchmark
  python -m benchmark.run_benchmark --server http://192.168.x.x:8080
  python -m benchmark.run_benchmark --mode hamib        # HAMIB only
  python -m benchmark.run_benchmark --mode baseline   # Baseline only

Output:
  benchmark_results/
    hamib_results.csv
    baseline_results.csv
    summary.json
    benchmark_comparison.png
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmark.runner import BenchmarkRunner
from benchmark.plotter import save_all


def main():
    parser = argparse.ArgumentParser(description="HAMIB vs Baseline benchmark")
    parser.add_argument("--server", default=None,
                        help="server URL (e.g. http://192.168.1.10:8080)")
    parser.add_argument("--mode", choices=["hamib", "baseline", "both"], default="both",
                        help="run mode (default: both)")
    args = parser.parse_args()

    runner = BenchmarkRunner(server_url=args.server)

    # Check server connectivity
    import httpx
    try:
        url = args.server or f"http://localhost:{8080}"
        r = httpx.get(f"{url}/health", timeout=5.0)
        r.raise_for_status()
        print(f"[OK] server connection confirmed: {r.json()}")
    except Exception as e:
        print(f"[ERROR] cannot connect to server: {e}")
        sys.exit(1)

    hamib_result = None
    baseline_result = None

    print("\n" + "=" * 60)
    if args.mode in ("hamib", "both"):
        print("■ HAMIB mode start")
        print("=" * 60)
        hamib_result = runner.run_hamib()

    print("\n" + "=" * 60)
    if args.mode in ("baseline", "both"):
        print("■ Baseline mode start")
        print("=" * 60)
        baseline_result = runner.run_baseline()

    runner.close()

    # Save results and draw graphs
    if hamib_result and baseline_result:
        save_all(hamib_result, baseline_result)
        _print_summary(hamib_result, baseline_result)
    elif hamib_result:
        from benchmark.plotter import OUTPUT_DIR, _save_csv
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        _save_csv(hamib_result, OUTPUT_DIR / "hamib_results.csv")
        print("\nHAMIB run only. Please run Baseline additionally.")
    elif baseline_result:
        from benchmark.plotter import OUTPUT_DIR, _save_csv
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        _save_csv(baseline_result, OUTPUT_DIR / "baseline_results.csv")
        print("\nBaseline run only. Please run HAMIB additionally.")


def _print_summary(hamib, baseline):
    import numpy as np

    print("\n" + "=" * 60)
    print("■ Benchmark summary")
    print("=" * 60)

    def recall_acc(r):
        hits = [t.recall_hit for t in r.turns if t.turn_type == "recall"]
        return sum(hits) / len(hits) if hits else 0.0

    hamib_acc = recall_acc(hamib)
    base_acc = recall_acc(baseline)
    hamib_tokens = np.mean([t.input_tokens for t in hamib.turns])
    base_tokens = np.mean([t.input_tokens for t in baseline.turns])
    hamib_ms = np.mean([t.inference_ms for t in hamib.turns])
    base_ms = np.mean([t.inference_ms for t in baseline.turns])
    hamib_cd = np.mean([t.cd_build_ms for t in hamib.turns])

    print(f"{'':25} {'HAMIB':>12} {'Baseline':>12}")
    print(f"  {'recall accuracy':23} {hamib_acc:>11.0%} {base_acc:>11.0%}")
    print(f"  {'avg input tokens':21} {hamib_tokens:>12,.0f} {base_tokens:>12,.0f}")
    print(f"  {'avg inference (ms)':22} {hamib_ms:>12,.0f} {base_ms:>12,.0f}")
    print(f"  {'client CD build (ms)':19} {hamib_cd:>12,.0f} {'N/A':>12}")
    print()
    token_reduction = (base_tokens - hamib_tokens) / base_tokens * 100
    print(f"  -> server input token reduction: {token_reduction:+.1f}%")
    print(f"  -> recall accuracy diff:         {(hamib_acc - base_acc):+.0%}")
    print()


if __name__ == "__main__":
    main()

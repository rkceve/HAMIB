"""
Compute observed agreement and Cohen's kappa between the two bundled
LongMemEval judges (Claude Opus 4.7 and GPT-5) on a per-item basis.

The two judge outputs are paired on `anonymous_id`. This script does NOT
require the private MAPPING files because agreement is computed directly on
the labels the two judges produced for the same blinded items.

Run from the repository root:
    python -m experiments.judges.analyze_inter_judge_agreement
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

# Reconfigure stdout to UTF-8 on Windows consoles (cp932 default) so that
# em-dash and other Unicode characters in the printed report do not crash.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
OPUS_OUT = HERE / "v10_paired_2026_05_21" / "judge_output_lme.json"
GPT5_OUT = HERE / "codex_gpt5_2026_05_25" / "judge_output_lme.json"
REPORT = HERE / "report_inter_judge_agreement.md"


def _load_labels(path: Path) -> tuple[str, dict[str, bool]]:
    data = json.load(open(path, encoding="utf-8"))
    judge_model = data.get("judge_model", "?")
    labels = {j["anonymous_id"]: bool(j["label"]) for j in data["judgments"]}
    return judge_model, labels


def cohens_kappa(a: list[int], b: list[int]) -> float:
    """Cohen's kappa for two equal-length binary label lists."""
    n = len(a)
    if n == 0:
        return 0.0
    p_o = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    p_a_true = sum(a) / n
    p_b_true = sum(b) / n
    p_e = p_a_true * p_b_true + (1 - p_a_true) * (1 - p_b_true)
    if abs(1 - p_e) < 1e-12:
        return 0.0
    return (p_o - p_e) / (1 - p_e)


def main() -> None:
    opus_model, opus_labels = _load_labels(OPUS_OUT)
    gpt5_model, gpt5_labels = _load_labels(GPT5_OUT)

    common = sorted(opus_labels.keys() & gpt5_labels.keys())
    n_common = len(common)
    n_opus_only = len(opus_labels.keys() - gpt5_labels.keys())
    n_gpt5_only = len(gpt5_labels.keys() - opus_labels.keys())

    opus = [int(opus_labels[k]) for k in common]
    gpt5 = [int(gpt5_labels[k]) for k in common]

    agree = sum(1 for a, b in zip(opus, gpt5, strict=True) if a == b)
    observed_agreement = agree / n_common if n_common else 0.0
    kappa = cohens_kappa(opus, gpt5)

    both_true = sum(1 for a, b in zip(opus, gpt5, strict=True) if a and b)
    both_false = sum(1 for a, b in zip(opus, gpt5, strict=True) if not a and not b)
    opus_only_true = sum(1 for a, b in zip(opus, gpt5, strict=True) if a and not b)
    gpt5_only_true = sum(1 for a, b in zip(opus, gpt5, strict=True) if b and not a)

    lines = []
    lines.append("# Inter-judge agreement — LongMemEval")
    lines.append("")
    lines.append(f"Judge A: `{opus_model}` (from `v10_paired_2026_05_21/judge_output_lme.json`)")
    lines.append(f"Judge B: `{gpt5_model}` (from `codex_gpt5_2026_05_25/judge_output_lme.json`)")
    lines.append("")
    lines.append("## Pairing")
    lines.append("")
    lines.append("| | count |")
    lines.append("|---|---|")
    lines.append(f"| items judged by both (anonymous_id present in both files) | {n_common} |")
    lines.append(f"| items judged only by Judge A | {n_opus_only} |")
    lines.append(f"| items judged only by Judge B | {n_gpt5_only} |")
    lines.append("")
    lines.append("## Agreement")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| observed agreement | {observed_agreement:.4f} ({agree}/{n_common}) |")
    lines.append(f"| Cohen's kappa | {kappa:.4f} |")
    lines.append("")
    lines.append("## 2x2 contingency (Judge A vs Judge B labels)")
    lines.append("")
    lines.append("| | B = true | B = false |")
    lines.append("|---|---|---|")
    lines.append(f"| **A = true**  | {both_true} | {opus_only_true} |")
    lines.append(f"| **A = false** | {gpt5_only_true} | {both_false} |")
    lines.append("")

    report = "\n".join(lines)
    open(REPORT, "w", encoding="utf-8").write(report)
    print(report)
    print(f"\n→ Written to {REPORT.relative_to(HERE.parent.parent)}")


if __name__ == "__main__":
    main()

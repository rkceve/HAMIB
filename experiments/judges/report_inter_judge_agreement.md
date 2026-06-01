# Inter-judge agreement — LongMemEval

Judge A: `claude-opus-4-7` (from `v10_paired_2026_05_21/judge_output_lme.json`)
Judge B: `gpt-5` (from `codex_gpt5_2026_05_25/judge_output_lme.json`)

## Pairing

| | count |
|---|---|
| items judged by both (anonymous_id present in both files) | 1000 |
| items judged only by Judge A | 0 |
| items judged only by Judge B | 0 |

## Agreement

| metric | value |
|---|---|
| observed agreement | 0.9130 (913/1000) |
| Cohen's kappa | 0.7801 |

## 2x2 contingency (Judge A vs Judge B labels)

| | B = true | B = false |
|---|---|---|
| **A = true**  | 228 | 42 |
| **A = false** | 45 | 685 |

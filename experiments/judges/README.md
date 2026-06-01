# judges — 9-channel isolation LLM-judge protocol

This directory contains the protocol and implementation for using an LLM as a
judge (LLM-as-judge) while physically preventing the judge from inferring the
identity of the systems under evaluation (HAMIB vs baseline).

The same protocol was run independently with **two judge models** — Claude Opus
4.7 (`v10_paired_2026_05_21/`) and GPT-5 (`codex_gpt5_2026_05_25/`). Both
reproduce the LongMemEval result (ratio 1.308× and 1.314× respectively,
McNemar p<0.005), which is what makes the headline number robust to the choice
of judge.

## What the 9 channels isolate

| # | Channel | Isolation method |
|---|---|---|
| 1 | Chat history | The judge runs in a fresh context and does not inherit any parent chat history |
| 2 | JUDGE_PROMPT content | Names of the systems under evaluation ("HAMIB", "baseline", "v10") never appear in the prompt |
| 3 | Input item field names | Identifying fields such as `mode` are excluded from the input JSON |
| 4 | meta fields | Hints the judge could use to infer identity (model name, run timestamp) are excluded from the input JSON |
| 5 | qid | Any qid that maps back to the system under evaluation is replaced with an `anonymous_id` (e.g. `lme_item_0001`) |
| 6 | anonymous_id order | The original output order is not preserved; items are shuffled with a fixed seed (LongMemEval: seed=46) |
| 7 | de-anonymization key | The `MAPPING_*.json` files are kept private and outside this repository, so the judge can never read them |
| 8 | Other files in the repo | The JUDGE_PROMPT explicitly forbids reading README files or any other file |
| 9 | Spawn-time prompt | The process that launches the judge conveys nothing about the evaluation goal or expected result (curiosity is explicitly flagged as bias) |

## Directory layout

```
experiments/judges/
├── README.md                              # this file
├── analyze_inter_judge_agreement.py       # observed agreement + Cohen's kappa across both judges
├── check_input_markers.py                 # scans judge_input_*.json for HAMIB-only response markers
├── v10_paired_2026_05_21/                 # Claude Opus 4.7 judge
│   ├── prepare_lme_v10_blinded.py         # generate blinded LongMemEval input
│   ├── analyze_paired.py                  # McNemar + bootstrap CI statistics
│   ├── JUDGE_PROMPT_lme.txt               # judge instructions (LongMemEval, 6 qtypes)
│   ├── judge_input_lme.json               # blinded input
│   ├── judge_output_lme.json              # judge output
│   └── report_lme_v10_paired.md           # aggregated report
└── codex_gpt5_2026_05_25/                 # GPT-5 judge (independent re-judging)
    ├── analyze_paired_gpt5.py
    ├── JUDGE_PROMPT_lme.txt
    ├── judge_input_lme.json
    ├── judge_output_lme.json
    ├── report_lme_gpt5.md
    └── CODEX_RUN_INSTRUCTIONS.md          # how the GPT-5 run was performed
```

## Where the data lives

| Purpose | Location |
|---|---|
| Raw model outputs (HAMIB and baseline, LongMemEval) | `../../results/longmemeval/` |
| Claude Opus judge inputs / outputs / reports | `v10_paired_2026_05_21/` |
| GPT-5 judge inputs / outputs / reports | `codex_gpt5_2026_05_25/` |

The quickest way to see the headline numbers is the `report_*.md` files. The
`judge_input_*.json` / `judge_output_*.json` pairs let you re-aggregate from
the raw judgments.

**Note**: the `MAPPING_*.json` de-anonymization keys are **not** included in
this repository (they are kept private). The published reports are
deterministic given the judge model.

## Inter-judge agreement

`analyze_inter_judge_agreement.py` computes observed agreement and Cohen's
kappa across the two judges by pairing on `anonymous_id`. It does not need the
private MAPPING files. Run from the repository root:

```bash
python -m experiments.judges.analyze_inter_judge_agreement
```

## Known caveat — response-marker leakage

In **1.40%** of LongMemEval items (14 of 1000), the underlying LLM
(Llama 3.3 70B) echoed HAMIB-specific input scaffolding tokens (`[PN1.0]`,
`[PN0.5]`, `<CONTEXT>`) verbatim into its own response field. Because the
baseline prompt never contains those tokens, an attentive judge could in
principle identify those 14 items as HAMIB by their response alone, partially
violating channel 3 of the isolation protocol on those items. The published
`judge_output_lme.json` files reflect judging the original (un-sanitized)
inputs.

`check_input_markers.py` re-discovers these items so a reader can verify the
count independently, and `prepare_lme_v10_blinded.py` now asserts that no
such markers appear in future input regenerations. Excluding the 14 affected
items leaves the ratio direction unchanged. See `../../README.md` "Known
caveat" for the user-facing disclosure.

## These scripts are reference implementations

`prepare_*.py` and `analyze_paired*.py` were written against the original repo
layout and the private `MAPPING_*.json` keys, so they will not run unchanged
in this public repository. They are published so you can read exactly how the
9-channel isolation is implemented. To run a full reproduction you would need
to either rewrite the path references or reproduce the original layout — see
the contact in `../../README.md`.

`analyze_inter_judge_agreement.py` and `check_input_markers.py` do NOT depend
on the private MAPPING files and run as-is on the public repository contents.

## Statistical methods

`analyze_paired*.py` compute:

- **McNemar one-sided test** — an exact computation via `scipy.stats.binomtest`
  of whether HAMIB > baseline on the paired comparison is non-random
- **Bootstrap 95% CI** — confidence intervals for the ratio (HAMIB acc / baseline
  acc) and the difference (HAMIB acc - baseline acc) over 10,000 iterations
- **per-qtype breakdown** — LongMemEval is broken down by its 6 qtypes

## Reusing this protocol for other research

For any LLM-judge evaluation, isolating all 9 channels is a precondition for
reliability. In particular:

1. Never reveal "which is the new method" inside the JUDGE_PROMPT
2. Do not include evaluated model names, versions, or codebase names in the input JSON
3. Keep the de-anonymization key physically invisible to the judge
4. Pass nothing about the evaluation goal or expected result from the spawning process
5. Fix the judge model (this protocol was verified to run deterministically with Claude Opus 4.7 and, independently, GPT-5)
6. Scan response fields for system-specific tokens before publishing (see `check_input_markers.py`)

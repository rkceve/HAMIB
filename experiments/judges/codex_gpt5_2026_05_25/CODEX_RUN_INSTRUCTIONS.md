# Codex GPT-5 judge run instructions

Use ChatGPT Plus + Codex CLI to have GPT-5 judge the same blinded input, and aggregate it independently of the Claude Opus 4.7 judgments (`v10_paired_2026_05_21`). The goal is to remove the concern of depending on a single judge model.

## Identical conditions — most important

The `JUDGE_PROMPT_lme.txt` in this directory is **character-for-character identical** to the same-named file in `v10_paired_2026_05_21` used for Claude Opus 4.7. The only difference is the input/output file paths (`v10_paired_2026_05_21` → `codex_gpt5_2026_05_25`). You can re-confirm this with:

```powershell
diff ..\v10_paired_2026_05_21\JUDGE_PROMPT_lme.txt JUDGE_PROMPT_lme.txt
# -> correct if only the two path lines (input/output) appear as differences
```

Changing even a single character of the prompt body (isolation rules, rubric, output spec, judging rules) breaks the identical-conditions experiment. Do not edit it.

Note that the JUDGE_PROMPT contains text forbidding Claude Code-specific tool names (Glob, Grep, LS); leave this unchanged too, for identical conditions. For GPT-5 it is a harmless "do not use tools that do not exist" instruction.

## Prerequisites

- ChatGPT Plus ($20/month) subscription
- Codex CLI installed (`npm install -g @openai/codex` or `winget install OpenAI.Codex`)
- `codex --version` works from the terminal
- All input files are local to this PC (this directory)

## Working directory

```
experiments/judges/codex_gpt5_2026_05_25
```

| File | Contents |
|---|---|
| `judge_input_lme.json` (~470 KB) | LongMemEval 1000-item blinded input |
| `JUDGE_PROMPT_lme.txt` | instructions for GPT-5 (identical to the Claude Opus version except paths) |
| `CODEX_RUN_INSTRUCTIONS.md` | this document |

## Run steps (LongMemEval)

1. Move to this directory in the terminal:
   ```powershell
   cd "experiments/judges/codex_gpt5_2026_05_25"
   ```

2. Launch the Codex CLI and pass `JUDGE_PROMPT_lme.txt` as the instructions:
   ```powershell
   codex --model gpt-5 --prompt-file JUDGE_PROMPT_lme.txt
   ```
   (CLI option names may differ across Codex versions; check with `codex --help`)

3. GPT-5 reads `judge_input_lme.json` and writes `judge_output_lme.json`. This takes a few minutes to a few tens of minutes.

4. Verify the output:
   ```powershell
   python -c "import json; d=json.load(open('judge_output_lme.json')); print(d['judge_model'], d['n_items'], len(d['judgments']))"
   ```
   Success if `n_items == 1000` and the length of `judgments` is 1000.

## Aggregation steps (after judging is complete)

`analyze_paired_gpt5.py` in this directory reuses the private `MAPPING_lme_v10_paired.json` from `../v10_paired_2026_05_21/_private/` (not included in the public repo) to de-anonymize the items and produce `report_lme_gpt5.md`. Run from the repository root:

```powershell
python -m experiments.judges.codex_gpt5_2026_05_25.analyze_paired_gpt5
```

For inter-judge agreement against the Claude Opus 4.7 run, see `../analyze_inter_judge_agreement.py`, which pairs the two judge outputs on `anonymous_id` and reports observed agreement and Cohen's kappa without needing the private MAPPING.

## If the Codex CLI is unavailable (via the browser ChatGPT)

1. Start a new chat at https://chatgpt.com (select GPT-5)
2. Paste the contents of `JUDGE_PROMPT_lme.txt` (it specifies absolute paths, but in the browser read those as file attachments)
3. Attach `judge_input_lme.json` (within the attachment limit)
4. Send: "Following the instructions above, judge every item in the attached JSON and return the result as JSON"
5. Save the JSON GPT-5 returns as `judge_output_lme.json`

Caution: the browser path has an output length limit, so it likely cannot return all 1000 items at once. In that case, split the input into 5 parts (200 items each), judge in 5 passes, and merge locally. If you need a splitting script, prepare one separately.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `judge_output_lme.json` is empty or cut off | Split the input into 2-5 parts, rerun, and merge |
| `n_items != 1000` | GPT-5 skipped some items; identify the missing anonymous_ids and re-judge |
| `judge_model` is empty / unknown | Manually set it to the actual model ID used |
| Codex CLI option names differ | Check the correct flags with `codex --help` |

## Expected cost

- Both the Codex CLI and browser ChatGPT are within the $20/month (ChatGPT Plus) allowance; no extra charge
- Time required: 30 minutes to 2 hours depending on GPT-5 throughput (1000 items)
- No cloud GPU is needed

## Wrap-up after completion

After aggregation, run `../analyze_inter_judge_agreement.py` to compute observed agreement and Cohen's kappa between the Claude Opus 4.7 (`v10_paired_2026_05_21`) and GPT-5 (this directory) judgments. If HAMIB's LongMemEval ratio (around 1.31×) reproduces in the same direction under both judges, the result can be argued to be "robust and independent of the judge model". If only one is significant, reconsider it as "a bias specific to that judge model".

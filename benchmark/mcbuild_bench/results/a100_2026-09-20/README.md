# mcbuild-bench — A100 run of 2026-09-20/21

Everything the run produced. Reader `Qwen/Qwen3.8-27B` bf16 (no quantization), summarizer
`Qwen/Qwen3.5-4B` bf16, judge TypeSafe Jev, one A100 80 GB PCIe pod (driver 570 / CUDA 12.8).
Corpus: 36 of 37 round trips of the redacted hackathon session (the retrospective round trip 36 is
excluded), 179,394 reader tokens, 96 questions (88 facts + 8 whose answer is not in the corpus).

## What is here

| path | contents |
|---|---|
| `all_numbers.json` | every measured number in one file: per cell the verdicts, pass counts, window composition, mean prompt tokens / attention FLOPs / wall ms / energy J, plus the manager's Jev and summarizer accounting |
| `scores.json`, `scores.md` | paired scoring of each cell against the full-transcript arm: McNemar (exact, one-sided, both directions) and a 10,000-iteration paired bootstrap (seed 47) of the pass-rate difference and ratio, plus compute ratios |
| `pod_scripts_as_run/` | the shell scripts exactly as they were executed on the pod, plus the bootstrap log |
| `raw/main/<cell>/` | per cell: `answers.json` (qid → answer), `answers.jsonl` (durable per-question checkpoint with timing and counters), `answers.meta.json`, `meta.json` (run meta + per-question record), `timing.jsonl`, `gpu.csv` (1 Hz utilisation / memory / power), `questions_subset.json` |
| `raw/main/*.log` | the stdout of every cell run |
| `raw/v4_36rt/` | the manager phase: `cd.json` (the correlation diagram), `jev_calls.*.jsonl.gz` (every judge request: question ids, tokens, latency, HTTP status, the answers themselves), `summarizer_calls.*.jsonl`, `gpu_manager.*.csv`, `build_cd.log`, `summarizer_server.log` |
| `raw/v4_3rt/` | the 3-round-trip trial that stopped on the summarizer length rule (kept as the record of that failure) |
| `raw/*.log` | pod-level logs: main run, resume, watchdog, kernel build, summarizer sweep |
| `raw_manifest.json` | sha256 and byte size of every file above; `archive_sha256` is the checksum of the tarball as it left the pod, verified after transfer |
| `v4_36rt/cd.json`, `v4_36rt/cd_tree.txt` | the diagram again, plus an indented sun → planet → satellite rendering for reading |
| `v2.json`, `v3.json` | the earlier load check and injection probe on this pod |

Files over 250 KB are stored gzipped; `all_numbers.json` and the scorer read either form.

## Headline numbers

Baseline (full transcript, no diagram, no bias): **94/96**, 86 of the 88 fact questions, 179,394
prompt tokens, 220.1 s and 65,435 J per question.

| W | w | pass | facts | prompt tokens | wall s | energy J | attention FLOPs |
|---|---|---|---|---|---|---|---|
| 8000 | 0.1 | 29/96 | 21/88 | 7,980 | 2.6 | 667 | 1.25e13 |
| 8000 | 0.3 | 24/96 | 16/88 | 7,980 | 2.7 | 684 | 1.25e13 |
| 8000 | 1.0 | 18/96 | 10/88 | 7,980 | 2.9 | 723 | 1.25e13 |
| 16000 | 0.1 | 41/96 | 33/88 | 15,982 | 5.2 | 1,442 | 5.02e13 |
| 16000 | 0.3 | 33/96 | 25/88 | 15,982 | 5.2 | 1,435 | 5.02e13 |
| 16000 | 1.0 | 20/96 | 12/88 | 15,982 | 5.4 | 1,500 | 5.02e13 |
| 32000 | 0.1 | **63/96** | 56/88 | 31,979 | 10.9 | 3,108 | 2.01e14 |
| 32000 | 0.3 | 58/96 | 51/88 | 31,979 | 10.8 | 3,096 | 2.01e14 |
| 32000 | 1.0 | 27/96 | 20/88 | 31,979 | 11.4 | 3,247 | 2.01e14 |

The `w = 0.0` rows are the control added afterwards (H32): same window, same diagram, same prompt,
mass vector scaled by zero.

| W | w=0 | w=0.1 | w=0.3 | w=1.0 |
|---|---|---|---|---|
| 8000 | 29/96 | 29/96 | 24/96 | 18/96 |
| 16000 | 41/96 | 41/96 | 33/96 | 20/96 |
| 32000 | 63/96 | 63/96 | 58/96 | 27/96 |

w = 0 and w = 0.1 are indistinguishable: zero questions gained or lost in the paired comparison at
every W, and the generated answer string is byte-identical on 96/96, 95/96 and 94/96 questions
respectively (the few differing strings score the same). The attention bias contributes nothing at
w = 0.1 and is harmful above it. Everything the cells recover comes from narrowing the window onto
the correlation diagram.

## Compute, all measured per question

`attn_flops_*` counts only the QK^T term of the 16 full-attention layers (the E1 definition), which
understates the reader's work; the `total` column below adds the linear-layer term `2 * 27e9 *
prompt_tokens` that all 64 layers pay.

| cell | pass | tokens | attention FLOPs | total FLOPs | wall s | energy J | J per correct answer |
|---|---|---|---|---|---|---|---|
| full transcript | 94/96 | 179,394 | 6.33e15 | 1.60e16 | 220.1 | 65,435 | 66,827 |
| W=32000, w=0.1 | 63/96 | 31,979 | 2.01e14 | 1.93e15 | 10.9 | 3,108 | 4,737 |
| W=16000, w=0.1 | 41/96 | 15,982 | 5.02e13 | 9.13e14 | 5.2 | 1,442 | 3,377 |
| W=8000, w=0.1 | 29/96 | 7,980 | 1.25e13 | 4.43e14 | 2.6 | 667 | 2,208 |

Relative to the baseline the best cell keeps 67 % of the answers for 12 % of the total FLOPs, 4.9 %
of the wall time and 4.8 % of the energy. Per CORRECT answer the cells cost 3 - 7 % of the baseline.
Over all 96 questions: baseline 5.87 h and 6.28 MJ; best cell 0.29 h and 0.30 MJ plus the manager's
one-time 1.75 h and 1.77 USD.

The full-transcript arm wins every cell (McNemar p < 0.001 in that direction everywhere), so equal
recall at lower compute is NOT demonstrated by this run. The best cell uses 3.2 % of the attention
FLOPs and 4.9 % of the wall time for 65.6 % of the questions against 97.9 %.

In every cell the window held the correlation diagram ALONE: `n_recent_rts` is 0, so no verbatim
round trip was ever shown to the reader. Every correct answer came from the diagram.

## Why the cells lost (31 questions the baseline answered and the best cell did not)

- 4 facts are absent from the diagram: the manager dropped them.
- 27 facts ARE in the diagram. The diagram is ~33,000 tokens, so even W=32000 evicted 173 of the 414
  planet lines by mass; at W=8000, 397 of 414 were evicted.

The bias strength is monotonically harmful across all three window sizes (0.1 > 0.3 > 1.0 at every
W). No w = 0 control was run, so this run cannot separate "the diagram helps" from "the bias hurts".

## Manager phase

1 h 45 m wall, 1,606 nodes (389 sun / 414 planet / 803 satellite) from 3,388 chunks; 1,083 chunks
dropped as not worth keeping, 699 absorbed into existing nodes, 92 node texts fell back to a
truncated chunk because the summarizer broke the 120-character rule twice. Jev: 10,571 requests,
42,153,677 input tokens, 9,795,924 output tokens, 1.77 USD at the published price, 0 unparsed, 0
defaulted, 0 retries. Time split: judge 3,473 s (55 %), local summarizer 2,652 s (42 %), other 168 s.

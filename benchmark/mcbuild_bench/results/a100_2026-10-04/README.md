# mcbuild-bench — second round, A100 run of 2026-10-04

Twelve reader-only cells on the inputs of the 2026-09-20/21 run, unchanged: the same redacted
corpus (36 round trips, 179,394 reader tokens), the same 96 questions (88 facts + 8 whose answer
is not in the corpus), the same correlation diagram (`../a100_2026-09-20/v4_36rt/cd.json`, not
rebuilt), the same reader `Qwen/Qwen3.8-27B` bf16 with the same chat-template prompt, greedy,
48 new tokens. No manager, no judge and no summarizer ran this round, so there is no build cost
here; the build cost of the diagram (0.694 MJ, 1 h 45 min, 1.77 USD) is in the first run's
`all_numbers.json` and applies to every diagram row below.

Two kinds of cell, decided by Ryosuke (DECISIONS H34, H35):

- `truncB_W<W>` — **truncation control**: the raw transcript cut to the budget W, newest round
  trips first, whole round trips only, no diagram, no bias. Answers the question "does the
  diagram beat simply keeping the most recent W tokens?"
- `proposed_W<W>_w<w>` — the diagram window as before, with two changes to the bias:
  satellites inherit their planet's mass (`--inject planet+satellites`) and the effective
  bias `w * mass` is clamped at 3.0 (`--bias-cap 3.0`). The 2026-09 run biased planet lines
  only, with no cap.

W in {8000, 16000, 32000}; w in {0.1, 0.3, 1.0}. The full-transcript baseline (94/96) and the
w = 0 cells are reused from the first run and were not re-run.

**Pod.** One RunPod A100 80 GB, this time the **SXM4** part (driver 580.126.16, CUDA 13.0;
torch 2.11.0+cu128, transformers 5.17.0, causal_conv1d built from source). The first run was on
an A100 80 GB **PCIe**. Wall time and energy are therefore comparable *within* this round
(diagram vs truncation at the same W) but not directly against the first run's baseline: the
SXM4 part is faster and draws more power, and the reader cells here are 10 % quicker and
15-20 % more energy per question than the same cells were in September. Accuracy is unaffected
by the pod.

## What is here

| path | contents |
|---|---|
| `all_numbers.json` | every measured number in one file: per cell the verdicts, pass counts, window composition, mean prompt tokens / attention FLOPs / wall ms / energy J, and for the diagram cells the bias footprint (biased positions, planet and satellite spans, bias calls) |
| `scores.json`, `scores.md` | paired scoring of each diagram cell against the full-transcript arm AND against the truncation control at the same W: exact one-sided McNemar in both directions and a 10,000-iteration paired bootstrap (seed 47) of the pass-rate difference, plus compute ratios |
| `pod_scripts_as_run/` | `bootstrap_round2.sh` (environment), `round2_run.sh` (the 12 cells), `watchdog_round2.sh` (unattended supervision) as they were on the pod when the run ended; see *How it ran* |
| `raw/round2/<cell>/` | per cell: `answers.json` (qid → answer), `answers.jsonl` (per-question checkpoint with timing and counters), `answers.meta.json`, `meta.json` (run meta + per-question record), `timing.jsonl`, `gpu.csv` (1 Hz utilisation / memory / power), `questions_subset.json` (diagram cells) |
| `raw/round2/*.log` | the stdout of every cell run |
| `raw/round2_run.log`, `raw/watchdog_round2.log` | the runner's and the watchdog's logs for the whole round |
| `raw/bootstrap.log`, `raw/weights.log`, `raw/ccd_build.log` | environment set-up, weight download, kernel build |
| `raw_manifest.json` | sha256 and byte size of every file above; `archive_sha256` is the checksum of the tarball as it left the pod, verified after transfer |
| `collect_numbers.py` | the script that produced `all_numbers.json` and `raw_manifest.json` from `raw/` and `scores.json`; run on the first round's folder it reproduces that round's `all_numbers.json` field for field |

No file is over 250 KB, so nothing is gzipped. The baseline's raw files are not duplicated;
the scorer reads them from `../a100_2026-09-20/raw/main/A_full` (see *Reproducing it*).

## Headline numbers

Correct answers out of 96, same 96 questions in every column. The baseline is the first run's
full-transcript arm (179,394 tokens, 94/96).

| W | truncation control (tokens actually used) | diagram w = 0.1 | w = 0.3 | w = 1.0 | first run, w = 0 | first run, w = 0.1 / 0.3 / 1.0 |
|---|---|---|---|---|---|---|
| 8,000 | 20 (5,229) | 30 | 29 | 29 | 29 | 29 / 24 / 18 |
| 16,000 | 25 (13,425) | 40 | 40 | 40 | 41 | 41 / 33 / 20 |
| 32,000 | 38 (31,007) | 61 | 62 | **63** | 63 | 63 / 58 / 27 |

Facts only (of 88), with the 8 "not in the corpus" questions shown separately:

| W | truncation: facts / absent | diagram w = 0.1 | w = 0.3 | w = 1.0 |
|---|---|---|---|---|
| 8,000 | 12 / 8 | 22 / 8 | 21 / 8 | 21 / 8 |
| 16,000 | 17 / 8 | 32 / 8 | 32 / 8 | 32 / 8 |
| 32,000 | 30 / 8 | 54 / 7 | 55 / 7 | 56 / 7 |

### Diagram vs truncation, paired on the same questions

| W | w | diagram | truncation | diagram only / truncation only | difference, 95 % CI | McNemar p (diagram > truncation) |
|---|---|---|---|---|---|---|
| 8,000 | 0.1 | 30 | 20 | 17 / 7 | +10 (+1, +20) | 0.032 |
| 8,000 | 0.3 | 29 | 20 | 16 / 7 | +9 (0, +18) | 0.047 |
| 8,000 | 1.0 | 29 | 20 | 16 / 7 | +9 (0, +18) | 0.047 |
| 16,000 | 0.1 | 40 | 25 | 21 / 6 | +15 (+5, +25) | 0.003 |
| 16,000 | 0.3 | 40 | 25 | 19 / 4 | +15 (+6, +24) | 0.001 |
| 16,000 | 1.0 | 40 | 25 | 19 / 4 | +15 (+6, +24) | 0.001 |
| 32,000 | 0.1 | 61 | 38 | 34 / 11 | +23 (+10, +35) | < 0.001 |
| 32,000 | 0.3 | 62 | 38 | — | +24 (+12, +36) | < 0.001 |
| 32,000 | 1.0 | 63 | 38 | — | +25 (+13, +37) | < 0.001 |

At every window the diagram answers more questions than the most recent transcript of the same
size, and the margin grows with the window: +10 at 8k, +15 at 16k, +23 to +25 at 32k. The
truncation control is not exclusion-filtered: a fact that falls inside its recent window counts
for it, which is exactly what a recency window is for.

The 11 questions only the truncation arm got at 32k are facts stated in the last 10 round trips
(and one "absent" question the diagram cell answered wrongly). The 34 only the diagram got are
spread over the whole session, including the first hour.

### The bias, second attempt

Satellite inheritance puts the bias on 71-78 % of the window's tokens (8k: 6,185 of 7,980;
16k: 12,053 of 15,982; 32k: 22,787 of 31,979), and with the cap at 3.0 the result no longer
depends on w:

- The collapse at large w is gone. In September w = 1.0 cost 11, 21 and 36 answers against w = 0
  at the three windows; here it costs 0, 1 and 0 (gains 0 at 32k). The cap did what it was
  meant to do.
- Nothing is gained either. The cells sit at the w = 0 level: 29-30 vs 29, 40 vs 41, 61-63 vs
  63. The strings do change with w (at 32k the answers to 14 of 96 questions differ between
  w = 0.1 and w = 1.0, and 8 differ between w = 0.1 here and w = 0.1 in September), but the
  changes cancel: at 32k the w = 0.1 cell lost f023 and f069 relative to September and gained
  nothing; w = 1.0 gained them back.
- The questions the diagram cells miss are the same ones as in September: present in the window
  as a terse satellite line and not used by the reader. Making the bias reach those lines did
  not make the reader use them.

So the measured effect of the attention bias, in both configurations tried, is zero within +-2
questions. What the diagram rows achieve against the full transcript and against truncation
comes from what is in the window, not from how the window is weighted.

The counters guard inside `run_arms` passed in every diagram cell: the bias was applied in the
16 full-attention layers on every decode step and skipped on prefill (`--prefill-scale 0.0`),
and the diagram cells' question subsets were the full 96 (no recent round trips in the window,
so no fact had to be excluded).

### Compute, this pod

Per question, mean over 96, as measured here (SXM4). The baseline row is from the first run on
the PCIe pod and is repeated only for orientation; the energy ratios in `scores.md` mix the two
pods and should be read with that in mind.

| cell | correct | prompt tokens | wall s | energy J | reading, whole run (MJ) | with the diagram build (MJ) |
|---|---|---|---|---|---|---|
| full transcript (first run, PCIe) | 94 | 179,394 | 220.1 | 65,435 | 6.282 | 6.282 |
| truncation, 32k | 38 | 31,007 | 9.3 | 3,516 | 0.338 | 0.338 |
| diagram, 32k, w = 1.0 | 63 | 31,979 | 9.8 | 3,672 | 0.352 | 1.046 |
| truncation, 16k | 25 | 13,425 | 3.9 | 1,341 | 0.129 | 0.129 |
| diagram, 16k, w = 0.1 | 40 | 15,982 | 4.7 | 1,698 | 0.163 | 0.857 |
| truncation, 8k | 20 | 5,229 | 1.9 | 500 | 0.048 | 0.048 |
| diagram, 8k, w = 0.1 | 30 | 7,980 | 2.4 | 815 | 0.078 | 0.772 |

Reading cost is set by the window size, so truncation and diagram cost the same to read at the
same W (the truncation windows are a little smaller because they hold whole round trips). The
diagram's extra cost is the one-time build; over 96 questions the diagram arm at 32k spends 3.1x
the energy of the truncation arm for 1.66x the correct answers, and the gap closes as the build
amortizes.

### A property of the truncation control worth knowing

The truncation window is composed of whole round trips, so it stops at the last one that fits:
5,229 of 8,000 tokens (3 round trips), 13,425 of 16,000 (4), 31,007 of 32,000 (10). At 8k the
control uses 65 % of the budget the diagram cell uses. A variant that fills the budget with a
partial round trip was not run; it is Fable's proposal and is not adopted.

## How it ran

Started 2026-10-04 00:07 UTC, all twelve cells complete at 02:28 UTC (2 h 21 min including twelve
model loads; the first load took 12 min from the network volume, later ones about 2 min from the
page cache). No cell failed, no restart and no stall occurred; the watchdog log has only its
start lines and the completion line.

Two supervision scripts were replaced on the pod at 00:38 UTC, during the third cell, at
Ryosuke's request that a stalled run be detected and restarted automatically: `round2_run.sh`
now reports a failed cell and continues instead of stopping the queue, and
`watchdog_round2.sh` also kills a runner that has written nothing for 25 minutes. The replaced
files were renamed over the old ones, so the runner instance that executed all twelve cells kept
reading the original script (identical cell command lines; only the failure handling differs);
the new watchdog supervised from 00:38 and never had to act. `pod_scripts_as_run/` holds the
files as they were at the end; the originals are the versions in the repository's history
before 2026-10-04.

## Reproducing it

```bash
# score the published cells (no GPU needed); the baseline is read from the first run's folder
python -m benchmark.mcbuild_bench.score_cells \
  --main benchmark/mcbuild_bench/results/a100_2026-10-04/raw/round2 \
  --a-dir ../../../a100_2026-09-20/raw/main/A_full \
  --questions benchmark/mcbuild_bench/data/questions.json \
  --out scores.json --md scores.md
```

The pod side is `pod/bootstrap_round2.sh` (environment, weights, kernel, self-test), then
`pod/round2_run.sh` under `pod/watchdog_round2.sh`. The first run's `A_full` directory and
`v4_36rt/cd.json` must be present under `/workspace/mcb/runs/` as the scripts expect.

# mcbuild-bench — status and runbook (2026-09-18 22:00)

Branch: `mcbuild-bench` (commit 019bdd9 + follow-ups). `main` untouched. Nothing pushed.

## Done and verified
| Item | Evidence |
|---|---|
| Decision ledger A–G + amendments H1–H14 | `DECISIONS.md` |
| Contract-first design (transformers 5.8.0 / Jev API / RunPod quotes) | `DESIGN.md` |
| Redacted corpus, 36 round trips used (round trip 36 excluded), 177,165 Qwen3.8 tokens | `data/session_redacted.{json,txt}`, `data/redaction_report.md` = CLEAN |
| Privacy audits: Fable sweep + Codex sol → luna → sol, every hit verified by Fable before applying | H12–H14 |
| Fact ledger 88 facts + 8 questions with no answer in the corpus = 96, machine-verified after every regeneration | `data/ledger_report.md` |
| Harness: SpecManager hooks, jev_client, summarizer_client, jev_judge, build_cd (--resume), windows, gpu_sampler, run_arms, compaction_c, jev_live_check | `tests/mcbuild` 163 tests; full suite 750; `verify_attention_math.py` OK; ruff clean |
| Adversarial reviews applied: Fable (12 findings) + Codex gpt-5.6-sol (14 findings) | H8–H11 |

## Gates before any GPU spend (in order)
1. **Ryosuke reads the corpus** `data/session_redacted.txt` and the questions `data/questions.json` (A4/B1 gates).
2. **V0 Jev live contract check** — PASSED 2026-09-20 (HTTP 200, all three answer shapes as documented, usage 426/73 tokens, 554 ms). Key file: Ryosuke's Documents (path only in memory, never in the repo). Re-run if the API changes:
   `$env:TYPESAFE_API_KEY="<key>"; python -m benchmark.mcbuild_bench.jev_live_check --accounting data/jev_live_check.jsonl` → must print `live contract check: OK`.
3. **Environment choice** (DESIGN §1): α = A6000 + INT4 + transformers 5.8.0 pinned (two venvs) / β = one 80 GB pod, BF16, transformers 5.17.0. Fable recommends β.
4. Remaining decisions: H6 (raw reader prompt, no chat template), H7 (oversized round trip in baseline C), H9 (Jev retry set = 429/529 only), H10 (missing usage → stop; implemented).
5. Codex GPT-6 Astra review rounds (every 5 h while limits allow; weekly limit resets 2026-09-19 ~11:45). Findings triaged by Fable, fixes committed on this branch.

## Pod runbook (after the gates)
1. Ryosuke: create the pod (template PyTorch/CUDA 12.x, volume ≥ 150 GB, expose TCP 22), register the public key, give Fable the connection triple (ip, port) and the key PATH as env `MCB_POD_KEY`.
2. Fable over ssh: clone branch, `uv venv`, install per DESIGN §9 (variant-specific), download weights, start vLLM summarizer.
3. V2 load checks → V3 injection probe (w grid, Ryosuke freezes it) → V4 manager on 3 round trips (Ryosuke inspects the CD, Jev cost extrapolated) → V5 arm A on 5 questions → main run → scoring locally.
4. Destructive pod operations (stop/terminate/delete volume) only with Ryosuke's approval each time.

## Known limitations (stated, not hidden)
- The reader prompt is the raw completion template of `run_reader.READER_PROMPT` (no chat template); identical across arms (H6).
- RT00's memory-file read is a fictional replacement, ~9k chars shorter than the original (H13); no question depends on it.
- `wall_ms_prefill/decode` come from a LogitsProcessor timestamp (first decode step), not from kernel-level timing.
- Attention FLOPs = QKᵀ term only, GQA ignored (E1 as decided).
- Codex reviews return PLAUSIBLE findings (its sandbox has no python); each is verified locally before a fix.

## Second measurement round — run 2026-10-04 (H35, H36); results in `results/a100_2026-10-04/`

12 cells on the unchanged corpus / questions / cd.json / prompt / windows, one A100 80 GB SXM4 pod,
00:07-02:28 UTC, no failure, no restart:
- truncation control `--arm B`, W in {8000, 16000, 32000} (H34): **20 / 25 / 38** of 96
  (windows of 5,229 / 13,425 / 31,007 tokens: whole round trips only);
- `--arm proposed --inject planet+satellites --bias-cap 3.0`, W in {8000, 16000, 32000},
  w in {0.1, 0.3, 1.0}: **30, 29, 29 / 40, 40, 40 / 61, 62, 63** of 96.
Baseline 94/96 and the w = 0 cells (29 / 41 / 63) were reused.

Two results. (1) The diagram beats the most recent transcript of the same size at every window:
+10 (p = 0.03), +15 (p = 0.003), +23 to +25 (p < 0.001) questions, paired McNemar. (2) The capped,
satellite-inherited bias removed the collapse at large w (September: 63 → 27 at 32k; now 61-63) and
gained nothing: every cell is within 2 questions of w = 0. Energy and wall time are not comparable
with the first run's PCIe pod (SXM4 is ~10 % faster and ~15-20 % more energy per question); within
this round, truncation and diagram cost the same to read at the same W. Details and the per-question
lists are in `results/a100_2026-10-04/README.md`.

What was verified before the run (kept for the record):

Verified without a GPU (2026-10-03): 957 unit tests; a dry run of round2_run.sh with a recording stand-in
for run_arms (12 command lines with the intended flags, a second pass skips all 12 as complete); a CPU
end-to-end of both cell types through the real CLI on the real corpus, questions and cd.json with the tiny
Qwen3.5 checkpoint (`tests/mcbuild/test_round2_cpu_e2e.py`): satellites receive positions, the effective
bias is clamped at 3.0, the counters guard passes, the subset and header carry the cap, and the scorer pairs
each cell with its truncation control. Not verifiable without the pod: the 16-layer counters on the real
reader (same logic as the first run) and, of course, the effect itself.

Expectation to keep in mind: with satellite inheritance the bias lands on ~60 % of the window's tokens. At
w = 1.0 the median effective bias is already the cap, so most biased tokens get a flat +3.0; a near-uniform
bias mainly down-weights the un-biased 40 % (suns, scaffold, the question), which is unlikely to help. The
informative cells are w = 0.1 and 0.3.

## Pending experiments (2026-10-01)

1. ~~Truncation control, `--arm B`~~ — run 2026-10-04 (H36): 20 / 25 / 38 of 96; the diagram
   wins at every window.
2. ~~Satellite inheritance for the bias, `--inject planet+satellites`~~ — run 2026-10-04 with
   the cap at 3.0 (H36): no gain, no collapse.
3. A window that holds the whole diagram (about 35k tokens), to find the method's ceiling.
4. (Fable's proposal, not adopted) a truncation variant that fills the budget with a partial
   round trip; the whole-round-trip control used 65 % / 84 % / 97 % of the 8k / 16k / 32k budgets.

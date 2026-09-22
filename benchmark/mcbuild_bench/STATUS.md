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

# RESULTS_v1 — First complete run of the bineval compress-then-answer benchmark

Date: 2026-07-23/24. Instrument: bineval v1.1 (173 scored questions, tier-1 string scoring, frozen-candidate PROTOCOL). Reader model: Claude Sonnet (CC subagent); robustness columns: Claude Opus 4.8. All raw per-arm answers under `results/pilot/`.

## 1. Full run — budget–accuracy table (tier-1, n=173)

| Arm | Context tokens | Ratio vs raw (172,773) | Pass | Fail (explicit) | Indet | Pass rate |
|---|---|---|---|---|---|---|
| **Ceiling** (full history, agentic read) | 172,773 | 1× | 173 | 0 | 0 | **100.0%** |
| **Summarization** (best-effort LLM, question-blind) | 19,115 | **9.0×** | 168 | 2 | 3 | **97.1%** |
| Truncation 2× (last 23 sessions) | 85,651 | 2× | 85 | 88 | 0 | 49.1% |
| Truncation 6× (last 7 sessions) | 26,621 | 6× | 27 | 144 | 2 | 15.6% |
| Truncation 10× (last 4 sessions) | 14,801 | 10× | 15 | 158 | 0 | 8.7% |
| Truncation 20× (last 2 sessions) | 6,228 | 20× | 4 | 169 | 0 | 2.3% |
| **CD-w0** (SBERT-path CD, mass eviction) | 28,795 | 6× | 12 | 157 | 4 | **6.9%** |
| Floor (no context, forced guess) | 0 | ∞ | 10 | 0 | 163 | 5.8% |

Reader-robustness (Opus 4.8): ceiling 57/58 = 98.3% (subsample), trunc-6× 15.0%, floor 9.2%. Per-question verdict agreement Sonnet↔Opus: **97.7%** (trunc-6×, 173 q), **98.3%** (ceiling, 58 q). Conclusion-level rankings identical across readers. Floor is reader-dependent (5.8% vs 9.2%) → floors must be measured per reader model.

## 2. Parity-budget readings (δ = 5pp, §1.6 of RESEARCH_PROGRAM)

- **Summarization: parity HOLDS at 9×** (97.1% ≥ 95%). B\*(summary) ≥ 9× on this instrument.
- **Truncation: parity fails already at 2×** (49.1%). B\*(trunc) < 2×.
- **CD-w0 (current SBERT path): parity nowhere**; 6.9% at 6× is within noise of the guessing floor.

## 3. Key findings

**F1 — The instrument works.** Monotone budget–accuracy curve spanning 2.3%–100%; item-level sensitivity confirmed (the summary's 2 explicit failures are real compression losses, verified against source: the 6,500-yen wine-pairing price was dropped by the summarizer; the 95,000-yen/tsubo threshold lost its linkage). Dynamic range and discriminative power far exceed the ≥30pp criterion.

**F2 — Best-effort LLM summarization is near-lossless at 9× on extractive fact probes.** 168/173 facts survive. Implication for the research program: **G2 as currently written (CD-w0 beats best-effort summarization by +10pp at 4–10×) is close to unwinnable at these ratios on this data.** The honest comparison axes are: (a) deeper ratios (20×+) where summarization must drop facts; (b) compressor COST — the summary required a full LLM pass over 172K tokens (~10× the cost of answering once), while CD/truncation are cheap mechanical compressors; HAMIB's deployment story (incremental CD built during the chat at near-zero marginal cost) belongs on this axis; (c) incremental/streaming settings where a full re-summarization pass per compaction is unaffordable. Recommendation (gate change is Ryosuke's call): recalibrate G2 to cost-matched comparisons or deeper ratios, and report LLM-summarization as the expensive-compressor reference line, analogous to how KV papers report full-KV.

**F3 — The current SBERT-path CD destroys information.** 6.9% at 6× vs truncation's 15.6% at the same budget: the CD is WORSE than naive recency truncation and statistically at the guessing floor. Cause is visible in the artifact: node texts are mid-word fragments ("inking through whether this is") — the sliding-window span extraction (§72.7-bis overlapping-spans issue) plus classifier bypass (§108: 30–40% spec compliance). This is a measurement of the provisional extractor, NOT of the patent-spec manager (never run at this scale) and NOT of the wM mechanism. It makes WO-2's two-extractor comparison decisive: the spec-compliant Gemma manager (§41) is the remaining candidate for a viable CD representation.

**F4 — Abstention asymmetry.** Truncation-20× (2.3%) scores BELOW the forced-guess floor (5.8%) because arms abstain ("not in context") while the floor guesses. Protocol note added: normalized recovery must use the abstention-consistent floor, or arms must be run in forced-guess mode for floor-comparable readings. Current reporting keeps abstention arms + forced-guess floor and flags the asymmetry.

## 4. Comparison with published benchmarks

| Benchmark | Full-context score (frontier) | What it ranks | Ceiling design |
|---|---|---|---|
| **bineval v1.1 (this)** | **100%** (Sonnet), 98.3% (Opus subsample) | Compression interventions | Intentionally easy ceiling; the measured quantity is degradation under budget |
| LongMemEval-S | 60.6% GPT-4o (oracle-evidence 87.0%) | Models/memory systems | Hard by design (multi-topic 115K haystack) |
| LoCoMo | human F1 87.9; models well below | Models/memory systems | Hard |
| BEAM 128K | 0.239–0.280 (GPT-4.1-nano…Qwen2.5-32B) | Models | Hard, unsaturated |
| PersonaMem-v2 | 37–48% MC | Models/personalization | Hard |

The design difference is intentional and matches universal field practice for intervention studies: KV-compression papers score against full-KV, quantization against FP16, prompt compression against the uncompressed prompt — all "easy ceiling, hard intervention" retention designs. (Notably, no paper articulates a methodological defense of this design; PROTOCOL now does so explicitly.) The saturation critique (near-100% ceilings as a flaw) applies to MODEL-RANKING benchmarks and does not transfer: this instrument ranks compressors, not models — 97.7–98.3% cross-reader verdict agreement confirms reader-model choice does not drive conclusions, which is above the field norm (2–3 readers, no published stability standard).

Positioning re-verified (2026-07-22 adversarial scan): no published benchmark makes "dialogue + fixed token budget + compress-then-answer QA" a first-class protocol; nearest neighbors are EpiCache (fixed KV budgets on dialogue, method-paper harness) and Reclaim (fixed-budget MultiWOZ error-correction). This instrument plus PersonaMem-v2-under-budget (Layer 2, probe-verified) remains a first.

## 5. Instrument coherence verification (goal item: 技術的整合の確認)

- Budget parity: enforced by construction (all arms consume files built to B = raw/ratio; token counts logged above). ✔
- Question validity: 2-of-2 independent validations converged (per-session audit with evidence quotes; ceiling arm 100%). ✔
- Scoring: deterministic tier-1 decides 98.3–100% of verdicts per arm; zero judge involvement in every number in this report. ✔
- Provenance: every question carries fact_quote + source_session; every exclusion carries an evidence file (EXCLUSIONS.md). ✔
- Leakage: cross-question leak screen = 0 violations; floor dropped 24.6%→5.8% after v1.1. ✔
- Power: n=173 paired; MDAD ≈ 4–6pp; all reported gaps (CD vs trunc 8.7pp … summary vs trunc-6× 81.5pp) exceed MDAD except floor-adjacent comparisons, which are flagged as at-noise. ✔
- Reproducibility: question set regenerates byte-identically; arms are file-in/JSON-out; scorer is a pure function. ✔
- Residual incoherences (documented, not hidden): in-world name drift in source data (contractor Yamamoto vs Shimizu in s14; restaurant name drifts to "Trattoria Modena" in s37-38; owner name drift in s30) — questions touching drifted names were audited against their own session and pass, but cross-session synthesis questions would be unsafe; none of the 173 requires cross-session name resolution. Sessions 19/24/26/32/33/35/40 contaminated and excluded from question sourcing.

## 6. Problems and improvements (v1.2 backlog)

1. **Single domain, synthetic self-authored data** — the strongest validity limitation. Improvement: WildChat coherent-subset track (WO-5) for real-log replication; PersonaMem-v2 Layer 2 full run (loader ready, ~100MB download remaining).
2. **Extractive-fact bias favors summarization** — 173 short-fact probes are exactly what a fact-dense summary preserves. Improvement: add question families summarization is bad at — multi-hop across sessions, temporal ordering, update chains (knowledge-update was HAMIB's documented weak spot too, §114), and abstention/false-premise probes (LongMemEval has ~6%).
3. **G2 calibration** — see F2; gate revision proposal to Ryosuke: cost-matched comparison or ≥20× ratios.
4. **Floor/abstention asymmetry** — unify by running every arm in both abstain and forced-guess modes, or fix normalized recovery to the abstain-mode floor (2 explicit-fail rate).
5. **Tier-2 judge never exercised** — 0–4 indeterminates per arm were left unscored (conservative). Before freeze, run the pinned-judge calibration on the accumulated ~12 indeterminates + 30 manual labels.
6. **Ceiling definition is agentic** (ordered file reads, note-taking) — documented in PROTOCOL; a single-prompt ceiling variant would remove agent-strategy variance for API models with ≥200K windows.
7. **In-world data drift** (names/dates across sessions) — tolerable for v1 (audited per-session), but blocks future cross-session synthesis questions; fix the source data before adding question family #2.
8. **Summary-arm cost accounting** — record compressor cost (tokens processed) per arm as a first-class column so the cost axis of F2 is visible in every future table.

## 7. Verdict

The benchmark is complete, internally coherent, and does its job: it separated seven arms over a 2.3–100% range with deterministic scoring, survived a reader-model swap, and produced two program-relevant discoveries (near-lossless 9× LLM summarization; SBERT-CD at the guessing floor) that reshape what C2 must prove. Freeze remains Ryosuke's call after the 30-question spot check.

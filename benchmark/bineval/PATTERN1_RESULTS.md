# PATTERN1_RESULTS — Three-way separation: bug vs manager quality vs format limit

Date: 2026-07-26/27. Instrument: bineval v1.1 (173 questions, tier-1 deterministic scoring). Reader: Claude Sonnet subagents. Arm B (4B-class local manager) deferred per D-11 (CPU-only machine); plan = bundle into the C3 Modal/Lambda session (~$1-2 add-on).

## Arms and results

| Arm | What it measures | Tokens | Ratio | Pass rate |
|---|---|---|---|---|
| A: SBERT-path CD, mass eviction @6× | Current implementation floor | 28,795 | 6× | **6.9%** |
| A: SBERT-path CD, random @6× | — | 28,787 | 6× | 5.8% |
| A: SBERT-path CD, recency @6× | — | 28,794 | 6× | 5.2% |
| **C: Oracle-manager CD, FULL** | **Format ceiling** (spec procedure ¶0038-0061 executed with frontier-model judgment; D-4 descriptive nodes; contaminated sessions skipped) | 18,414 | **9.4×** | **96.0%** |
| C: Oracle CD, mass eviction @ half | Mass informativeness on a HEALTHY structure | 9,205 | 18.8× | 52.0% |
| C: Oracle CD, random @ half | — | 9,207 | 18.8× | 56.1% |
| C: Oracle CD, recency @ half | — | 9,202 | 18.8× | 51.4% |
| Reference: best-effort flat summary | Expensive-compressor reference | 19,115 | 9.0× | 97.1% |
| Reference: ceiling / floor | — | 172,773 / 0 | 1× / ∞ | 100.0% / 5.8% |

Oracle CD structure: **1 sun / 32 planets / 207 satellites** from 33 clean sessions (7 contaminated skipped). SBERT CD structure at the same data: 664 suns / 661 planets / 3,121 satellites (no aggregation, mid-word fragment nodes).

## Verdicts

**V1 — The CD FORMAT is not the limit.** Oracle CD @9.4× scores 96.0% vs flat summary @9.0× 97.1% — a 1.1pp difference at matched compression, within noise (MDAD 4-6pp). The patent's hierarchical [PN] serialization carries information as well as the best flat summary when the nodes are descriptive fact sentences (D-4). The format-redesign branch (Pattern 4) is NOT needed.

**V2 — The current implementation is the dominant loss.** 6.9% (SBERT path) vs 96.0% (same procedure, competent judgment): an ~89pp gap attributable to the extraction layer — span fragmentation and zero topic aggregation, exactly the §72.7-bis / §108 defects. This is the bug/quality axis, and it accounts for essentially the entire loss.

**V3 — Satellite-count mass is NOT an eviction signal, even on a healthy structure.** G1-style comparison on the oracle CD @18.8×: mass − random = **−4.0pp** (McNemar discordant 38+45, p=0.51 — indistinguishable). On the degenerate SBERT CD: +1.1pp (also noise). Mechanism of the null: mass marks where discussion was DEEP, but the benchmark probes facts UNIFORMLY across all topics; uniform (random) retention matches a uniform query distribution at least as well as interest-weighted retention. Two implications, carefully separated:
  1. For the PROGRAM: the mass→eviction operationalization of C1 fails in both structures. Formal G1 evaluation is Ryosuke's, but the numbers leave little room.
  2. For the PATENT's actual claim: ¶0079-0080 defines mass as an interest signal guiding ATTENTION at answer time, on the premise that queries correlate with interest depth. The benchmark's uniform probing deliberately breaks that correlation — so this result does NOT falsify the interest-correlation premise; it shows mass adds nothing when queries are interest-independent. Testing the real premise needs (a) an interest-weighted question distribution variant (benchmark v1.2 item), and/or (b) C3's attention-time test where wM guides retrieval rather than deciding retention.

## Program implications

1. **Pattern 4 (format redesign) closed** — V1. The winning recipe for the representation already exists: spec procedure + D-4 descriptive nodes + aggregation. What is missing is a MANAGER that achieves it locally.
2. **The manager is the whole game (C2 side).** Arm B (Qwen3-4B/Gemma-3-4B under the coded pipeline) measures how close a deployable local manager gets to the 96% ceiling — bundled into the C3 GPU session (plan B-1, ~$1-2, per D-11).
3. **C1 reformulation needed before any further mass-eviction runs**: either adopt the interest-weighted probe variant, or shift the mass hypothesis entirely to C3 (attention guidance), where the patent actually places it. Gate wording change is Ryosuke's call.
4. Oracle-CD cost note: built by ~8 frontier-agent sessions over 33 sessions of chat (≈ the cost class of the flat summary). Like the summary reference, it is an EXPENSIVE compressor — the deployment claim still rests on the incremental, local, cheap manager (arm B).
5. Benchmark v1.2 backlog addition: interest-weighted question distribution (sample questions ∝ discussion depth of their topic) to test the patent's actual premise; keep the uniform set as the adversarial case.

## Artifacts
- Oracle CD: `results/oracle_cd/cd_state.json` (+ build log), serializations under `results/pilot/oracle_cd_*.txt`, answers `results/pilot/oracle_*_answers.json`
- SBERT eviction arms: `results/pilot/cd_{random,recency}_6x.txt` + answers
- Arm B plan: RESEARCH_PROGRAM.md §1.9 D-11 + WO-6 bundling note

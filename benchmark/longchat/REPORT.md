# HAMIB Long-Chat Benchmark — Sonnet 4.6 Compression Baseline

## Purpose

Measure how much information survives external context compression when running on Sonnet 4.6. This establishes the API-model baseline that HAMIB (run on Gemma + attention intervention, in a separate experiment) must eventually beat.

## Benchmark Design

- **Single chat, single topic** (matches HAMIB's CD-per-chat structure, unlike LongMemEval which mixes topics)
- **40 sessions** covering 17 months of a fictional restaurant opening journey ("Osteria Morishita" / Kenta Morishita)
- **172,773 tokens** raw context (tiktoken cl100k_base measurement)
- **16 questions** with verified verbatim gold answers, distributed across four difficulty axes:

| Category | Questions | Description |
|---|---|---|
| `early_static` | 4 | Answer appears ONLY in sessions 1-12 |
| `mid_static` | 4 | Answer appears ONLY in sessions 13-25 |
| `update_tracking` | 4 | Question asks for LATEST value of facts that changed |
| `synthesis` | 4 | Requires combining facts from sessions ≥10 apart |

## Evaluation Conditions

All three conditions run on Sonnet 4.6 via Claude Code subagent (subscription, no API billing).

1. **Full Context** (172K tokens): entire chat passed verbatim
2. **Truncation** (30K tokens budget): only the last ~8 sessions retained, earlier sessions dropped
3. **Summarization** (23K tokens): sessions 1-35 replaced with lossy summaries (Sonnet-generated; specific names, numbers, prices deliberately omitted) + last 5 sessions full

Compression ratio: ~6x (172K → 23-30K).

## Results

| Condition | Correct | Total | Accuracy |
|---|---|---|---|
| Full Context | 14 | 16 | **87.5%** |
| Truncation | 1 | 16 | **6.25%** |
| Summarization | 1 | 16 | **6.25%** |

### By Category

| Category | Full | Truncation | Summarization |
|---|---|---|---|
| early_static (S1-12) | 4/4 | 0/4 | 0/4 |
| mid_static (S13-25) | 4/4 | 0/4 | 0/4 |
| update_tracking | 4/4 | 1/4 | 1/4 |
| synthesis | 2/4 | 0/4 | 0/4 |

### Key Observations

- **Compression drops accuracy by 81 percentage points** (87.5% → 6.25%) at a 6x compression ratio
- **Truncation and summarization perform identically** — both effectively lose access to all facts in compressed/dropped sessions, only retaining what's in the last 5-8 full sessions
- **Only `Osteria Morishita` (current restaurant name)** survived compression across both conditions because it's repeatedly mentioned in late sessions
- **Sonnet correctly identified "not in context"** for 13/16 questions under compression — no hallucination, just honest information unavailability

### Full Condition Caveats

Full context scored 14/16 strict, not 16/16, due to format mismatches (not knowledge failure):
- Q13: gold `"12,000 yen per person"`, predicted `"12,000 yen"` (correct value, shorter form)
- Q14: gold `"four days"`, predicted `"4 days"` (number-word vs digit)

Lenient scoring: 16/16 (100%).

## Interpretation

The 81-percentage-point drop is the **information-loss cost** of external compression at 6x ratio when the model has no specialized memory mechanism. This is the gap that HAMIB's structured Correlation Diagram + mass-aware attention must eventually close — measured on a separate Gemma + GPU setup, not Sonnet.

### What this measures

- Sonnet 4.6's ability to recall facts from compressed conversational context
- The structural limit of lossy text compression: facts dropped at compression time cannot be recovered at query time
- A clean baseline for comparison against any memory-augmented system

### What this does NOT measure

- HAMIB itself — that requires open-weight model with attention intervention
- Cases where the question depends only on recent context (compression-friendly)
- Adversarial or noise-injected compression scenarios

## Data Quality Issues Noted

Generation artifacts identified during evaluation:
- Sessions 19, 24, 32, 33, 35 in the generated chat contain off-topic content (a separate "novel writing" thread that bled in during parallel generation)
- Q1 has internal chat inconsistency: session 1 says "11 years at Marubeni", session 39 says "fifteen years" (truncated/summarized retrieved the latter)
- Q13's "corporate dinner" is ambiguous: two distinct Marubeni events appear (12,000 yen in session 23, 18,000 yen in session 39)

These issues partially limit benchmark cleanliness but do not invalidate the directional finding. The compression effect (87.5% → 6.25%) is large enough to be robust to these noise sources.

## Files

- `restaurant_chat_v2.json` — patched 40-session chat (172,773 tokens)
- `restaurant_questions.json` — 16 evaluation questions with verified gold answers
- `session_summaries.json` — lossy summaries for sessions 1-35
- `eval_prompts/{full,truncated,summarized}.txt` — input prompts for each condition
- `eval_results/answers_{full,truncated,summarized}.txt` — Sonnet 4.6 responses
- `eval_results/scored.json` — full per-question scoring

## Next Step

Run HAMIB (Gemma + LoRA + mass-aware attention) on this same benchmark in a GPU-equipped environment. Compare HAMIB's accuracy to the 6.25% compression baseline established here. Any number meaningfully above 6.25% is HAMIB's measured contribution.

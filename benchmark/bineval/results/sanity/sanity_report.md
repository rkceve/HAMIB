# bineval sanity report (WO-0, sec.2.6 acceptance)

Instrument acceptance run: score the frozen tier-1 instrument on the three
EXISTING longchat conditions and confirm it reproduces the known ordering
**full >> truncated ~ summarized** with clear separation. An instrument that
cannot separate these is broken (RESEARCH_PROGRAM sec.2 rule 6).

- Questions: `questions_restaurant.json`, `--subset legacy` (the 16 re-expressed
  original questions, `rest_q01..16`, `legacy: true`).
- Answers: `benchmark/longchat/eval_results/answers_{full,truncated,summarized}.txt`
  (`A{n}:` line format; k-th scored question <-> A{k}).
- Scoring: tier 1 ONLY (`--tier2 none`); no judge, no API.
- Reproduce:
  ```
  python -m benchmark.bineval.score_binary \
      --answers benchmark/longchat/eval_results \
      --questions benchmark/bineval/questions_restaurant.json \
      --out benchmark/bineval/results/sanity/legacy_scored.json \
      --subset legacy
  ```

## Aggregate (tier-1 only)

| condition  | pass | fail | indeterminate | total | pass_rate_tier1 |
|------------|-----:|-----:|--------------:|------:|----------------:|
| full       |   16 |    0 |             0 |    16 |      **1.0000** |
| truncated  |    1 |   13 |             2 |    16 |      **0.0625** |
| summarized |    1 |   13 |             2 |    16 |      **0.0625** |

Ordering reproduced: full (1.0000) >> truncated (0.0625) ~ summarized (0.0625).
Separation = 0.9375 between full and the lossy arms; truncated == summarized to
within 0.0000. PASS.

`pass_rate_tier1` is defined as pass / total (an indeterminate counts as
not-yet-passed), so this rate is a conservative tier-1 floor; the two
indeterminate items per lossy arm could only move the rate up, not down, if a
tier-2 judge were run.

## Category breakdown (PASS counts; n = items per category)

| category         | n | full | truncated | summarized |
|------------------|--:|-----:|----------:|-----------:|
| early_static     | 4 |    4 |         0 |          0 |
| mid_static       | 4 |    4 |         0 |          0 |
| update_tracking  | 4 |    4 |         1 |          1 |
| synthesis        | 4 |    4 |         0 |          0 |

The single lossy-arm pass in both truncated and summarized is `rest_q09`
("Osteria Morishita", update_tracking) — the one fact that survived truncation
because the restaurant name recurs late in the history.

## The two known tier-1 false negatives now PASS in `full`

| qid      | gold                     | full answer  | verdict | via                                   |
|----------|--------------------------|--------------|---------|---------------------------------------|
| rest_q13 | "12,000 yen per person"  | "12,000 yen" | pass    | alias "12,000 yen" (unit-suffix)      |
| rest_q14 | "four days"              | "4 days"     | pass    | number-word<->digit canonicalization  |

Both were failures under the original `score_eval.py` strict substring match;
the extended tier-1 normalization (aliases + numeral table) fixes them. This is
WO-0 acceptance item (c).

## Indeterminate items routed to tier 2 (`pending_tier2.json`)

The 2 indeterminate items in each lossy arm are the internally-inconsistent
facts the exclusions target: the model asserts a concrete WRONG value, which is
neither a substring match nor a refusal marker, so tier 1 declines to score it
and queues it for a judge instead of guessing.

| arm        | qid      | gold                    | answer_fragment       |
|------------|----------|-------------------------|-----------------------|
| truncated  | rest_q01 | "11 years"              | "15 years"            |
| truncated  | rest_q13 | "12,000 yen per person" | "18,000 yen per person" |
| summarized | rest_q01 | "11 years"              | "fifteen years"       |
| summarized | rest_q13 | "12,000 yen per person" | "18,000 yen per person" |

(These are the generated-set exclusions #2 and #3 in EXCLUSIONS.md; the
instrument surfacing them here is the reason those facts are excluded from the
generated question set.)

## Note on running the *generated* set against these three answer files

The WO asks, as a secondary signal, to also run the new generated questions
against the same three answer files "ONLY for questions whose fact is actually
probed by the original 16 answers." The three legacy answer files contain
exactly 16 answers (`A1..A16`), each answering one of the 16 legacy PROMPTS, not
the generated prompts. There is no position- or qid-alignment between the
generated questions and these 16 answer lines, so scoring the generated set
against them would fabricate an alignment that does not exist. The primary and
sufficient sanity signal is therefore the legacy-16 run above (as the WO states:
"the main sanity signal is [the 16 legacy re-expressed]"). Generated-set
validation happens for real in WO-2, where an arm answers the generated set in
order and `A{k}` maps to the k-th scored generated question.

STATUS: sanity PASS. Instrument is ready for Ryosuke's 30-question spot-check
and freeze (see PROTOCOL.md).

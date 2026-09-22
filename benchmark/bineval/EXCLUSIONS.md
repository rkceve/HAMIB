# EXCLUSIONS — bineval question generation

Data-hygiene exclusions applied by `gen_questions.py` at generation time, per
RESEARCH_PROGRAM.md sec.2 rule 5 and the WO-0 exclusion list. Every excluded
question is kept in the JSON with `excluded: true` and an `exclusion_reason`
(so the record is auditable); scorers drop them by default and only include
them with `--include-excluded`.

The **legacy** `rest_q01..16` / `novel_q01..16` records are NOT excluded: they
are the sec.2.6 sanity anchor and must be scoreable against the three existing
answer files. Only the *generated* duplicates of the ambiguous facts below are
excluded. (So e.g. `rest_q01` legacy stays; the generated fact-question
`rest_s01_02` on the same "years at Marubeni" fact is excluded.)

Counts (restaurant): 33 generated questions excluded (novel: 0 — the bleed-in
issue is restaurant-only).

## 1. Whole-session exclusions — restaurant sessions 19, 24, 32, 33, 35
Reason: novel-topic bleed-in in the restaurant chat (RESEARCH_PROGRAM sec.2.5 /
2026-07-05 inventory). All 30 generated questions sourced from these five
sessions (6 each) are excluded regardless of the individual fact. This is a
session-granularity exclusion taken verbatim from the canon; the skeleton
`key_facts` for these sessions read as clean restaurant content, but the
contamination is in the underlying chat turns, so the whole session is dropped
to be safe.

- session 19 (Grand opening preparation): rest_s19_01 .. rest_s19_06
- session 24 (Year-end review): rest_s24_01 .. rest_s24_06
- session 32 (Equipment failure): rest_s32_01 .. rest_s32_06
- session 33 (Six-month milestone): rest_s33_01 .. rest_s33_06
- session 35 (Second location): rest_s35_01 .. rest_s35_06

NOTE: novel sessions 19/24/32/33/35 are NOT excluded — those session numbers
are legitimate in the novel chat (e.g. novel session 19 is the intended
profession-revision update-tracking fact). The exclusion applies to the
restaurant chat only.

## 2. "11 vs 15 years at Marubeni" (Q1)
Reason: internally inconsistent fact (RESEARCH_PROGRAM sec.2.5). The skeleton
says 11 years (session 1); truncation/summary model answers assert 15/fifteen.
The generated fact-question on this value is excluded.

- rest_s01_02  (gold "11 years")

## 3. "12,000 vs 18,000 yen Marubeni event" ambiguity (Q13)
Reason: the former employer (Marubeni) hosts two different private dinners at
two different per-person prices — session 23: 12,000 yen (24-guest December
dinner); session 39: 18,000 yen (14-person Sala Modena October event). A
question "what per-person price for the Marubeni event" is ambiguous, so BOTH
generated price questions are excluded.

- rest_s23_02  (gold "12,000 yen per person")
- rest_s39_03  (gold "18,000 yen per person")

## 4. "Trattoria Modena" name inconsistency
Reason: `new_sessions/session_37_en.json` and the late chat turns use
"Trattoria Modena" where the restaurant is "Osteria Morishita"
(RESEARCH_PROGRAM sec.2.5). Any generated question whose `fact_quote` contains
the string "Trattoria Modena" is excluded automatically by `gen_questions.py`.
No authored skeleton fact currently carries that string (the name inconsistency
lives in the chat body, not the skeleton `key_facts`), so this rule matches 0
questions today; it is retained as a defensive filter so the exclusion survives
any future re-authoring from the contaminated sessions.

## Tier-2 pending (not an exclusion, recorded for context)
The two inconsistent-fact answers (15 years; 18,000 yen) that appear in the
truncated/summarized legacy answer files are surfaced by the scorer as
tier-1 `indeterminate` and written to `pending_tier2.json` rather than being
auto-scored. They are the reason facts 2 and 3 above are excluded from the
generated set: the instrument correctly refuses to silently resolve them.

---

# v1.1 (2026-07-08)

Pilot iteration-2 hygiene pass (see PROTOCOL.md "Validation results
(2026-07-07/08 pilot, iterations 1-2)"). All waves are keyed by qid in
`gen_questions.py` (`RESTAURANT_V11_*` tables / `NOVEL_V11_DROP_REASON`);
excluded items keep their qid and carry `excluded: true` + a reason.

## Exclusion waves

| wave | count | evidence file | reason |
|------|-------|---------------|--------|
| Restaurant sessions 26, 40 (whole-session) | 12 | `results/audit/audit_batch_*.json` | session contamination confirmed in the ceiling/audit pass |
| World-guessable qids | 19 | `results/pilot/floor_leak_analysis.json` (`world_guessable`) | no-context floor arm produced the gold without the chat -> cannot separate conditions |
| `rest_s10_01` (self-leak) | 1 | `results/pilot/floor_leak_analysis.json` (`self_leak`) | the question text contains its own gold |
| `rest_s15_05` (yes/no + non-unique) | 1 | `results/audit/mechanical_screens.json` (`yes_no_form`) + audit | yes/no surface form; two equally valid answers (focaccia / roasted secondi) |
| `rest_s07_06` (non-unique gold) | 1 | `results/audit/audit_batch_0.json` | session names four capital sources (savings, uncle investment, equipment financing, JFC loan); "what other source" has no single answer (gold_correct=no, uniquely_determined=no; ceiling arm answered "uncle investment") |
| Novel generated layer (whole) | 151 | (source-data defect) | novel layer dropped: `novel_chat.json` sessions 1-18 duplicate the restaurant chat, so the true novel thread is only ~20K tokens; qids kept, `excluded: true` |

New restaurant exclusions in v1.1: **34** (12 + 19 + 3). Combined with the 33
pre-v1.1 exclusions -> **67 restaurant generated excluded**.

## Gold correction

- `rest_s13_04`: gold `"4 years"` -> `"3 years"`, aliases `["three years"]`.
  Evidence: audit `rest_s13_04` and the ceiling arm both answered "3 years"; the
  session says "first three [years] as sous chef and the final year as head
  chef", so 3 years sous-chef tenure is correct and the old "4 years" was the
  total. (`results/audit/audit_batch_*.json`, ceiling `results/pilot/`.)

The second candidate gold defect, `rest_s07_06` (ceiling arm "uncle investment"
vs gold "loan"), was NOT corrected: the audit shows the gold is not uniquely
determined (four capital sources), so it is excluded rather than re-golded (see
the table above).

## Rewords

**52** generated restaurant questions reworded to LongMemEval-style indirect
references (`RESTAURANT_V11_REWORDS`; question text only, qid and gold
unchanged) so that no included question's gold/alias appears in another included
question's text. 51 came from the pilot leak analysis
(`results/pilot/leak_pairs.json`); **1 extension** (`rest_s12_02`) was added in
this pass — the empirical `leak_pairs.json` missed that `rest_s01_04`'s gold
"Ginza Trattoria" appeared verbatim in `rest_s12_02`'s text (floor arm did not
happen to guess it). Acceptance check (generated-included scope): **0**
cross-leaks.

## Final counts (after v1.1)

- **Restaurant**: 256 total = 16 legacy + 240 generated; included **189**
  (16 legacy + **173** generated-included), excluded **67** generated.
- **Novel**: 167 total = 16 legacy + 151 generated; included **16** (legacy
  sanity anchor only), excluded **151** (all generated, layer dropped).
- **Primary scored set** = 173 restaurant generated-included questions (the 16 +
  16 legacy records remain the sec.2.6 sanity anchor, scored against the three
  existing answer files, not part of the context-condition comparison).

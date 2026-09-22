"""Generate the frozen binary-question sets from session skeleton key_facts.

This script IS the record of question generation (RESEARCH_PROGRAM sec.2 rule 1):
the question data is authored inline below (RESTAURANT_QUESTIONS / NOVEL_QUESTIONS),
derived ONLY from the skeletons' key_facts (ground truth by construction), never
from any model output. Running the script deterministically writes:
    benchmark/bineval/questions_restaurant.json   (legacy rest_q01..16 + generated)
    benchmark/bineval/questions_novel.json         (legacy novel_q01..16 + generated)

No LLM / network calls happen here. Re-running reproduces byte-identical JSON.

Schema per item (RESEARCH_PROGRAM sec.4 WO-0):
    {
      qid, source_session, fact_quote, question, gold_short,
      tier1_aliases: [...], excluded: bool, exclusion_reason: str,
      legacy: bool          # true only for the 16 re-expressed original questions
    }

Exclusions (see EXCLUSIONS.md) applied at generation time:
  * RESTAURANT sessions 19, 24, 32, 33, 35  -> whole-session exclusion
    (novel-topic bleed-in in the restaurant chat, per canon sec.2.5).
  * "11 vs 15 years at Marubeni"  (the generated form of legacy Q1) -> excluded.
  * "12,000 vs 18,000 yen Marubeni event" (generated forms in s23 & s39) -> excluded.
  * Anything referencing "Trattoria Modena" (name inconsistency).
  * v1.1 (2026-07-08, pilot iteration 2): RESTAURANT sessions 26, 40
    (session contamination); 19 world-guessable qids; rest_s10_01 (self-leak);
    rest_s15_05 (yes/no + non-unique); rest_s07_06 (non-unique gold). See the
    RESTAURANT_V11_* tables below and the EXCLUSIONS.md "v1.1" section.
The legacy rest_q01..16 / novel_q01..16 records are retained un-excluded so the
sec.2.6 sanity acceptance run can score them against the three existing answer
files; only the *generated* duplicates of the ambiguous facts are excluded.

v1.1 also rewords ~52 generated restaurant questions to LongMemEval-style
indirect references (RESTAURANT_V11_REWORDS; question text only, qid/gold
unchanged) and corrects one gold (RESTAURANT_V11_GOLD_FIXES: rest_s13_04).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LONGCHAT = ROOT.parent / "longchat"

# whole-session exclusions for the RESTAURANT chat only (canon sec.2.5)
RESTAURANT_EXCLUDED_SESSIONS = {19, 24, 32, 33, 35}
BLEED_REASON = "session flagged for novel-topic bleed-in in restaurant chat (canon sec.2.5)"
TRATTORIA_REASON = "references 'Trattoria Modena' (name inconsistency vs 'Osteria Morishita')"

# ---------------------------------------------------------------------------
# v1.1 revision (2026-07-08) — pilot iteration-2 hygiene pass. See EXCLUSIONS.md
# "v1.1" section and PROTOCOL.md "Validation results (2026-07-07/08 pilot)".
# All changes below are keyed by the deterministically-derived qid
# (rest_s{session:02d}_{counter:02d}); the underlying RESTAURANT_GEN rows keep
# their position (and therefore their qid), so regeneration stays byte-stable.
# ---------------------------------------------------------------------------

# (a) Additional whole-session exclusions: sessions 26 and 40 showed
#     session-contamination in the ceiling/audit pass (results/audit).
RESTAURANT_V11_EXCLUDED_SESSIONS = {26, 40}
V11_SESSION_REASON = (
    "session contamination confirmed in the 2026-07-07/08 audit "
    "(results/audit/audit_batch_*.json); whole session dropped (v1.1)"
)

# (b) Point exclusions keyed by qid -> reason.
#   - 19 world_guessable qids: the no-context floor arm produced the gold WITHOUT
#     the chat, so they cannot separate context conditions
#     (results/pilot/floor_leak_analysis.json).
#   - rest_s10_01: the question text itself contains its own gold (self-leak,
#     results/pilot/floor_leak_analysis.json "self_leak").
#   - rest_s15_05: yes/no surface form and non-unique answer
#     (results/audit/mechanical_screens.json "yes_no_form";
#      audit uniquely_determined=no).
#   - rest_s07_06: gold not uniquely determined — the session names four capital
#     sources (savings, uncle investment, equipment financing, JFC loan), so
#     "combined savings with what other source?" has no single answer; the
#     ceiling arm answered "uncle investment" while gold is "loan"
#     (results/audit/audit_batch_0.json rest_s07_06, gold_correct=no,
#      uniquely_determined=no).
_WORLD_GUESSABLE = [
    "rest_s01_01", "rest_s04_06", "rest_s06_06", "rest_s08_05", "rest_s08_06",
    "rest_s09_03", "rest_s09_05", "rest_s10_03", "rest_s12_05", "rest_s14_04",
    "rest_s20_05", "rest_s21_03", "rest_s21_06", "rest_s29_03", "rest_s30_04",
    "rest_s31_06", "rest_s34_06", "rest_s37_04", "rest_s38_04",
]
RESTAURANT_V11_POINT_EXCLUDED = {
    **{q: "floor arm guessed the gold with no context (world-guessable; "
          "results/pilot/floor_leak_analysis.json)" for q in _WORLD_GUESSABLE},
    "rest_s10_01": "self-leak: question text contains its own gold "
                   "(results/pilot/floor_leak_analysis.json 'self_leak')",
    "rest_s15_05": "yes/no surface form and non-unique answer "
                   "(results/audit/mechanical_screens.json 'yes_no_form'; "
                   "audit uniquely_determined=no)",
    "rest_s07_06": "gold not uniquely determined: session names four capital "
                   "sources, so 'what other source' has no single answer "
                   "(results/audit/audit_batch_0.json rest_s07_06)",
}

# (c) Gold corrections keyed by qid -> (new_gold, new_aliases). Applied AFTER
#     exclusion resolution (only meaningful for still-included qids).
#   - rest_s13_04: session says "first three [years] as sous chef and the final
#     year as head chef" -> 3 years as sous chef (audit rest_s13_04 and the
#     ceiling arm both answered "3 years"; old gold "4 years" was the total).
RESTAURANT_V11_GOLD_FIXES = {
    "rest_s13_04": ("3 years", ["three years"]),
}

# (d) Rewords keyed by qid -> (expected_old_question, new_question). LongMemEval
#     style: the specific entity that answers ANOTHER included question is
#     replaced by a unique role/context reference, keeping the SAME gold
#     answerable. Verified with the scoped cross-leak acceptance check
#     (generated-included set): zero cross-leaks remain. The expected_old text is
#     asserted at build time so a drift in the source row fails loudly rather
#     than silently mis-rewording.
RESTAURANT_V11_REWORDS = {
    # --- Tanabe (accountant) ---
    "rest_s03_06": (
        "On which street address is accountant Tanabe's office located?",
        "On which street address is the office of Kenta's accountant located?",
    ),
    # --- Hamada (real estate broker) ---
    "rest_s04_03": (
        "Which college friend introduced Kenta to the real estate broker Hamada?",
        "Which college friend introduced Kenta to his real estate broker?",
    ),
    "rest_s04_05": (
        "For which agency does Kenta's real estate broker Hamada work?",
        "For which agency does Kenta's real estate broker work?",
    ),
    # --- Ueno (produce broker), leaked via Tanabe answer 'Tanabe' ---
    "rest_s10_06": (
        "Through whom was produce broker Ueno introduced to Kenta?",
        "Through which of Kenta's advisers was the produce broker introduced to him?",
    ),
    # --- Nakata Shoji (cellar/wine supplier) ---
    "rest_s11_02": (
        "Who is Kenta's wine account manager at Nakata Shoji?",
        "Who is Kenta's wine account manager at the cellar supplier?",
    ),
    # --- Mariko Nakata (wine account manager) ---
    "rest_s11_03": (
        "What is the name of the sommelier friend who introduced Mariko Nakata to Kenta?",
        "What is the name of the sommelier friend who introduced Kenta's wine account manager to him?",
    ),
    # --- Petrelli (head chef) ---
    "rest_s13_02": (
        "How old is head chef Marco Petrelli?",
        "How old is the head chef?",
    ),
    "rest_s13_03": (
        "From which Italian city is head chef Marco Petrelli?",
        "From which Italian city is the head chef?",
    ),
    "rest_s13_04": (
        "For how many years was Petrelli sous chef at Ristorante Acqua Pazza?",
        "For how many years was the head chef a sous chef at Ristorante Acqua Pazza before joining?",
    ),
    "rest_s13_05": (
        "What monthly salary was head chef Petrelli hired at?",
        "What monthly salary was the head chef hired at?",
    ),
    "rest_s13_06": (
        "On what date did head chef Petrelli start?",
        "On what date did the head chef start?",
    ),
    "rest_s25_05": (
        "Which low-margin dish did Petrelli drop from the menu in January?",
        "Which low-margin dish did the head chef drop from the menu in January?",
    ),
    "rest_s25_06": (
        "Which new dish did Petrelli add to the menu in January?",
        "Which new dish did the head chef add to the menu in January?",
    ),
    "rest_s27_01": (
        "During what dates was Petrelli on vacation?",
        "During what dates was the head chef on vacation?",
    ),
    "rest_s27_02": (
        "Who ran the kitchen during Petrelli's February vacation?",
        "Who ran the kitchen during the head chef's February vacation?",
    ),
    "rest_s27_05": (
        "Which dish was dropped temporarily during Petrelli's vacation coverage?",
        "Which dish was dropped temporarily during the head chef's vacation coverage?",
    ),
    "rest_s27_06": (
        "During which dates was lunch service paused for Petrelli's coverage period?",
        "During which dates was lunch service paused for the head chef's coverage period?",
    ),
    "rest_s28_03": (
        "Which restaurant critic made an unexpected visit during Petrelli's absence?",
        "Which restaurant critic made an unexpected visit during the head chef's absence?",
    ),
    "rest_s29_05": (
        "On what date did Petrelli return from vacation?",
        "On what date did the head chef return from vacation?",
    ),
    "rest_s37_05": (
        "Who was identified as head-chef successor if Petrelli leaves?",
        "Who was identified as head-chef successor if the head chef leaves?",
    ),
    "rest_s38_05": (
        "What is Petrelli's new monthly base salary after the extension?",
        "What is the head chef's new monthly base salary after his contract extension?",
    ),
    # --- Nakajima (sous chef) ---
    "rest_s16_02": (
        "What monthly salary was sous chef Nakajima hired at?",
        "What monthly salary was the sous chef hired at?",
    ),
    "rest_s27_03": (
        "What temporary monthly pay bump did Nakajima get while running the kitchen?",
        "What temporary monthly pay bump did the sous chef get while running the kitchen?",
    ),
    "rest_s28_05": (
        "What was the average dinner ticket time under Nakajima during coverage?",
        "What was the average dinner ticket time under the sous chef during coverage?",
    ),
    "rest_s38_01": (
        "To what monthly salary was Nakajima raised in the performance-review adjustments?",
        "To what monthly salary was the sous chef raised in the performance-review adjustments?",
    ),
    # --- Sawada (service manager) ---
    "rest_s16_04": (
        "What monthly salary was service manager Sawada hired at?",
        "What monthly salary was the service manager hired at?",
    ),
    "rest_s18_06": (
        "Which table did service manager Sawada flag as having an acoustic issue?",
        "Which table did the service manager flag as having an acoustic issue?",
    ),
    # --- 80 soft-opening guests (s17_03 gold 80) ---
    "rest_s18_04": (
        "How many of the 80 soft-opening guests rated the meal 8/10 or higher?",
        "How many of the soft-opening guests rated the meal 8/10 or higher?",
    ),
    # --- three quotes (s06_05) leaked as bare '3' ---
    "rest_s20_04": (
        "What was the restaurant's first Tabelog rating after 3 reviews?",
        "What was the restaurant's very first Tabelog rating, recorded after only a handful of reviews?",
    ),
    "rest_s31_03": (
        "To what price was the 3-course lunch raised in the April 2026 revision?",
        "To what price was the longer of the two lunch course sets raised in the April 2026 revision?",
    ),
    # --- Sara Komatsu (replacement part-time server) ---
    "rest_s21_05": (
        "How old is Sara Komatsu, the replacement part-time server?",
        "How old is the replacement part-time server hired that season?",
    ),
    # --- Mrs. Yoneda (dishwasher) ---
    "rest_s23_03": (
        "At what hourly rate was Mrs. Yoneda hired as dishwasher?",
        "At what hourly rate was the dishwasher hired?",
    ),
    "rest_s23_04": (
        "How many hours nightly does dishwasher Mrs. Yoneda work?",
        "How many hours nightly does the dishwasher work?",
    ),
    # --- 3.61 / 24 reviews mutual pair (s28_01 gold 3.61, s28_02 gold 24 reviews) ---
    "rest_s28_01": (
        "To what Tabelog rating did the restaurant climb after 24 reviews?",
        "To what Tabelog rating did the restaurant climb by the time it had accumulated roughly two dozen reviews?",
    ),
    "rest_s28_02": (
        "How many Tabelog reviews had the restaurant received when it reached 3.61?",
        "How many Tabelog reviews had the restaurant accumulated at the rating check-in that followed the head chef's February vacation?",
    ),
    # --- Hiroko Sano (restaurant critic) ---
    "rest_s28_04": (
        "For which publication does critic Hiroko Sano write?",
        "For which publication does the restaurant critic who paid the unexpected visit write?",
    ),
    "rest_s30_06": (
        "In which publication was the Sano review published?",
        "In which publication was the restaurant critic's review published?",
    ),
    # --- Asahi Shimbun Weekly (the critic's publication) ---
    "rest_s28_06": (
        "When was the Asahi Shimbun critic's article expected to be published?",
        "When was the visiting critic's article expected to be published?",
    ),
    "rest_s30_01": (
        "On what date was the Asahi Shimbun review published?",
        "On what date was the critic's review actually published?",
    ),
    "rest_s30_02": (
        "What was the headline of the Asahi Shimbun review (Azabu's Quiet ...)?",
        "What was the headline of the critic's published review?",
    ),
    # --- tagliatelle al ragu bolognese (signature dish) ---
    "rest_s15_01": (
        "What was the initial price of the signature tagliatelle al ragu bolognese?",
        "What was the initial price of the restaurant's signature hand-cut pasta dish?",
    ),
    "rest_s31_01": (
        "To what price was the tagliatelle al ragu bolognese raised in the April 2026 revision?",
        "To what price was the signature hand-cut pasta dish raised in the April 2026 revision?",
    ),
    # --- Yamamoto Koumuten / Tetsuo Yamamoto (chosen construction company + its head) ---
    "rest_s36_02": (
        "What was Yamamoto Koumuten's cost estimate for the private-room conversion?",
        "What was the chosen construction company's cost estimate for the private-room conversion?",
    ),
    # --- Mr. Endo (labor consultant) ---
    "rest_s37_03": (
        "What fee did labor consultant Mr. Endo charge for the employee handbook?",
        "What fee did the labor consultant charge for the employee handbook?",
    ),
    # --- Yui Tachibana (part-timer) ---
    "rest_s38_02": (
        "To what hourly rate was Yui Tachibana raised in the performance-review adjustments?",
        "To what hourly rate was the longest-tenured part-time server raised in the performance-review adjustments?",
    ),
    # --- 24 reviews alias leaking into '24 months' (s38_03) ---
    "rest_s38_03": (
        "What retention bonus was formalized at 24 months?",
        "What retention bonus was formalized under the two-year tenure milestone?",
    ),
    # --- Sala Modena (private dining room) ---
    "rest_s39_01": (
        "On what date did Marubeni book the Sala Modena private room?",
        "On what date did Marubeni book the restaurant's private dining room?",
    ),
    "rest_s39_02": (
        "For how many people did Marubeni book the Sala Modena dinner?",
        "For how many people did Marubeni book the private-room dinner?",
    ),
    "rest_s39_04": (
        "What venue fee applies to the Sala Modena private room?",
        "What venue fee applies to the private dining room?",
    ),
    "rest_s39_05": (
        "At what price do the Sala Modena wine pairings start?",
        "At what price do the private-room wine pairings start?",
    ),
    "rest_s39_06": (
        "How many private events were pre-booked for September in the Sala Modena?",
        "How many private events were pre-booked for September in the private dining room?",
    ),
    # --- EXTENSION (this agent, 2026-07-08): 'Ginza Trattoria' (rest_s01_04 gold)
    #     leaked into rest_s12_02's text; the empirical leak_pairs.json missed it
    #     because the floor arm did not happen to guess it. Removed the literal
    #     name while keeping gold "Azabu" uniquely resolvable. ---
    "rest_s12_02": (
        "In which district is the restaurant located, making the name 'Ginza Trattoria' confusing?",
        "In which district is the restaurant located, a mismatch that made its original working name confusing?",
    ),
}

# ---------------------------------------------------------------------------
# Legacy 16 (re-expressed original questions). gold/answer copied verbatim from
# benchmark/longchat/restaurant_questions.json + novel_questions.json.
# These carry legacy=True and stay un-excluded (sanity anchor).
# ---------------------------------------------------------------------------

LEGACY_RESTAURANT = [
    # (qid, session, fact_quote, question, gold_short, aliases)
    ("rest_q01", 1, "working at Marubeni Corporation for 11 years",
     "How many years had Kenta Morishita worked at Marubeni Corporation before deciding to open a restaurant?",
     "11 years", ["11"]),
    ("rest_q02", 3, "office is at 2-7-3 Shibuya, Daiichi-Bunko Building 5F",
     "In which building and floor is the office of Kenta's accountant Hiroshi Tanabe located?",
     "Daiichi-Bunko Building 5F", ["Daiichi-Bunko Building", "5F"]),
    ("rest_q03", 5, "key money came out to 2,922,000 yen",
     "What was the key money amount Kenta paid when signing the lease for the Higashi-Azabu restaurant space?",
     "2,922,000 yen", ["2,922,000"]),
    ("rest_q04", 11, "we settled on 47 labels for the opening list",
     "How many distinct wine labels were included in Osteria Morishita's opening cellar order from Nakata Shoji?",
     "47 labels", ["47"]),
    ("rest_q05", 13, "most recent position was at Ristorante Acqua Pazza",
     "At which Daikanyama restaurant had head chef Marco Petrelli worked most recently before joining Osteria Morishita?",
     "Ristorante Acqua Pazza", ["Acqua Pazza"]),
    ("rest_q06", 14, "The health inspector's name was Mr. Ogawa",
     "What was the name of the health inspector who conducted the initial permit inspection of Osteria Morishita?",
     "Mr. Ogawa", ["Ogawa"]),
    ("rest_q07", 15, "2,400 yen for the tagliatelle",
     "What was the original price of the signature tagliatelle al ragu bolognese when the menu was first finalized?",
     "2,400 yen", ["2,400"]),
    ("rest_q08", 23, "bringing her in at 1,400 yen per hour",
     "What hourly rate was Mrs. Yoneda hired at when she joined Osteria Morishita as dishwasher in December 2025?",
     "1,400 yen per hour", ["1,400 yen", "1,400"]),
    ("rest_q09", 12, "Decision to rename from 'Ginza Trattoria' to 'Osteria Morishita'",
     "What is the current name of Kenta's restaurant? (It was changed from the original working name.)",
     "Osteria Morishita", []),
    ("rest_q10", 31, "raised from 2,400 to 2,700 yen",
     "What is the current price of the tagliatelle al ragu bolognese after the April 2026 menu revision?",
     "2,700 yen", ["2,700"]),
    ("rest_q11", 28, "currently sitting at a 3.61 rating after 24 reviews",
     "What is Osteria Morishita's most recent Tabelog rating mentioned in the chat? (It started at 3.42 and rose over time.)",
     "3.61", []),
    ("rest_q12", 31, "the 2-course went up to 2,000 yen",
     "What is the current price of the 2-course lunch set at Osteria Morishita after the April 2026 revision? (It was originally 1,800 yen.)",
     "2,000 yen", ["2,000"]),
    ("rest_q13", 23, "We landed on a 12,000 yen per person fixed menu",
     "Kenta worked for years at a large trading company before opening his restaurant. That same former employer later hosted a private dinner at Osteria Morishita. What per-person price was charged for that corporate dinner?",
     "12,000 yen per person", ["12,000 yen", "12,000"]),
    ("rest_q14", 14, "four days past the contractual target",
     "The renovation of Kenta's Higashi-Azabu property was completed late. How many days past the projected completion date did the renovation finish?",
     "four days", ["4 days", "four", "4"]),
    ("rest_q15", 11, "Mariko introduced by sommelier friend Reiko Igarashi from Hotel Okura",
     "Kenta's wine account manager Mariko Nakata was originally introduced to Kenta by a sommelier friend. Where did that sommelier friend work?",
     "Hotel Okura", ["Okura"]),
    ("rest_q16", 29, "Cantina Morishita ... currently has 23 members signed",
     "When Kenta reviewed Q1 2026 results, how many members had signed up for the wine club Cantina Morishita?",
     "23 members", ["23"]),
]

LEGACY_NOVEL = [
    ("novel_q01", 1, "working title 'The Hollow Years'",
     "What was the original working title Maya was using for her novel when she first described the concept in session 1?",
     "The Hollow Years", ["Hollow Years"]),
    ("novel_q02", 1, "protagonist Eleanor Reyes, a marine biologist",
     "What was Eleanor Reyes's profession as described in the original session 1 concept pitch?",
     "marine biologist", []),
    ("novel_q03", 1, "initial word count target 50000",
     "What word count target did Maya set for her novel in session 1?",
     "50,000 words", ["50,000", "50000"]),
    ("novel_q04", 1, "setting: Stonington, Maine",
     "In what Maine town is the novel set?",
     "Stonington, Maine", ["Stonington"]),
    ("novel_q05", 18, "at Annie Bloom's Books for eleven years",
     "At which bookstore did beta reader Quentin Marsh work, according to session 18?",
     "Annie Bloom's Books", ["Annie Bloom"]),
    ("novel_q06", 20, "tentatively chose 'Salt and Compass' as new working title",
     "What new working title did Maya tentatively choose during the title brainstorm session in session 20?",
     "Salt and Compass", []),
    ("novel_q07", 14, "revised word count after structural pass: 81200",
     "What was the revised word count of the manuscript after Maya completed the structural revision pass in session 14?",
     "81200", ["81,200"]),
    ("novel_q08", 15, "provided 12-question feedback form",
     "How many questions did Maya include in the feedback form she sent to beta readers in session 15?",
     "12-question feedback form", ["12-question", "12 questions", "12"]),
    ("novel_q09", 37, "final title locked as 'The Eelgrass Year'",
     "What is the FINAL locked title of Maya's novel as confirmed in session 37?",
     "The Eelgrass Year", ["Eelgrass Year"]),
    ("novel_q10", 19, "revise Eleanor's profession from marine biologist to marine ecologist studying eelgrass die-off",
     "What is Eleanor Reyes's profession in the revised manuscript, after the change implemented in session 19?",
     "marine ecologist studying eelgrass die-off", ["marine ecologist"]),
    ("novel_q11", 30, "the research vessel renamed Cordelia",
     "What is the name of Eleanor's research vessel in the final pre-submission manuscript (session 30), after it was renamed?",
     "Cordelia", []),
    ("novel_q12", 37, "publication date pushed from March 9 2027 to September 14, 2027",
     "What is the FINAL confirmed publication date for The Eelgrass Year, after it was moved from the originally proposed date?",
     "September 14, 2027", ["September 14 2027"]),
    ("novel_q13", 38, "confirmed launch event for September 16, 2027",
     "Quentin Marsh offered to host a launch event for Maya's novel. What specific date was that launch event confirmed for in session 38?",
     "September 16, 2027", ["September 16 2027"]),
    ("novel_q14", 5, "Eleanor's research vessel named Persephone",
     "Eleanor's research vessel was introduced under a name in session 5. What was that original name before it was renamed in session 30?",
     "Persephone", []),
    ("novel_q15", 14, "Lillian reveals she always blamed Camille for father's stress",
     "When Eleanor visits her mother Lillian in session 14, which sister does Lillian reveal she always blamed for their father's stress?",
     "Camille", []),
    ("novel_q16", 30, "final pre-submission word count 89700",
     "What was the actual word count of the manuscript when it was sent to editors on submission in session 30?",
     "89700", ["89,700"]),
]

# ---------------------------------------------------------------------------
# Generated restaurant questions (authored from skeleton key_facts, ~6/session).
# Format per row: (session, fact_quote, question, gold_short, aliases,
#                  excluded_override, reason_override)
# excluded_override=None means "derive from session rule"; a string forces
# exclusion with that reason (used for the ambiguous Marubeni facts).
# ---------------------------------------------------------------------------

E = None  # shorthand: no per-fact exclusion override (session rule applies)

# v1.1 (2026-07-08): the whole NOVEL generated layer is dropped. The novel chat
# source (novel_chat.json) has a data defect — sessions 1-18 duplicate the
# restaurant chat, so the true novel thread is only ~20K tokens and cannot carry
# the intended long-context load. qids are kept (excluded=true) so the record is
# auditable and can be revived if the source is rebuilt.
NOVEL_V11_DROP_REASON = (
    "novel layer dropped v1.1: novel_chat.json data defect (sessions 1-18 "
    "duplicate restaurant chat; true novel thread ~20K tokens)"
)

RESTAURANT_GEN = [
    # ---- Session 1 ----
    (1, "User name: Kenta Morishita, age 34",
     "What is the age of Kenta Morishita, the user planning the restaurant, as stated at the start?",
     "34", [], E, ""),
    (1, "Quitting job at Marubeni Corporation after 11 years",
     "How many years did Kenta work at Marubeni Corporation before quitting?",
     "11 years", ["11"], "the '11 vs 15 years at Marubeni' inconsistency (legacy Q1); generated duplicate excluded", ""),
    (1, "Savings: 28 million yen",
     "How much personal savings did Kenta have when he started planning the restaurant?",
     "28 million yen", ["28 million"], E, ""),
    (1, "Initial restaurant name idea: 'Ginza Trattoria'",
     "What was Kenta's initial working name idea for the restaurant in session 1?",
     "Ginza Trattoria", [], E, ""),
    (1, "studied at Hotel School of Bologna for 3 weeks",
     "For how long did Kenta study at the Hotel School of Bologna during his 2019 trip?",
     "3 weeks", ["three weeks"], E, ""),
    (1, "Wife: Ayako Morishita",
     "What is the name of Kenta's wife who supports the restaurant plan?",
     "Ayako Morishita", ["Ayako"], E, ""),
    # ---- Session 2 ----
    (2, "Visited 14 Italian restaurants in 2 weeks",
     "How many Italian restaurants did Kenta visit during his two weeks of market research?",
     "14", [], E, ""),
    (2, "Average ticket size target: 8,500 yen per person dinner",
     "What dinner average ticket size per person did Kenta target during market research?",
     "8,500 yen", ["8,500"], E, ""),
    (2, "Lunch target: 1,800 yen set",
     "What lunch set price did Kenta target during market research?",
     "1,800 yen", ["1,800"], E, ""),
    (2, "Benchmark restaurants: Ristorante Aso in Daikanyama",
     "Which Daikanyama benchmark restaurant did Kenta name during market research (Ristorante ...)?",
     "Ristorante Aso", ["Aso"], E, ""),
    (2, "Identified gap: northern Italian (Emilia-Romagna) underrepresented",
     "Which region of Italian cuisine did Kenta identify as an underrepresented gap in Tokyo?",
     "Emilia-Romagna", ["northern Italian"], E, ""),
    (2, "Heinz Beck in Marunouchi",
     "In which Tokyo district did Kenta name Heinz Beck as a benchmark restaurant?",
     "Marunouchi", [], E, ""),
    # ---- Session 3 ----
    (3, "Projected monthly revenue at full capacity: 4.2 million yen",
     "What monthly revenue at full capacity did Kenta's business plan project?",
     "4.2 million yen", ["4.2 million"], E, ""),
    (3, "Breakeven: 38 covers per day average",
     "How many covers per day on average did the business plan set as breakeven?",
     "38 covers", ["38"], E, ""),
    (3, "Working with accountant: Hiroshi Tanabe of Tanabe & Partners",
     "What is the name of Kenta's accountant?",
     "Hiroshi Tanabe", ["Tanabe"], E, ""),
    (3, "Loan plan: Japan Finance Corporation, applying for 15 million yen at 1.85%",
     "How much loan did Kenta initially apply for from Japan Finance Corporation?",
     "15 million yen", ["15 million"], E, ""),
    (3, "applying for 15 million yen at 1.85%",
     "At what interest rate did Kenta initially apply for the Japan Finance Corporation loan?",
     "1.85%", ["1.85"], E, ""),
    (3, "Tanabe office address: 2-7-3 Shibuya",
     "On which street address is accountant Tanabe's office located?",
     "2-7-3 Shibuya", ["Shibuya"], E, ""),
    # ---- Session 4 ----
    (4, "Scouted 23 properties via real estate broker Yusuke Hamada of Plaza Homes Commercial",
     "How many properties did Kenta scout with his real estate broker?",
     "23 properties", ["23"], E, ""),
    (4, "real estate broker Yusuke Hamada of Plaza Homes Commercial",
     "What is the name of the real estate broker who helped Kenta scout properties?",
     "Yusuke Hamada", ["Hamada"], E, ""),
    (4, "Hamada cellphone introduced through college friend Daichi Sasaki",
     "Which college friend introduced Kenta to the real estate broker Hamada?",
     "Daichi Sasaki", ["Sasaki"], E, ""),
    (4, "Ruled out Ginza due to rent (above 95,000 yen per tsubo)",
     "Above what per-tsubo rent did Kenta rule out Ginza as a location?",
     "95,000 yen per tsubo", ["95,000 yen", "95,000"], E, ""),
    (4, "broker Yusuke Hamada of Plaza Homes Commercial",
     "For which agency does Kenta's real estate broker Hamada work?",
     "Plaza Homes Commercial", ["Plaza Homes"], E, ""),
    (4, "Areas considered: Nishi-Azabu, Higashi-Azabu, Hiroo, Yoyogi-Uehara, Kagurazaka",
     "Name one of the neighborhoods Kenta considered besides Higashi-Azabu (e.g. Yoyogi-...).",
     "Yoyogi-Uehara", ["Nishi-Azabu", "Hiroo", "Kagurazaka"], E, ""),
    # ---- Session 5 ----
    (5, "Signed lease at 3-14-7 Higashi-Azabu, Minato-ku, 1st floor of Azabu Heights Building",
     "In which building did Kenta sign the lease for his restaurant?",
     "Azabu Heights Building", ["Azabu Heights"], E, ""),
    (5, "Floor area: 64.5 square meters (about 19.5 tsubo)",
     "What is the floor area in square meters of the leased restaurant space?",
     "64.5 square meters", ["64.5"], E, ""),
    (5, "Rent: 487,000 yen per month",
     "What is the monthly rent for Kenta's restaurant space?",
     "487,000 yen", ["487,000"], E, ""),
    (5, "plus 73,000 yen common area fee",
     "What is the monthly common area fee for the restaurant space?",
     "73,000 yen", ["73,000"], E, ""),
    (5, "Lease term: 4 years renewable",
     "What is the lease term length for the restaurant space?",
     "4 years", ["four years"], E, ""),
    (5, "Previous tenant: small French bistro that closed October 2024",
     "What kind of business was the previous tenant of the restaurant space?",
     "French bistro", ["bistro"], E, ""),
    # ---- Session 6 ----
    (6, "Selected Yamamoto Koumuten led by Tetsuo Yamamoto",
     "Which contractor did Kenta select for the renovation?",
     "Yamamoto Koumuten", ["Yamamoto"], E, ""),
    (6, "led by Tetsuo Yamamoto",
     "Who leads the contractor firm Kenta selected for renovation?",
     "Tetsuo Yamamoto", ["Yamamoto"], E, ""),
    (6, "Renovation budget: 8.4 million yen",
     "What was the initial renovation budget?",
     "8.4 million yen", ["8.4 million"], E, ""),
    (6, "Yamamoto suggested supplier 'Forno Bravo Japan' for the wood-fired oven",
     "Which supplier did the contractor suggest for the wood-fired oven?",
     "Forno Bravo Japan", ["Forno Bravo"], E, ""),
    (6, "Three quotes from contractors: Kenchiku Plus, Yamamoto Koumuten, Sasaki Design Build",
     "How many contractor quotes did Kenta obtain for the renovation?",
     "three quotes", ["3 quotes", "three", "3"], E, ""),
    (6, "use open kitchen with brick-faced pizza oven facing dining room",
     "What kitchen layout did the contractor recommend (with the oven facing the dining room)?",
     "open kitchen", [], E, ""),
    # ---- Session 7 ----
    (7, "Japan Finance Corporation approved 12 million yen (3 million below request) at 1.95%",
     "How much did the Japan Finance Corporation ultimately approve for Kenta's loan?",
     "12 million yen", ["12 million"], E, ""),
    (7, "approved 12 million yen (3 million below request) at 1.95%",
     "At what interest rate was Kenta's Japan Finance Corporation loan finally approved?",
     "1.95%", ["1.95"], E, ""),
    (7, "Total opening capital secured: 36 million yen",
     "What total opening capital did Kenta secure after the loan approval?",
     "36 million yen", ["36 million"], E, ""),
    (7, "12 million yen (3 million below request)",
     "By how much was Kenta's approved loan below his original request?",
     "3 million", ["3 million yen", "three million"], E, ""),
    (7, "Decision to scale back interior finishing to compensate",
     "What did Kenta decide to scale back to compensate for the smaller loan?",
     "interior finishing", [], E, ""),
    (7, "28M savings + 12M loan minus expenses already incurred",
     "The 36 million opening capital combined savings with what other source?",
     "loan", [], E, ""),
    # ---- Session 8 ----
    (8, "Renovation started June 9, 2025, projected completion August 22",
     "On what date was the renovation projected to be completed (at kickoff)?",
     "August 22", [], E, ""),
    (8, "Renovation started June 9, 2025",
     "On what date did the renovation start?",
     "June 9, 2025", ["June 9"], E, ""),
    (8, "Floor: white oak engineered plank from Toyo Wood Works",
     "What flooring material was chosen for the restaurant?",
     "white oak engineered plank", ["white oak"], E, ""),
    (8, "white oak engineered plank from Toyo Wood Works",
     "Which company supplied the restaurant's flooring?",
     "Toyo Wood Works", ["Toyo Wood"], E, ""),
    (8, "Walls: limestone-finish plaster (Italian product Marmorino imported via Tokai Sangyo)",
     "What Italian wall-plaster product was used in the restaurant?",
     "Marmorino", [], E, ""),
    (8, "single accessible unisex instead of split",
     "What bathroom layout did the architect suggest instead of a split layout?",
     "single accessible unisex", ["unisex"], E, ""),
    # ---- Session 9 ----
    (9, "Wood-fired pizza oven ordered from Forno Bravo Japan: model 'Modena 100', cost 1.6 million yen",
     "What is the model name of the wood-fired pizza oven Kenta ordered?",
     "Modena 100", [], E, ""),
    (9, "model 'Modena 100', cost 1.6 million yen",
     "How much did the wood-fired pizza oven cost?",
     "1.6 million yen", ["1.6 million"], E, ""),
    (9, "Pasta extruder: La Monferrina P3, cost 580,000 yen",
     "What model of pasta extruder did Kenta buy?",
     "La Monferrina P3", ["La Monferrina"], E, ""),
    (9, "Pasta extruder: La Monferrina P3, cost 580,000 yen",
     "How much did the pasta extruder cost?",
     "580,000 yen", ["580,000"], E, ""),
    (9, "Refrigeration: 2 Hoshizaki upright units, total 720,000 yen",
     "Which brand of refrigeration units did Kenta buy?",
     "Hoshizaki", [], E, ""),
    (9, "Decided against rotisserie - too niche for menu",
     "What piece of equipment did Kenta decide against as too niche for the menu?",
     "rotisserie", [], E, ""),
    # ---- Session 10 ----
    (10, "Olive oil supplier: Solleone Japan, sourcing Frantoio Franci EVOO at 4,200 yen per 500ml",
     "Which olive oil brand did Kenta source at opening (Frantoio ...)?",
     "Frantoio Franci", ["Frantoio"], E, ""),
    (10, "Frantoio Franci EVOO at 4,200 yen per 500ml",
     "What was the price per 500ml of the opening olive oil?",
     "4,200 yen", ["4,200"], E, ""),
    (10, "Flour: Caputo 00 imported through Suzuki Foods, 3,800 yen per 25kg sack",
     "Which flour did Kenta use, imported through Suzuki Foods?",
     "Caputo 00", ["Caputo"], E, ""),
    (10, "Produce: morning delivery from Tsukiji-Ota Market via broker Kenji Ueno of Ueno Seika",
     "What is the name of the produce broker who delivers to the restaurant each morning?",
     "Kenji Ueno", ["Ueno"], E, ""),
    (10, "3,800 yen per 25kg sack",
     "What was the price per 25kg sack of the restaurant's flour?",
     "3,800 yen", ["3,800"], E, ""),
    (10, "Ueno introduced through accountant Tanabe",
     "Through whom was produce broker Ueno introduced to Kenta?",
     "Tanabe", ["accountant Tanabe"], E, ""),
    # ---- Session 11 ----
    (11, "Wine supplier: Nakata Shoji",
     "What is the name of Kenta's wine supplier company?",
     "Nakata Shoji", [], E, ""),
    (11, "Account manager: Mariko Nakata, daughter of company president",
     "Who is Kenta's wine account manager at Nakata Shoji?",
     "Mariko Nakata", ["Mariko"], E, ""),
    (11, "Mariko introduced by sommelier friend Reiko Igarashi from Hotel Okura",
     "What is the name of the sommelier friend who introduced Mariko Nakata to Kenta?",
     "Reiko Igarashi", ["Igarashi"], E, ""),
    (11, "Initial cellar: 280 bottles across 47 labels",
     "How many bottles were in the initial wine cellar?",
     "280 bottles", ["280"], E, ""),
    (11, "60% Italian, 25% French, 15% Japanese",
     "What percentage of the initial cellar was Italian wine?",
     "60%", ["60"], E, ""),
    (11, "Opening cost for wine inventory: 1.85 million yen",
     "What was the opening cost for the wine inventory?",
     "1.85 million yen", ["1.85 million"], E, ""),
    # ---- Session 12 ----
    (12, "Decision to rename from 'Ginza Trattoria' to 'Osteria Morishita'",
     "To what name did Kenta rename the restaurant in session 12?",
     "Osteria Morishita", [], E, ""),
    (12, "'Ginza Trattoria' confusing since restaurant is in Azabu",
     "In which district is the restaurant located, making the name 'Ginza Trattoria' confusing?",
     "Azabu", [], E, ""),
    (12, "Logo designed by Aiko Fujimoto, freelance designer, fee 320,000 yen",
     "Who designed the restaurant's logo?",
     "Aiko Fujimoto", ["Fujimoto"], E, ""),
    (12, "fee 320,000 yen",
     "What fee did the logo designer charge?",
     "320,000 yen", ["320,000"], E, ""),
    (12, "Brand colors: terracotta and ivory",
     "What two brand colors were chosen for the restaurant?",
     "terracotta and ivory", ["terracotta", "ivory"], E, ""),
    (12, "Domain registered: osteria-morishita.tokyo",
     "What web domain did Kenta register for the restaurant?",
     "osteria-morishita.tokyo", [], E, ""),
    # ---- Session 13 ----
    (13, "Hired head chef: Marco Petrelli, 38 years old, Italian national from Modena",
     "What is the name of the head chef Kenta hired?",
     "Marco Petrelli", ["Petrelli"], E, ""),
    (13, "Marco Petrelli, 38 years old",
     "How old is head chef Marco Petrelli?",
     "38", [], E, ""),
    (13, "Italian national from Modena",
     "From which Italian city is head chef Marco Petrelli?",
     "Modena", [], E, ""),
    (13, "Petrelli previously sous chef at Ristorante Acqua Pazza in Daikanyama for 4 years",
     "For how many years was Petrelli sous chef at Ristorante Acqua Pazza?",
     "4 years", ["four years"], E, ""),
    (13, "Salary: 720,000 yen monthly, 13-month bonus structure",
     "What monthly salary was head chef Petrelli hired at?",
     "720,000 yen", ["720,000"], E, ""),
    (13, "Petrelli will start September 1, 2025",
     "On what date did head chef Petrelli start?",
     "September 1, 2025", ["September 1"], E, ""),
    # ---- Session 14 ----
    (14, "Renovation completed August 26 (4 days late)",
     "On what date was the renovation actually completed?",
     "August 26", [], E, ""),
    (14, "Final renovation cost: 8.87 million yen (470,000 yen over budget)",
     "What was the final total renovation cost?",
     "8.87 million yen", ["8.87 million"], E, ""),
    (14, "470,000 yen over budget",
     "By how much did the final renovation cost exceed budget?",
     "470,000 yen", ["470,000"], E, ""),
    (14, "Health permit inspector: Mr. Ogawa, Minato Ward Health Department",
     "From which ward's health department did inspector Mr. Ogawa come?",
     "Minato Ward", ["Minato"], E, ""),
    (14, "Overage due to electrical panel upgrade required by inspection",
     "What upgrade caused the renovation budget overage?",
     "electrical panel upgrade", ["electrical panel"], E, ""),
    (14, "renovation finished ... four days past the contractual target",
     "How many days late was the renovation completed?",
     "4 days", ["four days", "four", "4"], E, ""),
    # ---- Session 15 ----
    (15, "Signature dish: tagliatelle al ragu bolognese, hand-cut, 2,400 yen",
     "What was the initial price of the signature tagliatelle al ragu bolognese?",
     "2,400 yen", ["2,400"], E, ""),
    (15, "Final menu: 6 antipasti, 5 primi (3 pasta + 2 risotto), 4 secondi, 3 desserts",
     "How many antipasti were on the finalized menu?",
     "6 antipasti", ["6"], E, ""),
    (15, "Lunch menu: 2 courses 1,800 yen, 3 courses 2,400 yen",
     "What was the initial price of the 2-course lunch menu?",
     "1,800 yen", ["1,800"], E, ""),
    (15, "Dinner average ticket projection revised down: 7,800 yen per person",
     "To what per-person figure was the dinner average ticket projection revised down at menu finalization?",
     "7,800 yen", ["7,800"], E, ""),
    (15, "Pizza: not on menu - oven used for focaccia and roasted secondi only",
     "Was pizza on the finalized menu? (State what the oven is used for instead.)",
     "focaccia", ["not on menu", "roasted secondi"], E, ""),
    (15, "Signature dish: tagliatelle al ragu bolognese, hand-cut",
     "What is the signature pasta dish on the menu?",
     "tagliatelle al ragu bolognese", ["tagliatelle"], E, ""),
    # ---- Session 16 ----
    (16, "Sous chef hired: Haruki Nakajima, 29 years old, previously at Trattoria Goccia",
     "What is the name of the sous chef Kenta hired?",
     "Haruki Nakajima", ["Nakajima"], E, ""),
    (16, "Nakajima salary: 380,000 yen monthly",
     "What monthly salary was sous chef Nakajima hired at?",
     "380,000 yen", ["380,000"], E, ""),
    (16, "Service manager: Eiko Sawada, 41 years old, hotel background, salary 460,000 yen",
     "What is the name of the service manager Kenta hired?",
     "Eiko Sawada", ["Sawada"], E, ""),
    (16, "salary 460,000 yen",
     "What monthly salary was service manager Sawada hired at?",
     "460,000 yen", ["460,000"], E, ""),
    (16, "Total staff at opening: 5 full-time, 2 part-time",
     "How many full-time staff did the restaurant have at opening?",
     "5 full-time", ["5"], E, ""),
    (16, "Two part-time servers: Yui Tachibana (university student) and Naoki Hirose",
     "Name one of the two part-time servers hired at opening (Yui ...).",
     "Yui Tachibana", ["Naoki Hirose", "Tachibana", "Hirose"], E, ""),
    # ---- Session 17 ----
    (17, "Soft opening dates: October 8-11, 2025 (four days, dinner only)",
     "On what dates was the soft opening held?",
     "October 8-11, 2025", ["October 8-11", "October 8"], E, ""),
    (17, "four days, dinner only",
     "How many days did the soft opening run?",
     "four days", ["4 days", "four", "4"], E, ""),
    (17, "Invited guests: 80 total, 20 per night",
     "How many guests in total were invited to the soft opening?",
     "80", [], E, ""),
    (17, "80 total, 20 per night",
     "How many guests were invited per night during the soft opening?",
     "20", [], E, ""),
    (17, "food media (3 critics)",
     "How many food critics were among the invited soft-opening guests?",
     "3 critics", ["3", "three"], E, ""),
    (17, "Wine pairing offered but separately charged at cost",
     "How was wine pairing handled at the soft opening?",
     "charged at cost", ["separately charged"], E, ""),
    # ---- Session 18 ----
    (18, "kitchen ticket times averaged 38 minutes (target 22)",
     "What was the average kitchen ticket time during the soft opening?",
     "38 minutes", ["38"], E, ""),
    (18, "target 22",
     "What was the target kitchen ticket time at the soft opening?",
     "22 minutes", ["22"], E, ""),
    (18, "pasta water station too small, bottleneck on plating",
     "What was identified as the cause of the soft-opening kitchen bottleneck?",
     "pasta water station too small", ["pasta water station"], E, ""),
    (18, "73 of 80 guests rated meal 8/10 or higher",
     "How many of the 80 soft-opening guests rated the meal 8/10 or higher?",
     "73", [], E, ""),
    (18, "Food critic Akira Mochizuki from Tokyo Calendar published positive note on Instagram",
     "Which food critic from Tokyo Calendar posted a positive note about the soft opening?",
     "Akira Mochizuki", ["Mochizuki"], E, ""),
    (18, "table 7 has acoustic issue near AC vent",
     "Which table did service manager Sawada flag as having an acoustic issue?",
     "table 7", [], E, ""),
    # ---- Session 19 (EXCLUDED session) ----
    (19, "Grand opening date set: November 4, 2025",
     "On what date was the grand opening set?",
     "November 4, 2025", ["November 4"], E, ""),
    (19, "Pasta station expanded by adding second induction burner (cost 180,000 yen)",
     "What was added to expand the pasta station before grand opening?",
     "second induction burner", ["induction burner"], E, ""),
    (19, "cost 180,000 yen",
     "How much did expanding the pasta station cost?",
     "180,000 yen", ["180,000"], E, ""),
    (19, "Reservation system: TableCheck, 32,000 yen monthly base",
     "Which reservation system did the restaurant adopt?",
     "TableCheck", [], E, ""),
    (19, "TableCheck, 32,000 yen monthly base + transaction fees",
     "What is the monthly base fee of the restaurant's reservation system?",
     "32,000 yen", ["32,000"], E, ""),
    (19, "Table 7 acoustic issue fixed with felt panel installation",
     "How was the table 7 acoustic issue fixed before grand opening?",
     "felt panel installation", ["felt panel"], E, ""),
    # ---- Session 20 ----
    (20, "First three days: dinner covers 22, 26, 24 against target 28",
     "What was the dinner cover target during grand opening week?",
     "28", [], E, ""),
    (20, "Average ticket actual: dinner 8,100 yen (above projection)",
     "What was the actual dinner average ticket during grand opening week?",
     "8,100 yen", ["8,100"], E, ""),
    (20, "lunch 1,950 yen",
     "What was the actual lunch average ticket during grand opening week?",
     "1,950 yen", ["1,950"], E, ""),
    (20, "First Tabelog rating after 3 reviews: 3.42",
     "What was the restaurant's first Tabelog rating after 3 reviews?",
     "3.42", [], E, ""),
    (20, "Grand opening November 4 went smoothly, fully booked",
     "How booked was the restaurant on its grand opening day, November 4?",
     "fully booked", ["fully"], E, ""),
    (20, "Lunch service started November 5",
     "On what date did lunch service start during grand opening week?",
     "November 5", [], E, ""),
    # ---- Session 21 ----
    (21, "Part-time server Naoki Hirose quit after 2 weeks",
     "Which part-time server quit after two weeks, having found a higher-paying job?",
     "Naoki Hirose", ["Hirose"], E, ""),
    (21, "Replacement: Sara Komatsu, 23, started November 17",
     "Who replaced the part-time server who quit?",
     "Sara Komatsu", ["Komatsu"], E, ""),
    (21, "Sous chef Nakajima reports back pain - reducing his hours temporarily",
     "What health issue did sous chef Nakajima report in week 2 of operations?",
     "back pain", [], E, ""),
    (21, "Dishwasher candidate: Mrs. Yoneda, 58, lives in same building",
     "What is the name of the dishwasher candidate who lives in the same building?",
     "Mrs. Yoneda", ["Yoneda"], E, ""),
    (21, "Sara Komatsu, 23",
     "How old is Sara Komatsu, the replacement part-time server?",
     "23", [], E, ""),
    (21, "Mrs. Yoneda, 58",
     "How old is Mrs. Yoneda, the dishwasher candidate?",
     "58", [], E, ""),
    # ---- Session 22 ----
    (22, "November revenue: 3.42 million yen (target 4.0M)",
     "What was the restaurant's revenue in its first full month, November?",
     "3.42 million yen", ["3.42 million"], E, ""),
    (22, "Loss for month: 312,000 yen",
     "What was the restaurant's loss in November?",
     "312,000 yen", ["312,000"], E, ""),
    (22, "Cash runway: roughly 14 months at current burn",
     "Roughly how many months of cash runway did the restaurant have after November?",
     "14 months", ["14"], E, ""),
    (22, "Costs: food 31%, labor 38%, rent+utilities 18%, other 9%",
     "What percentage of costs was labor in the November financial review?",
     "38%", ["38"], E, ""),
    (22, "change opening hours - cut Monday lunch",
     "Which meal service did Kenta cut to change opening hours after November?",
     "Monday lunch", [], E, ""),
    (22, "food 31%",
     "What percentage of costs was food in the November financial review?",
     "31%", ["31"], E, ""),
    # ---- Session 23 ----
    (23, "Marubeni Corporation (user's old employer) booked December 22 private dinner: 24 guests, 12,000 yen per person fixed menu",
     "How many guests attended the Marubeni private dinner on December 22?",
     "24 guests", ["24"], E, ""),
    (23, "12,000 yen per person fixed menu",
     "What per-person price did the Marubeni December private dinner charge?",
     "12,000 yen per person", ["12,000 yen", "12,000"],
     "the '12,000 vs 18,000 yen Marubeni event' ambiguity (Q13); Marubeni-event price excluded", ""),
    (23, "Hired Mrs. Yoneda as dishwasher, 1,400 yen per hour, 4 hours nightly",
     "At what hourly rate was Mrs. Yoneda hired as dishwasher?",
     "1,400 yen per hour", ["1,400 yen", "1,400"], E, ""),
    (23, "4 hours nightly",
     "How many hours nightly does dishwasher Mrs. Yoneda work?",
     "4 hours", ["four hours"], E, ""),
    (23, "Christmas menu launched: 5 courses 9,800 yen",
     "What was the price of the 5-course Christmas menu?",
     "9,800 yen", ["9,800"], E, ""),
    (23, "Total December private bookings: 4 events, projected 1.1 million yen",
     "How many private booking events were projected for December in total?",
     "4 events", ["4", "four"], E, ""),
    # ---- Session 24 (EXCLUDED session) ----
    (24, "December revenue: 4.95 million yen (best month so far)",
     "What was the restaurant's December revenue, its best month so far?",
     "4.95 million yen", ["4.95 million"], E, ""),
    (24, "Holiday bookings exceeded plan by 38%",
     "By what percentage did December holiday bookings exceed plan?",
     "38%", ["38"], E, ""),
    (24, "First year-end staff bonus distributed: 1 month salary equivalent for FT, 30,000 yen for PT",
     "What year-end bonus did part-time staff receive?",
     "30,000 yen", ["30,000"], E, ""),
    (24, "Petrelli requested 2-week vacation in February to visit family in Modena",
     "How long a vacation did Petrelli request for February?",
     "2-week", ["2 weeks", "two weeks"], E, ""),
    (24, "Sawada proposed launching wine club membership in spring",
     "Who proposed launching a wine club membership in spring?",
     "Sawada", [], E, ""),
    (24, "1 month salary equivalent for FT",
     "What year-end bonus did full-time staff receive?",
     "1 month salary", ["one month salary", "1 month"], E, ""),
    # ---- Session 25 ----
    (25, "January first week revenue down 42% versus December average",
     "By what percentage was January's first-week revenue down versus the December average?",
     "42%", ["42"], E, ""),
    (25, "changed from paper to Square Plus inventory module",
     "To what inventory system did the restaurant switch in January?",
     "Square Plus inventory module", ["Square Plus"], E, ""),
    (25, "Subscription: 18,000 yen monthly",
     "What is the monthly subscription cost of the new inventory module?",
     "18,000 yen", ["18,000"], E, ""),
    (25, "Discovered 8% food waste, target reduction to 4%",
     "What food-waste percentage did the restaurant discover in January?",
     "8%", ["8"], E, ""),
    (25, "dropping branzino dish (low margin), adding guinea fowl",
     "Which low-margin dish did Petrelli drop from the menu in January?",
     "branzino", [], E, ""),
    (25, "adding guinea fowl",
     "Which new dish did Petrelli add to the menu in January?",
     "guinea fowl", [], E, ""),
    # ---- Session 26 ----
    (26, "Wine club name: Cantina Morishita",
     "What is the name of the restaurant's wine club?",
     "Cantina Morishita", [], E, ""),
    (26, "Membership fee: 36,000 yen annual",
     "What is the annual membership fee for the wine club?",
     "36,000 yen", ["36,000"], E, ""),
    (26, "Target: 40 members by April",
     "How many wine-club members did Kenta target by April?",
     "40 members", ["40"], E, ""),
    (26, "Benefits: monthly tasting dinner, 10% off bottle list",
     "What discount off the bottle list do wine-club members get?",
     "10%", ["10"], E, ""),
    (26, "Sawada designed program; Mariko Nakata sourcing exclusive labels",
     "Who designed the wine-club program?",
     "Sawada", [], E, ""),
    (26, "Mariko Nakata sourcing exclusive labels",
     "Who is sourcing exclusive labels for the wine club?",
     "Mariko Nakata", ["Mariko"], E, ""),
    # ---- Session 27 ----
    (27, "Petrelli on vacation February 14-28, 2026",
     "During what dates was Petrelli on vacation?",
     "February 14-28", ["February 14"], E, ""),
    (27, "Nakajima running kitchen during absence, given temporary 80,000 yen monthly bump",
     "Who ran the kitchen during Petrelli's February vacation?",
     "Nakajima", [], E, ""),
    (27, "given temporary 80,000 yen monthly bump",
     "What temporary monthly pay bump did Nakajima get while running the kitchen?",
     "80,000 yen", ["80,000"], E, ""),
    (27, "Hired part-time line cook for coverage: Toshiya Inoue, culinary school graduate",
     "Who was hired as a part-time line cook for vacation coverage?",
     "Toshiya Inoue", ["Inoue"], E, ""),
    (27, "dropped guinea fowl temporarily",
     "Which dish was dropped temporarily during Petrelli's vacation coverage?",
     "guinea fowl", [], E, ""),
    (27, "Lunch service paused February 17-21",
     "During which dates was lunch service paused for Petrelli's coverage period?",
     "February 17-21", ["February 17"], E, ""),
    # ---- Session 28 ----
    (28, "Tabelog rating climbed to 3.61 after 24 reviews",
     "To what Tabelog rating did the restaurant climb after 24 reviews?",
     "3.61", [], E, ""),
    (28, "Tabelog rating climbed to 3.61 after 24 reviews",
     "How many Tabelog reviews had the restaurant received when it reached 3.61?",
     "24 reviews", ["24"], E, ""),
    (28, "Unexpected visit by Hiroko Sano, restaurant critic from Asahi Shimbun Weekly",
     "Which restaurant critic made an unexpected visit during Petrelli's absence?",
     "Hiroko Sano", ["Sano"], E, ""),
    (28, "restaurant critic from Asahi Shimbun Weekly",
     "For which publication does critic Hiroko Sano write?",
     "Asahi Shimbun Weekly", ["Asahi Shimbun"], E, ""),
    (28, "dinner ticket times 24 minutes average",
     "What was the average dinner ticket time under Nakajima during coverage?",
     "24 minutes", ["24"], E, ""),
    (28, "Article publication expected late March",
     "When was the Asahi Shimbun critic's article expected to be published?",
     "late March", [], E, ""),
    # ---- Session 29 ----
    (29, "Wine club Cantina Morishita: 23 members signed, target 40 by April",
     "How many members had signed up for the wine club Cantina Morishita by the Q1 review?",
     "23 members", ["23"], E, ""),
    (29, "Q1 revenue total: 11.8 million yen versus plan 12.5 million",
     "What was the restaurant's Q1 revenue total?",
     "11.8 million yen", ["11.8 million"], E, ""),
    (29, "New supplier added: Acetaia Giusti via Petrelli direct",
     "Which balsamic supplier did Petrelli add directly after his vacation?",
     "Acetaia Giusti", ["Giusti"], E, ""),
    (29, "saves 18% versus current import",
     "By what percentage does the new balsamic supplier save versus the current import?",
     "18%", ["18"], E, ""),
    (29, "Petrelli returned March 1",
     "On what date did Petrelli return from vacation?",
     "March 1", [], E, ""),
    (29, "Considering raising prices on 4 menu items",
     "On how many menu items was Kenta considering raising prices at the Q1 review?",
     "4 menu items", ["4", "four"], E, ""),
    # ---- Session 30 ----
    (30, "Sano review published March 19, 2026 in Asahi Shimbun Weekly",
     "On what date was the Asahi Shimbun review published?",
     "March 19, 2026", ["March 19"], E, ""),
    (30, "Headline: 'Azabu's Quiet Modena: A Trattoria With Conviction'",
     "What was the headline of the Asahi Shimbun review (Azabu's Quiet ...)?",
     "Azabu's Quiet Modena", ["Quiet Modena"], E, ""),
    (30, "Reservation requests spiked 4x in 48 hours",
     "By what factor did reservation requests spike in the 48 hours after the review?",
     "4x", ["4", "four"], E, ""),
    (30, "Now booked 3 weeks out for Friday/Saturday dinner",
     "How far out was the restaurant booked for Friday/Saturday dinner after the review?",
     "3 weeks", ["three weeks"], E, ""),
    (30, "Review highlights: ragu bolognese, wine list, 'unfussy precision'",
     "Which two-word phrase did the review use to praise the restaurant's style ('unfussy ...')?",
     "unfussy precision", [], E, ""),
    (30, "Sano review published March 19, 2026 in Asahi Shimbun Weekly",
     "In which publication was the Sano review published?",
     "Asahi Shimbun Weekly", ["Asahi Shimbun"], E, ""),
    # ---- Session 31 ----
    (31, "Tagliatelle al ragu bolognese raised from 2,400 to 2,700 yen",
     "To what price was the tagliatelle al ragu bolognese raised in the April 2026 revision?",
     "2,700 yen", ["2,700"], E, ""),
    (31, "Lunch courses raised: 2-course to 2,000 yen",
     "To what price was the 2-course lunch raised in the April 2026 revision?",
     "2,000 yen", ["2,000"], E, ""),
    (31, "3-course went up to 2,600 yen",
     "To what price was the 3-course lunch raised in the April 2026 revision?",
     "2,600 yen", ["2,600"], E, ""),
    (31, "dinner average ticket target raised to 9,200 yen",
     "To what figure was the dinner average ticket target raised in April 2026?",
     "9,200 yen", ["9,200"], E, ""),
    (31, "Olive oil switched from Frantoio Franci to a less expensive blend from Solleone (saving 22% on cost)",
     "By what percentage did switching olive oil save on cost in the April 2026 adjustment?",
     "22%", ["22"], E, ""),
    (31, "Menu prices revised April 1",
     "On what date were the menu prices revised in 2026?",
     "April 1", [], E, ""),
    # ---- Session 32 (EXCLUDED session) ----
    (32, "Wood-fired oven Modena 100 cracked on stone deck April 13",
     "On what date did the wood-fired oven crack?",
     "April 13", [], E, ""),
    (32, "Emergency repair by Forno Bravo Japan technician Mr. Saito",
     "Which Forno Bravo Japan technician performed the emergency oven repair?",
     "Mr. Saito", ["Saito"], E, ""),
    (32, "Repair cost: 340,000 yen, 5-day downtime",
     "What was the cost of the emergency oven repair?",
     "340,000 yen", ["340,000"], E, ""),
    (32, "5-day downtime",
     "How many days of downtime did the oven crack cause?",
     "5-day", ["5 days", "five days", "5"], E, ""),
    (32, "Insurance through Tokio Marine Restaurant Plus policy covered 70% of repair",
     "What percentage of the oven repair did insurance cover?",
     "70%", ["70"], E, ""),
    (32, "Insurance through Tokio Marine Restaurant Plus policy",
     "Which insurance policy covered the oven repair?",
     "Tokio Marine Restaurant Plus", ["Tokio Marine"], E, ""),
    # ---- Session 33 (EXCLUDED session) ----
    (33, "Service manager Sawada renegotiated to 510,000 yen monthly salary (raise of 50,000)",
     "To what monthly salary did service manager Sawada renegotiate at the six-month mark?",
     "510,000 yen", ["510,000"], E, ""),
    (33, "raise of 50,000",
     "By how much was Sawada's monthly salary raised at the six-month mark?",
     "50,000 yen", ["50,000"], E, ""),
    (33, "Petrelli bonus structure adjusted: 4% of revenue above 4.5M monthly threshold",
     "Above what monthly revenue threshold does Petrelli's adjusted bonus apply?",
     "4.5M", ["4.5 million"], E, ""),
    (33, "Wine club Cantina Morishita reached 42 members (exceeded April target)",
     "How many members did the wine club Cantina Morishita reach by the six-month mark?",
     "42 members", ["42"], E, ""),
    (33, "Petrelli bonus structure adjusted: 4% of revenue above 4.5M",
     "What percentage of revenue above threshold is Petrelli's adjusted bonus?",
     "4%", ["4"], E, ""),
    (33, "Considering second seating system for Friday/Saturday",
     "What seating system was under consideration for Friday/Saturday at the six-month mark?",
     "second seating", [], E, ""),
    # ---- Session 34 ----
    (34, "Customer database (via TableCheck) shows 412 unique guests, 87 repeat customers",
     "How many unique guests did the customer database show at the six-month celebration?",
     "412 unique guests", ["412"], E, ""),
    (34, "87 repeat customers",
     "How many repeat customers did the customer database show at the six-month celebration?",
     "87", [], E, ""),
    (34, "Top repeat customer: Mr. Hiroyuki Tachibana (10 visits), Azabu resident",
     "Who is the restaurant's top repeat customer?",
     "Hiroyuki Tachibana", ["Tachibana"], E, ""),
    (34, "Mr. Hiroyuki Tachibana (10 visits)",
     "How many visits had the top repeat customer made by the six-month celebration?",
     "10 visits", ["10", "ten"], E, ""),
    (34, "Revenue cumulative: 28.4 million yen since opening",
     "What was the cumulative revenue since opening at the six-month celebration?",
     "28.4 million yen", ["28.4 million"], E, ""),
    (34, "Six-month anniversary May 4 marked with private gathering for 30 regulars",
     "For how many regulars was the six-month anniversary private gathering held?",
     "30 regulars", ["30"], E, ""),
    # ---- Session 35 (EXCLUDED session) ----
    (35, "Hamada (broker from Plaza Homes) showed property in Tomigaya, 18.2 tsubo, 580,000 yen rent",
     "In which neighborhood did broker Hamada show a property for a possible second location?",
     "Tomigaya", [], E, ""),
    (35, "property in Tomigaya, 18.2 tsubo, 580,000 yen rent",
     "What was the monthly rent of the Tomigaya property shown for a second location?",
     "580,000 yen", ["580,000"], E, ""),
    (35, "property in Tomigaya, 18.2 tsubo",
     "What was the size in tsubo of the Tomigaya second-location property?",
     "18.2 tsubo", ["18.2"], E, ""),
    (35, "revisit second location after fiscal year end September 2026",
     "After what point did Kenta decide to revisit the second-location decision?",
     "September 2026", ["fiscal year end September 2026"], E, ""),
    (35, "launch private dining room conversion in existing space",
     "What did Kenta decide to pursue instead of a second location?",
     "private dining room conversion", ["private dining room"], E, ""),
    (35, "Petrelli interested in expansion as long as Modena focus preserved",
     "Under what condition was Petrelli interested in expansion?",
     "Modena focus preserved", ["Modena focus"], E, ""),
    # ---- Session 36 ----
    (36, "Convert storage area at rear into 8-seat private room",
     "How many seats will the new private dining room have?",
     "8-seat", ["8 seats", "eight", "8"], E, ""),
    (36, "Cost estimate from Yamamoto Koumuten: 1.4 million yen",
     "What was Yamamoto Koumuten's cost estimate for the private-room conversion?",
     "1.4 million yen", ["1.4 million"], E, ""),
    (36, "Loss of storage requires basement rental: 38,000 yen monthly nearby",
     "What monthly cost is the nearby basement rental needed after the storage conversion?",
     "38,000 yen", ["38,000"], E, ""),
    (36, "Construction window: 4 weeks, scheduled July 14 to August 9",
     "How many weeks was the private-room construction window?",
     "4 weeks", ["four weeks"], E, ""),
    (36, "Private room name: Sala Modena, premium pricing",
     "What is the name of the new private dining room?",
     "Sala Modena", [], E, ""),
    (36, "scheduled July 14 to August 9",
     "On what date was the private-room construction scheduled to start?",
     "July 14", [], E, ""),
    # ---- Session 37 ----
    (37, "Total staff now: 7 full-time, 4 part-time",
     "How many full-time staff did the restaurant have at the HR-formalization stage?",
     "7 full-time", ["7"], E, ""),
    (37, "Employee handbook drafted with help from labor consultant Mr. Endo, fee 280,000 yen",
     "Which labor consultant helped draft the employee handbook?",
     "Mr. Endo", ["Endo"], E, ""),
    (37, "labor consultant Mr. Endo, fee 280,000 yen",
     "What fee did labor consultant Mr. Endo charge for the employee handbook?",
     "280,000 yen", ["280,000"], E, ""),
    (37, "First formal performance reviews scheduled July 1",
     "On what date were the first formal performance reviews scheduled?",
     "July 1", [], E, ""),
    (37, "if Petrelli ever leaves, Nakajima identified as successor",
     "Who was identified as head-chef successor if Petrelli leaves?",
     "Nakajima", [], E, ""),
    (37, "Added social insurance enrollment for all FT",
     "What benefit was added for all full-time staff during HR formalization?",
     "social insurance enrollment", ["social insurance"], E, ""),
    # ---- Session 38 ----
    (38, "Nakajima raised to 440,000 yen (from 380,000)",
     "To what monthly salary was Nakajima raised in the performance-review adjustments?",
     "440,000 yen", ["440,000"], E, ""),
    (38, "Yui Tachibana raised to 1,500 yen per hour",
     "To what hourly rate was Yui Tachibana raised in the performance-review adjustments?",
     "1,500 yen per hour", ["1,500 yen", "1,500"], E, ""),
    (38, "Two-year retention bonus formalized: 200,000 yen at 24 months",
     "What retention bonus was formalized at 24 months?",
     "200,000 yen", ["200,000"], E, ""),
    (38, "Petrelli signed two-year extension to October 2027",
     "Until when did Petrelli's two-year contract extension run?",
     "October 2027", [], E, ""),
    (38, "Petrelli new monthly base: 780,000 yen",
     "What is Petrelli's new monthly base salary after the extension?",
     "780,000 yen", ["780,000"], E, ""),
    (38, "Performance reviews completed for 7 FT staff",
     "For how many full-time staff were performance reviews completed?",
     "7 FT staff", ["7"], E, ""),
    # ---- Session 39 ----
    (39, "Marubeni (former employer) booked Sala Modena October 14 for 14-person dinner",
     "On what date did Marubeni book the Sala Modena private room?",
     "October 14", [], E, ""),
    (39, "booked Sala Modena October 14 for 14-person dinner",
     "For how many people did Marubeni book the Sala Modena dinner?",
     "14-person", ["14 people", "14"], E, ""),
    (39, "Pricing: 18,000 yen per person minimum plus venue fee 50,000 yen",
     "What per-person minimum did the Sala Modena Marubeni event charge?",
     "18,000 yen per person", ["18,000 yen", "18,000"],
     "the '12,000 vs 18,000 yen Marubeni event' ambiguity (Q13); Marubeni-event price excluded", ""),
    (39, "venue fee 50,000 yen",
     "What venue fee applies to the Sala Modena private room?",
     "50,000 yen", ["50,000"], E, ""),
    (39, "Wine pairings start at 6,500 yen",
     "At what price do the Sala Modena wine pairings start?",
     "6,500 yen", ["6,500"], E, ""),
    (39, "Pre-bookings already secured: 6 private events for September",
     "How many private events were pre-booked for September in the Sala Modena?",
     "6 private events", ["6", "six"], E, ""),
    # ---- Session 40 ----
    (40, "Tabelog rating now 3.78 with 91 reviews",
     "What was the restaurant's Tabelog rating at the one-year reflection?",
     "3.78", [], E, ""),
    (40, "Tabelog rating now 3.78 with 91 reviews",
     "How many Tabelog reviews did the restaurant have at the one-year reflection?",
     "91 reviews", ["91"], E, ""),
    (40, "Sala Modena opens August 12, 2026",
     "On what date does the Sala Modena private room open?",
     "August 12, 2026", ["August 12"], E, ""),
    (40, "Cumulative loss zeroed out in July 2026, now operating profitably",
     "In what month did the restaurant's cumulative loss zero out?",
     "July 2026", [], E, ""),
    (40, "from idea to thriving restaurant in ~17 months",
     "Over roughly how many months did the restaurant go from idea to thriving, per the reflection?",
     "17 months", ["17"], E, ""),
    (40, "Year-two goals: second location decision by November, possible cookbook collaboration",
     "By what month is the year-two second-location decision targeted?",
     "November", [], E, ""),
]

# ---------------------------------------------------------------------------
# Generated novel questions (~5/session x 30). No whole-session exclusions apply
# to the novel chat; the bleed-in sessions are a RESTAURANT-only issue.
# ---------------------------------------------------------------------------

NOVEL_GEN = [
    # ---- Session 1 ----
    (1, "working title 'The Hollow Years'",
     "What working title was Maya using for her novel in session 1?",
     "The Hollow Years", ["Hollow Years"], E, ""),
    (1, "protagonist Eleanor Reyes, a marine biologist",
     "What is the name of the novel's protagonist?",
     "Eleanor Reyes", ["Eleanor"], E, ""),
    (1, "setting: Stonington, Maine",
     "In which Maine town is the novel set?",
     "Stonington, Maine", ["Stonington"], E, ""),
    (1, "antagonist: her estranged sister Camille Reyes",
     "Who is the novel's antagonist, Eleanor's estranged sister?",
     "Camille Reyes", ["Camille"], E, ""),
    (1, "initial word count target 50000",
     "What was Maya's initial word count target in session 1?",
     "50,000 words", ["50,000", "50000"], E, ""),
    (1, "central conflict: a family lighthouse inheritance dispute",
     "What is the central conflict of the novel?",
     "lighthouse inheritance dispute", ["inheritance dispute"], E, ""),
    # ---- Session 2 ----
    (2, "12-chapter outline",
     "How many chapters were in the outline Maya drafted in session 2?",
     "12-chapter", ["12 chapters", "12"], E, ""),
    (2, "Act 2 climax: lighthouse fire",
     "What event is the Act 2 climax of the novel?",
     "lighthouse fire", [], E, ""),
    (2, "grandmother named Iris Penhallow",
     "What is the name of the grandmother character in the novel?",
     "Iris Penhallow", ["Iris"], E, ""),
    (2, "Act 3 resolution: sisters reconcile through grandmother's journal",
     "Through what object do the sisters reconcile in the Act 3 resolution?",
     "grandmother's journal", ["journal"], E, ""),
    (2, "Act 1 ends with arrival in Stonington",
     "With what event does Act 1 of the novel end?",
     "arrival in Stonington", ["arrival"], E, ""),
    # ---- Session 3 ----
    (3, "Eleanor age 38, divorced, no children",
     "How old is Eleanor Reyes in the character backstory?",
     "38", [], E, ""),
    (3, "Camille age 41, two kids, lives in Boston",
     "In which city does Eleanor's sister Camille live?",
     "Boston", [], E, ""),
    (3, "father Hector Reyes died 2019 of pancreatic cancer",
     "What is the name of Eleanor's father in the backstory?",
     "Hector Reyes", ["Hector"], E, ""),
    (3, "mother Lillian Reyes lives in assisted care in Bangor",
     "In which town does Eleanor's mother Lillian live in assisted care?",
     "Bangor", [], E, ""),
    (3, "Eleanor's best friend: Davinia Okafor, a Portland-based veterinarian",
     "What is the profession of Eleanor's best friend Davinia Okafor?",
     "veterinarian", [], E, ""),
    # ---- Session 4 ----
    (4, "Chapter 1 word count 4200",
     "What was the word count of Chapter 1 in session 4?",
     "4200", ["4,200"], E, ""),
    (4, "opening scene: Eleanor receiving lawyer's letter in Portland",
     "In which city does the opening scene take place, where Eleanor receives a lawyer's letter?",
     "Portland", [], E, ""),
    (4, "lawyer character named Theodore Whitcomb of Whitcomb & Beale",
     "What is the name of the lawyer character introduced in session 4?",
     "Theodore Whitcomb", ["Whitcomb"], E, ""),
    (4, "feedback: cut opening weather paragraph",
     "What did the session 4 feedback suggest cutting from the opening?",
     "opening weather paragraph", ["weather paragraph"], E, ""),
    (4, "introduce Camille's voice via voicemail earlier",
     "How did the feedback suggest introducing Camille's voice earlier?",
     "voicemail", [], E, ""),
    # ---- Session 5 ----
    (5, "interviewed retired lobsterman Earl Stenholm via Zoom",
     "What is the name of the retired lobsterman Maya interviewed for research?",
     "Earl Stenholm", ["Stenholm"], E, ""),
    (5, "lighthouse modeled on real Mark Island Light",
     "On which real lighthouse is the novel's lighthouse modeled?",
     "Mark Island Light", ["Mark Island"], E, ""),
    (5, "Eleanor's research vessel named Persephone",
     "What is the original name of Eleanor's research vessel, introduced in session 5?",
     "Persephone", [], E, ""),
    (5, "added seasonal detail: novel set across one autumn",
     "Across which single season is the novel set?",
     "one autumn", ["autumn"], E, ""),
    (5, "read three books on Maine lobstering culture",
     "How many books on Maine lobstering culture did Maya read for research?",
     "three books", ["3 books", "three", "3"], E, ""),
    # ---- Session 6 ----
    (6, "diner named The Salt Cod Cafe",
     "What is the name of the diner introduced in the chapter 3 scene?",
     "The Salt Cod Cafe", ["Salt Cod Cafe"], E, ""),
    (6, "added local sheriff character: Sheriff Margaret 'Maggie' Boudreau",
     "What is the name of the local sheriff character added in session 6?",
     "Margaret Boudreau", ["Maggie Boudreau", "Boudreau"], E, ""),
    (6, "combined word count chapters 1-3: 13800",
     "What was the combined word count of chapters 1-3 in session 6?",
     "13800", ["13,800"], E, ""),
    (6, "introduced antagonist Camille in chapter 3 diner scene",
     "In which chapter's diner scene is antagonist Camille introduced?",
     "chapter 3", [], E, ""),
    (6, "first sister confrontation in chapter 3",
     "In which chapter does the first sister confrontation occur?",
     "chapter 3", [], E, ""),
    # ---- Session 7 ----
    (7, "decided to add subplot: Eleanor reconnects with high school flame Owen Trask",
     "What is the name of Eleanor's high school flame added as a subplot in session 7?",
     "Owen Trask", ["Owen"], E, ""),
    (7, "Owen is a wooden boat builder",
     "What is Owen Trask's occupation?",
     "wooden boat builder", ["boat builder"], E, ""),
    (7, "raised target word count from 50000 to 80000",
     "To what figure did Maya raise the target word count in session 7?",
     "80000", ["80,000"], E, ""),
    (7, "added a secondary antagonist: developer Brent Halloran of Halloran Coastal LLC",
     "What is the name of the secondary antagonist developer added in session 7?",
     "Brent Halloran", ["Halloran"], E, ""),
    (7, "developer Brent Halloran of Halloran Coastal LLC",
     "For which company does the developer antagonist Brent Halloran work?",
     "Halloran Coastal LLC", ["Halloran Coastal"], E, ""),
    # ---- Session 8 ----
    (8, "Iris Penhallow's journal entries dated 1962-1971",
     "What year range do grandmother Iris Penhallow's journal entries span?",
     "1962-1971", ["1962"], E, ""),
    (8, "grandmother had a stillborn son named Wendell in 1965",
     "What is the name of the grandmother's stillborn son revealed as a family secret?",
     "Wendell", [], E, ""),
    (8, "chapter 5 contains the lighthouse-keeper journal discovery",
     "In which chapter does the lighthouse-keeper journal discovery occur?",
     "chapter 5", [], E, ""),
    (8, "chapter 5 word count 5400",
     "What was the word count of chapter 5 in session 8?",
     "5400", ["5,400"], E, ""),
    (8, "stillborn son named Wendell in 1965",
     "In what year did the grandmother have her stillborn son, per the journal?",
     "1965", [], E, ""),
    # ---- Session 9 ----
    (9, "confirmed close third-person POV throughout",
     "What point of view did Maya confirm for the novel in session 9?",
     "close third-person", ["third-person"], E, ""),
    (9, "journal excerpts as italicized fragments",
     "How are the journal excerpts to be presented in the novel?",
     "italicized fragments", ["italicized"], E, ""),
    (9, "considered Toni Morrison and Marilynne Robinson as influences",
     "Name one of the two authors Maya considered as influences in session 9 (Toni ...).",
     "Toni Morrison", ["Marilynne Robinson", "Morrison", "Robinson"], E, ""),
    (9, "narrator filtered through Eleanor",
     "Through which character is the narrator filtered?",
     "Eleanor", [], E, ""),
    (9, "decided against journal chapters being first-person",
     "Did Maya decide to make the journal chapters first-person?",
     "against first-person", ["decided against", "no"], E, ""),
    # ---- Session 10 ----
    (10, "developer Brent Halloran reveals offer of 1.8 million for lighthouse property",
     "How much did developer Brent Halloran offer for the lighthouse property?",
     "1.8 million", ["1.8 million dollars"], E, ""),
    (10, "total word count now 47000",
     "What was the total word count reported in session 10?",
     "47000", ["47,000"], E, ""),
    (10, "lighthouse fire in chapter 8",
     "In which chapter does the lighthouse fire occur?",
     "chapter 8", [], E, ""),
    (10, "Owen Trask first appears chapter 6",
     "In which chapter does Owen Trask first appear?",
     "chapter 6", [], E, ""),
    (10, "Camille initially wants to accept the offer",
     "Who initially wants to accept the developer's offer for the property?",
     "Camille", [], E, ""),
    # ---- Session 11 ----
    (11, "decided pivotal scene: shared swim at Crescent Beach in chapter 10",
     "At which beach does the pivotal shared-swim reconciliation scene take place?",
     "Crescent Beach", [], E, ""),
    (11, "added object motif: grandmother's brass compass",
     "What object motif did Maya add in session 11?",
     "grandmother's brass compass", ["brass compass", "compass"], E, ""),
    (11, "compass passes between sisters three times in novel",
     "How many times does the compass pass between the sisters in the novel?",
     "three times", ["3 times", "three", "3"], E, ""),
    (11, "shared swim at Crescent Beach in chapter 10",
     "In which chapter is the pivotal shared-swim scene set?",
     "chapter 10", [], E, ""),
    (11, "Eleanor's profession noted as marine biologist still",
     "As of session 11, what was Eleanor's profession still noted as?",
     "marine biologist", [], E, ""),
    # ---- Session 12 ----
    (12, "completed first draft June 9, 2025",
     "On what date did Maya complete the first draft?",
     "June 9, 2025", ["June 9"], E, ""),
    (12, "final first-draft word count 83400",
     "What was the final first-draft word count?",
     "83400", ["83,400"], E, ""),
    (12, "12 chapters plus epilogue",
     "The completed first draft had 12 chapters plus what?",
     "epilogue", [], E, ""),
    (12, "epilogue set 3 years after main events",
     "How many years after the main events is the epilogue set?",
     "3 years", ["three years"], E, ""),
    (12, "celebrating with self at Powell's Books",
     "At which bookstore did Maya celebrate finishing the first draft?",
     "Powell's Books", ["Powell"], E, ""),
    # ---- Session 13 ----
    (13, "plan three revision passes: structural, line, polish",
     "How many revision passes did Maya plan in session 13?",
     "three revision passes", ["3 passes", "three", "3"], E, ""),
    (13, "deleted entire subplot about a missing dog",
     "What subplot did Maya delete in session 13?",
     "missing dog", ["dog"], E, ""),
    (13, "tightened Owen Trask appearances from 9 scenes to 6",
     "To how many scenes did Maya tighten Owen Trask's appearances in session 13?",
     "6 scenes", ["6", "six"], E, ""),
    (13, "raised target word count to 95000",
     "To what figure did Maya raise the target word count in session 13?",
     "95000", ["95,000"], E, ""),
    (13, "estimated 8 weeks for full revision",
     "How many weeks did Maya estimate for the full revision?",
     "8 weeks", ["eight weeks"], E, ""),
    # ---- Session 14 ----
    (14, "added new chapter (now chapter 9) about Eleanor visiting mother Lillian in Bangor care home",
     "In which town's care home does Eleanor visit her mother in the new chapter 9?",
     "Bangor", [], E, ""),
    (14, "Lillian reveals she always blamed Camille for father's stress",
     "Whom does Lillian reveal she always blamed for their father's stress?",
     "Camille", [], E, ""),
    (14, "revised word count after structural pass: 81200",
     "What was the revised word count after the structural pass in session 14?",
     "81200", ["81,200"], E, ""),
    (14, "cut 6000 words from middle act",
     "How many words did Maya cut from the middle act in the structural pass?",
     "6000", ["6,000"], E, ""),
    (14, "reordered chapters 4 and 5 to put journal discovery earlier",
     "Which two chapters did Maya reorder to put the journal discovery earlier?",
     "chapters 4 and 5", ["4 and 5"], E, ""),
    # ---- Session 15 ----
    (15, "recruited five beta readers",
     "How many beta readers did Maya recruit in session 15?",
     "five beta readers", ["5 beta readers", "five", "5"], E, ""),
    (15, "indie bookseller Quentin Marsh of Annie Bloom's Books",
     "At which bookstore does beta reader Quentin Marsh work?",
     "Annie Bloom's Books", ["Annie Bloom"], E, ""),
    (15, "provided 12-question feedback form",
     "How many questions were in the feedback form Maya sent to beta readers?",
     "12-question", ["12 questions", "12"], E, ""),
    (15, "asked for 4-week turnaround",
     "What turnaround time did Maya ask beta readers for?",
     "4-week", ["4 weeks", "four weeks"], E, ""),
    (15, "beta readers: Davinia Okafor (friend), Priya Ramanathan (writing group)",
     "Name one beta reader from Maya's writing group (Priya ...).",
     "Priya Ramanathan", ["Marcus Holloway", "Priya", "Marcus"], E, ""),
    # ---- Session 16 ----
    (16, "Davinia recommended making Eleanor a marine ecologist studying eelgrass die-off",
     "What profession did Davinia recommend changing Eleanor to?",
     "marine ecologist", ["marine ecologist studying eelgrass die-off"], E, ""),
    (16, "Davinia returned feedback in 10 days",
     "In how many days did Davinia return her beta feedback?",
     "10 days", ["ten days"], E, ""),
    (16, "Davinia confused by the lawyer Theodore Whitcomb's motives in chapter 4",
     "About which character's motives in chapter 4 was Davinia confused?",
     "Theodore Whitcomb", ["Whitcomb"], E, ""),
    (16, "Davinia loved opening 30 pages",
     "How many opening pages did Davinia say she loved?",
     "30 pages", ["30"], E, ""),
    (16, "Davinia suggested protagonist's profession felt thin and underused",
     "What did Davinia say about the protagonist's profession?",
     "thin and underused", ["thin", "underused"], E, ""),
    # ---- Session 17 ----
    (17, "Priya flagged anachronistic word 'doomscroll' in 1965 journal entry",
     "Which anachronistic word did Priya flag in a 1965 journal entry?",
     "doomscroll", [], E, ""),
    (17, "Marcus Holloway thought Owen Trask was the strongest minor character",
     "Which minor character did Marcus Holloway think was the strongest?",
     "Owen Trask", ["Owen"], E, ""),
    (17, "Marcus suggested an ambiguous ending rather than full reconciliation",
     "What kind of ending did Marcus suggest instead of full reconciliation?",
     "ambiguous ending", ["ambiguous"], E, ""),
    (17, "both flagged that the antagonist Brent Halloran felt cartoonish",
     "How did both Priya and Marcus describe the antagonist Brent Halloran?",
     "cartoonish", [], E, ""),
    (17, "Priya Ramanathan thought chapter 7 dialogue felt expository",
     "Which chapter's dialogue did Priya think felt expository?",
     "chapter 7", [], E, ""),
    # ---- Session 18 ----
    (18, "Quentin Marsh said it reminded him of Elizabeth Strout meets Andre Dubus III",
     "Quentin Marsh compared the novel to Elizabeth Strout meets which author?",
     "Andre Dubus III", ["Andre Dubus"], E, ""),
    (18, "Quentin offered to host launch event if it gets published",
     "What did Quentin Marsh offer to host if the book gets published?",
     "launch event", [], E, ""),
    (18, "Jennifer Calloway praised the prose but found pacing slow until chapter 6",
     "Until which chapter did Jennifer Calloway find the pacing slow?",
     "chapter 6", [], E, ""),
    (18, "Jennifer's biggest note: title 'The Hollow Years' too generic",
     "What was Jennifer's biggest note about the title 'The Hollow Years'?",
     "too generic", ["generic"], E, ""),
    (18, "Jennifer suggested cutting chapters 2 and 3 down by half",
     "Which chapters did Jennifer suggest cutting down by half?",
     "chapters 2 and 3", ["2 and 3"], E, ""),
    # ---- Session 19 ----
    (19, "revise Eleanor's profession from marine biologist to marine ecologist studying eelgrass die-off",
     "To what profession was Eleanor revised in session 19?",
     "marine ecologist studying eelgrass die-off", ["marine ecologist"], E, ""),
    (19, "decided to soften antagonist Brent Halloran by giving him a sick wife",
     "How did Maya decide to soften antagonist Brent Halloran in session 19?",
     "giving him a sick wife", ["sick wife"], E, ""),
    (19, "did NOT change ending to ambiguous",
     "Did Maya change the ending to ambiguous in session 19?",
     "did NOT change", ["no", "did not change"], E, ""),
    (19, "kept Owen Trask scenes as Marcus suggested",
     "Whose suggestion led Maya to keep the Owen Trask scenes?",
     "Marcus", [], E, ""),
    (19, "agreed title 'The Hollow Years' needs to change",
     "About which title did Maya agree it needs to change in session 19?",
     "The Hollow Years", ["Hollow Years"], E, ""),
    # ---- Session 20 ----
    (20, "tentatively chose 'Salt and Compass' as new working title",
     "What new working title did Maya tentatively choose in session 20?",
     "Salt and Compass", [], E, ""),
    (20, "brainstormed 30 candidate titles",
     "How many candidate titles did Maya brainstorm in session 20?",
     "30 candidate titles", ["30"], E, ""),
    (20, "shortlist: 'Eelgrass Light', 'The Tender of Mark Island', 'The Penhallow Inheritance', 'Salt and Compass'",
     "Name one shortlisted title from session 20 besides 'Salt and Compass' (Eelgrass ...).",
     "Eelgrass Light", ["The Penhallow Inheritance", "The Tender of Mark Island"], E, ""),
    (20, "added a scene of Eleanor sampling eelgrass off the Persephone",
     "What is Eleanor sampling in the new scene off the Persephone?",
     "eelgrass", [], E, ""),
    (20, "the eelgrass research subplot now informs chapter 6 in detail",
     "Which chapter does the eelgrass research subplot now inform in detail?",
     "chapter 6", [], E, ""),
    # ---- Session 21 ----
    (21, "now using Vellum for formatting",
     "Which software is Maya now using for formatting?",
     "Vellum", [], E, ""),
    (21, "started line edit on October 12",
     "On what date did Maya start the line edit?",
     "October 12", [], E, ""),
    (21, "averaging 8 pages per day at line-edit pace",
     "How many pages per day was Maya averaging at line-edit pace?",
     "8 pages", ["eight pages"], E, ""),
    (21, "found and fixed the doomscroll anachronism Priya flagged",
     "Which anachronism did Maya find and fix during the line edit?",
     "doomscroll", [], E, ""),
    (21, "cut 4 adverbs per page average",
     "How many adverbs per page on average did Maya cut in the line edit?",
     "4 adverbs", ["four adverbs", "4"], E, ""),
    # ---- Session 22 ----
    (22, "drafted 250-word query letter",
     "How many words was the query letter Maya drafted?",
     "250-word", ["250 words", "250"], E, ""),
    (22, "comp titles: 'Olive Kitteridge' and 'House of Sand and Fog'",
     "Name one comp title Maya used in the query letter (Olive ...).",
     "Olive Kitteridge", ["House of Sand and Fog"], E, ""),
    (22, "current title 'Salt and Compass'",
     "What was the novel's title as of the query-letter session 22?",
     "Salt and Compass", [], E, ""),
    (22, "word count after line edits: 92100",
     "What was the word count after the line edits in session 22?",
     "92100", ["92,100"], E, ""),
    (22, "logline focuses on sister rivalry over inheritance and ecological stakes",
     "On what two themes does the query logline focus (sister rivalry and ...)?",
     "ecological stakes", ["ecological"], E, ""),
    # ---- Session 23 ----
    (23, "researched 40 literary agents",
     "How many literary agents did Maya research in session 23?",
     "40 literary agents", ["40"], E, ""),
    (23, "shortlisted 15 to query in first batch",
     "How many agents did Maya shortlist to query in the first batch?",
     "15", [], E, ""),
    (23, "Renata Goldfarb of Goldfarb Literary",
     "Name the agent from Goldfarb Literary in Maya's first query batch.",
     "Renata Goldfarb", ["Goldfarb"], E, ""),
    (23, "all five have made deals with debut authors in past 18 months",
     "Within what recent time window had all five first-batch agents made debut-author deals?",
     "past 18 months", ["18 months"], E, ""),
    (23, "Joon-Ho Park of Crescent Atlantic Agency",
     "For which agency does Joon-Ho Park work?",
     "Crescent Atlantic Agency", ["Crescent Atlantic"], E, ""),
    # ---- Session 24 ----
    (24, "sent first batch of 5 queries November 24, 2025",
     "On what date did Maya send the first batch of queries?",
     "November 24, 2025", ["November 24"], E, ""),
    (24, "Renata Goldfarb requested partial within 6 days",
     "Within how many days did Renata Goldfarb request a partial?",
     "6 days", ["six days"], E, ""),
    (24, "Anders Krishnamurthy auto-rejected within 48 hours",
     "Which agent auto-rejected Maya within 48 hours?",
     "Anders Krishnamurthy", ["Krishnamurthy"], E, ""),
    (24, "sent first batch of 5 queries",
     "How many queries were in the first batch Maya sent?",
     "5 queries", ["five queries", "5"], E, ""),
    (24, "Sloane Whitford-Jeong, Hattie Macreadie, and Joon-Ho Park have not responded",
     "Name one first-batch agent who had not responded (Hattie ...).",
     "Hattie Macreadie", ["Sloane Whitford-Jeong", "Macreadie"], E, ""),
    # ---- Session 25 ----
    (25, "Joon-Ho Park requested full manuscript December 8",
     "On what date did Joon-Ho Park request the full manuscript?",
     "December 8", [], E, ""),
    (25, "Hattie Macreadie passed with form rejection on December 6",
     "Which agent passed with a form rejection on December 6?",
     "Hattie Macreadie", ["Macreadie"], E, ""),
    (25, "sent partial (first 50 pages) to Renata Goldfarb",
     "How many pages were in the partial Maya sent to Renata Goldfarb?",
     "50 pages", ["fifty pages", "50"], E, ""),
    (25, "sent second batch of 5 queries",
     "How many queries were in the second batch Maya sent?",
     "5 queries", ["five queries", "5"], E, ""),
    (25, "second batch of 5 queries to: Mireille Asante of Northbeam Literary",
     "Name the agent from Northbeam Literary in the second query batch.",
     "Mireille Asante", ["Asante"], E, ""),
    # ---- Session 26 ----
    (26, "Renata Goldfarb offered representation December 22, 2025",
     "On what date did Renata Goldfarb offer representation?",
     "December 22, 2025", ["December 22"], E, ""),
    (26, "Renata Goldfarb requested full manuscript December 18",
     "On what date did Renata Goldfarb request the full manuscript?",
     "December 18", [], E, ""),
    (26, "decided to give other agents until January 12 to respond",
     "Until what date did Maya give other agents to respond after the offer?",
     "January 12", [], E, ""),
    (26, "Joon-Ho Park, Pavithra Sundaram, and Esme Calabrese all stepped up with reads",
     "Name one agent besides Renata who stepped up with a read after the offer (Pavithra ...).",
     "Pavithra Sundaram", ["Joon-Ho Park", "Esme Calabrese", "Sundaram"], E, ""),
    (26, "now under 'offer of representation' notification to others",
     "What notification status was Maya under after Renata's offer?",
     "offer of representation", [], E, ""),
    # ---- Session 27 ----
    (27, "chose Renata Goldfarb of Goldfarb Literary",
     "Which agent did Maya ultimately choose to represent her?",
     "Renata Goldfarb", ["Goldfarb"], E, ""),
    (27, "Pavithra Sundaram also offered representation January 5, 2026",
     "On what date did Pavithra Sundaram also offer representation?",
     "January 5, 2026", ["January 5"], E, ""),
    (27, "Joon-Ho Park ultimately passed citing too crowded literary list",
     "Which agent passed citing a too-crowded literary list?",
     "Joon-Ho Park", ["Park"], E, ""),
    (27, "Esme Calabrese passed citing 'voice mismatch'",
     "What reason did Esme Calabrese give for passing?",
     "voice mismatch", [], E, ""),
    (27, "had calls with Renata Goldfarb (1 hour) and Pavithra Sundaram (45 minutes)",
     "How long was Maya's call with Pavithra Sundaram?",
     "45 minutes", ["45"], E, ""),
    # ---- Session 28 ----
    (28, "Renata sent 9-page editorial letter January 18",
     "How many pages was Renata's editorial letter?",
     "9-page", ["9 pages", "nine pages", "9"], E, ""),
    (28, "Renata flagged chapter 5 journal device as overused",
     "Which chapter's journal device did Renata flag as overused?",
     "chapter 5", [], E, ""),
    (28, "Renata recommended adding a present-day chapter from Camille's POV",
     "From which character's POV did Renata recommend adding a present-day chapter?",
     "Camille", [], E, ""),
    (28, "Renata sent 9-page editorial letter January 18",
     "On what date did Renata send the editorial letter?",
     "January 18", [], E, ""),
    (28, "Renata suggested cutting Owen Trask subplot in half",
     "Whose subplot did Renata suggest cutting in half?",
     "Owen Trask", ["Owen"], E, ""),
    # ---- Session 29 ----
    (29, "tentative new title 'The Eelgrass Year' replacing 'Salt and Compass'",
     "What tentative new title replaced 'Salt and Compass' in session 29?",
     "The Eelgrass Year", ["Eelgrass Year"], E, ""),
    (29, "adding one Camille POV chapter as chapter 11",
     "As which chapter number was the new Camille POV chapter added?",
     "chapter 11", [], E, ""),
    (29, "cutting Owen Trask scenes to 4 from 6",
     "To how many scenes were Owen Trask's appearances cut in session 29?",
     "4", ["four"], E, ""),
    (29, "Renata Goldfarb approved 'The Eelgrass Year' on February 2",
     "On what date did Renata approve 'The Eelgrass Year'?",
     "February 2", [], E, ""),
    (29, "trimming journal device to 5 italicized fragments total",
     "To how many italicized fragments was the journal device trimmed?",
     "5 italicized fragments", ["5 fragments", "5", "five"], E, ""),
    # ---- Session 30 ----
    (30, "final pre-submission word count 89700",
     "What was the final pre-submission word count in session 30?",
     "89700", ["89,700"], E, ""),
    (30, "Persephone the research vessel renamed Cordelia",
     "To what name was the research vessel Persephone renamed in session 30?",
     "Cordelia", [], E, ""),
    (30, "Renata's submission list of 14 editors finalized",
     "How many editors were on Renata's finalized submission list?",
     "14 editors", ["14"], E, ""),
    (30, "submission planned for February 25, 2026",
     "On what date was the manuscript submission planned?",
     "February 25, 2026", ["February 25"], E, ""),
    (30, "comp titles updated to 'Olive Kitteridge' and 'Migrations' by Charlotte McConaghy",
     "Which Charlotte McConaghy novel became an updated comp title in session 30?",
     "Migrations", [], E, ""),
]


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def _legacy_items(rows: list[tuple]) -> list[dict]:
    items = []
    for qid, session, quote, question, gold, aliases in rows:
        items.append({
            "qid": qid,
            "source_session": session,
            "fact_quote": quote,
            "question": question,
            "gold_short": gold,
            "tier1_aliases": list(aliases),
            "excluded": False,
            "exclusion_reason": "",
            "legacy": True,
        })
    return items


def _generated_items(
    rows: list[tuple],
    prefix: str,
    excluded_sessions: set[int],
    *,
    force_exclude_all_reason: str | None = None,
    v11_excluded_sessions: set[int] | None = None,
    v11_point_excluded: dict[str, str] | None = None,
    v11_gold_fixes: dict[str, tuple[str, list[str]]] | None = None,
    v11_rewords: dict[str, tuple[str, str]] | None = None,
) -> list[dict]:
    v11_excluded_sessions = v11_excluded_sessions or set()
    v11_point_excluded = v11_point_excluded or {}
    v11_gold_fixes = v11_gold_fixes or {}
    v11_rewords = v11_rewords or {}
    seen_reword: set[str] = set()

    items = []
    counters: dict[int, int] = {}
    for session, quote, question, gold, aliases, excl_override, reason_override in rows:
        counters[session] = counters.get(session, 0) + 1
        qid = f"{prefix}_s{session:02d}_{counters[session]:02d}"
        excluded = False
        reason = ""
        # whole-layer drop wins over everything (v1.1 novel-layer drop).
        if force_exclude_all_reason is not None:
            excluded = True
            reason = force_exclude_all_reason
        # per-fact override wins (e.g. ambiguous Marubeni price / years).
        elif isinstance(excl_override, str):
            excluded = True
            reason = excl_override
        elif "Trattoria Modena" in quote:
            excluded = True
            reason = TRATTORIA_REASON
        elif session in excluded_sessions:
            excluded = True
            reason = BLEED_REASON
        elif session in v11_excluded_sessions:
            excluded = True
            reason = V11_SESSION_REASON
        elif qid in v11_point_excluded:
            excluded = True
            reason = v11_point_excluded[qid]

        # v1.1 reword (question text only; qid/gold unchanged). Applied even to
        # excluded items so the JSON record stays consistent; the assert guards
        # against silent drift between this map and the source row.
        if qid in v11_rewords:
            expected_old, new_question = v11_rewords[qid]
            if question != expected_old:
                raise ValueError(
                    f"reword drift for {qid}: source question {question!r} "
                    f"does not match expected {expected_old!r}"
                )
            question = new_question
            seen_reword.add(qid)

        # v1.1 gold fix (only meaningful for still-included qids).
        if qid in v11_gold_fixes:
            gold, aliases = v11_gold_fixes[qid]

        items.append({
            "qid": qid,
            "source_session": session,
            "fact_quote": quote,
            "question": question,
            "gold_short": gold,
            "tier1_aliases": list(aliases),
            "excluded": excluded,
            "exclusion_reason": reason,
            "legacy": False,
        })

    unseen = set(v11_rewords) - seen_reword
    if unseen:
        raise ValueError(f"reword qids not found in rows: {sorted(unseen)}")
    return items


def build_restaurant() -> list[dict]:
    return (
        _legacy_items(LEGACY_RESTAURANT)
        + _generated_items(
            RESTAURANT_GEN,
            "rest",
            RESTAURANT_EXCLUDED_SESSIONS,
            v11_excluded_sessions=RESTAURANT_V11_EXCLUDED_SESSIONS,
            v11_point_excluded=RESTAURANT_V11_POINT_EXCLUDED,
            v11_gold_fixes=RESTAURANT_V11_GOLD_FIXES,
            v11_rewords=RESTAURANT_V11_REWORDS,
        )
    )


def build_novel() -> list[dict]:
    return (
        _legacy_items(LEGACY_NOVEL)
        + _generated_items(
            NOVEL_GEN, "novel", set(),
            force_exclude_all_reason=NOVEL_V11_DROP_REASON,
        )
    )


def _write(path: Path, items: list[dict]) -> None:
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    restaurant = build_restaurant()
    novel = build_novel()
    _write(ROOT / "questions_restaurant.json", restaurant)
    _write(ROOT / "questions_novel.json", novel)

    for name, items in (("restaurant", restaurant), ("novel", novel)):
        legacy = sum(1 for q in items if q["legacy"])
        gen = sum(1 for q in items if not q["legacy"])
        excl = sum(1 for q in items if q["excluded"])
        print(
            f"{name}: {len(items)} total "
            f"(legacy={legacy}, generated={gen}, excluded={excl})"
        )


if __name__ == "__main__":
    main()

"""Unit tests for the bineval tier-1 scorer (WO-0 acceptance).

Covers: base normalization, number-word<->digit conversion, unit-suffix
tolerance, scrub of [PN/<CONTEXT> tokens, refusal->fail, the two known
false-negative fixes, subset/position answer mapping, and aggregate math.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.bineval import score_binary as sb

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# normalization
# --------------------------------------------------------------------------

def test_normalize_lower_and_punct():
    assert sb.normalize("Osteria Morishita!") == "osteria morishita"


def test_normalize_dash_fold():
    # en-dash and full-width dash both fold to ascii hyphen
    assert sb.normalize("October 8–11") == sb.normalize("October 8-11")


def test_normalize_nfkc_fullwidth_digits():
    assert sb.normalize("２０００") == "2000"  # fullwidth 2000


# --------------------------------------------------------------------------
# number-word <-> digit
# --------------------------------------------------------------------------

@pytest.mark.parametrize("word,digit", [
    ("four", "4"), ("one", "1"), ("twelve", "12"), ("twenty", "20"), ("zero", "0"),
])
def test_numeric_canon_word_to_digit(word, digit):
    assert sb._numeric_canon(sb.normalize(word)) == digit


def test_numeric_canon_phrase():
    assert sb._numeric_canon(sb.normalize("four days")) == "4 days"


def test_num_variants_bidirectional():
    assert sb._num_variants("four") == {"four", "4"}
    assert sb._num_variants("4") == {"4", "four"}
    assert sb._num_variants("bologna") == {"bologna"}


# --------------------------------------------------------------------------
# scrub
# --------------------------------------------------------------------------

def test_scrub_pn_tokens():
    assert sb.scrub("The rating is [PN3.2] 3.61 now") == "The rating is 3.61 now"


def test_scrub_bare_pn():
    # bare "[PN" without closing bracket is still removed
    assert "[PN" not in sb.scrub("value [PN and more")


def test_scrub_context_tags():
    out = sb.scrub("<CONTEXT>stuff</CONTEXT> answer: 47 labels")
    assert "context" not in out.lower()
    assert "47 labels" in out


# --------------------------------------------------------------------------
# tier1_match: aliases, unit suffix, the two false negatives
# --------------------------------------------------------------------------

def test_tier1_exact_gold():
    ok, alias = sb.tier1_match("Osteria Morishita", [], "It is called Osteria Morishita.")
    assert ok and alias is None


def test_tier1_false_negative_q13_unit_suffix():
    # gold "12,000 yen per person" vs pred "12,000 yen" -> alias rescue
    ok, alias = sb.tier1_match(
        "12,000 yen per person", ["12,000 yen", "12,000"], "A13: 12,000 yen"
    )
    assert ok
    assert alias == "12,000 yen"


def test_tier1_false_negative_q14_number_word():
    # gold "four days" vs pred "4 days" -> numeric canon on both sides
    ok, _alias = sb.tier1_match("four days", ["4 days", "four", "4"], "A14: 4 days")
    assert ok


def test_tier1_number_word_reverse():
    # gold digit, prediction spelled out
    ok, _ = sb.tier1_match("4 days", [], "It finished four days late.")
    assert ok


def test_tier1_no_match_on_wrong_value():
    ok, alias = sb.tier1_match("12,000 yen", ["12,000"], "18,000 yen per person")
    assert not ok and alias is None


def test_tier1_empty_answer_no_match():
    ok, _ = sb.tier1_match("47 labels", ["47"], "")
    assert not ok


# --------------------------------------------------------------------------
# refusal / no-answer detection
# --------------------------------------------------------------------------

@pytest.mark.parametrize("marker", [
    "not in context", "not mentioned", "unknown", "N/A", "none",
    "not specified", "  Not In Context  ",
])
def test_is_no_answer_true(marker):
    assert sb.is_no_answer(marker)


@pytest.mark.parametrize("real", [
    "12,000 yen", "Osteria Morishita", "15 years", "none of the wine was Italian",
])
def test_is_no_answer_false_on_real_answers(real):
    # a real answer that merely *contains* a marker word is not a refusal
    assert not sb.is_no_answer(real)


# --------------------------------------------------------------------------
# score_condition: verdict routing + aggregate
# --------------------------------------------------------------------------

def _q(qid, gold, aliases):
    return {"qid": qid, "question": f"q-{qid}", "gold_short": gold,
            "tier1_aliases": aliases}


def test_score_condition_pass_fail_indeterminate():
    questions = [
        _q("a", "47 labels", ["47"]),          # A1 pass
        _q("b", "Mr. Ogawa", ["Ogawa"]),       # A2 refusal -> fail
        _q("c", "12,000 yen", []),             # A3 wrong concrete -> indeterminate
        _q("d", "four days", ["4 days"]),      # A4 pass via number word
    ]
    by_pos = {1: "47 labels", 2: "not in context", 3: "18,000 yen", 4: "4 days"}
    res = sb.score_condition(questions, by_pos, {}, sb.judge_none)
    agg = res["aggregate"]
    assert agg["pass"] == 2
    assert agg["fail"] == 1
    assert agg["indeterminate"] == 1
    assert agg["total"] == 4
    assert agg["pass_rate_tier1"] == 0.5  # 2/4
    verd = {it["qid"]: it["verdict"] for it in res["items"]}
    assert verd == {"a": "pass", "b": "fail", "c": "indeterminate", "d": "pass"}
    # the indeterminate item is queued for tier 2
    assert [p["qid"] for p in res["pending_tier2"]] == ["c"]


def test_score_condition_qid_keyed_answers_take_precedence():
    questions = [_q("x", "Cordelia", [])]
    # positional says wrong, qid map says right -> qid wins
    res = sb.score_condition(questions, {1: "Persephone"}, {"x": "renamed Cordelia"},
                             sb.judge_none)
    assert res["aggregate"]["pass"] == 1


def test_pluggable_judge_resolves_indeterminate():
    questions = [_q("z", "marine ecologist", [])]
    by_pos = {1: "she studies underwater plants"}  # paraphrase, no substring

    def judge(_question, _gold, _answer):
        return True

    res = sb.score_condition(questions, by_pos, {}, judge)
    assert res["aggregate"]["pass"] == 1
    assert res["items"][0]["tier"] == 2


# --------------------------------------------------------------------------
# select_questions: subset + exclusion filtering
# --------------------------------------------------------------------------

def _mkq(qid, legacy, excluded):
    return {"qid": qid, "legacy": legacy, "excluded": excluded,
            "gold_short": "x", "question": "q", "tier1_aliases": []}


def test_select_legacy_subset():
    qs = [_mkq("l1", True, False), _mkq("g1", False, False)]
    out = sb.select_questions(qs, "legacy", include_excluded=False)
    assert [q["qid"] for q in out] == ["l1"]


def test_select_generated_drops_excluded():
    qs = [_mkq("g1", False, False), _mkq("g2", False, True)]
    out = sb.select_questions(qs, "generated", include_excluded=False)
    assert [q["qid"] for q in out] == ["g1"]


def test_select_include_excluded():
    qs = [_mkq("g1", False, False), _mkq("g2", False, True)]
    out = sb.select_questions(qs, "generated", include_excluded=True)
    assert [q["qid"] for q in out] == ["g1", "g2"]


# --------------------------------------------------------------------------
# end-to-end against the real legacy answer files (sec.2.6 acceptance)
# --------------------------------------------------------------------------

def test_generated_files_exist_and_wellformed():
    for fn in ("questions_restaurant.json", "questions_novel.json"):
        p = ROOT / fn
        assert p.exists(), f"{fn} not generated; run gen_questions.py"
        items = json.loads(p.read_text(encoding="utf-8"))
        req = {"qid", "source_session", "fact_quote", "question", "gold_short",
               "tier1_aliases", "excluded", "exclusion_reason"}
        for it in items:
            assert req.issubset(it.keys())


def test_sanity_ordering_full_beats_truncated_summarized():
    """full >> truncated ~ summarized under tier-1 only (sec.2.6)."""
    questions = sb.load_questions(ROOT / "questions_restaurant.json")
    legacy = sb.select_questions(questions, "legacy", include_excluded=False)
    assert len(legacy) == 16
    eval_dir = ROOT.parent / "longchat" / "eval_results"
    rates = {}
    for cond in ("full", "truncated", "summarized"):
        txt = (eval_dir / f"answers_{cond}.txt").read_text(encoding="utf-8")
        by_pos = sb.parse_answer_lines(txt)
        res = sb.score_condition(legacy, by_pos, {}, sb.judge_none)
        rates[cond] = res["aggregate"]["pass_rate_tier1"]
    assert rates["full"] == 1.0
    assert rates["full"] > rates["truncated"] + 0.5
    assert rates["full"] > rates["summarized"] + 0.5
    assert abs(rates["truncated"] - rates["summarized"]) < 0.05


def test_two_known_false_negatives_pass_in_full():
    questions = sb.load_questions(ROOT / "questions_restaurant.json")
    legacy = sb.select_questions(questions, "legacy", include_excluded=False)
    eval_dir = ROOT.parent / "longchat" / "eval_results"
    by_pos = sb.parse_answer_lines((eval_dir / "answers_full.txt").read_text(encoding="utf-8"))
    res = sb.score_condition(legacy, by_pos, {}, sb.judge_none)
    verd = {it["qid"]: it["verdict"] for it in res["items"]}
    assert verd["rest_q13"] == "pass"   # "12,000 yen" vs gold "12,000 yen per person"
    assert verd["rest_q14"] == "pass"   # "4 days" vs gold "four days"


# --------------------------------------------------------------------------
# 2026-09-18 scoring fixes (mcbuild-bench item D): abstention, short golds,
# all-of aliases.  ``strict_short=False`` reproduces the old behaviour.
# --------------------------------------------------------------------------

def test_d1_abstention_only_matches_an_abstention_gold():
    # "unknown" used to pass gold "No" by substring ("no" in "unknown").
    ok, _ = sb.tier1_match("No", ["not shown"], "unknown")
    assert not ok
    ok, _ = sb.tier1_match("No, it was dropped", ["no"], "not in context")
    assert not ok
    # an absent-fact question: gold IS an abstention -> pass
    ok, alias = sb.tier1_match("unknown", ["not in context"], "unknown")
    assert ok and alias is None
    ok, _ = sb.tier1_match("unknown", ["not in context"], "Not in context.")
    assert ok  # gold "unknown" is itself an abstention -> gold-first match
    # old behaviour on request
    ok, _ = sb.tier1_match("No", [], "unknown", strict_short=False)
    assert ok


@pytest.mark.parametrize("gold, answer, expect", [
    ("4", "14", False),
    ("4", "4.5", False),
    ("4", "4", True),
    ("4", "The answer is 4.", True),
    ("4", "4 blocks", True),
    ("13", "13 m x 22 m", True),
    ("13", "113 tests", False),
    ("odd", "29, odd", True),
    ("odd", "oddly", False),
    ("y=0", "y = 0", True),
    ("y=0", "y = 05", False),
])
def test_d2_short_golds_match_with_boundaries(gold, answer, expect):
    ok, _ = sb.tier1_match(gold, [], answer)
    assert ok is expect
    # pre-fix: "4" was a substring of "14" and "4.5"
    assert sb.tier1_match("4", [], "14", strict_short=False)[0]


def test_d3_all_of_alias_requires_every_part():
    aliases = ["186 & 109 & 81"]
    ok, alias = sb.tier1_match("186 x 109 x 81", aliases, "186 by 109 by 81 blocks")
    assert ok and alias == "186 & 109 & 81"
    ok, _ = sb.tier1_match("186 x 109 x 81", aliases, "186 blocks")
    assert not ok
    ok, _ = sb.tier1_match("towers 4, spires 6", ["4 & 6"], "4 and 6")
    assert ok
    ok, _ = sb.tier1_match("towers 4, spires 6", ["4 & 6"], "4 and 16")
    assert not ok


def test_score_condition_forwards_strict_short():
    questions = [_q("a", "No", ["not shown"])]
    strict = sb.score_condition(questions, {1: "unknown"}, {}, sb.judge_none)
    assert strict["aggregate"]["fail"] == 1 and strict["aggregate"]["strict_short"] is True
    loose = sb.score_condition(questions, {1: "unknown"}, {}, sb.judge_none, strict_short=False)
    assert loose["aggregate"]["pass"] == 1 and loose["aggregate"]["strict_short"] is False


def test_cli_no_strict_short_flag(tmp_path):
    q = tmp_path / "q.json"
    q.write_text(json.dumps([{"qid": "a", "question": "?", "gold_short": "No",
                              "tier1_aliases": [], "excluded": False, "legacy": False}]),
                 encoding="utf-8")
    a = tmp_path / "ans.json"
    a.write_text(json.dumps({"a": "unknown"}), encoding="utf-8")
    out = tmp_path / "scored.json"
    assert sb.main(["--answers", str(a), "--questions", str(q), "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["aggregate"]["fail"] == 1
    assert sb.main(["--answers", str(a), "--questions", str(q), "--out", str(out),
                    "--no-strict-short"]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["aggregate"]["pass"] == 1

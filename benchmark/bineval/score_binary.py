"""Two-tier binary scorer for the bineval instrument (WO-0).

Tier 1  : NFKC-normalized substring / alias match. NO LLM calls.
          Base normalization is copied verbatim from
          `benchmark/longchat/score_eval.py:normalize` and then extended with
          a number-word<->digit table (1..20) and unit-suffix tolerance so the
          two known false negatives pass:
              gold "12,000 yen per person"  vs pred "12,000 yen"
              gold "four days"              vs pred "4 days"
Tier 2  : pluggable `judge_fn(question, gold_short, answer_text) -> bool | None`.
          Default `--tier2 none` ships NO api client; tier-1-indeterminate items
          are written to a sibling `pending_tier2.json` instead of being scored.

CLI:
    python -m benchmark.bineval.score_binary \
        --answers <file-or-dir> --questions <json> --out <json> \
        [--subset all|legacy|generated] [--include-excluded] [--tier2 none]

Answer sources accepted by --answers:
    * a single .txt file in `A{n}: text` line format (longchat convention);
      the k-th SCORED question (after subset/exclusion filtering, in file order)
      maps to `A{k}` exactly like score_eval.py enumerate(questions, 1).
    * a single .json file mapping {qid: answer_text}.
    * a directory: every *.txt is scored as its own condition and every *.json
      keyed file is merged (qid -> answer); results are emitted per condition.

Windows: all file I/O uses encoding="utf-8"; every print() is cp932-safe
(ASCII only, no emoji / box-drawing glyphs).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Callable, Optional

# --------------------------------------------------------------------------
# Tier-1 normalization  (base copied from benchmark/longchat/score_eval.py)
# --------------------------------------------------------------------------

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")
_DASHES = ("‐", "‑", "‒", "–", "—", "－")


def normalize(s: object) -> str:
    """NFKC + lowercase + dash-fold + punctuation-strip + whitespace-collapse.

    Identical behaviour to score_eval.normalize (the base of tier 1). Kept as a
    standalone copy on purpose: benchmark/longchat is not an importable package
    (no __init__.py) and RESEARCH_PROGRAM sec.2 says "extend", not "import".
    """
    s = unicodedata.normalize("NFKC", str(s).lower())
    for ch in _DASHES:
        s = s.replace(ch, "-")
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


# number-word <-> digit table for 1..20 (and zero), both directions.
_WORD2NUM = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}
_NUM2WORD = {v: k for k, v in _WORD2NUM.items()}


def _num_variants(token: str) -> set[str]:
    """All equivalent spellings of a single normalized token for numbers 0..20.

    "four" -> {"four", "4"};  "4" -> {"4", "four"};  otherwise -> {token}.
    """
    out = {token}
    if token in _WORD2NUM:
        out.add(_WORD2NUM[token])
    if token in _NUM2WORD:
        out.add(_NUM2WORD[token])
    return out


def _numeric_canon(s: str) -> str:
    """Rewrite an already-normalized string so number words 0..20 become digits.

    Applied to BOTH gold and prediction before substring test, so
    "four days" and "4 days" collapse to the same "4 days".
    """
    return " ".join(_WORD2NUM.get(tok, tok) for tok in s.split())


# --------------------------------------------------------------------------
# Scrub  (remove CD scaffolding tokens before matching / judging)
# --------------------------------------------------------------------------

# [PN{mass}] spans, bare "[PN", and <CONTEXT>...</CONTEXT> markers.
_PN_TOKEN = re.compile(r"\[PN[^\]]*\]?", re.IGNORECASE)
_CONTEXT_TAG = re.compile(r"</?context>", re.IGNORECASE)


def scrub(text: str) -> str:
    """Strip `[PN...` tokens and <CONTEXT>/</CONTEXT> tags from answer text."""
    text = _PN_TOKEN.sub(" ", text)
    text = _CONTEXT_TAG.sub(" ", text)
    return _WS.sub(" ", text).strip()


# Normalized no-answer / refusal markers. When the WHOLE scrubbed+normalized
# answer equals one of these, the model asserted it does not know / the fact is
# absent -> tier-1 FAIL (an unambiguous non-statement of the fact), never
# "indeterminate". This is what lets tier 1 alone reproduce the sec.2.6 ordering
# (full >> truncated ~ summarized) without any judge: the truncation/summary
# arms answer "not in context" on facts they dropped.
_NO_ANSWER_MARKERS = frozenset(
    normalize(m)
    for m in (
        "not in context",
        "not in the context",
        "not mentioned",
        "not mentioned in context",
        "not mentioned in the context",
        "not stated",
        "not specified",
        "not provided",
        "not available",
        "not found",
        "no information",
        "no answer",
        "unknown",
        "n a",  # "n/a" normalizes to "n a"
        "na",
        "none",
        "i don t know",
        "cannot determine",
        "can t determine",
        "unable to determine",
        "insufficient information",
        "not enough information",
    )
)


def is_no_answer(answer_text: str) -> bool:
    """True iff the whole scrubbed answer is a no-answer / refusal marker."""
    norm = normalize(scrub(answer_text))
    return norm in _NO_ANSWER_MARKERS


# --------------------------------------------------------------------------
# Tier 1 verdict
# --------------------------------------------------------------------------

# 2026-09-18 scoring fixes (mcbuild-bench item D). ``strict_short=True`` enables
# all three; ``--no-strict-short`` reproduces the pre-fix behaviour.
#   D1 abstention: a no-answer reply passes ONLY when the gold/alias itself is an
#      abstention marker ("unknown" no longer matches gold "No" by substring).
#   D2 short golds: a normalized candidate that is purely numeric or <= 3
#      characters must match on boundaries ``(?<![\w.])...(?![\w.])`` on a hay
#      that keeps DECIMAL points ("4" matches neither "14" nor "4.5").
#   D3 all-of aliases: a candidate containing " & " requires EVERY part to
#      match (each part under the same rules).
SHORT_MAX_CHARS = 3
ALL_OF_SEP = " & "
_PUNCT_KEEP_DOT = re.compile(r"[^\w\s.]", re.UNICODE)
_NON_DECIMAL_DOT = re.compile(r"(?<!\d)\.|\.(?!\d)")


def normalize_keep_decimal(s: object) -> str:
    """``normalize`` but a "." BETWEEN digits survives ("4.5" stays "4.5");
    every other dot is stripped like any punctuation."""
    s = unicodedata.normalize("NFKC", str(s).lower())
    for ch in _DASHES:
        s = s.replace(ch, "-")
    s = _PUNCT_KEEP_DOT.sub(" ", s)
    s = _NON_DECIMAL_DOT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def is_short_needle(needle: str) -> bool:
    """D2: purely numeric (spaces ignored) or at most SHORT_MAX_CHARS characters."""
    return needle.replace(" ", "").isdigit() or len(needle) <= SHORT_MAX_CHARS


def _match_simple(cand: str, hay: str, hay_dec: str, strict_short: bool) -> bool:
    if not strict_short:
        needle = _numeric_canon(normalize(cand))
        return bool(needle) and needle in hay
    needle_dec = _numeric_canon(normalize_keep_decimal(cand))
    if not needle_dec:
        return False
    if is_short_needle(needle_dec):
        pattern = r"(?<![\w.])" + re.escape(needle_dec) + r"(?![\w.])"
        return re.search(pattern, hay_dec) is not None
    needle = _numeric_canon(normalize(cand))
    return bool(needle) and needle in hay


def _match_candidate(cand: str, hay: str, hay_dec: str, strict_short: bool) -> bool:
    if strict_short and ALL_OF_SEP in cand:
        parts = [p for p in cand.split(ALL_OF_SEP) if p.strip()]
        return bool(parts) and all(_match_simple(p, hay, hay_dec, strict_short) for p in parts)
    return _match_simple(cand, hay, hay_dec, strict_short)


def tier1_match(
    gold_short: str,
    aliases: list[str],
    answer_text: str,
    *,
    strict_short: bool = True,
) -> tuple[bool, Optional[str]]:
    """Return (matched, matched_alias_or_None).

    A candidate (gold_short or any alias) matches when its numeric-canonical
    normalized form is a substring of the numeric-canonical normalized,
    scrubbed answer. Empty candidates never match. Unit-suffix tolerance is
    inherent: a shorter alias ("12,000 yen") is itself a candidate, so a
    prediction lacking the "per person" suffix still matches via that alias.

    ``strict_short`` (default True) adds the D1 / D2 / D3 rules documented
    above; False is the pre-2026-09-18 behaviour.
    """
    hay = _numeric_canon(normalize(scrub(answer_text)))
    if not hay:
        return (False, None)
    hay_dec = _numeric_canon(normalize_keep_decimal(scrub(answer_text)))
    # gold first (so matched_alias stays None when the full gold matched),
    # then declared aliases in order.
    candidates: list[tuple[str, Optional[str]]] = [(gold_short, None)]
    candidates += [(a, a) for a in aliases]
    if strict_short and is_no_answer(answer_text):
        # D1: an abstention passes only an abstention gold/alias.
        for cand, label in candidates:
            if normalize(cand) in _NO_ANSWER_MARKERS:
                return (True, label)
        return (False, None)
    for cand, label in candidates:
        if _match_candidate(cand, hay, hay_dec, strict_short):
            return (True, label)
    return (False, None)


# type of a pluggable tier-2 judge; returns True/False/None(=still indeterminate)
JudgeFn = Callable[[str, str, str], Optional[bool]]


def judge_none(question: str, gold_short: str, answer_text: str) -> Optional[bool]:
    """Default tier-2: no judge wired. Everything stays indeterminate."""
    return None


# --------------------------------------------------------------------------
# Answer loading
# --------------------------------------------------------------------------

_A_LINE = re.compile(r"\s*A(\d+)\s*[:\-]\s*(.+)", re.IGNORECASE)


def parse_answer_lines(text: str) -> dict[int, str]:
    """Parse `A{n}: text` lines -> {n: text}. Mirrors score_eval.parse_answers
    but without an upper bound (bound is applied by the caller via the subset).
    """
    answers: dict[int, str] = {}
    for line in text.strip().split("\n"):
        m = _A_LINE.match(line.strip())
        if m:
            answers[int(m.group(1))] = m.group(2).strip()
    return answers


def load_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"questions file {path} must be a JSON list")
    return data


def select_questions(
    questions: list[dict],
    subset: str,
    include_excluded: bool,
) -> list[dict]:
    """Filter by subset and exclusion flag, preserving source order.

    subset: 'all' | 'legacy' | 'generated'. 'legacy' keeps items with
    legacy==True (the WO's rest_q01..16 sanity anchor); 'generated' keeps the
    rest. Excluded items (data-hygiene) are dropped unless include_excluded.
    """
    out = []
    for q in questions:
        is_legacy = bool(q.get("legacy", False))
        if subset == "legacy" and not is_legacy:
            continue
        if subset == "generated" and is_legacy:
            continue
        if q.get("excluded", False) and not include_excluded:
            continue
        out.append(q)
    return out


# --------------------------------------------------------------------------
# Scoring one condition
# --------------------------------------------------------------------------

def truncate_words(text: str, max_words: Optional[int]) -> str:
    """Keep only the first ``max_words`` whitespace-separated words.

    2026-09-07 Fable review #6: tier-1 is a substring match with no length
    limit, so a reader that echoes the whole CD would pass every question. The
    C3 (wM) arms push the model toward copying CD text, so those arms are
    scored on a truncated answer. None = no truncation (bineval v1 behaviour).
    """
    if max_words is None:
        return text
    words = text.split()
    return " ".join(words[:max_words])


def multi_gold_answer_count(
    scored_questions: list[dict],
    answers: list[str],
) -> int:
    """Side metric: number of answers containing >= 2 DISTINCT golds of the
    question set (echoing indicator). Uses the same normalize() as tier-1."""
    golds = {normalize(q["gold_short"]) for q in scored_questions}
    golds = {g for g in golds if g}
    n = 0
    for ans in answers:
        na = normalize(ans)
        if sum(1 for g in golds if g in na) >= 2:
            n += 1
    return n


def score_condition(
    scored_questions: list[dict],
    answers_by_pos: dict[int, str],
    answers_by_qid: dict[str, str],
    judge_fn: JudgeFn,
    max_words: Optional[int] = None,
    *,
    strict_short: bool = True,
) -> dict:
    """Score an already-filtered question list against one answer source.

    Position mapping: the k-th scored question (1-indexed) <-> A{k}. A per-qid
    dict (from a .json answer file) takes precedence when the qid is present.
    ``strict_short`` is forwarded to ``tier1_match`` (D1-D3).
    """
    per_q = []
    pending = []
    n_pass = n_fail = n_indet = 0
    used_answers: list[str] = []
    for k, q in enumerate(scored_questions, 1):
        qid = q["qid"]
        if qid in answers_by_qid:
            ans = answers_by_qid[qid]
        else:
            ans = answers_by_pos.get(k, "")
        ans = truncate_words(ans, max_words)
        used_answers.append(ans)
        matched, alias = tier1_match(
            q["gold_short"], q.get("tier1_aliases", []), ans, strict_short=strict_short
        )
        if matched:
            n_pass += 1
            rec = {"qid": qid, "verdict": "pass", "tier": 1}
            if alias is not None:
                rec["matched_alias"] = alias
            per_q.append(rec)
            continue
        # tier-1 FAIL: empty answer or an explicit no-answer/refusal marker is an
        # unambiguous non-statement of the fact; no judge needed.
        if (not scrub(ans)) or is_no_answer(ans):
            n_fail += 1
            per_q.append({"qid": qid, "verdict": "fail", "tier": 1})
            continue
        # otherwise the answer is a concrete non-matching string (possible
        # paraphrase / wrong value) -> defer to tier 2 (default: indeterminate).
        verdict = judge_fn(q["question"], q["gold_short"], scrub(ans))
        if verdict is True:
            n_pass += 1
            per_q.append({"qid": qid, "verdict": "pass", "tier": 2})
        elif verdict is False:
            n_fail += 1
            per_q.append({"qid": qid, "verdict": "fail", "tier": 2})
        else:
            n_indet += 1
            per_q.append({"qid": qid, "verdict": "indeterminate", "tier": 1})
            pending.append({
                "qid": qid,
                "question": q["question"],
                "gold_short": q["gold_short"],
                "answer_fragment": scrub(ans),
            })
    total = len(scored_questions)
    # pass_rate_tier1 counts an indeterminate as not-yet-passed (denominator =
    # all scored items), so the reported rate is a conservative tier-1 floor.
    aggregate = {
        "pass": n_pass,
        "fail": n_fail,
        "indeterminate": n_indet,
        "total": total,
        "pass_rate_tier1": round(n_pass / total, 4) if total else 0.0,
        "max_words": max_words,
        "strict_short": bool(strict_short),
        "multi_gold_answers": multi_gold_answer_count(scored_questions, used_answers),
    }
    return {"items": per_q, "aggregate": aggregate, "pending_tier2": pending}


# --------------------------------------------------------------------------
# Answer-source discovery (file or directory)
# --------------------------------------------------------------------------

def gather_answer_sources(path: Path) -> list[tuple[str, dict[int, str], dict[str, str]]]:
    """Return [(condition_name, answers_by_pos, answers_by_qid), ...].

    A .txt file -> one condition keyed by A{n} position.
    A .json file mapping qid->text -> one condition keyed by qid.
    A directory  -> one condition per *.txt (named by stem). Directory mode is
                    intentionally .txt-only so unrelated sidecar JSON (e.g. an
                    old scored.json) is never mistaken for an answer file; pass
                    a keyed .json directly as --answers to score it.
    """
    sources: list[tuple[str, dict[int, str], dict[str, str]]] = []
    if path.is_dir():
        files = sorted(path.glob("*.txt"))
        if not files:
            raise FileNotFoundError(f"no .txt answer files under {path}")
        for f in files:
            sources.append((f.stem, parse_answer_lines(f.read_text(encoding="utf-8")), {}))
        return sources
    # single file
    if path.suffix.lower() == ".txt":
        sources.append((path.stem, parse_answer_lines(path.read_text(encoding="utf-8")), {}))
    elif path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        sources.append((path.stem, {}, {str(k): str(v) for k, v in obj.items()}))
    else:
        raise ValueError(f"unsupported answer file type: {path.suffix}")
    return sources


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _resolve_judge(name: str) -> JudgeFn:
    if name == "none":
        return judge_none
    raise SystemExit(
        f"tier2='{name}' is not available in this build; only 'none' ships. "
        "Wire a judge_fn programmatically and call score_condition() directly."
    )


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Two-tier binary scorer (bineval WO-0)")
    ap.add_argument("--answers", required=True, help="answer file (.txt/.json) or directory")
    ap.add_argument("--questions", required=True, help="questions JSON (bineval schema)")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--subset", choices=["all", "legacy", "generated"], default="all")
    ap.add_argument("--include-excluded", action="store_true",
                    help="also score items with excluded==true (default: skip)")
    ap.add_argument("--tier2", default="none",
                    help="tier-2 judge id; only 'none' ships (indeterminate -> pending_tier2.json)")
    ap.add_argument("--max-words", type=int, default=None,
                    help="score only the first N words of each answer (C3/wM arms: 32)")
    ap.add_argument("--no-strict-short", action="store_true",
                    help="pre-2026-09-18 tier-1: no abstention rule, no boundaries on "
                         "short golds, no ' & ' all-of aliases")
    args = ap.parse_args(argv)

    q_path = Path(args.questions)
    a_path = Path(args.answers)
    out_path = Path(args.out)

    questions = load_questions(q_path)
    scored_questions = select_questions(questions, args.subset, args.include_excluded)
    judge_fn = _resolve_judge(args.tier2)

    sources = gather_answer_sources(a_path)
    conditions: dict[str, dict] = {}
    all_pending: dict[str, list] = {}
    for name, by_pos, by_qid in sources:
        res = score_condition(scored_questions, by_pos, by_qid, judge_fn, max_words=args.max_words,
                              strict_short=not args.no_strict_short)
        conditions[name] = {"items": res["items"], "aggregate": res["aggregate"]}
        if res["pending_tier2"]:
            all_pending[name] = res["pending_tier2"]

    # single-condition runs flatten to top level (keeps score_eval-like shape);
    # multi-condition runs nest under "conditions".
    if len(conditions) == 1:
        (only_name, only_val), = conditions.items()
        output = {
            "condition": only_name,
            "subset": args.subset,
            "questions_file": str(q_path),
            "items": only_val["items"],
            "aggregate": only_val["aggregate"],
        }
    else:
        output = {
            "subset": args.subset,
            "questions_file": str(q_path),
            "conditions": conditions,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    # pending_tier2.json written next to --out (only when tier-1 left gaps)
    if all_pending:
        pend_path = out_path.parent / "pending_tier2.json"
        pend_payload = all_pending if len(all_pending) > 1 else next(iter(all_pending.values()))
        pend_path.write_text(json.dumps(pend_payload, ensure_ascii=False, indent=2),
                             encoding="utf-8")

    # cp932-safe console summary (ASCII only)
    print(f"scored questions: {len(scored_questions)} (subset={args.subset})")
    for name in sorted(conditions):
        agg = conditions[name]["aggregate"]
        print(
            f"  {name}: pass={agg['pass']} fail={agg['fail']} "
            f"indeterminate={agg['indeterminate']} "
            f"pass_rate_tier1={agg['pass_rate_tier1']:.4f}"
        )
    if all_pending:
        n = sum(len(v) for v in all_pending.values())
        print(f"  wrote {n} indeterminate item(s) to pending_tier2.json")
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

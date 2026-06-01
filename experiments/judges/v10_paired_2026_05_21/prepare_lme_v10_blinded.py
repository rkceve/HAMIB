"""
Prepare blinded input for the fix-applied HAMIB + baseline LongMemEval
paired re-judging using the OFFICIAL LongMemEval rubric (Wu et al., ICLR 2025).

Items are anonymized + shuffled (seed=46) so the judge cannot tell which
response came from HAMIB vs baseline.

LongMemEval items carry a native qtype field that is preserved here so the
judge applies the appropriate per-qtype rubric.

Run from repo root:
    python -m experiments.judges.v10_paired_2026_05_21.prepare_lme_v10_blinded
"""
from __future__ import annotations
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS = REPO_ROOT / "results" / "longmemeval"

V10_HAMIB = RESULTS / "longmemeval_hamib_sbert.json"
V9_BASE = RESULTS / "longmemeval_baseline.json"

# Regex used as a pre-write assertion. The HAMIB input template embeds tokens
# like "[PN1.0]" and "<CONTEXT>" that never appear in the baseline prompt;
# if the LLM echoes one of these into its response field, an attentive judge
# could read mode off the response alone. This was an issue in the originally
# published inputs (~0.8% of LongMemEval items); future regenerations must
# either drop or sanitize affected items. See ../README.md "Known caveat".
_HAMIB_MARKER_RE = re.compile(r"\[PN\d+(?:\.\d+)?\]|</?CONTEXT>|\bmass=\S+")

OUT_DIR = Path(__file__).resolve().parent
# Blinded input is judge-visible; MAPPING is kept in a separate _private subdir
# so a judge subagent reading the input directory does not stumble onto the
# de-anonymization labels.
PRIVATE_DIR = OUT_DIR / "_private"
BLINDED = OUT_DIR / "judge_input_lme.json"
MAPPING = PRIVATE_DIR / "MAPPING_lme_v10_paired.json"
SEED = 46


def main() -> None:
    v10 = json.load(open(V10_HAMIB, encoding="utf-8"))
    v9 = json.load(open(V9_BASE, encoding="utf-8"))

    all_items = []
    for it in v10["items"]:
        all_items.append({
            "_mode": "hamib",
            "_orig_i": it["i"],
            "_orig_qid": it["qid"],
            "_substring_label": bool(it["correct"]),
            "qid": it["qid"],
            "qtype": it["qtype"],
            "question": it["question"],
            "gold": str(it["gold"]),
            "response": it["response"],
        })
    for it in v9["items"]:
        all_items.append({
            "_mode": "baseline",
            "_orig_i": it["i"],
            "_orig_qid": it["qid"],
            "_substring_label": bool(it["correct"]),
            "qid": it["qid"],
            "qtype": it["qtype"],
            "question": it["question"],
            "gold": str(it["gold"]),
            "response": it["response"],
        })

    print(f"Total items to judge: {len(all_items)} (v10 HAMIB {len(v10['items'])} + v9 baseline {len(v9['items'])})")

    rng = random.Random(SEED)
    rng.shuffle(all_items)

    # Note: LME qid is an opaque hash (e.g., "eace081b") that may end in "_abs"
    # to signal an abstention question; we preserve the original qid because
    # the judge rubric depends on the "_abs" suffix detection. The qid itself
    # does NOT leak the mode (hamib vs baseline) — both modes share the same qid
    # for the same question. Anonymous_id drops the "v10" tag to avoid even
    # the weak hint of a version number.
    blinded_items = []
    mapping = []
    for new_i, it in enumerate(all_items, 1):
        anon_id = f"lme_item_{new_i:04d}"
        blinded_items.append({
            "anonymous_id": anon_id,
            "qtype": it["qtype"],
            "qid": it["qid"],
            "question": it["question"],
            "gold": it["gold"],
            "response": it["response"],
        })
        mapping.append({
            "anonymous_id": anon_id,
            "mode": it["_mode"],
            "orig_i": it["_orig_i"],
            "orig_qid": it["_orig_qid"],
            "qtype": it["qtype"],
            "substring_label": it["_substring_label"],
        })

    # Judge-visible meta is sanitized: no system identifiers, no source paths,
    # no references to the two compared systems. The judge must not be able to
    # infer the experimental purpose from the input file alone.
    blinded_out = {
        "meta": {
            "n_items": len(blinded_items),
            "qtype_policy": "each item carries its native qtype; apply the matching rubric",
        },
        "items": blinded_items,
    }
    # MAPPING is kept private (separate dir) and carries full provenance for
    # later de-anonymization. It must never be accessed by the judge.
    map_out = {
        "meta": {
            "purpose": "De-anonymization mapping (judge-invisible).",
            "shuffle_seed": SEED,
            "source_hamib": str(V10_HAMIB.relative_to(REPO_ROOT)),
            "source_baseline": str(V9_BASE.relative_to(REPO_ROOT)),
        },
        "mapping": mapping,
    }

    # Pre-write assertion: no HAMIB-specific input-template tokens may appear in
    # any response field, otherwise the judge could deduce mode from response
    # text alone. Fail loudly here so the regeneration can be re-run after
    # sanitizing the upstream raw outputs.
    leaked = [
        it["anonymous_id"]
        for it in blinded_items
        if _HAMIB_MARKER_RE.search(it["response"] or "")
    ]
    if leaked:
        raise AssertionError(
            f"HAMIB-specific markers leaked into {len(leaked)} response field(s); "
            f"first 5 anonymous_ids: {leaked[:5]}. Sanitize the upstream raw "
            "outputs (strip [PN<n>], <CONTEXT> and mass=<...>) and re-run."
        )

    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(blinded_out, open(BLINDED, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(map_out, open(MAPPING, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"Wrote {BLINDED.relative_to(REPO_ROOT)}: {len(blinded_items)} items, {os.path.getsize(BLINDED)/1024:.1f} KB")
    print(f"Wrote {MAPPING.relative_to(REPO_ROOT)}: {len(mapping)} entries")
    mc = Counter(m["mode"] for m in mapping)
    print(f"  mode distribution (HIDDEN from judge): {dict(mc)}")
    qc = Counter(it["qtype"] for it in blinded_items)
    print("  qtype distribution:")
    for k, v in sorted(qc.items()):
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()

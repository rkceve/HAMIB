"""
Scan judge_input_*.json files for HAMIB-only input-template markers that may have
leaked into LLM response fields.

The HAMIB prompt template wraps the correlation diagram in `<CONTEXT>` ... and
embeds priority-number tags like `[PN1.0]` and `[PN0.5]`. The baseline prompt
never contains these strings. If the underlying LLM echoes one of those tokens
into its response, an attentive judge could in principle identify those items
as HAMIB items by their response text alone — a partial violation of channel 3
of the 9-channel isolation protocol (see ../README.md).

This script enumerates the affected anonymous_ids in each bundled judge_input
file so a reader can independently verify the leakage count claimed in
../README.md "Known caveat".

Run from the repository root:
    python -m experiments.judges.check_input_markers
"""
from __future__ import annotations
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Same regex used as a pre-write assertion in prepare_lme_v10_blinded.py.
_HAMIB_MARKER_RE = re.compile(r"\[PN\d+(?:\.\d+)?\]|</?CONTEXT>|\bmass=\S+")

INPUT_FILES = [
    HERE / "v10_paired_2026_05_21" / "judge_input_lme.json",
    HERE / "codex_gpt5_2026_05_25" / "judge_input_lme.json",
]


def scan(path: Path) -> tuple[int, int, list[str]]:
    data = json.load(open(path, encoding="utf-8"))
    items = data.get("items", [])
    leaked_ids: list[str] = []
    for it in items:
        resp = it.get("response") or ""
        if _HAMIB_MARKER_RE.search(resp):
            leaked_ids.append(it.get("anonymous_id", "?"))
    return len(items), len(leaked_ids), leaked_ids


def main() -> None:
    for path in INPUT_FILES:
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        n_items, n_leaked, leaked_ids = scan(path)
        rel = path.relative_to(HERE.parent.parent)
        print(f"{rel}: {n_leaked}/{n_items} items contain HAMIB markers "
              f"({n_leaked / n_items * 100:.2f}%)")
        for aid in leaked_ids:
            print(f"  - {aid}")


if __name__ == "__main__":
    main()

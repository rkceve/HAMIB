"""PersonaMem-v2 loader — third-party layer for compress-then-answer eval.

Wraps the ungated HuggingFace dataset `bowen-upenn/PersonaMem-v2`
(CC-BY-4.0) so a compress-then-answer run can reuse its persona chat
histories and multiple-choice questions without re-deriving any structure.

Probe cache layout (a MINIMUM subset, not the full dataset):
    results/personamem_probe/
        benchmark.csv                    # benchmark/text/benchmark.csv (5000 rows)
        column_descriptions.md
        data/chat_history_128k/*.json    # a few linked history files

Question rows live in benchmark.csv (one MC question per row). Each row
carries a `chat_history_128k_link` that names a JSON file of role/content
turns; many rows share the same history (200 unique histories over 5000
rows in the full CSV).

A history JSON is {"metadata": {...}, "chat_history": [{role, content}, ...]}.
`total_tokens_in_chat_history_128k` in the CSV equals metadata.final_token_count.

`incorrect_answers` is stored as a JSON-encoded list of three strings.

All file I/O is utf-8. Console prints are ASCII-only (cp932-safe): every
line is passed through `_ascii()` before printing, so a Windows cp932
terminal never raises on the Unicode math symbols that appear in histories.

CLI:
    python -m benchmark.bineval.personamem_loader --probe

This module is import-safe: no I/O or model calls happen at import time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import tiktoken

# ---------------------------------------------------------------------------
# Probe cache locations (resolved relative to this file so the CLI needs no
# arguments). These point at the MINIMUM subset downloaded for the probe.
# ---------------------------------------------------------------------------
_PROBE_DIR = Path(__file__).resolve().parent / "results" / "personamem_probe"
_PROBE_CSV = _PROBE_DIR / "benchmark.csv"
_HISTORY_ROOT = _PROBE_DIR / "data" / "chat_history_128k"

_ENC_NAME = "cl100k_base"

# Fields we surface from each CSV row. Everything present in the row is kept
# in the returned dict; this list documents the load-bearing subset.
_QUESTION_FIELDS = (
    "persona_id",
    "user_query",
    "correct_answer",
    "incorrect_answers",
    "chat_history_128k_link",
    "chat_history_32k_link",
    "total_tokens_in_chat_history_128k",
    "topic_query",
    "preference",
    "conversation_scenario",
)


def _ascii(text: str) -> str:
    """cp932-safe console rendering: replace any non-ascii char with '?'."""
    return str(text).encode("ascii", "replace").decode("ascii")


def _token_counter():
    """tiktoken cl100k_base counter (matches build_cd_offline convention)."""
    enc = tiktoken.get_encoding(_ENC_NAME)
    return lambda text: len(enc.encode(text))


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------
def load_questions(csv_path: str | Path) -> list[dict]:
    """Load benchmark.csv into a list of question dicts (one per row).

    Uses the stdlib csv module (no pandas dependency) with utf-8. Numeric
    token columns are coerced to int where possible; `incorrect_answers`
    is decoded from its JSON string into a list of strings. A synthetic
    string `qid` (row index) is attached for deterministic option shuffling.
    """
    path = Path(csv_path)
    rows: list[dict] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, raw in enumerate(reader):
            row = dict(raw)
            row["qid"] = str(idx)
            row["incorrect_answers"] = _parse_incorrect(row.get("incorrect_answers"))
            for col in (
                "total_tokens_in_chat_history_128k",
                "total_tokens_in_chat_history_32k",
                "persona_id",
            ):
                if col in row:
                    row[col] = _to_int(row[col])
            rows.append(row)
    return rows


def _parse_incorrect(value) -> list[str]:
    """Decode the `incorrect_answers` JSON-list string into list[str].

    Returns whatever list JSON yields (expected length 3). On failure
    returns an empty list so callers can detect malformed rows.
    """
    if isinstance(value, list):
        return [str(x) for x in value]
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    return []


def _to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
def load_history(json_path: str | Path) -> list[dict]:
    """Load a chat-history JSON and return its list of {role, content} turns.

    Accepts either the PersonaMem-v2 wrapped form
    ({"metadata": ..., "chat_history": [...]}) or a bare list of turns.
    """
    path = Path(json_path)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        turns = data.get("chat_history", [])
    elif isinstance(data, list):
        turns = data
    else:
        turns = []
    return [t for t in turns if isinstance(t, dict) and "role" in t]


def history_text(turns: list[dict]) -> str:
    """Flatten turns into a single 'ROLE: content' transcript string."""
    lines: list[str] = []
    for t in turns:
        role = str(t.get("role", "")).strip()
        content = str(t.get("content", "")).strip()
        lines.append("%s: %s" % (role.upper(), content))
    return "\n".join(lines)


def resolve_history_path(link: str, probe_dir: str | Path = _PROBE_DIR) -> Path:
    """Map a `chat_history_128k_link` (a repo-relative path) to the probe cache.

    Links look like 'data/chat_history_128k/chat_history_..._personaN.json';
    the probe cache mirrors that layout under probe_dir.
    """
    base = Path(probe_dir)
    rel = str(link).lstrip("/")
    return base / rel


# ---------------------------------------------------------------------------
# Multiple-choice prompt assembly
# ---------------------------------------------------------------------------
_LETTERS = ("A", "B", "C", "D")


def _shuffled_options(question_row: dict) -> tuple[list[str], str]:
    """Return (options_in_display_order, correct_letter).

    The correct answer is placed among the (up to) three incorrect answers,
    then the four are ordered deterministically by a per-question seed derived
    from the qid hash, so shuffling is reproducible without global RNG state.
    """
    correct = str(question_row.get("correct_answer", ""))
    incorrect = [str(x) for x in question_row.get("incorrect_answers", [])]
    options = [correct, *incorrect]

    qid = str(question_row.get("qid", question_row.get("user_query", "")))
    seed = int(hashlib.sha256(qid.encode("utf-8")).hexdigest(), 16)

    # Deterministic order: sort options by a per-option hash mixed with the
    # question seed. Stable and independent of Python's hash randomization.
    def sort_key(opt: str):
        h = hashlib.sha256(("%d:%s" % (seed, opt)).encode("utf-8")).hexdigest()
        return int(h, 16)

    ordered = sorted(options, key=sort_key)
    correct_letter = _LETTERS[ordered.index(correct)]
    return ordered, correct_letter


def mc_prompt(question_row: dict, compressed_history: str) -> str:
    """Assemble a 4-option MC prompt from a question row + compressed history.

    Options are shuffled deterministically (seeded by the qid hash). The
    correct letter is recoverable via `correct_letter(question_row)`.
    """
    options, _correct_letter = _shuffled_options(question_row)
    query = str(question_row.get("user_query", "")).strip()

    parts: list[str] = []
    parts.append(
        "You are answering on behalf of a chatbot that has been talking with "
        "one user. Below is the (compressed) chat history with that user, "
        "followed by the user's new query and four candidate replies. Choose "
        "the single reply that is most consistent with the user's persona and "
        "stated preferences.\n"
    )
    parts.append("=== CHAT HISTORY (compressed) ===")
    parts.append(compressed_history.strip())
    parts.append("=== END CHAT HISTORY ===\n")
    parts.append("USER QUERY:")
    parts.append(query + "\n")
    parts.append("OPTIONS:")
    for letter, opt in zip(_LETTERS, options):
        parts.append("%s. %s" % (letter, str(opt).strip()))
    parts.append(
        "\nRespond with the letter (A, B, C, or D) of the best reply, and "
        "nothing else."
    )
    return "\n".join(parts)


def correct_letter(question_row: dict) -> str:
    """Return the letter (A-D) of the correct option for this question."""
    _options, letter = _shuffled_options(question_row)
    return letter


def extract_answer_letter(response: str) -> str | None:
    """Extract a single answer letter (A-D) from a model response, or None.

    Accepts forms like 'A', 'A.', '(B)', 'Answer: C', 'The answer is D'.
    Returns the first standalone A-D token found (case-insensitive), else None.
    """
    if not response:
        return None
    text = response.strip()
    # Fast path: leading letter.
    head = text[:3].upper()
    for letter in _LETTERS:
        if head.startswith(letter) and (
            len(head) == 1 or not head[1].isalpha()
        ):
            return letter
    # Scan for a standalone A-D token.
    upper = text.upper()
    for i, ch in enumerate(upper):
        if ch in _LETTERS:
            prev_ok = i == 0 or not upper[i - 1].isalpha()
            nxt_ok = i + 1 >= len(upper) or not upper[i + 1].isalpha()
            if prev_ok and nxt_ok:
                return ch
    return None


# ---------------------------------------------------------------------------
# Compression helper used by the smoke test (truncation baseline)
# ---------------------------------------------------------------------------
def truncate_history_by_ratio(
    turns: list[dict], keep_ratio: float, token_counter
) -> list[dict]:
    """Keep the LAST `keep_ratio` fraction of turns by cumulative token count.

    Walks turns from the end, accumulating token counts of 'ROLE: content'
    lines, and keeps turns until the budget (keep_ratio * total_tokens) is
    exceeded. Returns the kept turns in original order. This is a simple
    recency-truncation compression baseline (no CD, no model).
    """
    if not turns:
        return []
    per_turn = [token_counter("%s: %s" % (t.get("role", ""), t.get("content", ""))) for t in turns]
    total = sum(per_turn)
    budget = int(total * keep_ratio)
    kept_rev: list[dict] = []
    used = 0
    for t, cost in zip(reversed(turns), reversed(per_turn)):
        if used + cost > budget and kept_rev:
            break
        kept_rev.append(t)
        used += cost
    return list(reversed(kept_rev))


# ---------------------------------------------------------------------------
# CLI probe
# ---------------------------------------------------------------------------
def _print(line: str = "") -> None:
    print(_ascii(line))


def run_probe(csv_path: Path, history_root: Path) -> None:
    """Print row count, per-persona counts, token stats, and one-persona detail."""
    if not csv_path.exists():
        _print("ERROR: probe CSV not found: %s" % csv_path.as_posix())
        return

    counter = _token_counter()
    rows = load_questions(csv_path)

    _print("=== PersonaMem-v2 probe ===")
    _print("csv: %s" % csv_path.as_posix())
    _print("row count: %d" % len(rows))

    # per-persona question counts (top 5)
    per_persona: dict = {}
    for r in rows:
        pid = r.get("persona_id")
        per_persona[pid] = per_persona.get(pid, 0) + 1
    top5 = sorted(per_persona.items(), key=lambda kv: (-kv[1], str(kv[0])))[:5]
    _print("distinct personas: %d" % len(per_persona))
    _print("per-persona question counts (top 5):")
    for pid, n in top5:
        _print("  persona %s: %d questions" % (pid, n))

    # token stats over total_tokens_in_chat_history_128k
    toks = [
        r["total_tokens_in_chat_history_128k"]
        for r in rows
        if isinstance(r.get("total_tokens_in_chat_history_128k"), int)
    ]
    if toks:
        _print("total_tokens_in_chat_history_128k: min=%d median=%d max=%d" % (
            min(toks), int(statistics.median(toks)), max(toks)
        ))
    else:
        _print("total_tokens_in_chat_history_128k: (no integer values found)")

    # one persona detail: pick the first persona whose linked history file is
    # present in the probe cache.
    detail_row = None
    for r in rows:
        link = r.get("chat_history_128k_link")
        if link and resolve_history_path(link, csv_path.parent).exists():
            detail_row = r
            break

    _print("")
    if detail_row is None:
        _print("NOTE: no linked history file present in probe cache to detail.")
        return

    link = detail_row["chat_history_128k_link"]
    hpath = resolve_history_path(link, csv_path.parent)
    turns = load_history(hpath)
    htext = history_text(turns)
    htokens = counter(htext)
    q_for_persona = per_persona.get(detail_row.get("persona_id"), 0)

    _print("--- one persona detail ---")
    _print("persona_id: %s" % detail_row.get("persona_id"))
    _print("history file: %s" % hpath.name)
    _print("history turns: %d" % len(turns))
    _print("history token count (cl100k, ROLE: content transcript): %d" % htokens)
    _print("csv total_tokens_in_chat_history_128k for this row: %s" % (
        detail_row.get("total_tokens_in_chat_history_128k")
    ))
    _print("questions available for this persona: %d" % q_for_persona)


def run_smoke(csv_path: Path, out_path: Path) -> None:
    """Build a 6x truncation-compressed history for one persona and emit one
    complete MC prompt to `out_path`. Prints a short summary (cp932-safe)."""
    counter = _token_counter()
    rows = load_questions(csv_path)

    # first row whose linked history is present in the probe cache
    row = None
    for r in rows:
        link = r.get("chat_history_128k_link")
        if link and resolve_history_path(link, csv_path.parent).exists():
            row = r
            break
    if row is None:
        _print("SMOKE: no row with a present history file; cannot emit prompt.")
        return

    link = row["chat_history_128k_link"]
    hpath = resolve_history_path(link, csv_path.parent)
    turns = load_history(hpath)

    full_text = history_text(turns)
    full_tokens = counter(full_text)

    kept = truncate_history_by_ratio(turns, 1.0 / 6.0, counter)
    compressed = history_text(kept)
    compressed_tokens = counter(compressed)

    prompt = mc_prompt(row, compressed)
    letter = correct_letter(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(prompt)
        f.write("\n\n--- (probe annotation, not part of the model prompt) ---\n")
        f.write("qid: %s\n" % row.get("qid"))
        f.write("persona_id: %s\n" % row.get("persona_id"))
        f.write("correct_letter: %s\n" % letter)
        f.write("history_file: %s\n" % hpath.name)
        f.write("full_history_tokens(cl100k): %d\n" % full_tokens)
        f.write("compressed_history_tokens(cl100k, ~6x truncation): %d\n" % compressed_tokens)
        f.write("kept_turns: %d of %d\n" % (len(kept), len(turns)))

    _print("=== smoke test: one MC prompt written ===")
    _print("persona_id: %s  qid: %s" % (row.get("persona_id"), row.get("qid")))
    _print("history file: %s (%d turns)" % (hpath.name, len(turns)))
    _print("full history tokens (cl100k): %d" % full_tokens)
    _print("compressed (~6x truncation) tokens: %d  kept turns: %d/%d" % (
        compressed_tokens, len(kept), len(turns)
    ))
    _print("options: 4  correct_letter: %s" % letter)
    _print("wrote: %s" % out_path.as_posix())


def main() -> None:
    ap = argparse.ArgumentParser(
        description="PersonaMem-v2 probe loader / smoke test (probe cache only)."
    )
    ap.add_argument("--csv", default=str(_PROBE_CSV), help="path to benchmark.csv")
    ap.add_argument(
        "--history-root",
        default=str(_HISTORY_ROOT),
        help="root of the chat_history_128k probe cache",
    )
    ap.add_argument("--probe", action="store_true", help="print probe stats")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="emit one 6x-compressed MC prompt to results/personamem_probe/sample_prompt.txt",
    )
    ap.add_argument(
        "--out",
        default=str(_PROBE_DIR / "sample_prompt.txt"),
        help="output path for the smoke-test sample prompt",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    history_root = Path(args.history_root)

    if args.probe or not (args.probe or args.smoke):
        run_probe(csv_path, history_root)
    if args.smoke:
        _print("")
        run_smoke(csv_path, Path(args.out))


if __name__ == "__main__":
    main()

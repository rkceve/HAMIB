"""score_spec_run.py -- score a downloaded spec-run directory into scores.csv.

LOCAL ONLY: no GPU, no model, no network.  Every answers file produced by
``experiments/modal_spec_run.py`` phase 3 is scored with
``benchmark.bineval.score_binary.score_condition``.

WHY THIS FILE EXISTS (F4)
-------------------------
The documented per-cell command was::

    python -m benchmark.bineval.score_binary --answers <cell>.json ... --max-words 32

and ``score_binary``'s ``--subset`` defaults to ``all``.  The reader, however,
answers what ``run_reader.load_questions`` selects, whose default subset is
``generated``: 173 of the 189 non-excluded questions.  Scoring the reader's
answers with ``--subset all`` therefore puts 189 in the denominator and counts
the 16 legacy questions as unanswered failures -- every arm reported ~8 points
low, and the ERROR IS UNIFORM, so it does not even show up as an outlier.
Here ``--subset generated`` is the default and cannot be forgotten.

The exploratory cell answers only the first 20 questions
(``EXPLORATORY_CELL["max_questions"]``).  Scoring it against all 173 would
report a pass rate of at most 20/173 for a cell that answered everything it was
asked.  ``--only-answered`` (ON by default) restricts the question list to the
qids actually present in the answers file and records the count as
``n_questions``, so the exploratory cell is scored on its own 20.

CLI::

    python experiments/score_spec_run.py --run-dir <dir> \\
        --questions benchmark/bineval/questions_restaurant.json
    # or, equivalently:
    python experiments/modal_spec_run.py --score <dir>

Output: ``<run-dir>/scores.csv`` with one row per cell
(cell, arm, w, inject, prefill, n_questions, pass, total, pass_rate,
multi_gold_answers) plus ``<run-dir>/scores/<cell>.json`` with the full
per-question report.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow `python experiments/score_spec_run.py`
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_QUESTIONS = "benchmark/bineval/questions_restaurant.json"
# The reader generates at most 48 tokens but tier-1 is an unbounded substring
# match, so an answer that echoes the context would pass everything (score_binary
# truncate_words). 32 words is the value every published bineval number uses.
DEFAULT_MAX_WORDS = 32
DEFAULT_SUBSET = "generated"

CSV_COLUMNS = (
    "cell",
    "arm",
    "w",
    "inject",
    "prefill",
    "n_questions",
    "pass",
    "total",
    "pass_rate",
    "multi_gold_answers",
)


def answer_files(answers_dir: str | Path) -> list[Path]:
    """Every ``<cell>.json`` under ``answers/``, excluding the meta sidecars."""
    root = Path(answers_dir)
    if not root.is_dir():
        raise FileNotFoundError("no answers directory at %s" % root)
    return sorted(
        p for p in root.glob("*.json") if not p.name.endswith(".meta.json")
    )


def cell_fields(answers_path: str | Path) -> dict:
    """(arm, w, inject, prefill) for a cell, from its ``.meta.json`` when there.

    The sidecar is the authority (it is what the run actually did); the cell
    name is the fallback for a hand-assembled directory.
    """
    path = Path(answers_path)
    meta_path = path.with_suffix(".meta.json")
    out: dict[str, Any] = {
        "cell": path.stem, "arm": None, "w": None, "inject": None, "prefill": None,
    }
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        out["arm"] = meta.get("arm")
        out["w"] = meta.get("w")
        out["inject"] = meta.get("inject")
        out["prefill"] = meta.get("prefill_scale")
        out["cell"] = meta.get("cell", path.stem)
    if out["arm"] is None:
        # "<arm>__w<w>[__<inject>][__pf<scale>]"
        parts = path.stem.split("__")
        out["arm"] = parts[0]
        for part in parts[1:]:
            if part.startswith("w"):
                out["w"] = _maybe_float(part[1:])
            elif part.startswith("pf"):
                out["prefill"] = _maybe_float(part[2:])
            else:
                out["inject"] = part.replace("-", "+")
    return out


def _maybe_float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def score_answers_file(
    answers_path: str | Path,
    questions: list[dict],
    *,
    subset: str = DEFAULT_SUBSET,
    max_words: int | None = DEFAULT_MAX_WORDS,
    only_answered: bool = True,
) -> dict:
    """Score ONE answers file. Returns score_binary's report plus the cell fields."""
    from benchmark.bineval.score_binary import judge_none, score_condition, select_questions

    path = Path(answers_path)
    answers = {
        str(k): str(v)
        for k, v in json.loads(path.read_text(encoding="utf-8")).items()
    }
    scored = select_questions(questions, subset, False)
    if only_answered:
        # The exploratory cell answers 20 of 173; scoring it against all 173
        # would report a pass rate for questions it was never asked.
        scored = [q for q in scored if q["qid"] in answers]
    report = score_condition(scored, {}, answers, judge_none, max_words)
    report["cell_fields"] = cell_fields(path)
    report["subset"] = subset
    report["only_answered"] = only_answered
    return report


def report_row(report: dict) -> dict:
    """One csv row from a score_condition report."""
    agg = report["aggregate"]
    fields = report["cell_fields"]
    return {
        "cell": fields["cell"],
        "arm": fields["arm"],
        "w": fields["w"],
        "inject": fields["inject"],
        "prefill": fields["prefill"],
        "n_questions": agg["total"],
        "pass": agg["pass"],
        "total": agg["total"],
        "pass_rate": agg["pass_rate_tier1"],
        "multi_gold_answers": agg["multi_gold_answers"],
    }


def write_scores_csv(rows: list[dict], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in sorted(rows, key=lambda r: str(r["cell"])):
            writer.writerow(row)
    return path


def score_run_dir(
    run_dir: str | Path,
    *,
    questions: str | Path = REPO_ROOT / DEFAULT_QUESTIONS,
    subset: str = DEFAULT_SUBSET,
    max_words: int | None = DEFAULT_MAX_WORDS,
    only_answered: bool = True,
) -> dict:
    """Score every cell of a downloaded run directory.

    ``run_dir`` is the directory that holds ``answers/`` (i.e. the ``<run_id>``
    directory of a ``--download``), or ``answers/`` itself.
    """
    from benchmark.bineval.score_binary import load_questions

    root = Path(run_dir)
    answers_dir = root if root.name == "answers" else root / "answers"
    if root.name == "answers":
        root = root.parent
    all_questions = load_questions(Path(questions))

    rows: list[dict] = []
    for path in answer_files(answers_dir):
        report = score_answers_file(
            path, all_questions, subset=subset, max_words=max_words,
            only_answered=only_answered,
        )
        (root / "scores").mkdir(parents=True, exist_ok=True)
        (root / "scores" / path.name).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        rows.append(report_row(report))

    csv_path = write_scores_csv(rows, root / "scores.csv")
    return {"rows": rows, "csv": str(csv_path), "run_dir": str(root)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="score a spec-run directory (local)")
    p.add_argument("--run-dir", required=True,
                   help="the <run_id> directory holding answers/")
    p.add_argument("--questions", default=str(REPO_ROOT / DEFAULT_QUESTIONS))
    p.add_argument("--subset", choices=("all", "legacy", "generated"),
                   default=DEFAULT_SUBSET,
                   help="MUST stay 'generated' to match what the reader answered")
    p.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS)
    p.add_argument(
        "--all-questions", action="store_true",
        help="score every subset question even when the cell did not answer it "
             "(turns OFF --only-answered; the exploratory 20-question cell then "
             "reports a pass rate over 173)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    res = score_run_dir(
        args.run_dir, questions=args.questions, subset=args.subset,
        max_words=args.max_words, only_answered=not args.all_questions,
    )
    print("scored %d cells -> %s" % (len(res["rows"]), res["csv"]))
    for row in res["rows"]:
        print(
            "  %-40s %3d/%-3d  %.4f"
            % (row["cell"], row["pass"], row["total"], row["pass_rate"])
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

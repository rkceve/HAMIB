"""score_cells: paired scoring of a proposed cell against arm A on the cell's subset (V7 / H29)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.mcbuild_bench import score_cells

QS = [
    {"qid": "f001", "question": "Port?", "gold_short": "25565", "tier1_aliases": [], "kind": "value"},
    {"qid": "f002", "question": "Name?", "gold_short": "Paper", "tier1_aliases": ["PaperMC"], "kind": "value"},
    {"qid": "f003", "question": "Version?", "gold_short": "1.21.8", "tier1_aliases": [], "kind": "value"},
    {"qid": "a001", "question": "GPU?", "gold_short": "unknown", "tier1_aliases": ["not mentioned"],
     "kind": "absent"},
]


def _pq(L: int, n: int = 4, energy: float = 10.0) -> dict:
    return {"prompt_tokens": L, "attn_flops_prefill": 16 * 2 * 24 * L * L * 256,
            "attn_flops_decode": 16 * 2 * 24 * (L + n) * 256 * n, "wall_ms_total": float(L),
            "energy_joules": energy}


def _write(d: Path, answers: dict, L: int, qids: list[str], subset: dict | None = None) -> None:
    d.mkdir(parents=True)
    (d / "answers.json").write_text(json.dumps(answers), encoding="utf-8")
    meta = {"context_tokens": L, "per_question": {q: _pq(L) for q in qids}}
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if subset is not None:
        (d / "questions_subset.json").write_text(json.dumps(subset), encoding="utf-8")


@pytest.fixture()
def main_dir(tmp_path: Path) -> Path:
    (tmp_path / "questions.json").write_text(json.dumps(QS), encoding="utf-8")
    main = tmp_path / "main"
    # arm A answered all 4; f003 is A's only miss
    _write(main / "A_full", {"f001": "25565", "f002": "PaperMC", "f003": "1.20", "a001": "unknown"},
           180000, [q["qid"] for q in QS])
    # the cell dropped f002 (inside the window); proposed misses f001, gets f003
    _write(main / "proposed_W8000_w0.3", {"f001": "25566", "f003": "1.21.8", "a001": "unknown"}, 8000,
           ["f001", "f003", "a001"],
           subset={"qids": ["f001", "f003", "a001"], "dropped": ["f002"], "first_recent_rt": 30})
    return main


def test_cell_is_scored_on_its_subset_with_paired_stats(main_dir: Path) -> None:
    out = main_dir.parent / "scores.json"
    score_cells.main(["--main", str(main_dir), "--questions", str(main_dir.parent / "questions.json"),
                      "--out", str(out)])
    payload = json.loads(out.read_text(encoding="utf-8"))
    (cell,) = payload["cells"]
    assert cell["W"] == 8000 and cell["w"] == 0.3 and cell["n_questions"] == 3
    assert cell["dropped_in_window"] == ["f002"]
    assert cell["proposed"]["pass"] == 2 and cell["A"]["pass"] == 2  # f002 not counted for A either
    paired = cell["paired"]
    assert (paired["A_only"], paired["proposed_only"], paired["both_pass"], paired["both_fail"]) == (1, 1, 1, 0)
    assert paired["mcnemar_p_proposed_gt_A"] == pytest.approx(0.75)
    lo, hi = paired["bootstrap"]["diff_ci95"]
    assert lo <= 0.0 <= hi
    ratio = cell["compute_ratio_proposed_over_A"]
    assert ratio["prompt_tokens"] == pytest.approx(8000 / 180000)
    assert 0 < ratio["attn_flops_total"] < 0.01  # quadratic in L
    assert ratio["energy_joules"] == pytest.approx(1.0)


def test_missing_answer_in_arm_A_is_refused(main_dir: Path) -> None:
    a_path = main_dir / "A_full" / "answers.json"
    a = json.loads(a_path.read_text(encoding="utf-8"))
    del a["f003"]
    a_path.write_text(json.dumps(a), encoding="utf-8")
    with pytest.raises(ValueError, match="arm A has no answer"):
        score_cells.score_cell(main_dir / "proposed_W8000_w0.3", main_dir / "A_full", QS)


def test_mcnemar_matches_binomial() -> None:
    assert score_cells.mcnemar_one_sided(0, 0) == 1.0
    assert score_cells.mcnemar_one_sided(0, 5) == pytest.approx(0.5 ** 5)


def test_truncation_control_is_compared_when_present(main_dir: Path) -> None:
    """A ``truncB_W<W>`` directory is scored on the same subset and reported next to arm A;
    without it the result has no B keys (so earlier results re-score unchanged)."""
    out = main_dir.parent / "s.json"
    qfile = str(main_dir.parent / "questions.json")
    score_cells.main(["--main", str(main_dir), "--questions", qfile, "--out", str(out)])
    (cell,) = json.loads(out.read_text(encoding="utf-8"))["cells"]
    assert "B" not in cell and "paired_vs_B" not in cell

    # truncation gets the recent fact (f003) and the absent one, misses the old ones
    _write(main_dir / "truncB_W8000", {"f001": "unknown", "f002": "Paper", "f003": "1.21.8", "a001": "unknown"},
           8000, [q["qid"] for q in QS])
    md = main_dir.parent / "s.md"
    score_cells.main(["--main", str(main_dir), "--questions", qfile, "--out", str(out), "--md", str(md)])
    (cell,) = json.loads(out.read_text(encoding="utf-8"))["cells"]
    assert cell["b_dir"] == "truncB_W8000"
    assert cell["B"]["pass"] == 2            # f003 + a001 on the 3-question subset (f002 is excluded)
    pb = cell["paired_vs_B"]
    assert (pb["both_pass"], pb["both_fail"], pb["other_only"], pb["proposed_only"]) == (2, 1, 0, 0)
    assert pb["mcnemar_p_proposed_gt_other"] == 1.0
    assert "truncation pass" in md.read_text(encoding="utf-8")
    # arm A comparison is untouched by the extra column
    assert cell["A"]["pass"] == 2 and cell["paired"]["A_only"] == 1

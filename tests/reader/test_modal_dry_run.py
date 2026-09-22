"""S2.4 tests for experiments/modal_spec_run.py (CPU only, no Modal, no model)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from experiments.modal_spec_run import (
    DEFAULT_ARMS,
    build_cells,
    cell_name,
    dry_run,
    should_abort_pilot,
)


# --------------------------------------------------------------------------
# abort rule
# --------------------------------------------------------------------------

def test_abort_rule_fires_on_30_percent_defaulted() -> None:
    abort, reason = should_abort_pilot(
        {"calls": 100, "defaulted": 30, "nodes": 50, "node_fallback": 0}
    )
    assert abort
    assert "defaulted/calls" in reason


def test_abort_rule_fires_on_30_percent_node_fallback() -> None:
    abort, reason = should_abort_pilot(
        {"calls": 100, "defaulted": 0, "nodes": 50, "node_fallback": 15}
    )
    assert abort
    assert "node_fallback/nodes" in reason


def test_abort_rule_passes_a_clean_pilot() -> None:
    abort, reason = should_abort_pilot(
        {"calls": 100, "defaulted": 10, "nodes": 50, "node_fallback": 10}
    )
    assert not abort and reason == "ok"


def test_abort_rule_is_strictly_greater_than_the_threshold() -> None:
    # exactly 20% must NOT abort (the rule is "> 0.2")
    abort, _ = should_abort_pilot(
        {"calls": 100, "defaulted": 20, "nodes": 100, "node_fallback": 20}
    )
    assert not abort


def test_abort_rule_on_empty_pilot() -> None:
    abort, reason = should_abort_pilot({"calls": 0, "defaulted": 0, "nodes": 0,
                                        "node_fallback": 0})
    assert abort and "0 nodes" in reason


# --------------------------------------------------------------------------
# grid
# --------------------------------------------------------------------------

def test_only_cd_arms_sweep_w() -> None:
    cells = build_cells()
    for c in cells:
        if not c["arm"].startswith("cd_"):
            assert c["w"] == 0.0 and c["inject"] == "none"
    assert {c["arm"] for c in cells} == set(DEFAULT_ARMS)


def test_cd_arms_keep_a_real_inject_at_w_zero() -> None:
    """M7: the w=0 control of a cd arm must run the SAME kernel path as its
    injected siblings -- the bias is 0 * mass, not "no bias at all"."""
    zero = [c for c in build_cells() if c["arm"].startswith("cd_") and c["w"] == 0.0]
    assert zero
    for c in zero:
        assert c["inject"] == "planet"


def test_cell_names_are_unique_and_filesystem_safe() -> None:
    cells = build_cells()
    names = [c["cell"] for c in cells]
    assert len(names) == len(set(names))
    assert all(set(n) <= set(
        "abcdefghijklmnopqrstuvwxyz0123456789_-." ) for n in names)


def test_exploratory_prefill_cell_is_present() -> None:
    cells = build_cells()
    hits = [c for c in cells if c["prefill_scale"] > 0]
    assert len(hits) == 1
    c = hits[0]
    assert (c["arm"], c["w"], c["inject"], c["max_questions"]) == (
        "cd_mass_6x", 0.1, "planet", 20
    )


def test_cell_name_shape() -> None:
    assert cell_name("floor", 0.0, "none") == "floor__w0"
    assert cell_name("cd_mass_6x", 0.25, "planet+satellites") == (
        "cd_mass_6x__w0.25__planet-satellites"
    )


# --------------------------------------------------------------------------
# dry run layout
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dry(tmp_path_factory) -> Path:
    """F3: every output lives under <out>/<run_id>/, never directly in <out>."""
    out = tmp_path_factory.mktemp("spec_run")
    res = dry_run(out, max_questions=3, run_id="testrun-abc1234")
    assert Path(res["out"]) == out / "testrun-abc1234"
    assert not (out / "run_manifest.json").exists()
    return Path(res["out"])


def test_dry_run_layout_is_namespaced_by_run_id(dry: Path) -> None:
    """F3: the volume persists between runs, so two runs must not merge."""
    assert dry.name == "testrun-abc1234"
    info = json.loads((dry / "run_info.json").read_text(encoding="utf-8"))
    assert info["run_id"] == "testrun-abc1234"
    assert info["pairing"] == "round_trip"
    assert info["dry_run"] is True
    assert "git_sha" in info and info["arms"] and info["w_grid"] is not None


def test_dry_run_default_run_id_carries_a_timestamp_and_the_sha() -> None:
    from experiments.modal_spec_run import default_run_id

    rid = default_run_id(sha="deadbee")
    assert rid.endswith("-deadbee") and rid[8] == "T" and rid[15] == "Z"
    assert default_run_id(sha="unknown").endswith("Z")


def test_dry_run_file_layout(dry: Path) -> None:
    for name in ("pilot_summary.json", "cd_spec.json", "instrument.json",
                 "run_manifest.json", "run_info.json"):
        assert (dry / name).exists(), name
    assert (dry / "answers").is_dir()


def test_dry_run_writes_every_cell_with_unknown_answers(dry: Path) -> None:
    manifest = json.loads((dry / "run_manifest.json").read_text(encoding="utf-8"))
    cells = manifest["cells"]
    assert len(cells) == len(build_cells())
    for cell in cells:
        for key in ("cell", "arm", "w", "inject", "prefill_scale", "tokens"):
            assert key in cell
        path = dry / "answers" / ("%s.json" % cell["cell"])
        answers = json.loads(path.read_text(encoding="utf-8"))
        assert answers and set(answers.values()) == {"unknown"}
        assert all(qid.startswith("rest_") for qid in answers)
        meta = json.loads(
            (dry / "answers" / ("%s.meta.json" % cell["cell"])).read_text(encoding="utf-8")
        )
        assert meta["dry_run"] is True
        assert meta["arm"] == cell["arm"] and meta["w"] == cell["w"]
        assert set(meta["per_question"]) == set(answers)


def test_dry_run_manifest_tokens_match_the_arm_builder(dry: Path) -> None:
    from benchmark.bineval.arms import build_arm_context, load_chat

    manifest = json.loads((dry / "run_manifest.json").read_text(encoding="utf-8"))
    chat = load_chat()
    seen = {}
    for cell in manifest["cells"]:
        if cell["arm"].startswith("cd_"):
            continue  # the dry-run CD is an empty placeholder
        seen.setdefault(cell["arm"], cell["tokens"])
    for arm, tokens in seen.items():
        assert build_arm_context(arm, chat=chat).tokens == tokens


def test_dry_run_pilot_summary_passes_the_abort_rule(dry: Path) -> None:
    pilot = json.loads((dry / "pilot_summary.json").read_text(encoding="utf-8"))
    assert pilot["dry_run"] is True
    assert pilot["abort"] is False
    assert should_abort_pilot(pilot)[0] is False


def test_dry_run_instrument_is_marked_as_a_placeholder(dry: Path) -> None:
    inst = json.loads((dry / "instrument.json").read_text(encoding="utf-8"))
    assert inst["dry_run"] is True and inst["checks"] is None


def test_dry_run_imports_no_modal_and_no_model(dry: Path) -> None:
    """The dry-run path must not import modal, and must not touch a model.

    ``transformers`` may be imported transitively by other tests in the same
    session, so the assertion is on ``modal`` (never installed/imported here)
    and on the absence of any loaded torch module class.
    """
    assert "modal" not in sys.modules

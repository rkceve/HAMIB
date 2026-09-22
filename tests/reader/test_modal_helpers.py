"""Review fixes B1 / B3 / B4 / M3 / M4 / M5 / M6 for experiments/modal_spec_run.py.

CPU only: no Modal, no model, no network.  The one test that runs the driver
uses ``--extractor spec --judge fake`` on a five-turn synthetic chat.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import experiments.modal_spec_run as msr
from experiments.modal_spec_run import (
    DEFAULT_ARMS,
    DEFAULT_W_GRID,
    GPU_MEM_GB,
    IMAGE_IGNORE,
    LONG_ARM_MIN_GPU_GB,
    LONG_ARMS,
    _arm_tokens,
    build_cells,
    flatten_pilot_summary,
    max_planet_mass,
    parse_w_grid,
    resolve_arms,
    should_abort_pilot,
    vllm_command,
)

# --------------------------------------------------------------------------
# B1: the pilot summary the driver really writes
# --------------------------------------------------------------------------

CHAT = {
    "sessions": [
        {
            "turns": [
                {
                    "role": "user",
                    "content": (
                        "The restaurant plan is the topic. The budget matters "
                        "most. Rent is 500k yen."
                    ),
                },
                {
                    "role": "assistant",
                    "content": "Noted. The loan is 30M yen and the term is 7 years.",
                },
                {
                    "role": "user",
                    "content": "Marketing is a new pillar. Delivery costs 2k yen.",
                },
            ]
        }
    ]
}


@pytest.fixture(scope="module")
def pilot_summary(tmp_path_factory) -> dict:
    """Run the REAL driver and return the summary block it wrote."""
    from benchmark.bineval.build_cd_offline import main

    tmp = tmp_path_factory.mktemp("pilot")
    chat_path = tmp / "chat.json"
    chat_path.write_text(json.dumps(CHAT, ensure_ascii=False), encoding="utf-8")
    out = tmp / "cd_spec_pilot.json"
    argv = [
        "build_cd_offline", "--chat", str(chat_path), "--out", str(out),
        "--extractor", "spec", "--judge", "fake", "--max-sessions", "1",
    ]
    saved = sys.argv
    sys.argv = argv
    try:
        main()
    finally:
        sys.argv = saved
    return json.loads(out.read_text(encoding="utf-8"))["summary"]


def test_the_driver_writes_the_nested_shape_not_the_flat_one(pilot_summary) -> None:
    """This is the defect: the abort rule read keys the driver never writes."""
    assert "harness_calls" in pilot_summary and "harness_quality" in pilot_summary
    for flat_key in ("calls", "defaulted", "nodes", "node_fallback"):
        assert flat_key not in pilot_summary


def test_flattening_a_real_summary_feeds_the_abort_rule(pilot_summary) -> None:
    flat = flatten_pilot_summary(pilot_summary)
    assert flat["calls"] > 0
    assert flat["nodes"] > 0
    assert flat["defaulted"] == 0
    assert flat["node_fallback"] == 0
    abort, reason = should_abort_pilot(flat)
    assert not abort and reason == "ok"


def test_a_bad_real_summary_aborts(pilot_summary) -> None:
    """Same real shape, but with the quality counters of a broken judge."""
    broken = dict(pilot_summary)
    calls = sum(pilot_summary["harness_calls"].values())
    broken["harness_quality"] = dict(
        pilot_summary["harness_quality"], defaulted=calls, node_fallback=0
    )
    abort, reason = should_abort_pilot(flatten_pilot_summary(broken))
    assert abort and "defaulted/calls" in reason

    broken2 = dict(pilot_summary)
    broken2["harness_quality"] = dict(
        pilot_summary["harness_quality"], node_fallback=pilot_summary["total"]
    )
    abort2, reason2 = should_abort_pilot(flatten_pilot_summary(broken2))
    assert abort2 and "node_fallback/nodes" in reason2


def test_the_unflattened_summary_would_have_passed_a_broken_pilot(pilot_summary) -> None:
    """Regression guard: fed RAW, the rule sees zeros everywhere and only the
    "0 nodes" branch can fire -- which is exactly why the defect was invisible."""
    broken = dict(pilot_summary)
    broken["harness_quality"] = dict(
        pilot_summary["harness_quality"],
        defaulted=sum(pilot_summary["harness_calls"].values()),
    )
    raw_abort, raw_reason = should_abort_pilot(broken)
    assert not raw_abort or "defaulted/calls" not in raw_reason
    assert should_abort_pilot(flatten_pilot_summary(broken))[0]


def test_flatten_tolerates_a_missing_block() -> None:
    assert flatten_pilot_summary({}) == {
        "calls": 0, "defaulted": 0, "nodes": 0, "node_fallback": 0
    }


# --------------------------------------------------------------------------
# B3: the image ignore list
# --------------------------------------------------------------------------

def test_image_ignore_does_not_drop_the_models_package() -> None:
    """models/ is this repo's own source package (models/node.py); ignoring it
    made every remote import fail."""
    assert not any("models" in pattern for pattern in IMAGE_IGNORE)
    assert (msr.REPO_ROOT / "models" / "node.py").exists()


def test_image_ignore_drops_the_heavy_and_useless_paths() -> None:
    for pattern in (
        "benchmark/bineval/results/personamem_probe/**", "**/*.pdf", "**/*.html",
        "**/.git/**", "**/__pycache__/**", "wandb/**", "outputs/**",
    ):
        assert pattern in IMAGE_IGNORE


# --------------------------------------------------------------------------
# F7: the ignore list must not eat the two arm source files
# --------------------------------------------------------------------------

def test_the_blanket_results_ignore_is_gone() -> None:
    """It excluded results/pilot/, where summary_9x and oracle_cd_full live."""
    assert "benchmark/bineval/results/**" not in IMAGE_IGNORE


def test_the_two_arm_source_files_are_shipped() -> None:
    for rel in msr.IMAGE_REQUIRED_PATHS:
        assert (msr.REPO_ROOT / rel).exists(), rel
        assert not msr.is_ignored(rel), rel


def test_the_arm_builder_reads_exactly_those_files() -> None:
    """The paths in IMAGE_REQUIRED_PATHS are the ones arms.py opens."""
    src = (msr.REPO_ROOT / "benchmark" / "bineval" / "arms.py").read_text(
        encoding="utf-8"
    )
    assert "summary_6x.txt" in src and "oracle_cd_full.txt" in src


def test_the_heavy_results_subdirectories_are_still_ignored() -> None:
    for rel in (
        "benchmark/bineval/results/personamem_probe/probe.json",
        "benchmark/bineval/results/cd/restaurant_cd.json",
        "benchmark/bineval/results/audit/a.json",
        "benchmark/bineval/results/spec_run/answers/x.json",
        "benchmark/bineval/results/spec_run_2/answers/x.json",
    ):
        assert msr.is_ignored(rel), rel


def test_is_ignored_handles_the_glob_forms_this_module_relies_on() -> None:
    assert msr.is_ignored("a/b/__pycache__/x.pyc")
    assert msr.is_ignored("__pycache__/x.pyc")
    assert msr.is_ignored("doc.pdf") and msr.is_ignored("a/b/doc.pdf")
    assert msr.is_ignored("outputs/x/y.png")
    assert not msr.is_ignored("models/node.py")
    assert not msr.is_ignored("benchmark/bineval/arms.py")
    # a single * never crosses a directory separator
    assert not msr.is_ignored("a/b.pdf", ["*.pdf"])
    assert msr.is_ignored("b.pdf", ["*.pdf"])


# --------------------------------------------------------------------------
# B4: the long arms are gated
# --------------------------------------------------------------------------

def test_default_arms_exclude_the_long_ones() -> None:
    for arm in LONG_ARMS:
        assert arm not in DEFAULT_ARMS
    assert "trunc_6x" in DEFAULT_ARMS and "cd_mass_6x" in DEFAULT_ARMS


def test_include_long_arms_needs_an_explicit_gpu() -> None:
    with pytest.raises(SystemExit, match="requires an explicit --gpu"):
        resolve_arms(None, include_long=True, gpu=None)


def test_include_long_arms_rejects_a_small_gpu() -> None:
    with pytest.raises(SystemExit, match=">= 141GB"):
        resolve_arms(None, include_long=True, gpu="A100-80GB")


def test_include_long_arms_accepts_an_h200() -> None:
    assert GPU_MEM_GB["H200"] >= LONG_ARM_MIN_GPU_GB
    arms = resolve_arms(None, include_long=True, gpu="H200")
    for arm in LONG_ARMS:
        assert arm in arms
    assert set(DEFAULT_ARMS) <= set(arms)


def test_resolve_arms_without_the_flag_ignores_the_gpu() -> None:
    assert resolve_arms(None, include_long=False, gpu=None) == DEFAULT_ARMS


# --------------------------------------------------------------------------
# M3: the vLLM command
# --------------------------------------------------------------------------

def test_vllm_command_has_no_reasoning_parser() -> None:
    cmd = vllm_command("Qwen/Qwen3.8-27B", 8000)
    assert "--reasoning-parser" not in cmd


def test_vllm_command_carries_the_serving_knobs() -> None:
    cmd = vllm_command("Qwen/Qwen3.8-27B", 8123)
    joined = " ".join(cmd)
    assert "--gpu-memory-utilization 0.85" in joined
    assert "--max-num-seqs 16" in joined
    assert "--max-model-len 16384" in joined
    assert "--port 8123" in joined


def test_thinking_is_disabled_through_the_judge_body() -> None:
    from benchmark.bineval.build_cd_offline import DEFAULT_JUDGE_EXTRA_BODY

    body = json.loads(DEFAULT_JUDGE_EXTRA_BODY)
    assert body["chat_template_kwargs"]["enable_thinking"] is False


# --------------------------------------------------------------------------
# M4 / M5: pinned images, module-level functions
# --------------------------------------------------------------------------

def test_the_pins_are_exact() -> None:
    assert msr.VLLM_PIN.startswith("vllm==")
    assert msr.TORCH_PIN == "torch==2.11.*"
    assert msr.TRANSFORMERS_PIN == "transformers==5.8.*"


def test_modal_names_exist_at_module_level() -> None:
    """M5: Modal re-imports this module inside the container, so the functions
    must be module attributes -- not locals of a build_app() closure."""
    for name in ("app", "volume", "phase_manager", "phase_instrument",
                 "phase_reader", "image_vllm", "image_hf"):
        assert hasattr(msr, name), name
    assert not hasattr(msr, "build_app")


def test_modal_is_not_imported_on_this_machine() -> None:
    assert msr.modal_available() is False
    assert "modal" not in sys.modules


def test_modal_entrypoints_fail_loudly_without_modal() -> None:
    for argv in (["--deploy"], ["--run", "manager"], ["--download"]):
        with pytest.raises(SystemExit, match="modal is not installed"):
            msr.main(argv)


# --------------------------------------------------------------------------
# M6: the w grid
# --------------------------------------------------------------------------

def test_default_w_grid() -> None:
    assert DEFAULT_W_GRID == (0.0, 0.02, 0.05, 0.1, 0.2, 0.5)


def test_parse_w_grid() -> None:
    assert parse_w_grid("0,0.05,0.2") == (0.0, 0.05, 0.2)
    assert parse_w_grid("0.2, 0.05 ,0.2") == (0.05, 0.2)
    with pytest.raises(SystemExit):
        parse_w_grid("")
    with pytest.raises(SystemExit):
        parse_w_grid("-1")


def test_w_grid_override_reaches_the_cells() -> None:
    cells = build_cells(["cd_mass_6x"], parse_w_grid("0,0.3"), exploratory=None)
    assert sorted({c["w"] for c in cells}) == [0.0, 0.3]


def test_max_planet_mass_reads_the_marker_format() -> None:
    block = "<CONTEXT>\n[SN] a\n  [PN4.0] b\n    [RN] c\n  [PN12.5] d\n</CONTEXT>"
    assert max_planet_mass(block) == 12.5
    assert max_planet_mass("no markers here") == 0.0


# --------------------------------------------------------------------------
# legacy oracle arm is refused with w > 0
# --------------------------------------------------------------------------

def test_oracle_cd_full_is_refused_in_an_injected_cell() -> None:
    with pytest.raises(ValueError, match="legacy"):
        build_cells(["oracle_cd_full"], [0.5], exploratory=None)


def test_oracle_cd_full_is_allowed_as_a_text_only_arm() -> None:
    cells = build_cells(["oracle_cd_full"], [0.0], exploratory=None)
    assert cells[0]["w"] == 0.0 and cells[0]["inject"] == "none"


def test_validate_cell_refuses_a_hand_built_legacy_injection() -> None:
    from experiments.modal_spec_run import validate_cell

    with pytest.raises(ValueError, match="legacy"):
        validate_cell(
            {"cell": "x", "arm": "oracle_cd_full", "w": 0.1, "inject": "planet",
             "prefill_scale": 0.0, "max_questions": None}
        )
    ok = {"cell": "x", "arm": "oracle_cd_full", "w": 0.0, "inject": "none",
          "prefill_scale": 0.0, "max_questions": None}
    assert validate_cell(ok) is ok


# --------------------------------------------------------------------------
# _arm_tokens no longer swallows errors
# --------------------------------------------------------------------------

def test_arm_tokens_propagates_a_missing_chat() -> None:
    with pytest.raises(ValueError):
        _arm_tokens("trunc_6x", None, None)  # 'trunc' needs chat=


def test_arm_tokens_propagates_a_missing_cd_json() -> None:
    from benchmark.bineval.arms import load_chat

    with pytest.raises(ValueError):
        _arm_tokens("cd_mass_6x", load_chat(), None)


def test_arm_tokens_returns_none_only_for_an_optional_artifact(monkeypatch) -> None:
    from benchmark.bineval import arms as arms_mod

    def gone(path):
        raise FileNotFoundError("arm source file not found: %s" % path)

    monkeypatch.setattr(arms_mod, "_read_verbatim", gone)
    assert _arm_tokens("summary_9x", None, None) is None
    assert _arm_tokens("oracle_cd_full", None, None) is None


def test_arm_tokens_works_for_a_present_arm() -> None:
    from benchmark.bineval.arms import load_chat

    assert _arm_tokens("trunc_6x", load_chat(), None) > 0


# --------------------------------------------------------------------------
# the dry run writes to a temp dir and leaves the repo alone
# --------------------------------------------------------------------------

def test_dry_run_default_out_is_a_temp_dir(capsys, monkeypatch) -> None:
    import tempfile

    monkeypatch.setattr(msr, "DEFAULT_W_GRID", (0.0,))
    assert msr.main(["--dry-run", "--max-questions", "1", "--no-tokens"]) == 0
    printed = capsys.readouterr().out
    out = Path(printed.strip().split(" to ")[-1])
    assert out.exists() and out.is_dir()
    assert str(out).startswith(tempfile.gettempdir())
    assert msr.REPO_ROOT not in out.parents
    assert (out / "run_manifest.json").exists()

"""S1.4: the `--extractor spec` driver path of build_cd_offline.

No model, no network: `--judge fake` only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from _spec_helpers import make_manager, node_reply, same_text_judge

from benchmark.bineval.arms import cd_from_records
from benchmark.bineval.build_cd_offline import build_cd, main
from communication.cd_serializer import CDSerializer
from management.harness.spec_manager import make_spec_fake_judge
from models.correlation_diagram import CorrelationDiagram

# Two synthetic English sessions, one query turn (D-7) in each.
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
                {"role": "user", "content": "Tell me the rent"},  # D-7, English rule
            ]
        },
        {
            "turns": [
                {
                    "role": "user",
                    "content": "Marketing is a new pillar. Delivery costs 2k yen.",
                },
                {"role": "assistant", "content": "Understood, delivery is 2k yen."},
                {"role": "user", "content": "What is the delivery fee?"},  # D-7
            ]
        },
    ]
}

MARKER_LINE = re.compile(r"^(\[SN\] |  \[PN\d+(\.\d+)?\] |    \[RN\] )\S")


def _chat_file(tmp_path: Path) -> Path:
    path = tmp_path / "chat.json"
    path.write_text(json.dumps(CHAT, ensure_ascii=False), encoding="utf-8")
    return path


def _run_main(tmp_path, monkeypatch, *extra: str) -> Path:
    out = tmp_path / "cd_spec.json"
    argv = [
        "build_cd_offline",
        "--chat",
        str(_chat_file(tmp_path)),
        "--out",
        str(out),
        "--extractor",
        "spec",
        "--judge",
        "fake",
        *extra,
    ]
    monkeypatch.setattr("sys.argv", argv)
    main()
    return out


def test_driver_spec_fake_produces_a_non_empty_cd(tmp_path, monkeypatch, capsys) -> None:
    out = _run_main(tmp_path, monkeypatch)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["summary"]["total"] > 0
    assert payload["summary"]["failed_turns"] == 0
    # D-7 skipped the two query turns, so 6 messages were counted but only 4
    # reached the manager.
    # Spec 0036: the spec arm counts ROUND TRIPS (user+assistant pairs), not
    # messages: 6 messages -> 4 round trips here (query round trips included
    # in the count, skipped by D-7 before the manager sees them).
    assert payload["summary"]["turns"] == 4
    assert payload["summary"]["harness_calls"]["node"] > 0
    assert set(payload["summary"]["harness_quality"]) == {
        "unparsed",
        "defaulted",
        "node_fallback",
        "vanished",
        "attached",
    }
    assert payload["summary"]["harness_quality"]["node_fallback"] == 0

    printed = capsys.readouterr().out
    assert "using spec manager (judge=fake, shortlist_k=0, max_workers=1)" in printed
    assert "level markers: True" in printed
    marker_lines = [ln for ln in printed.splitlines() if ln.startswith(("[SN]", "  [PN", "    [RN]"))]
    assert marker_lines, printed
    for line in marker_lines:
        assert MARKER_LINE.match(line), line
    # Legacy "[PN{mass}]" on a top-level line would mean level_markers was off.
    assert not any(ln.startswith("[PN") for ln in printed.splitlines())

    # M2: the fake judge now yields a REAL three-level tree.  The printed smoke
    # block is budgeted to 6x and can keep a single line, so the marker check is
    # made on the FULL serialization of the written CD.
    assert payload["summary"]["sun"] > 0
    assert payload["summary"]["planet"] > 0
    assert payload["summary"]["satellite"] > 0
    block = CDSerializer(level_markers=True).to_context_block(
        cd_from_records(payload["nodes"])
    )
    lines = block.splitlines()[1:-1]
    assert any(ln.startswith("[SN] ") for ln in lines), block
    assert any(ln.startswith("  [PN") for ln in lines), block
    assert any(ln.startswith("    [RN] ") for ln in lines), block
    for line in lines:
        assert MARKER_LINE.match(line), line


def test_level_markers_can_be_turned_off(tmp_path, monkeypatch, capsys) -> None:
    _run_main(tmp_path, monkeypatch, "--no-level-markers")
    printed = capsys.readouterr().out
    assert "level markers: False" in printed
    assert "[SN]" not in printed


def test_judge_local_without_a_model_exits(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_cd_offline",
            "--chat",
            str(_chat_file(tmp_path)),
            "--out",
            str(tmp_path / "cd.json"),
            "--extractor",
            "spec",
            "--judge",
            "local",
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert "--judge local requires --judge-model" in str(excinfo.value)


def test_anthropic_and_openai_are_no_longer_cli_choices(tmp_path, monkeypatch) -> None:
    """Directive 5: no external API.  The backend CLASSES stay (they have their
    own unit tests in tests/harness/test_backends.py)."""
    for choice in ("anthropic", "openai"):
        monkeypatch.setattr(
            "sys.argv",
            [
                "build_cd_offline",
                "--chat",
                str(_chat_file(tmp_path)),
                "--out",
                str(tmp_path / "cd.json"),
                "--extractor",
                "spec",
                "--judge",
                choice,
            ],
        )
        with pytest.raises(SystemExit):
            main()


def test_planet_mass_floor_requires_the_spec_extractor(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_cd_offline",
            "--chat",
            str(_chat_file(tmp_path)),
            "--out",
            str(tmp_path / "cd.json"),
            "--extractor",
            "harness",
            "--judge",
            "fake",
            "--planet-mass-floor",
            "0.0",
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert "--planet-mass-floor requires --extractor spec" in str(excinfo.value)


# -- the fake judge itself ---------------------------------------------------


def test_spec_fake_judge_cycles_the_three_levels() -> None:
    """Contract change (M2): the fake judge used to make EVERY chunk a
    satellite, which produced a smoke diagram of promoted suns and not a single
    [PN line.  It now cycles sun / planet / satellite / satellite."""
    manager = make_manager(make_spec_fake_judge(120))
    levels = [
        manager.node_for_text(t).level.value
        for t in ("first", "second", "third", "fourth", "fifth")
    ]
    assert levels == ["sun", "planet", "satellite", "satellite", "sun"]


def test_spec_fake_judge_builds_a_three_level_tree() -> None:
    manager = make_manager(make_spec_fake_judge(120))
    base = CorrelationDiagram()
    manager.update(
        base,
        "The restaurant plan is the topic. The budget matters most. Rent is 500k yen.",
        "",
        0,
    )
    block = CDSerializer(level_markers=True).to_context_block(base)
    lines = block.splitlines()[1:-1]
    assert any(ln.startswith("[SN] ") for ln in lines), block
    assert any(ln.startswith("  [PN") for ln in lines), block
    assert any(ln.startswith("    [RN] ") for ln in lines), block


def test_spec_fake_judge_answers_no_to_q_same_and_to_repeat_belongs() -> None:
    judge = make_spec_fake_judge(120)
    assert judge.complete("Do A and B state the same matter?", max_tokens=8) == "no"
    belongs = (
        "Statement:\n<<<\nRent is 500k yen.\n>>>\nTopic:\n<<<\nBudget\n>>>\n"
        "Does the statement belong under this topic?"
    )
    # first candidate for this query -> yes, every later one -> no
    assert judge.complete(belongs, max_tokens=8) == "yes"
    assert judge.complete(belongs, max_tokens=8) == "no"


# -- all three markers really appear for a full tree -------------------------


def test_spec_manager_tree_serializes_with_all_three_markers() -> None:
    """The fake judge yields suns only (by design, S1.3), so the [PN/[RN lines
    are proved here with the scripted judge that builds a real tree."""

    def node(text: str) -> str:
        if "restaurant plan" in text:
            return node_reply(text, 90, 10, 10)
        if "budget" in text:
            return node_reply(text, 10, 90, 10)
        return node_reply(text, 10, 10, 90)

    manager = make_manager(same_text_judge(node=node))
    base = CorrelationDiagram()
    manager.update(
        base,
        "The restaurant plan is the topic. The budget matters most. Rent is 500k yen.",
        "",
        0,
    )
    block = CDSerializer(level_markers=True).to_context_block(base)
    lines = block.splitlines()[1:-1]
    assert any(ln.startswith("[SN] ") for ln in lines)
    assert any(ln.startswith("  [PN") for ln in lines)
    assert any(ln.startswith("    [RN] ") for ln in lines)
    for line in lines:
        assert MARKER_LINE.match(line), line


def test_build_cd_drives_the_spec_manager_directly() -> None:
    manager = make_manager(make_spec_fake_judge(120))
    cd, n_turns, failed = build_cd(CHAT, None, apply_d7=True, manager=manager)
    assert n_turns == 6
    assert failed == 0
    assert len(cd) > 0
    assert manager.normalize_calls == 4  # 6 messages minus the 2 D-7 query turns

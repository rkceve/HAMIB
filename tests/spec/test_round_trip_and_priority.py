"""Spec 0036 (one update per round trip) and planet-only eviction priority."""

from __future__ import annotations

from benchmark.bineval.build_cd_offline import _round_trip_iter, build_cd
from communication.cd_serializer import CDSerializer
from models.correlation_diagram import CorrelationDiagram
from models.node import Node, NodeLevel


def test_round_trip_iter_pairs_user_with_following_assistant() -> None:
    chat = {
        "sessions": [
            {
                "turns": [
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"},
                    {"role": "user", "content": "u2"},
                    {"role": "user", "content": "u3"},
                    {"role": "assistant", "content": "a3"},
                    {"role": "user", "content": "u4"},
                ]
            },
            {"turns": [{"role": "assistant", "content": "a5"}]},
        ]
    }
    assert list(_round_trip_iter(chat)) == [
        (0, "u1", "a1"),
        (0, "u2", ""),
        (0, "u3", "a3"),
        (0, "u4", ""),
        (1, "", "a5"),
    ]


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def update(self, base, user_text, assistant_text, turn):
        self.calls.append((user_text, assistant_text, turn))


def test_build_cd_pairs_round_trips_and_applies_d7_to_the_pair() -> None:
    chat = {
        "sessions": [
            {
                "turns": [
                    {"role": "user", "content": "The rent is 500k yen."},
                    {"role": "assistant", "content": "Noted, 500k yen."},
                    {"role": "user", "content": "What is the rent?"},
                    {"role": "assistant", "content": "The rent is 500k yen."},
                ]
            }
        ]
    }
    rec = _Recorder()
    _cd, n_turns, failed = build_cd(
        chat, None, apply_d7=True, manager=rec, pair_round_trips=True
    )
    assert failed == 0 and n_turns == 2
    # the query round trip (question + its answer) is skipped entirely
    assert rec.calls == [("The rent is 500k yen.", "Noted, 500k yen.", 0)]
    rec2 = _Recorder()
    build_cd(chat, None, apply_d7=True, manager=rec2, pair_round_trips=False)
    assert len(rec2.calls) == 3  # legacy: per message, only the query message skipped


def _cd() -> CorrelationDiagram:
    cd = CorrelationDiagram()
    s1 = Node(text="Topic A", level=NodeLevel.SUN, mass=0.0, created_turn=0)
    cd.add_sun(s1)
    p_small = Node(text="Small planet", level=NodeLevel.PLANET, mass=0.0, created_turn=0)
    cd.add_planet(p_small, s1.node_id)
    cd.add_satellite(
        Node(text="s1", level=NodeLevel.SATELLITE, mass=0.0, created_turn=0), p_small.node_id
    )
    s2 = Node(text="Topic B", level=NodeLevel.SUN, mass=0.0, created_turn=1)
    cd.add_sun(s2)
    p_big = Node(text="Big planet", level=NodeLevel.PLANET, mass=0.0, created_turn=1)
    cd.add_planet(p_big, s2.node_id)
    for i in range(3):
        cd.add_satellite(
            Node(text=f"b{i}", level=NodeLevel.SATELLITE, mass=0.0, created_turn=1),
            p_big.node_id,
        )
    cd.normalize(planet_mass_floor=0.0)
    return cd


def test_mass_eviction_uses_planet_mass_only_in_marker_mode() -> None:
    cd = _cd()
    # Inflate Topic A's INTERNAL sun mass: the legacy path follows it, the
    # spec path must ignore it and follow the planet masses (1 vs 3).
    cd.suns[0].sun.mass = 99.0

    def counter(text: str) -> int:
        return len([ln for ln in text.splitlines() if ln and not ln.startswith("<")])

    budget = 3  # sun + planet + one satellite
    legacy = CDSerializer(level_markers=False).to_context_block_budgeted(
        cd, budget, "mass", counter
    )
    spec = CDSerializer(level_markers=True).to_context_block_budgeted(
        cd, budget, "mass", counter
    )
    legacy_lines = [ln for ln in legacy.splitlines() if not ln.startswith("<")]
    spec_lines = [ln for ln in spec.splitlines() if not ln.startswith("<")]
    # legacy: the inflated sun mass wins the first slot
    assert legacy_lines[0].endswith("Topic A")
    # spec: Topic B (its best planet has mass 3) comes first, then that planet,
    # then one of its satellites; Topic A never fits in a 3-line budget.
    assert spec_lines[0] == "[SN] Topic B"
    assert spec_lines[1] == "  [PN3.0] Big planet"
    assert spec_lines[2].startswith("    [RN] b")
    assert "Topic A" not in spec

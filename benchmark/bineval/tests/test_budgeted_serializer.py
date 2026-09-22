"""Unit tests for WO-1: budgeted CD serialization + created_turn plumbing.

Coverage (RESEARCH_PROGRAM sec.4 WO-1 acceptance + work-order test list):
(a) budget respected: serialized block token count <= budget on a 3-level CD;
(b) ancestor invariant holds under all 3 policies;
(c) every policy with a huge budget reproduces to_context_block byte-identically;
(d) same seed -> identical output; different seed -> different kept-set (random);
(e) created_turn defaults to -1, to_dict/from_dict round-trips it, from_dict on a
    dict WITHOUT the key yields -1;
(f) classifier stamps created_turn from chunk.turn (extractor mocked; no SBERT).

No network / no SBERT / no gen model is used anywhere in this file.
"""

from __future__ import annotations

import pytest
import tiktoken

from communication.cd_serializer import CDSerializer
from management.graph_builder import GraphBuilder
from management.node_classifier import NodeClassifier
from management.text_chunker import Chunk
from models.correlation_diagram import CorrelationDiagram
from models.node import Coordinates, Node, NodeLevel


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _token_counter():
    enc = tiktoken.get_encoding("cl100k_base")
    return lambda text: len(enc.encode(text))


def _node(text: str, level: NodeLevel, mass: float, created_turn: int = -1) -> Node:
    return Node(text=text, level=level, mass=mass, created_turn=created_turn)


def _build_cd(n_suns: int, n_planets: int, n_sats: int) -> CorrelationDiagram:
    """Deterministic n_suns x n_planets x n_sats CD with distinct texts, masses,
    and created_turn values (so mass/recency orderings are well-defined).
    """
    cd = CorrelationDiagram()
    turn = 0
    for si in range(n_suns):
        sun = _node(f"sun-{si}", NodeLevel.SUN, mass=100.0 - si, created_turn=turn)
        cd.add_sun(sun)
        turn += 1
        for pi in range(n_planets):
            planet = _node(
                f"sun{si}-planet-{pi}", NodeLevel.PLANET,
                mass=50.0 - pi - 10 * si, created_turn=turn,
            )
            cd.add_planet(planet, sun.node_id)
            turn += 1
            for ti in range(n_sats):
                sat = _node(
                    f"sun{si}-planet{pi}-sat-{ti}", NodeLevel.SATELLITE,
                    mass=1.0 + 0.01 * ti, created_turn=turn,
                )
                cd.add_satellite(sat, planet.node_id)
                turn += 1
    return cd


def _kept_ids(cd: CorrelationDiagram, block: str) -> set[str]:
    """Map the emitted node lines back to node_ids by matching text."""
    text_to_id = {n.text: n.node_id for n in cd.all_nodes()}
    ids: set[str] = set()
    for ln in block.splitlines():
        if not ln or ln in ("<CONTEXT>", "</CONTEXT>"):
            continue
        # line format: (indent)"[PN{mass}] {text}"; recover text after first "] "
        after = ln.split("] ", 1)[1] if "] " in ln else ln
        if after in text_to_id:
            ids.add(text_to_id[after])
    return ids


def _ancestor_ok(cd: CorrelationDiagram, kept: set[str]) -> bool:
    """Every kept node must have its full parent chain kept."""
    parent_of = {n.node_id: n.parent_id for n in cd.all_nodes()}
    for nid in kept:
        p = parent_of.get(nid)
        while p is not None:
            if p not in kept:
                return False
            p = parent_of.get(p)
    return True


# ---------------------------------------------------------------------------
# (a) budget respected
# ---------------------------------------------------------------------------

def test_budget_respected_all_policies():
    cd = _build_cd(2, 3, 3)
    ser = CDSerializer()
    tc = _token_counter()
    full = tc(ser.to_context_block(cd))
    budget = full // 3
    for policy in ("mass", "random", "recency"):
        block = ser.to_context_block_budgeted(cd, budget, policy, tc, seed=0)
        assert tc(block) <= budget, f"{policy}: {tc(block)} > {budget}"


def test_budget_tiny_yields_empty_wrapper():
    # A budget below any single node (but not below the wrapper) returns the
    # bare wrapper; a budget below the wrapper itself is a ValueError
    # (mcbuild_bench Astra round 3 item 7: never return a block over budget).
    cd = _build_cd(1, 2, 2)
    ser = CDSerializer()
    tc = _token_counter()
    wrapper = tc("<CONTEXT>\n</CONTEXT>")
    with pytest.raises(ValueError, match="wrapper"):
        ser.to_context_block_budgeted(cd, budget_tokens=wrapper - 1, policy="mass", token_counter=tc)
    block = ser.to_context_block_budgeted(cd, budget_tokens=wrapper, policy="mass", token_counter=tc)
    assert tc(block) <= wrapper
    node_lines = [ln for ln in block.splitlines() if ln not in ("<CONTEXT>", "</CONTEXT>") and ln]
    assert node_lines == []


# ---------------------------------------------------------------------------
# (b) ancestor invariant
# ---------------------------------------------------------------------------

def test_ancestor_invariant_all_policies():
    cd = _build_cd(2, 3, 2)
    ser = CDSerializer()
    tc = _token_counter()
    full = tc(ser.to_context_block(cd))
    # try several budget fractions so eviction actually bites at different depths
    for frac in (2, 3, 5, 8):
        budget = full // frac
        for policy in ("mass", "random", "recency"):
            block = ser.to_context_block_budgeted(cd, budget, policy, tc, seed=1)
            kept = _kept_ids(cd, block)
            assert _ancestor_ok(cd, kept), f"{policy} frac={frac}: ancestor broken"


# ---------------------------------------------------------------------------
# (c) huge budget == to_context_block byte-identical, every policy
# ---------------------------------------------------------------------------

def test_huge_budget_byte_identical_all_policies():
    cd = _build_cd(2, 3, 3)
    ser = CDSerializer()
    tc = _token_counter()
    reference = ser.to_context_block(cd)
    huge = 10_000_000
    for policy in ("mass", "random", "recency"):
        block = ser.to_context_block_budgeted(cd, huge, policy, tc, seed=7)
        assert block == reference, f"{policy}: not byte-identical to to_context_block"


# ---------------------------------------------------------------------------
# (d) determinism + seed sensitivity for random
# ---------------------------------------------------------------------------

def test_same_seed_identical_output():
    cd = _build_cd(2, 4, 3)
    ser = CDSerializer()
    tc = _token_counter()
    budget = tc(ser.to_context_block(cd)) // 3
    for policy in ("mass", "random", "recency"):
        a = ser.to_context_block_budgeted(cd, budget, policy, tc, seed=42)
        b = ser.to_context_block_budgeted(cd, budget, policy, tc, seed=42)
        assert a == b, f"{policy}: not deterministic under same seed"


def test_different_seed_changes_random_kept_set():
    # CD large enough that a forced-eviction random kept-set differs across seeds
    cd = _build_cd(3, 4, 3)
    ser = CDSerializer()
    tc = _token_counter()
    budget = tc(ser.to_context_block(cd)) // 3
    seen: set[frozenset[str]] = set()
    for seed in range(6):
        block = ser.to_context_block_budgeted(cd, budget, "random", tc, seed=seed)
        seen.add(frozenset(_kept_ids(cd, block)))
    # at least two distinct kept-sets across 6 seeds (collision very unlikely)
    assert len(seen) >= 2


# ---------------------------------------------------------------------------
# (e) created_turn dataclass + round-trip
# ---------------------------------------------------------------------------

def test_created_turn_defaults_to_minus_one():
    n = Node(text="x", level=NodeLevel.SUN, mass=1.0)
    assert n.created_turn == -1


def test_created_turn_roundtrips_via_dict():
    n = Node(
        text="hello", level=NodeLevel.PLANET, mass=3.0, node_id="abc123",
        parent_id="par", coordinates=Coordinates(1, 2, -1), created_turn=17,
    )
    d = n.to_dict()
    assert d["created_turn"] == 17
    back = Node.from_dict(d)
    assert back.created_turn == 17
    assert back.node_id == "abc123"
    assert back.parent_id == "par"


def test_from_dict_without_key_yields_minus_one():
    # simulate an OLD JSON node lacking created_turn
    legacy = {
        "node_id": "old1",
        "text": "legacy node",
        "level": "satellite",
        "mass": 1.0,
        "parent_id": "p",
        "coordinates": {"sun_idx": 0, "planet_idx": 0, "satellite_idx": 0},
    }
    n = Node.from_dict(legacy)
    assert n.created_turn == -1


# ---------------------------------------------------------------------------
# (f) classifier stamps created_turn from chunk.turn (extractor mocked)
# ---------------------------------------------------------------------------

def _mock_extractor(_text: str) -> list[dict]:
    # one sun-level fact per chunk; deterministic, no network
    return [{"text": "ALPHA -> CRANE", "level": "sun", "parent_hint": ""}]


def test_classifier_stamps_created_turn_from_chunk_turn():
    clf = NodeClassifier()
    builder = GraphBuilder()
    cd = CorrelationDiagram()
    chunk = Chunk(text="ALPHA code is CRANE", source="user", turn=41)
    clf.classify(chunk, cd, _mock_extractor, builder=builder)
    nodes = list(cd.all_nodes())
    assert nodes, "expected at least one node created"
    assert all(n.created_turn == 41 for n in nodes), (
        f"created_turn not stamped from chunk.turn: {[n.created_turn for n in nodes]}"
    )


def test_classifier_update_mass_preserves_original_created_turn():
    """UPDATE_MASS must NOT overwrite created_turn: a node seen again on a later
    turn keeps its birth turn (management/node_classifier.py + graph_builder.py:34-37).

    Similarity is injected (set_llm_similarity_fn) so no SBERT/network loads: the
    similarity backend returns 1.0 for identical text, forcing the UPDATE_MASS path
    on the second classify without touching sentence-transformers.
    """
    from utils import similarity as sim

    sim.set_llm_similarity_fn(lambda a, b: 1.0 if a == b else 0.0)
    try:
        clf = NodeClassifier()
        builder = GraphBuilder()
        cd = CorrelationDiagram()
        # turn 5 births the node
        clf.classify(Chunk("ALPHA is CRANE", "user", 5), cd, _mock_extractor, builder=builder)
        birth_turns = {n.node_id: n.created_turn for n in cd.all_nodes()}
        assert birth_turns, "expected a node at birth"
        # turn 9 re-mentions the SAME text -> similarity 1.0 -> UPDATE_MASS path
        clf.classify(Chunk("ALPHA is CRANE", "user", 9), cd, _mock_extractor, builder=builder)
        for n in cd.all_nodes():
            if n.node_id in birth_turns:
                assert n.created_turn == birth_turns[n.node_id] == 5, (
                    f"UPDATE_MASS overwrote created_turn to {n.created_turn}"
                )
    finally:
        sim.set_llm_similarity_fn(None)

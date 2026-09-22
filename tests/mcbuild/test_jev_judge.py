"""jev_judge adapters against replies shaped like the DESIGN.md 3 verbatim JSON."""

from __future__ import annotations

import pytest

from benchmark.mcbuild_bench.errors import JevStop
from benchmark.mcbuild_bench.jev_judge import (
    LEVEL_SCORES,
    MAX_SUN_CHOICES,
    JevCounters,
    JevJudgeLLM,
    JevKeepFn,
    JevNodeFn,
    JevSimilarityJudge,
    Q_AXES,
    Q_KEEP,
    argmax_level,
    choice_of,
    noul_of,
    planet_question,
    project_requests,
    sun_question,
)
from management.harness.prompts import K_BELONGS, K_NODE, K_SAME
from management.harness.spec_manager import level_from_scores
from models.node import NodeLevel
from tests.mcbuild._fakes import (
    TAG_DROP,
    TAG_PLANET,
    TAG_SUN,
    FakeJev,
    FakeSummarizer,
    choice_answer,
    noul_answer,
    score_answer,
)

def _alpha_name(prefix: str, i: int) -> str:
    """Alphabetic candidate texts (FakeJev's topic word must be isalpha)."""
    return prefix + chr(ord("A") + i // 26) + chr(ord("a") + i % 26)


# The DESIGN.md 3 response example, verbatim.
DOC_RESPONSE = {
    "model": "jev-latest",
    "answers": {
        "is_urgent": {"type": "noul", "noul": 0.92},
        "department": {
            "type": "choice",
            "choice": "technical",
            "probabilities": {"billing": 0.08, "technical": 0.85, "sales": 0.07},
            "confidence": 0.82,
        },
        "frustration": {
            "type": "score",
            "score": 1.6,
            "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
            "probabilities": {"0": 0.05, "1": 0.3, "2": 0.65},
            "confidence": 0.78,
        },
    },
    "usage": {"input_tokens": 312, "output_tokens": 48},
}


# -- pure parsers ---------------------------------------------------------------------


def test_parsers_on_the_verbatim_doc_json() -> None:
    a = DOC_RESPONSE["answers"]
    assert noul_of(a["is_urgent"]) == 0.92
    assert choice_of(a["department"]) == "technical"
    assert argmax_level(a["frustration"], n_levels=3) == 2


def test_argmax_tie_goes_to_lower_index() -> None:
    assert argmax_level(score_answer({"1": 0.4, "3": 0.4, "0": 0.2})) == 1
    assert argmax_level(score_answer({"0": 0.2, "1": 0.2, "2": 0.2, "3": 0.2, "4": 0.2})) == 0


def test_level_mapping_is_10_plus_20k() -> None:
    assert LEVEL_SCORES == (10, 30, 50, 70, 90)


def test_missing_fields_raise_jevstop() -> None:
    with pytest.raises(JevStop):
        noul_of({"type": "noul"})
    with pytest.raises(JevStop):
        choice_of({"type": "choice", "probabilities": {"a": 1.0}})
    with pytest.raises(JevStop):
        argmax_level({"type": "score", "score": 1.6})
    # F3: a missing level is probability 0, not a stop ...
    assert argmax_level(
        {"type": "score", "probabilities": {"0": 0.2, "1": 0.2, "2": 0.2, "3": 0.4}}
    ) == 3
    # ... but no key at all, or an index beyond the 5 LEVEL_SCORES, stops.
    with pytest.raises(JevStop):
        argmax_level({"type": "score", "probabilities": {}})
    with pytest.raises(JevStop):
        argmax_level({"type": "score", "probabilities": {"5": 1.0}})


def test_parsers_are_the_jev_client_ones_and_type_check() -> None:
    """F3/F12: jev_judge must not re-implement loose parsers; odd payloads
    raise JevStop, never TypeError / ValueError."""
    from benchmark.mcbuild_bench import jev_client

    assert noul_of is jev_client.noul_of
    assert choice_of is jev_client.choice_of
    with pytest.raises(JevStop):
        noul_of({"type": "noul", "noul": None})
    with pytest.raises(JevStop):
        noul_of({"type": "noul", "noul": "0.9"})
    with pytest.raises(JevStop):
        choice_of({"type": "choice", "choice": None})
    with pytest.raises(JevStop):
        argmax_level({"type": "score", "probabilities": {"0": None}})
    with pytest.raises(JevStop):
        argmax_level({"type": "score", "probabilities": {"0": "0.5"}})


def test_keep_fn_null_noul_is_jevstop_not_typeerror() -> None:
    jev = FakeJev(override={"keep": lambda s, q: {"type": "noul", "noul": None}})
    keep = JevKeepFn(JevNodeFn(jev, FakeSummarizer()))
    with pytest.raises(JevStop):
        keep("some chunk")
    assert keep.counters.unparsed == {K_NODE: 1}


def test_sun_and_planet_question_shapes_and_batch_offset() -> None:
    q = sun_question(["Topic A", "Topic B"])
    assert q["type"] == "choice"
    assert q["criteria"] == {"s0": "Topic A", "s1": "Topic B", "new_topic": "None of the listed topics"}
    assert len(sun_question(["t"] * MAX_SUN_CHOICES)["criteria"]) == 255
    # H22 (a): keys carry the GLOBAL candidate index of the batch
    assert list(sun_question(["x", "y"], offset=254)["criteria"]) == ["s254", "s255", "new_topic"]
    p = planet_question(["Item A", "Item B"])
    assert p["type"] == "choice"
    assert p["instructions"] == (
        "Which of the listed items is this excerpt a detail or sub-point of? Pick none if it "
        "belongs under none of them."
    )
    assert p["criteria"] == {"p0": "Item A", "p1": "Item B", "none": "None of the listed items"}
    assert list(planet_question(["x"], offset=300)["criteria"]) == ["p300", "none"]
    # a batch larger than 254 is a programming error (most_similar never builds one)
    with pytest.raises(ValueError, match="254"):
        sun_question(["t"] * (MAX_SUN_CHOICES + 1))
    with pytest.raises(ValueError, match="254"):
        planet_question(["t"] * (MAX_SUN_CHOICES + 1))


def test_question_texts_match_design() -> None:
    assert Q_KEEP["type"] == "noul"
    assert set(Q_KEEP["criteria"]) == {"true", "false"}
    for axis, q in Q_AXES.items():
        assert q["type"] == "score" and len(q["criteria"]) == 5, axis
    assert Q_AXES["detail"]["instructions"] == "How specific is this excerpt?"


# -- adapters ---------------------------------------------------------------------------


def test_keep_fn_noul_cut_at_half() -> None:
    jev = FakeJev(override={"keep": lambda s, q: noul_answer(0.5 if "half" in s else 0.49)})
    keep = JevKeepFn(JevNodeFn(jev, FakeSummarizer()))
    assert keep("exactly half") is True
    assert keep("just below") is False
    assert keep.counters.calls == {K_NODE: 2}  # H1: one request per chunk, counted once
    assert keep.counters.defaulted == {}
    assert keep.counters.input_tokens > 0
    # One request carrying keep AND the three axes, raw chunk as the state (D3).
    state, questions = jev.requests[0]
    assert state == "exactly half"
    assert list(questions) == ["keep", "comprehensiveness", "independence", "detail"]


def test_node_fn_three_axes_one_request_then_summarizer() -> None:
    """H1 (D3 one-request contract): keep_fn(text) then node_fn(text) on the
    same chunk = ONE Jev request with {keep, 3 axes}; the summarizer runs
    only for the kept chunk, after the node scores are read."""
    jev, summ = FakeJev(), FakeSummarizer()
    node_fn = JevNodeFn(jev, summ)
    keep = JevKeepFn(node_fn)
    text = f"{TAG_SUN} Hackathon plan overview"
    assert keep(text) is True
    summary, scores = node_fn(text)
    assert len(jev.requests) == 1
    assert list(jev.requests[0][1]) == ["keep", "comprehensiveness", "independence", "detail"]
    assert scores == {"comprehensiveness": 90, "independence": 10, "detail": 10}
    assert level_from_scores(scores) is NodeLevel.SUN
    assert summary == "Hackathon plan overview" and summ.calls == [text]
    assert keep(f"{TAG_PLANET} Server port fact") is True
    _, planet_scores = node_fn(f"{TAG_PLANET} Server port fact")
    assert level_from_scores(planet_scores) is NodeLevel.PLANET
    assert keep("A precise value 25565") is True
    _, sat_scores = node_fn("A precise value 25565")
    assert level_from_scores(sat_scores) is NodeLevel.SATELLITE
    assert len(jev.requests) == 3
    assert node_fn.counters.calls == {K_NODE: 3}
    assert node_fn.summarizer_calls == 3


def test_node_fn_before_keep_fn_is_an_ordering_error() -> None:
    """spec_manager calls keep_fn first, then node_fn on the same text; the
    reverse (or a different text) is a bug, not a second request."""
    jev, summ = FakeJev(), FakeSummarizer()
    node_fn = JevNodeFn(jev, summ)
    with pytest.raises(RuntimeError, match="keep_fn"):
        node_fn("never asked")
    JevKeepFn(node_fn)("asked text")
    with pytest.raises(RuntimeError, match="keep_fn"):
        node_fn("another text")
    assert len(jev.requests) == 1 and summ.calls == []


def test_dropped_chunk_never_reaches_the_summarizer() -> None:
    jev, summ = FakeJev(), FakeSummarizer()
    node_fn = JevNodeFn(jev, summ)
    keep = JevKeepFn(node_fn)
    assert keep(f"{TAG_DROP} raw log line") is False
    assert summ.calls == [] and node_fn.summarizer_calls == 0
    assert len(jev.requests) == 1


def test_jevstop_raised_by_ask_counts_unparsed_for_the_kind() -> None:
    """H2: a JevStop coming out of jev.ask itself is booked under the kind."""

    class Dead(FakeJev):
        def ask(self, state, questions):
            raise JevStop("429 after 3 attempts")

    node_fn = JevNodeFn(Dead(), FakeSummarizer())
    with pytest.raises(JevStop):
        JevKeepFn(node_fn)("chunk")
    assert node_fn.counters.unparsed == {K_NODE: 1}
    sim = JevSimilarityJudge(Dead())
    with pytest.raises(JevStop):
        sim.most_similar("a", ["b"], K_SAME)
    assert sim.runner.unparsed == {K_SAME: 1}


def test_node_fn_missing_axis_answer_is_unparsed_and_stops() -> None:
    class DropDetail(FakeJev):
        def ask(self, state, questions):
            result = super().ask(state, questions)
            result["answers"].pop("detail")
            return result

    node_fn = JevNodeFn(DropDetail(), FakeSummarizer())
    with pytest.raises(JevStop):  # every answer is parsed when the request returns
        JevKeepFn(node_fn)("some chunk")
    assert node_fn.counters.unparsed == {K_NODE: 1}
    assert node_fn.counters.defaulted == {}


def test_similarity_same_single_candidate_stays_pairwise_noul() -> None:
    """H23: one candidate keeps the pairwise `same` Noul (one request either way)."""
    jev = FakeJev()
    sim = JevSimilarityJudge(jev)
    assert sim.most_similar("x", [], K_SAME) == (-1, 0.0) and jev.requests == []
    assert sim.most_similar("b", ["b"], K_SAME) == (0, 1.0)
    assert jev.requests == [("A: b\nB: b", {"same": {"type": "noul",
                                                      "instructions": "Do A and B state the same matter?"}})]
    assert sim.most_similar("z", ["a"], K_SAME) == (-1, 0.0)
    assert sim.runner.calls == {K_SAME: 2}


def test_similarity_same_many_candidates_is_one_choice_batch() -> None:
    """H23 (Ryosuke, 2026-09-20): K_SAME over > 1 candidates is ONE Choice
    request per batch (spec 0042), counted under K_SAME, state = the query."""
    jev = FakeJev()
    sim = JevSimilarityJudge(jev)
    cands = ["Hackathon planning", "Server configuration", "Scale decision"]
    assert sim.most_similar("Server facts", cands, K_SAME) == (1, 1.0)
    assert len(jev.requests) == 1
    state, questions = jev.requests[0]
    assert state == "Server facts" and list(questions) == ["same"]
    assert list(questions["same"]["criteria"]) == ["m0", "m1", "m2", "none"]
    assert questions["same"]["criteria"]["m2"] == "Scale decision"
    assert sim.most_similar("nothing matches here", cands, K_SAME) == (-1, 0.0)
    assert sim.runner.calls == {K_SAME: 2} and sim.runner.unparsed == {}


def test_similarity_300_same_candidates_are_two_choice_batches_in_order() -> None:
    cands = [_alpha_name("Matter", i) for i in range(300)]
    jev = FakeJev()
    sim = JevSimilarityJudge(jev)
    assert sim.most_similar("about %s here" % cands[260], cands, K_SAME) == (260, 1.0)
    assert [list(q) for _, q in jev.requests] == [["same"], ["same"]]
    first, second = (q["same"]["criteria"] for _, q in jev.requests)
    assert list(first) == ["m%d" % i for i in range(254)] + ["none"]
    assert list(second) == ["m%d" % i for i in range(254, 300)] + ["none"]
    assert first["m0"] == cands[0] and second["m299"] == cands[299]
    jev2 = FakeJev()
    assert JevSimilarityJudge(jev2).most_similar("about %s here" % cands[7], cands, K_SAME) == (7, 1.0)
    assert len(jev2.requests) == 1
    jev3 = FakeJev()
    sim3 = JevSimilarityJudge(jev3)
    assert sim3.most_similar("nothing matches", cands, K_SAME) == (-1, 0.0)
    assert len(jev3.requests) == 2 and sim3.runner.calls == {K_SAME: 2}


def test_same_choice_with_an_unoffered_key_stops_under_k_same() -> None:
    bad = FakeJev(override={"same": lambda s, q: choice_answer("m99", list(q["criteria"]))})
    sim = JevSimilarityJudge(bad)
    with pytest.raises(JevStop):
        sim.most_similar("q", ["Item a", "Item b"], K_SAME)
    assert sim.runner.unparsed == {K_SAME: 1}


def test_similarity_belongs_single_candidate_uses_noul() -> None:
    jev = FakeJev()
    sim = JevSimilarityJudge(jev)
    assert sim.most_similar("Server port is 25565", ["Server topic"], K_BELONGS) == (0, 1.0)
    assert list(jev.requests[0][1]) == ["belongs"]
    assert sim.most_similar("unrelated words", ["Server topic"], K_BELONGS) == (-1, 0.0)
    assert sim.runner.calls == {K_BELONGS: 2}


def test_similarity_belongs_many_candidates_is_one_sun_choice() -> None:
    jev = FakeJev()
    suns = ["Hackathon planning", "Server configuration", "Scale decision"]
    sim = JevSimilarityJudge(jev, sun_texts_fn=lambda: set(suns))
    assert sim.most_similar("The Server runs Paper", suns, K_BELONGS) == (1, 1.0)
    assert len(jev.requests) == 1
    state, questions = jev.requests[0]
    assert state == "The Server runs Paper" and list(questions) == ["sun"]
    assert set(questions["sun"]["criteria"]) == {"s0", "s1", "s2", "new_topic"}
    assert sim.most_similar("nothing matches here", suns, K_BELONGS) == (-1, 0.0)
    assert sim.runner.calls == {K_BELONGS: 2}
    assert sim.runner.defaulted == {}


def test_similarity_300_suns_are_two_choice_batches_in_order() -> None:
    """H22 (a): > 254 suns no longer stop the run; the candidates are asked in
    batches of <= 254 (candidate order), the first non-new_topic answer wins
    and its key maps back to the GLOBAL candidate index."""
    suns = [_alpha_name("Topic", i) for i in range(300)]
    jev = FakeJev()
    sim = JevSimilarityJudge(jev, sun_texts_fn=lambda: set(suns))
    assert sim.most_similar("about %s here" % suns[260], suns, K_BELONGS) == (260, 1.0)
    assert [list(q) for _, q in jev.requests] == [["sun"], ["sun"]]
    first, second = (q["sun"]["criteria"] for _, q in jev.requests)
    assert list(first) == ["s%d" % i for i in range(254)] + ["new_topic"]
    assert list(second) == ["s%d" % i for i in range(254, 300)] + ["new_topic"]
    assert first["s0"] == suns[0] and second["s299"] == suns[299]
    # a hit in the first batch stops there
    jev2 = FakeJev()
    sim2 = JevSimilarityJudge(jev2, sun_texts_fn=lambda: set(suns))
    assert sim2.most_similar("about %s here" % suns[7], suns, K_BELONGS) == (7, 1.0)
    assert len(jev2.requests) == 1
    # no hit anywhere -> every batch asked, (-1, 0.0)
    jev3 = FakeJev()
    sim3 = JevSimilarityJudge(jev3, sun_texts_fn=lambda: set(suns))
    assert sim3.most_similar("nothing matches", suns, K_BELONGS) == (-1, 0.0)
    assert len(jev3.requests) == 2 and sim3.runner.calls == {K_BELONGS: 2}
    assert sim3.runner.unparsed == {}


def test_similarity_unknown_choice_or_missing_field_stops() -> None:
    bad = FakeJev(override={"sun": lambda s, q: choice_answer("s99", list(q["criteria"]))})
    with pytest.raises(JevStop):
        JevSimilarityJudge(bad, sun_texts_fn=lambda: {"a", "b"}).most_similar(
            "q", ["a", "b"], K_BELONGS
        )
    missing = FakeJev(override={"same": lambda s, q: {"type": "noul"}})
    sim = JevSimilarityJudge(missing)
    with pytest.raises(JevStop):
        sim.most_similar("q", ["a"], K_SAME)
    assert sim.runner.unparsed == {K_SAME: 1}


def test_shared_counters_and_usage_accumulate() -> None:
    counters = JevCounters()
    jev = FakeJev()
    node_fn = JevNodeFn(jev, FakeSummarizer(), counters)
    JevKeepFn(node_fn)("fact")
    node_fn("fact")
    JevSimilarityJudge(jev, counters).most_similar("fact", ["fact"], K_SAME)
    assert counters.calls == {K_NODE: 1, K_SAME: 1}
    assert counters.total_calls() == 2
    assert counters.input_tokens == sum(100 + len(s) // 4 for s, _ in jev.requests)
    assert counters.output_tokens == 24
    assert counters.total_defaulted() == 0


def test_judge_llm_never_completes() -> None:
    with pytest.raises(NotImplementedError):
        JevJudgeLLM().complete("any prompt", max_tokens=8)


def test_drop_tag_is_not_kept() -> None:
    assert JevKeepFn(JevNodeFn(FakeJev(), FakeSummarizer()))(f"{TAG_DROP} raw log line") is False


# -- H22 (a) (replaces H8): planet candidates are matched with Choice batches too ----


def test_belongs_with_300_planet_candidates_is_two_planet_choice_batches() -> None:
    """GraphMerger attach over ALL planets (0060): ONE Choice per batch of
    <= 254 candidates with the planet wording, never one Noul per candidate."""
    planets = [_alpha_name("Item", i) for i in range(300)]
    jev = FakeJev()
    sim = JevSimilarityJudge(jev, sun_texts_fn=lambda: {"Only sun"})
    assert sim.most_similar("a satellite fact", planets, K_BELONGS) == (-1, 0.0)
    assert [list(q) for _, q in jev.requests] == [["planet"], ["planet"]]
    assert all(s == "a satellite fact" for s, _ in jev.requests)  # the query is the state
    first, second = (q["planet"]["criteria"] for _, q in jev.requests)
    assert list(first) == ["p%d" % i for i in range(254)] + ["none"]
    assert list(second) == ["p%d" % i for i in range(254, 300)] + ["none"]
    assert second["p299"] == planets[299]
    assert sim.runner.calls == {K_BELONGS: 2}

    # the first batch whose answer is not none wins; a hit in batch 2 maps to the global index
    jev2 = FakeJev()
    sim2 = JevSimilarityJudge(jev2, sun_texts_fn=lambda: set())
    assert sim2.most_similar("detail of %s" % planets[260], planets, K_BELONGS) == (260, 1.0)
    assert len(jev2.requests) == 2
    jev3 = FakeJev()
    sim3 = JevSimilarityJudge(jev3)
    assert sim3.most_similar("detail of %s" % planets[7], planets, K_BELONGS) == (7, 1.0)
    assert len(jev3.requests) == 1


def test_planet_choice_with_an_unoffered_key_stops() -> None:
    bad = FakeJev(override={"planet": lambda s, q: choice_answer("p99", list(q["criteria"]))})
    sim = JevSimilarityJudge(bad)
    with pytest.raises(JevStop):
        sim.most_similar("q", ["Item a", "Item b"], K_BELONGS)
    assert sim.runner.unparsed == {K_BELONGS: 1}


def test_belongs_with_all_sun_candidates_is_exactly_one_choice() -> None:
    jev = FakeJev()
    suns = ["Hackathon planning", "Server configuration"]
    sim = JevSimilarityJudge(jev, sun_texts_fn=lambda: set(suns))
    assert sim.most_similar("The Server runs Paper", suns, K_BELONGS) == (1, 1.0)
    assert len(jev.requests) == 1 and list(jev.requests[0][1]) == ["sun"]


def test_belongs_with_mixed_candidates_takes_the_planet_wording() -> None:
    jev = FakeJev()
    sim = JevSimilarityJudge(jev, sun_texts_fn=lambda: {"Hackathon planning"})
    mixed = ["Hackathon planning", "Server configuration"]  # second is not a sun
    assert sim.most_similar("Server facts", mixed, K_BELONGS) == (1, 1.0)
    assert [list(q) for _, q in jev.requests] == [["planet"]]


def test_belongs_without_sun_texts_fn_uses_the_planet_wording() -> None:
    jev = FakeJev()
    sim = JevSimilarityJudge(jev)
    assert sim.most_similar("Server facts", ["Hackathon planning", "Server configuration"], K_BELONGS) == (1, 1.0)
    assert [list(q) for _, q in jev.requests] == [["planet"]]
    # a single candidate still takes the `belongs` Noul
    assert sim.most_similar("Server facts", ["Server configuration"], K_BELONGS) == (0, 1.0)
    assert list(jev.requests[-1][1]) == ["belongs"]


# -- item 6a (Astra round 3): the EXACT question dictionaries of DESIGN.md 3 ---------

# Literal copies of the DESIGN.md 3 question texts (ASCII hyphens in `keep`).
DESIGN_Q_KEEP = {
    "type": "noul",
    "instructions": (
        "Does this excerpt contain information worth remembering later - a fact, value, "
        "setting, result, decision, definition, instruction or requirement - even if it "
        "appears inside code, command output or logs? Answer no only for content with no "
        "lasting information (boilerplate, progress noise, repeated listings)."
    ),
    "criteria": {
        "true": "Contains at least one concrete fact, value, decision or instruction worth recalling later",
        "false": "No lasting information: boilerplate, progress noise, or repetition",
    },
}
DESIGN_Q_COMPREHENSIVENESS = {
    "type": "score",
    "instructions": "How broad is the matter this excerpt states?",
    "criteria": [
        "A single detail of something larger",
        "A minor point",
        "A self-standing point",
        "A major theme with several parts",
        "The overarching topic of a whole discussion",
    ],
}
DESIGN_Q_INDEPENDENCE = {
    "type": "score",
    "instructions": "Can this excerpt be understood on its own?",
    "criteria": [
        "Meaningless without its surrounding context",
        "Mostly dependent on context",
        "Partly self-contained",
        "Mostly self-contained",
        "Fully self-contained",
    ],
}
DESIGN_Q_DETAIL = {
    "type": "score",
    "instructions": "How specific is this excerpt?",
    "criteria": ["Very general", "General", "Moderately specific", "Specific", "A precise concrete detail"],
}
DESIGN_Q_SUN = {
    "type": "choice",
    "instructions": "Which existing topic does this excerpt belong to? Pick new_topic if none fits.",
    "criteria": {
        "s0": "Hackathon planning",
        "s1": "Server configuration",
        "new_topic": "None of the listed topics",
    },
}
DESIGN_Q_PLANET = {
    "type": "choice",
    "instructions": (
        "Which of the listed items is this excerpt a detail or sub-point of? Pick none if it "
        "belongs under none of them."
    ),
    "criteria": {
        "p0": "Hackathon RCON port is 25575",
        "p1": "Server runs Paper 1.21.8",
        "none": "None of the listed items",
    },
}
DESIGN_Q_SAME = {"type": "noul", "instructions": "Do A and B state the same matter?"}
DESIGN_Q_SAME_CHOICE = {
    "type": "choice",
    "instructions": (
        "Which of the listed items states the same matter as this excerpt? Pick none if "
        "no item states the same matter."
    ),
    "criteria": {
        "m0": "Hackathon RCON port is 25575",
        "m1": "Server runs Paper 1.21.8",
        "none": "No listed item states the same matter",
    },
}
DESIGN_Q_BELONGS = {
    "type": "noul",
    "instructions": "Is A a detail or sub-point that belongs under topic B?",
}


def test_classification_request_is_exactly_the_design_questions() -> None:
    jev = FakeJev()
    JevKeepFn(JevNodeFn(jev, FakeSummarizer()))("raw chunk text")
    assert jev.requests == [
        (
            "raw chunk text",
            {
                "keep": DESIGN_Q_KEEP,
                "comprehensiveness": DESIGN_Q_COMPREHENSIVENESS,
                "independence": DESIGN_Q_INDEPENDENCE,
                "detail": DESIGN_Q_DETAIL,
            },
        )
    ]


def test_sun_same_and_belongs_requests_are_exactly_the_design_questions() -> None:
    jev = FakeJev()
    suns = ["Hackathon planning", "Server configuration"]
    sim = JevSimilarityJudge(jev, sun_texts_fn=lambda: set(suns))
    sim.most_similar("The Server runs Paper", suns, K_BELONGS)
    sim.most_similar("Hackathon RCON port is 25575", ["Hackathon planning"], K_BELONGS)
    sim.most_similar("Scale is two to one", ["Scale is 2:1"], K_SAME)
    sim.most_similar(
        "RCON listens on 25575", ["Hackathon RCON port is 25575", "Server runs Paper 1.21.8"], K_BELONGS
    )
    assert jev.requests == [
        ("The Server runs Paper", {"sun": DESIGN_Q_SUN}),
        ("A: Hackathon RCON port is 25575\nB: Hackathon planning", {"belongs": DESIGN_Q_BELONGS}),
        ("A: Scale is two to one\nB: Scale is 2:1", {"same": DESIGN_Q_SAME}),
        ("RCON listens on 25575", {"planet": DESIGN_Q_PLANET}),
    ]


def test_same_choice_request_is_exactly_the_design_question() -> None:
    """H23: the `same` Choice batch, literal dict (DESIGN.md 3)."""
    jev = FakeJev()
    sim = JevSimilarityJudge(jev)
    sim.most_similar(
        "RCON listens on 25575", ["Hackathon RCON port is 25575", "Server runs Paper 1.21.8"], K_SAME
    )
    assert jev.requests == [("RCON listens on 25575", {"same": DESIGN_Q_SAME_CHOICE})]
    from benchmark.mcbuild_bench.jev_judge import same_question

    assert same_question(["a", "b"], 5) == {
        "type": "choice",
        "instructions": DESIGN_Q_SAME_CHOICE["instructions"],
        "criteria": {"m5": "a", "m6": "b", "none": "No listed item states the same matter"},
    }
    with pytest.raises(ValueError):
        same_question(["x"] * 255)


def test_design_md_keep_instruction_uses_the_ascii_hyphen() -> None:
    """DESIGN.md 3 and Q_KEEP must agree byte for byte (the em-dash was replaced);
    H22 (e): the wording keeps facts found inside tool output."""
    from pathlib import Path

    design = Path("benchmark/mcbuild_bench/DESIGN.md").read_text(encoding="utf-8")
    flat = " ".join(design.split())
    assert Q_KEEP == DESIGN_Q_KEEP
    assert Q_KEEP["instructions"] in flat
    assert Q_KEEP["criteria"]["true"] in flat and Q_KEEP["criteria"]["false"] in flat
    assert "worth keeping for later —" not in flat
    assert "rather than code, tool output" not in Q_KEEP["instructions"]
    # the planet wording and the batch rule are documented too
    from benchmark.mcbuild_bench.jev_judge import Q_PLANET_INSTRUCTIONS

    assert Q_PLANET_INSTRUCTIONS in flat
    from benchmark.mcbuild_bench.jev_judge import Q_SAME_CHOICE_INSTRUCTIONS

    assert Q_SAME_CHOICE_INSTRUCTIONS in flat


# -- item 1 (Astra round 3): worst-case request projection ----------------------------


def test_project_requests_formula() -> None:
    """H22 (a) + H23 cost model: every `belongs` AND every `same` decision over
    k candidates costs ceil(k / 254) Choice requests (1 Noul when k == 1, which
    is also one request)."""
    from benchmark.mcbuild_bench.jev_judge import choice_batches, within_turn_requests

    assert [choice_batches(k) for k in (0, 1, 254, 255, 508, 509)] == [0, 1, 1, 2, 2, 3]
    # within-turn (build_provisional): S provisional suns, P planets, T satellites of
    # the turn; planets ask ceil(S/254) each, satellites ceil(P/254) each, maximised.
    assert within_turn_requests(0) == 0 and within_turn_requests(1) == 0
    assert within_turn_requests(2) == 1      # S=1, P=1
    assert within_turn_requests(4) == 3      # S=1, P=1, T=2
    assert within_turn_requests(300) == 343  # S=1, P=255, T=44: 255*1 + 44*2
    assert project_requests(0, 0, 0) == 0
    # one chunk on an empty CD: 1 classification + 0 within-turn +
    #   1 * (same: ceil(1/254) + belongs batches: ceil(1/254) planets + ceil(1/254) suns)
    assert project_requests(1, 0, 0) == 1 + 0 + 1 * (1 + 1 + 1)
    # 4 chunks, CD with 2 suns / 3 planets / 5 satellites:
    #   4 classification + 3 within-turn + 4 * (ceil(14/254) + ceil(7/254) + ceil(6/254))
    assert project_requests(4, 2, 3, n_satellites=5) == 4 + 3 + 4 * (1 + 1 + 1)
    # 300 planets in the CD: `same` over up to 302 candidates is 2 batches, not 302 Nouls;
    # the orphan planet/satellite `belongs` is 2 batches, not 300 Nouls
    assert project_requests(1, 1, 300) == 1 + 0 + 1 * (2 + 2 + 1)
    with pytest.raises(ValueError):
        project_requests(-1, 0, 0)


# -- H23: input-token / USD estimate (an ESTIMATE, not the budget guard) --------------


def test_estimate_input_tokens_arithmetic_on_small_numbers() -> None:
    from benchmark.mcbuild_bench.jev_judge import estimate_input_tokens

    est = estimate_input_tokens(
        4, 0.5, avg_option_tokens=10, avg_chunk_tokens=100, question_tokens=50,
        instruction_tokens=20, n_round_trips=1,
    )
    assert est["estimate"] is True and est["kept_nodes"] == 2
    t = est["terms"]
    # classification: 4 requests x (100 chunk + 50 question) tokens
    assert t["classification"] == {"requests": 4, "input_tokens": 600}
    # within-turn: 2 kept nodes in one turn, each one decision over ~1 candidate
    #   -> 2 requests x (20 instruction + 10 state + 1 * 10 option)
    assert t["within_turn"] == {"requests": 2, "input_tokens": 80}
    # same: node 0 faces 0 candidates (no request), node 1 faces 1 -> 1 request
    #   x (20 + 10 + 1 * 10)
    assert t["same"] == {"requests": 1, "input_tokens": 40}
    # belongs: node 0 -> 0; node 1 faces 1 candidate split over sun + planet decisions
    #   -> 2 requests: ceil(1/2)=1 candidate each -> 2 x (20 + 10 + 10)
    assert t["belongs"] == {"requests": 2, "input_tokens": 80}
    assert est["input_tokens_total"] == 600 + 80 + 40 + 80
    assert est["requests_total"] == 4 + 2 + 1 + 2
    assert est["price_per_mtok_usd"] == 0.042
    assert est["usd"] == pytest.approx(800 * 0.042 / 1_000_000)
    # no kept node -> only the classification term costs anything
    zero = estimate_input_tokens(3, 0.0, avg_chunk_tokens=10, question_tokens=0)
    assert zero["kept_nodes"] == 0 and zero["input_tokens_total"] == 30
    with pytest.raises(ValueError):
        estimate_input_tokens(3, 1.5)
    with pytest.raises(ValueError):
        estimate_input_tokens(-1, 0.5)


def test_unusable_summarizer_text_returns_none_for_the_node_fallback_path() -> None:
    """H28: after its one retry the summarizer client raises
    SummarizerNodeTextUnusable; JevNodeFn returns None so SpecManager counts a
    node_fallback (truncated chunk text) instead of the run stopping.  Any other
    SummarizerStop (unreachable server) still propagates."""
    from benchmark.mcbuild_bench.errors import SummarizerNodeTextUnusable, SummarizerStop
    from benchmark.mcbuild_bench.jev_judge import JevNodeFn

    class Summ:
        def __init__(self, exc):
            self.exc = exc

        def summarize(self, text):
            raise self.exc

    jev = FakeJev()
    node_fn = JevNodeFn(jev, Summ(SummarizerNodeTextUnusable("too long twice")))
    node_fn.ask_chunk(f"{TAG_PLANET} chunk A")
    assert node_fn(f"{TAG_PLANET} chunk A") is None
    assert node_fn.summarizer_calls == 1

    node_fn = JevNodeFn(jev, Summ(SummarizerStop("HTTP 500")))
    node_fn.ask_chunk(f"{TAG_PLANET} chunk B")
    with pytest.raises(SummarizerStop):
        node_fn(f"{TAG_PLANET} chunk B")

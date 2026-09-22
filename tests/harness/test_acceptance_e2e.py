"""Fable acceptance test: every harness step with a semantic FakeJudge.

Unlike the driver's all-"no" fake judge, this judge answers from simple,
deterministic rules so that a two-turn Japanese conversation exercises
extraction, faithfulness, all three classification axes, provisional linking,
case-1 merge (vanish + attach), case-2/3 orphan handling, and 0062 mass counting.
shortlist_k=0 so no embedding model is ever loaded.
"""

from __future__ import annotations

import json
import re

from management.harness import FakeJudge, HarnessConfig, HarnessManager
from management.harness.backends import first_fenced_span, strip_think_blocks
from management.harness.prompts import FENCE_CLOSE, FENCE_OPEN
from models.correlation_diagram import CorrelationDiagram
from models.node import NodeLevel

_FENCES = re.compile(
    re.escape(FENCE_OPEN) + r"\n(.*?)\n" + re.escape(FENCE_CLOSE), re.DOTALL
)

TOPIC = "夕食の献立についての相談"
PILLAR_A = "ユーザーは甲殻類にアレルギーがある"
PILLAR_B = "ユーザーの冷蔵庫には鶏肉がある"
DETAIL_A1 = "アレルギーの症状は発疹が出ることである"
DETAIL_B1 = "鶏肉は300グラムある"
DETAIL_B2 = "鶏肉の消費期限は9月8日である"


def _policy(prompt: str) -> str | None:
    spans = _FENCES.findall(prompt)
    if "JSON array" in prompt:
        # Extraction: one statement per sentence of the chunk.
        text = first_fenced_span(prompt)
        sents = [s.strip() for s in text.split("。") if s.strip()]
        return json.dumps(sents, ensure_ascii=False)
    if "supported by the source" in prompt:
        return "yes" if "でたらめ" not in spans[0] else "no"
    if "title or the summary" in prompt:
        return "yes" if "について" in spans[0] else "no"
    if "new pillar of the discussion" in prompt:
        s = spans[0]
        return "yes" if ("アレルギー" in s or "冷蔵庫" in s) and not re.search(r"\d", s) else "no"
    if "concrete step" in prompt:
        return "yes" if re.search(r"\d|症状", spans[0]) else "no"
    if "belong under this topic" in prompt:
        statement, topic = spans[0], spans[1]
        if "について" in topic:
            # a pillar belongs to the dinner topic only if it is about the dinner
            return "yes" if re.search(r"アレルギー|冷蔵庫|鶏肉|献立", statement) else "no"
        key = "アレルギー" if "アレルギー" in topic else "鶏肉" if "鶏肉" in topic else None
        return "yes" if key and key in statement else "no"
    if "state the same matter" in prompt:
        return "yes" if spans[0] == spans[1] else "no"
    if "Does the topic change" in prompt:
        return "yes"
    return "no"


def _manager() -> tuple[HarnessManager, FakeJudge]:
    judge = FakeJudge(policy=_policy)
    cfg = HarnessConfig(shortlist_k=0, boundary_check=True, faithfulness_check=True)
    return HarnessManager(judge, config=cfg), judge


def _planet_by_text(cd: CorrelationDiagram, text: str):
    for se in cd.suns:
        for pe in se.planets:
            if pe.planet.text == text:
                return pe
    raise AssertionError(f"planet not found: {text}")


def test_two_turn_conversation_builds_spec_tree() -> None:
    mgr, judge = _manager()
    base = CorrelationDiagram()

    # Turn 1: topic + two pillars + one detail, plus an unsupported statement.
    t1 = f"{TOPIC}。{PILLAR_A}。{DETAIL_A1}。{PILLAR_B}。でたらめな文。"
    r1 = mgr.update(base, t1, "", turn=0)
    assert r1.statements == 5
    assert r1.dropped_unsupported == 1
    assert len(base.suns) == 1 and base.suns[0].sun.text == TOPIC
    planets = {pe.planet.text for pe in base.suns[0].planets}
    assert planets == {PILLAR_A, PILLAR_B}
    assert [s.text for s in _planet_by_text(base, PILLAR_A).satellites] == [DETAIL_A1]
    # 0062: planet mass = satellite count (min 1).
    assert _planet_by_text(base, PILLAR_A).planet.mass == 1.0
    assert _planet_by_text(base, PILLAR_B).planet.mass == 1.0
    assert mgr.normalize_calls == 1

    # Turn 2: repeats PILLAR_B (must vanish, 0056) and adds two details under it;
    # one orphan detail about the allergy links to the existing planet (0060).
    t2 = f"{PILLAR_B}。{DETAIL_B1}。{DETAIL_B2}。"
    r2 = mgr.update(base, t2, "", turn=1)
    assert len(base.suns) == 1  # no spurious promotion to sun
    pb = _planet_by_text(base, PILLAR_B)
    assert {s.text for s in pb.satellites} == {DETAIL_B1, DETAIL_B2}
    assert pb.planet.mass == 2.0  # 0062 satellite count
    assert _planet_by_text(base, PILLAR_A).planet.mass == 1.0
    assert r2.merged >= 1  # PILLAR_B vanished into the existing planet
    assert mgr.normalize_calls == 2

    # Orphan satellite with no matching planet/sun -> promoted to sun (0061 last branch).
    r3 = mgr.update(base, "電車は10分遅れた。", "", turn=2)
    assert r3.promoted == 1
    assert any(se.sun.text == "電車は10分遅れた" for se in base.suns)
    assert all(se.sun.level is NodeLevel.SUN for se in base.suns)

    # Every prompt is a single question with the fenced source present.
    assert all(FENCE_OPEN in p and FENCE_CLOSE in p for p in judge.prompts)
    kinds = set(mgr.totals)
    assert {"extract", "supported", "comprehensive", "independent", "detail", "belongs", "same"} <= kinds


def test_strip_think_blocks() -> None:
    assert strip_think_blocks("<think>long reasoning</think>\nyes") == "yes"
    assert strip_think_blocks("yes") == "yes"
    # H5 (behaviour change): an UNTERMINATED <think> means the answer was never
    # emitted, so the whole reply is discarded instead of reading an answer out
    # of the reasoning trace.
    assert strip_think_blocks("<think>\nunterminated\n\nno") == ""
    assert strip_think_blocks("<think>only reasoning") == ""
    # H5: a closing tag without an opening one (the template ate the opener) --
    # drop everything up to and including the LAST </think>.
    assert strip_think_blocks("reasoning\n</think>\nno") == "no"
    assert strip_think_blocks("a</think>b</think>\nyes") == "yes"

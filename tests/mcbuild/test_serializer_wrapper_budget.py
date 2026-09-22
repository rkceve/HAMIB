"""Item 7 (Astra round 3): to_context_block_budgeted refuses a budget below the wrapper cost."""

from __future__ import annotations

import pytest

from communication.cd_serializer import CDSerializer
from models.correlation_diagram import CorrelationDiagram
from models.node import Node, NodeLevel


def _cd() -> CorrelationDiagram:
    cd = CorrelationDiagram()
    sun = Node(text="topic", level=NodeLevel.SUN, mass=1.0)
    cd.add_sun(sun)
    cd.add_planet(Node(text="fact", level=NodeLevel.PLANET, mass=2.0), sun.node_id)
    return cd


def _words(text: str) -> int:
    return len(text.split())


def test_budget_below_wrapper_cost_raises_with_both_numbers() -> None:
    ser = CDSerializer(level_markers=True)
    wrapper = _words("<CONTEXT>\n</CONTEXT>")
    assert wrapper == 2
    with pytest.raises(ValueError, match=r"budget_tokens=1 .*wrapper.*2"):
        ser.to_context_block_budgeted(_cd(), 1, "mass", _words)
    with pytest.raises(ValueError):
        ser.to_context_block_budgeted(_cd(), 0, "recency", _words)


def test_budget_equal_to_wrapper_cost_returns_the_bare_wrapper() -> None:
    ser = CDSerializer(level_markers=True)
    assert ser.to_context_block_budgeted(_cd(), 2, "mass", _words) == "<CONTEXT>\n</CONTEXT>"
    assert ser.to_context_block_budgeted(_cd(), 4, "mass", _words) == "<CONTEXT>\n[SN] topic\n</CONTEXT>"

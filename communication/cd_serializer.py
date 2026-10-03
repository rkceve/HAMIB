"""CDSerializer: turn a CorrelationDiagram into the text block placed at the
top of the prompt, and into the JSON payload for the server.

Default format: one node per line, indented by level, each prefixed with a
"[PN{mass}]" token that the server finds to build the mass matrix.
  <CONTEXT>
  [PN10.0] machine learning
    [PN5.0] neural networks
      [PN1.0] backpropagation
  </CONTEXT>

With ``level_markers`` only planets carry a mass, because the spec defines
mass for planets only:
  [SN] sun text / [PN{mass}] planet text / [RN] satellite text
(SN / PN / RN are the spec's labels for sun, planet and satellite nodes.)
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Callable, Iterable, Literal

from models.correlation_diagram import CorrelationDiagram
from models.node import Node
from utils.config import get


@dataclass
class _NodeRecord:
    """One node in diagram order, with what the budgeted serializer needs."""
    node: Node
    parent_id: str | None
    indent: int
    order_index: int


class CDSerializer:
    def __init__(self, level_markers: bool | None = None):
        """level_markers: True for the "[SN] / [PN{mass}] / [RN]" format, False
        for "[PN{mass}]" on every line, None to read config
        ``tokenization.level_markers`` (default False)."""
        self._prefix: str = get("tokenization", "prefix", "[PN")
        self._suffix: str = get("tokenization", "suffix", "]")
        self._precision: int = get("tokenization", "mass_precision", 1)
        self._level_markers: bool = (
            bool(get("tokenization", "level_markers", False))
            if level_markers is None
            else level_markers
        )

    def to_context_block(self, cd: CorrelationDiagram) -> str:
        """Return the whole diagram as a <CONTEXT>...</CONTEXT> block."""
        return self._render(self._records(cd))

    def _node_line(self, text: str, mass: float, indent: int) -> str:
        if self._level_markers:
            if indent == 0:
                token = "[SN]"
            elif indent == 1:
                token = f"[PN{round(mass, self._precision)}]"
            else:
                token = "[RN]"
        else:
            token = f"{self._prefix}{round(mass, self._precision)}{self._suffix}"
        return "  " * indent + f"{token} {text}"

    def _render(self, records: Iterable[_NodeRecord]) -> str:
        lines = ["<CONTEXT>"]
        lines += [self._node_line(r.node.text, r.node.mass, r.indent) for r in records]
        lines.append("</CONTEXT>")
        return "\n".join(lines)

    @staticmethod
    def _records(cd: CorrelationDiagram) -> list[_NodeRecord]:
        """Every node in diagram order (sun, its planets, each planet's satellites)."""
        records: list[_NodeRecord] = []
        for se in cd.suns:
            records.append(_NodeRecord(se.sun, None, 0, len(records)))
            for pe in se.planets:
                records.append(_NodeRecord(pe.planet, se.sun.node_id, 1, len(records)))
                for sat in pe.satellites:
                    records.append(_NodeRecord(sat, pe.planet.node_id, 2, len(records)))
        return records

    # ── budgeted serialization ─────────────────────────────────────────

    def to_context_block_budgeted(
        self,
        cd: CorrelationDiagram,
        budget_tokens: int,
        policy: Literal["mass", "random", "recency"],
        token_counter: Callable[[str], int],
        seed: int = 0,
    ) -> str:
        """Like ``to_context_block``, but keeps only as many nodes as fit in
        ``budget_tokens``, counted by ``token_counter`` over the whole block
        (wrapper lines and indentation included).

        ``policy`` sets which nodes are kept first:
          "mass":    highest mass first; ties: newer created_turn, then diagram order
          "recency": newest created_turn first; ties: diagram order
          "random":  a fresh shuffle each pass, from random.Random(seed)

        A node can only be kept if its parent is kept. Each pass keeps the
        first candidate, in policy order, that still fits, then starts over,
        since keeping a node makes its children candidates. Stops when
        nothing more fits. Kept nodes are written in diagram order, so with
        enough budget the result equals ``to_context_block(cd)`` for every
        policy.

        Raises ValueError for an unknown policy, or if the budget is smaller
        than the empty <CONTEXT></CONTEXT> wrapper.
        """
        if policy not in ("mass", "random", "recency"):
            raise ValueError(
                f"policy must be one of 'mass','random','recency', got {policy!r}"
            )
        wrapper_tokens = token_counter(self._render([]))
        if budget_tokens < wrapper_tokens:
            raise ValueError(
                f"budget_tokens={budget_tokens} is smaller than the wrapper cost of "
                f"{wrapper_tokens} tokens (<CONTEXT> + </CONTEXT> lines): no block "
                "within the budget exists"
            )

        records = self._records(cd)

        # With level markers only planets have a mass, so rank by planet mass:
        # a planet by its own, a satellite by its planet's, a sun by its
        # largest planet's (0 if it has none). Otherwise rank by node.mass.
        priority: dict[str, float] = {}
        if self._level_markers:
            for se in cd.suns:
                for pe in se.planets:
                    priority[pe.planet.node_id] = pe.planet.mass
                    for sat in pe.satellites:
                        priority[sat.node_id] = pe.planet.mass
                priority[se.sun.node_id] = max([0.0] + [pe.planet.mass for pe in se.planets])

        kept_ids: set[str] = set()
        rng = random.Random(seed)

        def mass_key(r: _NodeRecord) -> float:
            return priority.get(r.node.node_id, r.node.mass)

        def fits(r: _NodeRecord) -> bool:
            return token_counter(self._emit_kept(records, kept_ids | {r.node.node_id})) <= budget_tokens

        while True:
            candidates = [
                r for r in records
                if r.node.node_id not in kept_ids
                and (r.parent_id is None or r.parent_id in kept_ids)
            ]
            if not candidates:
                break

            if policy == "random":
                rng.shuffle(candidates)
            elif policy == "mass":
                candidates.sort(key=lambda r: (-mass_key(r), -r.node.created_turn, r.order_index))
            else:  # recency
                candidates.sort(key=lambda r: (-r.node.created_turn, r.order_index))

            chosen = next((r for r in candidates if fits(r)), None)
            if chosen is None:
                break
            kept_ids.add(chosen.node.node_id)

        return self._emit_kept(records, kept_ids)

    def _emit_kept(self, records: list[_NodeRecord], keep: set[str]) -> str:
        """The block containing only the records whose node id is in ``keep``."""
        return self._render(r for r in records if r.node.node_id in keep)

    def to_api_payload(self, cd: CorrelationDiagram) -> dict:
        """Payload for the server: ``node_list`` (the nodes it builds the mass
        matrix from) and ``context_block``."""
        node_list = [
            {
                "node_id": n.node_id,
                "text": n.text,
                "level": n.level.value,
                "mass": n.mass,
                "token_repr": n.token_repr(self._precision),
            }
            for n in cd.all_nodes()
        ]
        return {
            "node_list": node_list,
            "context_block": self.to_context_block(cd),
        }

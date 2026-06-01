"""
CDSerializer: convert a CorrelationDiagram into a server-bound payload.

The "[PN{mass}] special token" scheme:
  Each node of the correlation diagram is embedded at the start of the
  prompt as text carrying a [PN{mass}] prefix. The server detects those
  token positions and builds the M matrix from them.

Format (block inserted at the start of the prompt):
  <CONTEXT>
  [PN10.0] Machine learning
    [PN5.0] Neural network
      [PN1.0] Backpropagation
  ...
  </CONTEXT>
"""
from __future__ import annotations

from models.correlation_diagram import CorrelationDiagram
from utils.config import get


class CDSerializer:
    def __init__(self):
        self._prefix: str = get("tokenization", "prefix", "[PN")
        self._suffix: str = get("tokenization", "suffix", "]")
        self._precision: int = get("tokenization", "mass_precision", 1)

    def to_context_block(self, cd: CorrelationDiagram) -> str:
        """
        Convert the CD into a <CONTEXT>...</CONTEXT> block string.
        Prepended to the prompt before sending.
        """
        lines = ["<CONTEXT>"]
        for se in cd.suns:
            lines.append(self._node_line(se.sun.text, se.sun.mass, indent=0))
            for pe in se.planets:
                lines.append(self._node_line(pe.planet.text, pe.planet.mass, indent=1))
                for sat in pe.satellites:
                    lines.append(self._node_line(sat.text, sat.mass, indent=2))
        lines.append("</CONTEXT>")
        return "\n".join(lines)

    def _node_line(self, text: str, mass: float, indent: int) -> str:
        token = f"{self._prefix}{round(mass, self._precision)}{self._suffix}"
        return "  " * indent + f"{token} {text}"

    def to_api_payload(self, cd: CorrelationDiagram) -> dict:
        """
        Payload sent to the server API.
        node_list: structured data the server uses to build the M matrix.
        """
        node_list = []
        for n in cd.all_nodes():
            node_list.append({
                "node_id": n.node_id,
                "text": n.text,
                "level": n.level.value,
                "mass": n.mass,
                "token_repr": n.token_repr(self._precision),
            })
        return {
            "node_list": node_list,
            "context_block": self.to_context_block(cd),
        }

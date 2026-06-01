"""
MMatrixBuilder: builds the M matrix (shape: seq_len x seq_len) from a list
of ParsedNode and a token sequence.

Status: UNUSED in the default inference pipeline. The live HAMIBSession path
uses a 1D mass vector instead (see HAMIBSession._build_mass_vector). This 2D
M-matrix builder is retained for short-context experiments; no caller invokes
set_m_matrix() anywhere in this repository.

M definition:
  - shape: (seq_len, seq_len)
  - starts as a zero matrix
  - mass is added along the column dimension (the attended-to side)
  - i.e. M[:, j] += mass  (j = position of a [PN{mass}] token)

Attention modification:
  scores += w * M
  (w is attention.mass_weight from config)
"""
from __future__ import annotations
import torch
from server.cd_parser import ParsedNode, find_pn_positions
from utils.config import get


class MMatrixBuilder:
    def __init__(self):
        self._mass_weight: float = get("attention", "mass_weight", 1.0)

    def build(
        self,
        seq_len: int,
        node_list: list[ParsedNode],
        input_ids: list[int],
        tokenizer,
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Returns M: FloatTensor of shape (seq_len, seq_len) on `device`.
        M[:, j] += mass for each [PN{mass}] token at position j.
        """
        M = torch.zeros(seq_len, seq_len, dtype=torch.float32, device=device)

        pn_positions = find_pn_positions(input_ids, tokenizer)
        for pos, mass in pn_positions:
            if pos < seq_len:
                M[:, pos] += mass * self._mass_weight

        return M

    def build_from_context_block(
        self,
        seq_len: int,
        input_ids: list[int],
        tokenizer,
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Build the M matrix by scanning the [PN{mass}] tokens in the context block.
        Simplified variant that does not require a node_list.
        """
        return self.build(seq_len, [], input_ids, tokenizer, device)

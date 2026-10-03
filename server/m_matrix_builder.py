"""
MMatrixBuilder: builds the M matrix (shape seq_len x seq_len) from a ParsedNode
list and the token sequence.

Patent definition:
  - M has shape (seq_len, seq_len)
  - it starts as a zero matrix
  - mass is added along the column (the attended-to side)
  - i.e. M[:, j] += mass  (j = position of the [PN{mass}] token)

Attention modification:
  scores += w * M
  (w is attention.mass_weight in config)
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

        DEAD CODE: experiment L showed that the 2D M matrix collapses to 0%, and
        the live path uses only the 1D mass vector (server/mass_vector.py).
        Nothing calls this class, but its double-multiplication bug is fixed.

        Fix: this used to store ``mass * self._mass_weight``, but build_mass_bias
        multiplies by ``mass_weight`` again in 2D mode, so the effective weight
        was w^2. M now holds the raw mass only; w is applied only in
        build_mass_bias.
        """
        M = torch.zeros(seq_len, seq_len, dtype=torch.float32, device=device)

        pn_positions = find_pn_positions(input_ids, tokenizer)
        for pos, mass in pn_positions:
            if pos < seq_len:
                M[:, pos] += mass

        return M

    def build_from_context_block(
        self,
        seq_len: int,
        input_ids: list[int],
        tokenizer,
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Build the M matrix by scanning the [PN{mass}] tokens in context_block.
        Simplified version that needs no node_list.
        """
        return self.build(seq_len, [], input_ids, tokenizer, device)

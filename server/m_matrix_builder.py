"""
MMatrixBuilder: ParsedNode リストとトークン列から
Mマトリクス（shape: seq_len × seq_len）を構築する。

特許の定義:
  - M の形状: (seq_len, seq_len)
  - ゼロ行列から始める
  - column 方向（見られる側 = attended-to）に mass を加算
  - つまり M[:, j] += mass  （j = [PN{mass}] トークンの位置）

Attention 修正式:
  scores += w * M
  （w は config の attention.mass_weight）
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

        DEAD CODE (2026-09-06): 実験Lで 2D M行列は 0% に崩壊することが確認され、
        本番経路は 1D マスベクトル (server/mass_vector.py) に一本化されている。
        このクラスはどこからも呼ばれていないが、 F3 で見つかった二重乗算の
        欠陥だけは直してある。

        F3 修正: 以前はここで ``mass * self._mass_weight`` としていたが、
        build_mass_bias が 2D モードで再度 ``mass_weight * M`` を掛けるため、
        実効の重みが w^2 になっていた。 M は素の mass だけを持ち、
        w の乗算は build_mass_bias 側にだけ存在するのが正しい。
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
        context_block 内の [PN{mass}] トークンをスキャンしてMマトリクスを構築。
        node_list が不要な簡易版。
        """
        return self.build(seq_len, [], input_ids, tokenizer, device)

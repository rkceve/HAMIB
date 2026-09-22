"""mass_vector: [PN{mass}] 位置リストから 1D マスベクトルを組み立てる共通ヘルパー。

D-3 (RESEARCH_PROGRAM.md §1.9) が要求する正規化/上限層をここに一本化する:

    value = min(cap, mass * scale)

呼び出し側 (server/cms_session.py, server/main.py) は以前それぞれ手書きの
ループで ``vec[pos] += mass`` していた。 上限がどこにも掛かっておらず、
同一位置に複数のマーカーが当たると値が加算で膨らむため D-1 (effective bias
<= ~3.0) を無音で踏み越えていた。 このモジュールはその 2 点を直す:

  * 上限: ``min(cap, mass * scale)``  (D-3)
  * 衝突: 加算ではなく ``max`` (同じ位置に 2 つのマーカーが来ても膨らまない)

cap / scale の既定値は config から読む:
    get("attention", "mass_cap", 3.0)
    get("attention", "mass_scale", 1.0)
"""
from __future__ import annotations

import torch


def positions_to_mass_vector(
    positions: list[tuple[int, float]],
    seq_len: int,
    *,
    cap: float,
    scale: float = 1.0,
    device: torch.device | str | None = None,
) -> torch.Tensor | None:
    """(位置, mass) のリストを 1D マスベクトル (shape: (seq_len,)) に変換する。

    Args:
        positions: ``find_pn_positions`` 等が返す (token_position, mass) のリスト。
        seq_len:   ベクトル長 (= プロンプトのトークン数)。
        cap:       D-3 の上限。 各要素は ``min(cap, mass * scale)`` になる。
        scale:     mass に掛ける正規化係数。
        device:    生成先デバイス。

    Returns:
        ``positions`` が空のときだけ None。 それ以外は必ず Tensor を返す
        (範囲外の位置しかない場合はゼロベクトル)。

    同一位置に複数のマーカーが当たった場合は **max** を取る (加算しない)。
    加算すると上限 cap を超えてしまい D-1 を破るため。
    """
    if not positions:
        return None

    # F8 (perf): build on the CPU, then move once.  Assigning element by element
    # into a CUDA tensor costs one host<->device synchronisation PER POSITION
    # (the ``float(vec[pos])`` read is itself a device->host copy), and the
    # reader does this for every question of every cell.  The values are
    # identical: same cap, same scale, same max-on-collision rule.
    values: dict[int, float] = {}
    for pos, mass in positions:
        if 0 <= pos < seq_len:
            value = min(cap, float(mass) * scale)
            if value > values.get(pos, 0.0):
                values[pos] = value

    vec = torch.zeros(seq_len, dtype=torch.float32)
    if values:
        idx = torch.tensor(sorted(values), dtype=torch.long)
        vec[idx] = torch.tensor(
            [values[i] for i in sorted(values)], dtype=torch.float32
        )
    return vec.to(device) if device is not None else vec

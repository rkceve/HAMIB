"""Build the 1D mass vector from ``(token_position, mass)`` pairs.

Each entry is ``min(cap, mass * scale)``. When several markers hit the same
position the largest value wins; summing them could push it past the cap.
Callers read cap and scale from config:
``get("attention", "mass_cap", 3.0)`` and ``get("attention", "mass_scale", 1.0)``.
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
    """Return a float32 vector of shape ``(seq_len,)`` holding the mass at each position.

    Each value is ``min(cap, mass * scale)``; positions outside ``[0, seq_len)``
    are ignored and on a collision the largest value wins. Returns None only
    when ``positions`` is empty (out-of-range positions alone give a zero
    vector).
    """
    if not positions:
        return None

    # Build on the CPU and move once at the end: writing element by element
    # into a CUDA tensor would cost one host-device sync per position.
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

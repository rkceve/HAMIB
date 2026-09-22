"""pytest wrapper around verify_attention_math.py (one test per check).

The checks themselves live in verify_attention_math.py so that the same code can be
run standalone (`python verify_attention_math.py`) and under pytest.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import verify_attention_math as vam  # noqa: E402


@pytest.mark.parametrize("check", vam.CHECKS, ids=[c.__name__ for c in vam.CHECKS])
def test_check(check) -> None:
    check()


def test_all_checks_registered() -> None:
    # 10 original checks + 8b (F1 bool mask) + 11-16 (real closure, sliding guard,
    # mass-vector cap, spans/levels)
    assert len(vam.CHECKS) == 17
    names = {c.__name__ for c in vam.CHECKS}
    for required in (
        "check_8b_bool_mask_conversion",
        "check_11_closure_bool_mask_decode",
        "check_12_closure_scale_and_gqa",
        "check_13_closure_float_mask_and_counters",
        "check_14_sliding_window_alignment",
        "check_15_mass_vector_cap",
        "check_16_spans_and_levels",
    ):
        assert required in names, required


def test_main_returns_zero() -> None:
    assert vam.main() == 0


def test_main_does_not_leak_deterministic_algorithms() -> None:
    """F9(d): the global torch determinism flag must be restored after main()."""
    before = torch.are_deterministic_algorithms_enabled()
    assert vam.main() == 0
    assert torch.are_deterministic_algorithms_enabled() == before


def test_sdpa_is_restored_after_the_checks() -> None:
    """Every check that patches sdpa restores it in a finally block."""
    import torch.nn.functional as F

    for check in vam.CHECKS:
        check()
    assert F.scaled_dot_product_attention is torch._C._nn.scaled_dot_product_attention

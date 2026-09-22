"""gpu_sampler.energy_joules on a hand-written nvidia-smi CSV fixture."""

from __future__ import annotations

import time

import pytest

from benchmark.mcbuild_bench.gpu_sampler import (
    NVIDIA_SMI_CMD,
    energy_joules,
    parse_samples,
    parse_timestamp,
    wait_for_samples,
)

HEADER = "timestamp, utilization.gpu [%], utilization.memory [%], memory.used [MiB], power.draw [W]"
ROWS = [
    ("2026/09/17 12:00:00.000", 100.0),
    ("2026/09/17 12:00:01.000", 200.0),
    ("2026/09/17 12:00:02.000", 300.0),
    ("2026/09/17 12:00:03.000", 100.0),
    ("2026/09/17 12:00:04.500", 100.0),
]


def _write(tmp_path, rows=ROWS, extra_lines=()):
    lines = [HEADER]
    for ts, p in rows:
        lines.append("%s, 87 %%, 40 %%, 20000 MiB, %.2f W" % (ts, p))
    lines.extend(extra_lines)
    path = tmp_path / "gpu.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_energy_matches_hand_computed_trapezoid(tmp_path) -> None:
    path = _write(tmp_path)
    t0 = parse_timestamp(ROWS[0][0])
    t1 = parse_timestamp(ROWS[-1][0])
    # trapezoids: (100+200)/2*1 + (200+300)/2*1 + (300+100)/2*1 + (100+100)/2*1.5
    #           = 150 + 250 + 200 + 150 = 750 J
    assert energy_joules(path, t0, t1) == pytest.approx(750.0, abs=1e-6)


def test_energy_uses_only_samples_inside_the_span(tmp_path) -> None:
    path = _write(tmp_path)
    t0 = parse_timestamp(ROWS[1][0])
    t1 = parse_timestamp(ROWS[3][0])
    # (200+300)/2 + (300+100)/2 = 250 + 200
    assert energy_joules(path, t0, t1) == pytest.approx(450.0, abs=1e-6)


def test_uncovered_span_raises(tmp_path) -> None:
    path = _write(tmp_path, rows=ROWS[:1])
    t0 = parse_timestamp(ROWS[0][0])
    with pytest.raises(ValueError):
        energy_joules(path, t0 - 1, t0 + 1)
    # a span that starts before the first sample is not extrapolated
    full = _write(tmp_path)
    first = parse_timestamp(ROWS[0][0])
    with pytest.raises(ValueError):
        energy_joules(full, first - 0.5, first + 1.0)
    with pytest.raises(ValueError):
        energy_joules(full, first + 1.0, first + 0.5)


def test_endpoints_are_interpolated_for_sub_second_spans(tmp_path) -> None:
    full = _write(tmp_path)
    t = parse_timestamp(ROWS[1][0])  # power 200 at t, 300 at t+1
    # [t+0.25, t+0.75]: power 225 -> 275, mean 250 over 0.5 s = 125 J
    assert energy_joules(full, t + 0.25, t + 0.75) == pytest.approx(125.0, abs=1e-6)
    # a span across a sample: [t-0.5, t+0.5] = (150+200)/2*0.5 + (200+250)/2*0.5 = 87.5 + 112.5
    assert energy_joules(full, t - 0.5, t + 0.5) == pytest.approx(200.0, abs=1e-6)


def test_header_and_na_rows_are_skipped(tmp_path) -> None:
    path = _write(tmp_path, extra_lines=["2026/09/17 12:00:05.000, 87 %, 40 %, 20000 MiB, [N/A]"])
    assert len(parse_samples(path)) == len(ROWS)


def test_timestamp_fraction_is_kept() -> None:
    a = parse_timestamp("2026/09/17 12:00:00.000")
    b = parse_timestamp("2026/09/17 12:00:00.250")
    assert b - a == pytest.approx(0.25, abs=1e-6)


def test_command_matches_decisions_e3() -> None:
    assert NVIDIA_SMI_CMD == [
        "nvidia-smi",
        "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw",
        "--format=csv", "-lms", "1000",
    ]


# -- item 7: energy coverage waits -----------------------------------------------


def test_wait_for_samples_returns_true_once_enough_numeric_rows_exist(tmp_path) -> None:
    import threading

    path = tmp_path / "gpu.csv"
    path.write_text(HEADER + "\n", encoding="utf-8")
    # header only: not a sample
    assert wait_for_samples(path, 1, timeout_s=0.05, poll_s=0.01) is False

    def writer() -> None:
        time.sleep(0.05)
        with path.open("a", encoding="utf-8") as f:
            f.write("2026/09/17 12:00:05.000, 87 %, 40 %, 20000 MiB, [N/A]\n")  # not numeric
            f.write("%s, 87 %%, 40 %%, 20000 MiB, 100.00 W\n" % ROWS[0][0])
            f.write("%s, 87 %%, 40 %%, 20000 MiB, 100.00 W\n" % ROWS[1][0])

    threading.Thread(target=writer).start()
    assert wait_for_samples(path, 2, timeout_s=5.0, poll_s=0.01) is True
    assert len(parse_samples(path)) == 2


def test_wait_for_samples_after_ts_counts_only_later_samples(tmp_path) -> None:
    path = _write(tmp_path)
    t_last = parse_timestamp(ROWS[-1][0])
    assert wait_for_samples(path, 1, timeout_s=0.05, poll_s=0.01, after_ts=t_last) is True
    assert wait_for_samples(path, 1, timeout_s=0.05, poll_s=0.01, after_ts=t_last + 0.001) is False
    # a missing file is simply "no samples yet"
    assert wait_for_samples(tmp_path / "absent.csv", 1, timeout_s=0.02, poll_s=0.01) is False

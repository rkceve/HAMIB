"""gpu_sampler.py — nvidia-smi power sampler and per-span energy (DECISIONS E3, DESIGN §6).

``GpuSampler(path).start()`` launches::

    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw
               --format=csv -lms 1000

with stdout redirected to ``path``; ``.stop()`` terminates it.  ``energy_joules``
is a pure function over the resulting CSV (testable with a hand-written fixture):
it integrates ``power.draw`` [W] over ``[t0, t1]`` with the trapezoid rule.

CSV shape (nvidia-smi ``--format=csv``)::

    timestamp, utilization.gpu [%], utilization.memory [%], memory.used [MiB], power.draw [W]
    2026/09/17 12:00:00.123, 87 %, 40 %, 20000 MiB, 250.12 W

Timestamps are LOCAL time in ``%Y/%m/%d %H:%M:%S.%f`` and are converted with
``time.mktime`` plus the fractional second, so ``t0``/``t1`` are ``time.time()``
values taken on the same machine.
"""

from __future__ import annotations

import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import IO

NVIDIA_SMI_CMD = [
    "nvidia-smi",
    "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw",
    "--format=csv",
    "-lms",
    "1000",
]
TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M:%S.%f"


class GpuSampler:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._proc: subprocess.Popen | None = None
        self._fh: IO[str] | None = None

    def start(self) -> None:
        if self._proc is not None:
            raise RuntimeError("GpuSampler already started")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self._proc = subprocess.Popen(
            NVIDIA_SMI_CMD, stdout=self._fh, stderr=subprocess.STDOUT
        )

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._proc = None
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def parse_timestamp(text: str) -> float:
    """``2026/09/17 12:00:00.123`` (local time) -> epoch seconds (float)."""
    dt = datetime.strptime(text.strip(), TIMESTAMP_FORMAT)
    return time.mktime(dt.timetuple()) + dt.microsecond / 1e6


def parse_samples(csv_path: str | Path) -> list[tuple[float, float]]:
    """``[(epoch_seconds, power_watts), ...]`` from an nvidia-smi CSV.

    The header line and any row whose power field is not a number (``[N/A]``,
    ``[Not Supported]``) are skipped.
    """
    out: list[tuple[float, float]] = []
    for line in Path(csv_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cells = [c.strip() for c in line.split(",")]
        if len(cells) < 5 or cells[0].lower() == "timestamp":
            continue
        power_field = cells[4].split()[0] if cells[4] else ""
        try:
            ts = parse_timestamp(cells[0])
            power = float(power_field)
        except ValueError:
            continue
        out.append((ts, power))
    return out


def wait_for_samples(
    csv_path: str | Path,
    n: int,
    timeout_s: float,
    poll_s: float,
    *,
    after_ts: float | None = None,
) -> bool:
    """Poll ``csv_path`` until it holds at least ``n`` numeric power samples
    (with timestamp >= ``after_ts`` when given); True on success, False on
    timeout.  A missing file counts as zero samples.  Pure apart from the
    clock, so run_arms can inject/skip it in tests (item 7).
    """
    if n <= 0:
        return True
    deadline = time.monotonic() + float(timeout_s)
    path = Path(csv_path)
    while True:
        count = 0
        if path.exists():
            samples = parse_samples(path)
            if after_ts is not None:
                samples = [s for s in samples if s[0] >= after_ts]
            count = len(samples)
        if count >= n:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(max(0.0, float(poll_s)))


def _interp(a: tuple[float, float], b: tuple[float, float], t: float) -> float:
    (ta, pa), (tb, pb) = a, b
    if tb == ta:
        return pa
    return pa + (pb - pa) * (t - ta) / (tb - ta)


def energy_joules(csv_path: str | Path, t0: float, t1: float) -> float:
    """∫ power.draw dt over [t0, t1] by the trapezoid rule.

    Power at the two endpoints is linearly interpolated from the bracketing
    samples (no extrapolation: the span must lie inside the sampled range,
    otherwise ValueError).  Interior samples are used as-is.  A span shorter
    than the 1 s sampling period is therefore still measurable.
    """
    if t1 <= t0:
        raise ValueError("t1 must be greater than t0 (got %.3f, %.3f)" % (t0, t1))
    samples = sorted(parse_samples(csv_path))
    if len(samples) < 2 or samples[0][0] > t0 or samples[-1][0] < t1:
        raise ValueError(
            "span [%.3f, %.3f] is not covered by the %d power samples"
            % (t0, t1, len(samples))
        )
    before = max(s for s in samples if s[0] <= t0)
    after_t0 = min(s for s in samples if s[0] >= t0)
    before_t1 = max(s for s in samples if s[0] <= t1)
    after = min(s for s in samples if s[0] >= t1)
    points = [(t0, _interp(before, after_t0, t0))]
    points += [s for s in samples if t0 < s[0] < t1]
    points.append((t1, _interp(before_t1, after, t1)))
    total = 0.0
    for (ta, pa), (tb, pb) in zip(points, points[1:]):
        total += 0.5 * (pa + pb) * (tb - ta)
    return total

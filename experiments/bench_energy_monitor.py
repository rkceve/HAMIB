"""
GPU + CPU + RAM time-series sampler (for energy measurement)

Usage:
  with EnergyMonitor() as em:
      run_some_work()
  print(em.summary())
    -> {gpu_J: 1234.5, cpu_J_est: 567.8, peak_gpu_mem_mb: 4096, ...}

GPU power:
  Sample nvidia-smi --query-gpu=power.draw every 0.1s -> ∫ p dt = J

CPU:
  psutil.cpu_percent() every 0.1s × estimated TDP (i7 65W default) -> approximate J

RAM:
  psutil.Process().memory_info().rss peak
"""
from __future__ import annotations
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional

import psutil


@dataclass
class EnergyMonitor:
    sample_interval: float = 0.2   # seconds
    cpu_tdp_w: float = 65.0        # default i7 desktop ~65W

    def __post_init__(self):
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.t_start: float = 0
        self.t_end: float = 0
        self.gpu_power_samples: list[tuple[float, float]] = []  # (t, W)
        self.cpu_percent_samples: list[tuple[float, float]] = []
        self.ram_rss_samples: list[tuple[float, int]] = []      # (t, bytes)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False

    def start(self):
        self.t_start = time.perf_counter()
        self._stop.clear()
        self.gpu_power_samples = []
        self.cpu_percent_samples = []
        self.ram_rss_samples = []
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.t_end = time.perf_counter()

    def _query_gpu_power_w(self) -> Optional[float]:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=power.draw",
                 "--format=csv,noheader,nounits"],
                timeout=1.5,
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            # Assume a single GPU; if multiple lines, use the first
            line = out.splitlines()[0] if out else "0"
            return float(line)
        except Exception:
            return None

    def _loop(self):
        proc = psutil.Process()
        proc.cpu_percent(interval=None)  # prime
        while not self._stop.is_set():
            t = time.perf_counter() - self.t_start
            gp = self._query_gpu_power_w()
            if gp is not None:
                self.gpu_power_samples.append((t, gp))
            self.cpu_percent_samples.append((t, proc.cpu_percent(interval=None)))
            self.ram_rss_samples.append((t, proc.memory_info().rss))
            self._stop.wait(self.sample_interval)

    def summary(self) -> dict:
        # trapezoidal integration of GPU power → energy in Joules
        def integrate(samples: list[tuple[float, float]]) -> float:
            if len(samples) < 2:
                return 0.0
            E = 0.0
            for (t0, v0), (t1, v1) in zip(samples, samples[1:]):
                dt = t1 - t0
                E += 0.5 * (v0 + v1) * dt
            return E

        gpu_J = integrate(self.gpu_power_samples)
        # CPU: cpu_percent is per-core. 100% on N cores = N×TDP/N approx
        # Simplification: % × TDP gives a rough estimate
        cpu_samples_scaled = [(t, (pct / 100.0) * self.cpu_tdp_w)
                               for t, pct in self.cpu_percent_samples]
        cpu_J_est = integrate(cpu_samples_scaled)

        ram_peak_mb = max((b for _, b in self.ram_rss_samples), default=0) / 1024 / 1024
        gpu_w_mean = (sum(v for _, v in self.gpu_power_samples) / len(self.gpu_power_samples)
                      if self.gpu_power_samples else 0)
        gpu_w_peak = max((v for _, v in self.gpu_power_samples), default=0)
        cpu_pct_mean = (sum(v for _, v in self.cpu_percent_samples) / len(self.cpu_percent_samples)
                         if self.cpu_percent_samples else 0)
        cpu_pct_peak = max((v for _, v in self.cpu_percent_samples), default=0)
        dur_s = self.t_end - self.t_start if self.t_end else 0

        return {
            "duration_s": round(dur_s, 1),
            "gpu_energy_J": round(gpu_J, 1),
            "gpu_power_mean_W": round(gpu_w_mean, 1),
            "gpu_power_peak_W": round(gpu_w_peak, 1),
            "cpu_energy_J_est": round(cpu_J_est, 1),
            "cpu_percent_mean": round(cpu_pct_mean, 1),
            "cpu_percent_peak": round(cpu_pct_peak, 1),
            "ram_peak_mb": round(ram_peak_mb, 0),
            "n_samples_gpu": len(self.gpu_power_samples),
            "n_samples_cpu": len(self.cpu_percent_samples),
        }


if __name__ == "__main__":
    import time as _t
    print("Self-test: idle 3 sec")
    with EnergyMonitor() as m:
        _t.sleep(3)
    import json
    print(json.dumps(m.summary(), ensure_ascii=False, indent=2))

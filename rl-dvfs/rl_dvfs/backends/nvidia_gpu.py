"""Backend GPU NVIDIA (EKSPERIMENTAL, belum diuji di hardware) via nvidia-smi. Butuh root + driver proprietary.

Aktuator : nvidia-smi -lgc <min>,<max>   (reset: -rgc)
Sensor   : nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw
"""
import subprocess
import time

from ..types import Observation

N_LEVELS = 6


def _smi(*args):
    return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, check=True).stdout.strip()


class NvidiaGpuBackend:
    n_levels = N_LEVELS

    def __init__(self, interval=0.5, dry_run=False, gpu=0, latency_server=None):
        self.interval, self.dry_run, self.gpu, self.latency = interval, dry_run, gpu, latency_server
        out = _smi("-i", str(gpu), "--query-gpu=clocks.max.graphics,power.limit",
                   "--format=csv,noheader,nounits")
        max_clk, plimit = [float(x) for x in out.split(",")]
        min_clk = 300.0
        self.levels = [int(min_clk + i * (max_clk - min_clk) / (N_LEVELS - 1)) for i in range(N_LEVELS)]
        self.power_ref = plimit
        self.level = N_LEVELS - 1

    def set_level(self, level):
        self.level = level
        if not self.dry_run:
            _smi("-i", str(self.gpu), "-lgc", f"{self.levels[0]},{self.levels[level]}")

    def restore(self):
        if not self.dry_run:
            _smi("-i", str(self.gpu), "-rgc")

    def wait(self):
        time.sleep(self.interval)

    def observe(self):
        out = _smi("-i", str(self.gpu), "--query-gpu=utilization.gpu,temperature.gpu,power.draw",
                   "--format=csv,noheader,nounits")
        util, temp, power = [float(x) for x in out.split(",")]
        util /= 100.0
        ratio = self.latency.ratio() if self.latency else None
        return Observation(util, temp, power, util / 0.85 if ratio is None else ratio)

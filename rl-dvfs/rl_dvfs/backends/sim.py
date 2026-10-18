"""Simulator CPU + beban inferensi bursty. Dipakai untuk pelatihan offline tanpa root."""
import random

from ..types import Observation

SIM_FREQS_KHZ = [800_000, 1_200_000, 1_600_000, 2_000_000, 2_400_000, 2_800_000]


class SimBackend:
    n_levels = len(SIM_FREQS_KHZ)
    power_ref = 15.0

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.level = self.n_levels - 1
        self.temp = 40.0
        self.busy = False

    def set_level(self, level: int) -> None:
        self.level = level

    def wait(self) -> None:
        pass

    def observe(self) -> Observation:
        r = self.rng
        if self.busy:
            self.busy = r.random() > 0.12
        else:
            self.busy = r.random() < 0.08
        demand = r.uniform(0.4, 0.95) if self.busy else r.uniform(0.05, 0.2)

        f_rel = SIM_FREQS_KHZ[self.level] / SIM_FREQS_KHZ[-1]
        util = min(1.0, demand / f_rel)
        base = 0.55 if self.busy else 0.25
        latency_ratio = (base / f_rel) * max(1.0, demand / f_rel)
        volt = 0.6 + 0.4 * f_rel
        power = 1.0 + 14.0 * f_rel * volt ** 2 * (0.3 + 0.7 * util)
        self.temp += 0.15 * (40.0 + 3.0 * power - self.temp)
        return Observation(util, self.temp, power, latency_ratio)

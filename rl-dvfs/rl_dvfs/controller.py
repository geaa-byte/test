"""Controller: menghubungkan backend (sensor+aktuator) dengan agen RL.

State  : (bucket utilisasi, bucket suhu, bucket latensi, level frekuensi saat ini)
Aksi   : 0=turun satu level, 1=tahan, 2=naik satu level, 3=boost ke level maksimum
Reward : -(daya ternormalisasi) - penalti pelanggaran latensi - penalti termal - biaya switching
"""
from dataclasses import dataclass

from .agent import QAgent
from .types import Observation

ACTIONS = ("down", "hold", "up", "boost")


@dataclass
class RewardWeights:
    power: float = 1.0
    latency: float = 4.0
    thermal: float = 1.0
    switch: float = 0.03
    temp_soft_limit: float = 80.0


class DVFSController:
    def __init__(self, backend, agent: QAgent, learn: bool = True,
                 weights: RewardWeights = None):
        self.backend = backend
        self.agent = agent
        self.learn = learn
        self.w = weights or RewardWeights()
        self.level = backend.n_levels - 1          # mulai dari performa penuh (aman)
        backend.set_level(self.level)
        self.p_ref = max(backend.power_ref, 1e-6)
        self.obs = backend.observe()
        self.state = self.encode(self.obs)

    # ---- diskretisasi ----
    def encode(self, o: Observation):
        u = min(4, int(o.util * 5))
        t = 0 if o.temp_c < 70 else (1 if o.temp_c < 85 else 2)
        l = 0 if o.latency_ratio < 0.7 else (1 if o.latency_ratio <= 1.0 else 2)
        return (u, t, l, self.level)

    def reward(self, o: Observation, action: int) -> float:
        self.p_ref = max(self.p_ref, o.power_w)
        r = -self.w.power * (o.power_w / self.p_ref)
        r -= self.w.latency * max(0.0, o.latency_ratio - 1.0)
        r -= self.w.thermal * max(0.0, (o.temp_c - self.w.temp_soft_limit) / 10.0)
        if action in (0, 2, 3):
            r -= self.w.switch
        return r

    def _apply(self, action: int) -> None:
        top = self.backend.n_levels - 1
        if action == 0:
            self.level = max(0, self.level - 1)
        elif action == 2:
            self.level = min(top, self.level + 1)
        elif action == 3:
            self.level = top
        self.backend.set_level(self.level)

    def step(self):
        a = self.agent.act(self.state, greedy=not self.learn)
        self._apply(a)
        self.backend.wait()
        o = self.backend.observe()
        s2 = self.encode(o)
        r = self.reward(o, a)
        if self.learn:
            self.agent.update(self.state, a, r, s2)
        self.state, self.obs = s2, o
        return o, a, r

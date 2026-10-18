"""Agen Q-learning tabular (tanpa dependensi eksternal)."""
import json
import random
from typing import Dict, List, Tuple

State = Tuple[int, ...]
HOLD = 1  # indeks aksi "hold"; dipakai sebagai default untuk state yang belum pernah dilihat


class QAgent:
    def __init__(self, n_actions: int = 4, alpha: float = 0.1, gamma: float = 0.9,
                 epsilon: float = 1.0, eps_min: float = 0.02, eps_decay: float = 0.98,
                 seed=None):
        self.n_actions = n_actions
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.eps_min, self.eps_decay = epsilon, eps_min, eps_decay
        self.rng = random.Random(seed)
        self.q: Dict[State, List[float]] = {}

    def _row(self, s: State) -> List[float]:
        return self.q.setdefault(s, [0.0] * self.n_actions)

    def act(self, s: State, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_actions)
        row = self._row(s)
        best = max(row)
        ties = [i for i, v in enumerate(row) if v == best]
        return HOLD if HOLD in ties else self.rng.choice(ties)

    def update(self, s: State, a: int, r: float, s2: State) -> None:
        row = self._row(s)
        target = r + self.gamma * max(self._row(s2))
        row[a] += self.alpha * (target - row[a])

    def decay(self) -> None:
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    # ---- persistensi ----
    def save(self, path: str, meta: dict = None) -> None:
        data = {
            "meta": meta or {},
            "n_actions": self.n_actions,
            "q": {",".join(map(str, k)): v for k, v in self.q.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str, **kw) -> "QAgent":
        with open(path) as f:
            data = json.load(f)
        agent = cls(n_actions=data["n_actions"], **kw)
        agent.q = {tuple(int(x) for x in k.split(",")): v for k, v in data["q"].items()}
        agent.meta = data.get("meta", {})
        return agent

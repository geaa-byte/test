"""Pelatihan dan evaluasi di simulator + baseline pembanding."""
from .agent import QAgent
from .backends.sim import SimBackend
from .controller import DVFSController


def train(episodes=400, steps=300, seed=0, log=None):
    agent = QAgent(seed=seed, eps_decay=0.985)
    for ep in range(episodes):
        ctl = DVFSController(SimBackend(seed=ep), agent, learn=True)
        total = 0.0
        for _ in range(steps):
            total += ctl.step()[2]
        agent.decay()
        if log and (ep + 1) % max(1, episodes // 10) == 0:
            log(f"episode {ep + 1}/{episodes}  reward={total / steps:.3f}  eps={agent.epsilon:.3f}")
    return agent


def _policy_performance(obs):
    return 3  # selalu boost -> frekuensi maksimum


def _policy_ondemand(obs):
    if obs.util > 0.8:
        return 3
    if obs.util < 0.3:
        return 0
    return 1


def evaluate(agent=None, episodes=30, steps=300, seed0=10_000):
    """Mengembalikan dict {nama: (daya_rata2_W, %pelanggaran_latensi)}."""
    results = {}
    names = {"performance": _policy_performance, "ondemand-like": _policy_ondemand}
    if agent is not None:
        names["rl-dvfs"] = None
    for name, pol in names.items():
        pw = viol = n = 0
        for ep in range(episodes):
            ctl = DVFSController(SimBackend(seed=seed0 + ep),
                                 agent if pol is None else QAgent(), learn=False)
            for _ in range(steps):
                if pol is not None:
                    ctl._apply(pol(ctl.obs))
                    o = ctl.backend.observe()
                    ctl.obs = o
                else:
                    o = ctl.step()[0]
                pw += o.power_w
                viol += o.latency_ratio > 1.0
                n += 1
        results[name] = (pw / n, 100.0 * viol / n)
    return results

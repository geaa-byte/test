from rl_dvfs.agent import QAgent
from rl_dvfs.backends.linux_cpu import LinuxCpuBackend
from rl_dvfs.backends.sim import SimBackend
from rl_dvfs.bench import evaluate, train
from rl_dvfs.controller import DVFSController


def test_agent_learns_bandit():
    ag = QAgent(n_actions=2, epsilon=0.3, seed=1)
    for _ in range(500):
        a = ag.act((0,))
        ag.update((0,), a, 1.0 if a == 1 else 0.0, (0,))
    assert ag.act((0,), greedy=True) == 1


def test_save_load(tmp_path):
    ag = QAgent()
    ag.update((1, 2, 3, 4), 2, 1.0, (1, 2, 3, 4))
    p = tmp_path / "m.json"
    ag.save(str(p))
    assert QAgent.load(str(p)).q == ag.q


def test_controller_steps():
    ctl = DVFSController(SimBackend(0), QAgent(seed=0))
    for _ in range(50):
        ctl.step()
    assert 0 <= ctl.level < ctl.backend.n_levels


def test_rl_saves_power_vs_performance():
    agent = train(episodes=150, steps=200)
    agent.epsilon = 0.0
    res = evaluate(agent, episodes=10, steps=200)
    assert res["rl-dvfs"][0] < res["performance"][0]


def test_linux_backend_fake_sysfs(tmp_path):
    cpu = tmp_path / "sys/devices/system/cpu/cpu0/cpufreq"
    cpu.mkdir(parents=True)
    for n, v in {"cpuinfo_min_freq": 800000, "cpuinfo_max_freq": 3000000,
                 "scaling_min_freq": 800000, "scaling_max_freq": 3000000}.items():
        (cpu / n).write_text(str(v))
    (tmp_path / "proc").mkdir()
    (tmp_path / "proc/stat").write_text("cpu 10 0 10 80 0 0 0 0\n")
    b = LinuxCpuBackend(root=str(tmp_path))
    b.set_level(0)
    assert int((cpu / "scaling_max_freq").read_text()) == b.levels[0]
    b.restore()
    assert int((cpu / "scaling_max_freq").read_text()) == 3000000
    assert 0.0 <= b.observe().util <= 1.0

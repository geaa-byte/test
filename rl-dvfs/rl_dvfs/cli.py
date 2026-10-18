"""rl-dvfs: DVFS berbasis Reinforcement Learning untuk Linux."""
import argparse
import os
import signal
import sys
import time

from . import __version__
from .agent import QAgent
from .bench import evaluate, train
from .controller import ACTIONS, DVFSController
from .latency import DEFAULT_SOCK, LatencyServer


def cmd_train(a):
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    agent = train(a.episodes, a.steps, a.seed, log=print)
    agent.save(a.out, meta={"n_levels": 6, "version": __version__})
    print(f"model disimpan: {a.out}")
    agent.epsilon = 0.0
    _print_eval(agent)


def cmd_eval(a):
    agent = QAgent.load(a.model, epsilon=0.0) if a.model else None
    _print_eval(agent)


def _print_eval(agent):
    res = evaluate(agent)
    print(f"\n{'policy':<15}{'daya rata2 (W)':>16}{'pelanggaran latensi':>22}")
    for k, (p, v) in res.items():
        print(f"{k:<15}{p:>16.2f}{v:>21.1f}%")


def cmd_run(a):
    if a.device == "cpu":
        from .backends.linux_cpu import LinuxCpuBackend as B
    else:
        from .backends.nvidia_gpu import NvidiaGpuBackend as B
    if not a.dry_run and os.geteuid() != 0:
        sys.exit("Butuh root untuk mengubah frekuensi. Jalankan dengan sudo, atau pakai --dry-run.")

    srv = None
    if not a.no_latency_socket:
        srv = LatencyServer(a.socket)
        srv.start()
    backend = B(interval=a.interval, dry_run=a.dry_run, latency_server=srv)
    if a.model and os.path.exists(a.model):
        agent = QAgent.load(a.model, epsilon=a.epsilon)
    else:
        print("model tidak ditemukan -> mulai dari nol (eksplorasi lebih tinggi)")
        agent = QAgent(epsilon=max(a.epsilon, 0.3))
    ctl = DVFSController(backend, agent, learn=not a.no_learn)

    stop = False

    def _sig(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    print(f"rl-dvfs berjalan (dry_run={a.dry_run}); level freq(kHz)={backend.levels}")
    last_save = time.monotonic()
    try:
        while not stop:
            o, act, r = ctl.step()
            if a.verbose:
                print(f"util={o.util:4.0%} T={o.temp_c:4.1f}C P={o.power_w:5.1f}W "
                      f"lat={o.latency_ratio:4.2f} -> {ACTIONS[act]:<5} "
                      f"f={backend.levels[ctl.level] / 1000:.0f}MHz r={r:+.2f}")
            if a.model and not a.no_learn and time.monotonic() - last_save > 60:
                agent.save(a.model, meta={"n_levels": 6})
                last_save = time.monotonic()
    finally:
        backend.restore()
        if srv:
            srv.stop()
        if a.model and not a.no_learn:
            os.makedirs(os.path.dirname(a.model) or ".", exist_ok=True)
            agent.save(a.model, meta={"n_levels": 6})
        print("frekuensi dikembalikan ke pengaturan awal.")


def main(argv=None):
    p = argparse.ArgumentParser(prog="rl-dvfs", description=__doc__)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="latih agen di simulator")
    t.add_argument("--episodes", type=int, default=400)
    t.add_argument("--steps", type=int, default=300)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default="models/qtable.json")
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("eval", help="bandingkan RL vs baseline di simulator")
    e.add_argument("--model")
    e.set_defaults(fn=cmd_eval)

    r = sub.add_parser("run", help="jalankan daemon pada hardware nyata")
    r.add_argument("--device", choices=["cpu", "nvidia"], default="cpu")
    r.add_argument("--model", default="models/qtable.json")
    r.add_argument("--interval", type=float, default=0.5)
    r.add_argument("--epsilon", type=float, default=0.02)
    r.add_argument("--no-learn", action="store_true", help="hanya eksploitasi, tanpa belajar online")
    r.add_argument("--dry-run", action="store_true", help="jangan menulis ke sysfs")
    r.add_argument("--socket", default=DEFAULT_SOCK)
    r.add_argument("--no-latency-socket", action="store_true")
    r.add_argument("-v", "--verbose", action="store_true")
    r.set_defaults(fn=cmd_run)

    a = p.parse_args(argv)
    a.fn(a)
    return 0

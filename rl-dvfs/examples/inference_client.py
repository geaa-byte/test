"""Contoh aplikasi inferensi yang melapor latensi ke daemon rl-dvfs.

Ganti `fake_inference()` dengan pemanggilan model nyata (onnxruntime, llama.cpp, PyTorch, dll.).
"""
import random
import time

from rl_dvfs.latency import LatencyReporter


def fake_inference():
    n = 220
    a = [[random.random() for _ in range(n)] for _ in range(n)]
    return sum(sum(r) for r in a)  # beban CPU sintetis


reporter = LatencyReporter(target_ms=60)  # target latensi (SLA) per inferensi

for i in range(200):
    with reporter.track():
        fake_inference()
    time.sleep(random.choice([0.0, 0.0, 0.5]))  # beban bursty

"""Umpan balik latensi dari proses inferensi AI ke daemon lewat UNIX datagram socket.

Aplikasi inferensi (tanpa root) cukup memakai LatencyReporter; daemon menjalankan LatencyServer.
"""
import json
import os
import socket
import threading
import time
from contextlib import contextmanager
from typing import Optional

DEFAULT_SOCK = "/run/rl-dvfs/latency.sock"


class LatencyServer:
    def __init__(self, path: str = DEFAULT_SOCK):
        self.path = path
        self._ema: Optional[float] = None
        self._t = 0.0
        self._stop = threading.Event()
        self._sock = None
        self._thread = None

    def start(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if os.path.exists(self.path):
            os.unlink(self.path)
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._sock.bind(self.path)
        os.chmod(self.path, 0o666)  # agar aplikasi non-root bisa melapor
        self._sock.settimeout(0.5)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                data = self._sock.recv(1024)
                msg = json.loads(data)
                ratio = float(msg["latency_ms"]) / float(msg["target_ms"])
            except (socket.timeout, ValueError, KeyError, TypeError):
                continue
            except OSError:
                break
            self._ema = ratio if self._ema is None else 0.5 * self._ema + 0.5 * ratio
            self._t = time.monotonic()

    def ratio(self, max_age: float = 2.0) -> Optional[float]:
        """Rasio latensi terbaru, atau None jika tidak ada laporan baru (tidak ada inferensi aktif)."""
        if self._ema is not None and time.monotonic() - self._t <= max_age:
            return self._ema
        return None

    def stop(self) -> None:
        self._stop.set()
        if self._sock:
            self._sock.close()
        try:
            os.unlink(self.path)
        except OSError:
            pass


class LatencyReporter:
    def __init__(self, target_ms: float, path: str = DEFAULT_SOCK):
        self.target_ms = target_ms
        self.path = path
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)

    def report(self, latency_ms: float) -> None:
        try:
            self._sock.sendto(
                json.dumps({"latency_ms": latency_ms, "target_ms": self.target_ms}).encode(),
                self.path)
        except OSError:
            pass  # daemon tidak berjalan: abaikan

    @contextmanager
    def track(self):
        t0 = time.perf_counter()
        yield
        self.report((time.perf_counter() - t0) * 1000.0)

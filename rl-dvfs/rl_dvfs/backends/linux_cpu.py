"""Backend CPU Linux lewat sysfs cpufreq.

Aktuator : menulis scaling_max_freq (cap frekuensi) di semua core -> kernel governor
           (schedutil/ondemand) tetap bekerja di bawah cap yang dipilih agen RL.
Sensor   : /proc/stat (util), /sys/class/thermal (suhu), RAPL / baterai (daya).
Butuh root untuk menulis. Gunakan dry_run=True untuk hanya membaca.
"""
import glob
import time
from pathlib import Path

from ..types import Observation

N_LEVELS = 6


def _read(p):
    try:
        return Path(p).read_text().strip()
    except OSError:
        return None


class LinuxCpuBackend:
    n_levels = N_LEVELS

    def __init__(self, interval: float = 0.5, dry_run: bool = False, root: str = "/",
                 latency_server=None, power_ref: float = 15.0):
        self.root = Path(root)
        self.interval = interval
        self.dry_run = dry_run
        self.latency = latency_server
        self.power_ref = power_ref
        self.level = N_LEVELS - 1

        pat = str(self.root / "sys/devices/system/cpu/cpu[0-9]*/cpufreq")
        self.cpus = sorted(Path(p) for p in glob.glob(pat))
        if not self.cpus:
            raise RuntimeError("cpufreq tidak ditemukan (VM tanpa cpufreq? driver belum dimuat?)")

        c0 = self.cpus[0]
        self.hw_min = int(_read(c0 / "cpuinfo_min_freq"))
        self.hw_max = int(_read(c0 / "cpuinfo_max_freq"))
        avail = _read(c0 / "scaling_available_frequencies")
        if avail:
            freqs = sorted(int(x) for x in avail.split())
        else:  # intel_pstate (mode active) tidak menyediakan daftar -> buat grid 100 MHz
            freqs = list(range(self.hw_min, self.hw_max + 1, 100_000))
            if freqs[-1] != self.hw_max:
                freqs.append(self.hw_max)
        n = N_LEVELS
        self.levels = [freqs[round(i * (len(freqs) - 1) / (n - 1))] for i in range(n)]

        self._orig = {c: (_read(c / "scaling_min_freq"), _read(c / "scaling_max_freq"))
                      for c in self.cpus}
        self._prev_stat = self._read_stat()
        self._rapl = self._find_rapl()
        self._prev_energy = self._read_energy()
        self._prev_t = time.monotonic()

    # ---- aktuator ----
    def set_level(self, level: int) -> None:
        self.level = level
        if self.dry_run:
            return
        f = self.levels[level]
        for c in self.cpus:
            cur_min = int(_read(c / "scaling_min_freq") or 0)
            if f < cur_min:
                (c / "scaling_min_freq").write_text(str(f))
            (c / "scaling_max_freq").write_text(str(f))

    def restore(self) -> None:
        if self.dry_run:
            return
        for c, (mn, mx) in self._orig.items():
            try:
                if mx:
                    (c / "scaling_max_freq").write_text(mx)
                if mn:
                    (c / "scaling_min_freq").write_text(mn)
            except OSError:
                pass

    def wait(self) -> None:
        time.sleep(self.interval)

    # ---- sensor ----
    def _read_stat(self):
        line = (_read(self.root / "proc/stat") or "cpu 0 0 0 0 0 0 0 0").splitlines()[0]
        v = [int(x) for x in line.split()[1:9]]
        return sum(v) - v[3] - v[4], sum(v)  # busy, total

    def _util(self) -> float:
        busy, total = self._read_stat()
        pb, pt = self._prev_stat
        self._prev_stat = (busy, total)
        return 0.0 if total == pt else max(0.0, min(1.0, (busy - pb) / (total - pt)))

    def _temp(self) -> float:
        temps = []
        for p in glob.glob(str(self.root / "sys/class/thermal/thermal_zone*/temp")):
            v = _read(p)
            if v and v.lstrip("-").isdigit() and 0 < int(v) / 1000 < 150:
                temps.append(int(v) / 1000)
        return max(temps) if temps else 50.0

    def _find_rapl(self):
        p = self.root / "sys/class/powercap/intel-rapl:0/energy_uj"
        return p if _read(p) is not None else None

    def _read_energy(self):
        return int(_read(self._rapl)) if self._rapl else None

    def _battery_power(self):
        for b in glob.glob(str(self.root / "sys/class/power_supply/BAT*")):
            pw = _read(b + "/power_now")
            if pw and pw.isdigit() and int(pw) > 0:
                return int(pw) / 1e6
            cur, volt = _read(b + "/current_now"), _read(b + "/voltage_now")
            if cur and volt and cur.isdigit() and volt.isdigit() and int(cur) > 0:
                return int(cur) * int(volt) / 1e12
        return None

    def _power(self, util: float, dt: float) -> float:
        e = self._read_energy()
        if e is not None and self._prev_energy is not None and dt > 0 and e >= self._prev_energy:
            p = (e - self._prev_energy) / 1e6 / dt
            self._prev_energy = e
            return p
        self._prev_energy = e
        bp = self._battery_power()
        if bp is not None:
            return bp
        f_rel = self.levels[self.level] / self.hw_max  # fallback: model estimasi
        return 2.0 + 13.0 * util * f_rel ** 2.5

    def observe(self) -> Observation:
        now = time.monotonic()
        dt, self._prev_t = now - self._prev_t, now
        util = self._util()
        ratio = self.latency.ratio() if self.latency else None
        if ratio is None:
            ratio = util / 0.85  # proxy: util tinggi di bawah cap => kemungkinan latensi naik
        return Observation(util, self._temp(), self._power(util, dt), ratio)

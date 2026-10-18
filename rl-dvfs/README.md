# rl-dvfs — DVFS berbasis Reinforcement Learning untuk inferensi AI

Daemon Linux yang memakai **Q-learning** untuk menentukan kapan frekuensi CPU (dan eksperimental GPU NVIDIA)
dinaikkan atau diturunkan saat menjalankan inferensi AI, supaya **hemat daya/baterai tanpa melanggar target latensi**.

```
 ┌────────────┐  latensi inferensi   ┌──────────────┐  state   ┌─────────┐
 │ Aplikasi AI│ ───────────────────► │  rl-dvfs     │ ───────► │ Q-agent │
 │ (reporter) │   (UNIX socket)      │  controller  │ ◄─────── │         │
 └────────────┘                      └──────┬───────┘  aksi    └─────────┘
                                            │ scaling_max_freq / nvidia-smi -lgc
                                     ┌──────▼───────┐
                                     │ CPU / GPU    │  sensor: /proc/stat, thermal, RAPL, baterai
                                     └──────────────┘
```

## Cara kerja RL

| Komponen | Isi |
|---|---|
| **State** | bucket utilisasi (5) × suhu (3) × rasio latensi (3) × level frekuensi saat ini (6) |
| **Aksi** | `down` (turun 1 level), `hold`, `up` (naik 1 level), `boost` (langsung ke maksimum) |
| **Reward** | `-daya/daya_ref − 4·max(0, latensi/target − 1) − penalti suhu − biaya switching` |
| **Aktuator** | menulis `scaling_max_freq` (cap) — governor kernel tetap berjalan di bawah cap |

Tanpa aplikasi yang melapor latensi, rasio latensi diperkirakan dari utilisasi (proxy). Dengan `LatencyReporter`,
agen mendapat sinyal SLA yang sebenarnya, jadi hasilnya jauh lebih akurat.

## Instalasi (Debian)

```bash
sudo apt update && sudo apt install -y python3 python3-venv git
git clone https://github.com/<username>/rl-dvfs.git && cd rl-dvfs
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Tidak ada dependensi runtime (hanya stdlib Python).

## Pemakaian

```bash
# 1) Latih di simulator (tanpa root) + bandingkan dengan baseline
rl-dvfs train --episodes 400 --out models/qtable.json

# 2) Coba dulu tanpa mengubah frekuensi (hanya baca sensor)
rl-dvfs run --model models/pretrained_sim.json --dry-run -v

# 3) Jalankan sungguhan (butuh root)
sudo .venv/bin/rl-dvfs run --model models/qtable.json -v

# 4) Di terminal lain: aplikasi inferensi yang melapor latensi
python examples/inference_client.py
```

Contoh hasil evaluasi di simulator (`rl-dvfs train`):

```
policy           daya rata2 (W)   pelanggaran latensi
performance                8.62                  0.0%
ondemand-like              6.29                  4.0%
rl-dvfs                    5.88                  1.6%
```

> Angka ini dari **simulator**, bukan hardware. Ukur ulang di mesinmu (lihat bagian Benchmark).

### Integrasi di aplikasi inferensi

```python
from rl_dvfs.latency import LatencyReporter
rep = LatencyReporter(target_ms=60)
with rep.track():
    output = model(input)   # onnxruntime / torch / llama.cpp, dll.
```

### Service systemd

```bash
sudo mkdir -p /opt/rl-dvfs && sudo cp -r . /opt/rl-dvfs && cd /opt/rl-dvfs
sudo python3 -m venv .venv && sudo .venv/bin/pip install .
sudo cp systemd/rl-dvfs.service /etc/systemd/system/
sudo systemctl enable --now rl-dvfs
```

## Keamanan & catatan penting

- Daemon hanya menulis `scaling_min_freq`/`scaling_max_freq` dan **mengembalikan nilai asli** saat berhenti (SIGINT/SIGTERM).
  Jika proses mati paksa (`kill -9`), reset manual: `echo <freq_max_asli> | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq`
  atau reboot.
- Selalu mulai dengan `--dry-run`. Pantau suhu pada percobaan pertama.
- Driver `intel_pstate` mode *active* dan `amd-pstate` punya perilaku sendiri; cap `scaling_max_freq` tetap dihormati, tetapi
  jumlah level frekuensi diperoleh dari grid 100 MHz jika `scaling_available_frequencies` tidak ada.
- Pengukuran daya: RAPL (`/sys/class/powercap/intel-rapl:0/energy_uj`, mungkin butuh root) → baterai (`power_now`) →
  model estimasi. Pengukuran nyata sangat disarankan agar reward akurat.
- Model dari simulator adalah *titik awal*; jalankan tanpa `--no-learn` agar agen beradaptasi dengan hardware nyata.
- Backend GPU NVIDIA (`--device nvidia`) **eksperimental dan belum diuji di hardware**.

## Benchmark di hardware nyata

Bandingkan energi per batch inferensi: jalankan beban yang sama dengan governor `schedutil`/`powersave`
lalu dengan `rl-dvfs`, catat daya via `powertop`, `turbostat`, atau `upower -i /org/freedesktop/UPower/devices/battery_BAT0`.

## Struktur repo

```
rl_dvfs/
  agent.py          Q-learning tabular + simpan/muat JSON
  controller.py     state, aksi, reward, loop kontrol
  latency.py        socket umpan balik latensi (server + reporter)
  bench.py          training & evaluasi di simulator
  backends/
    sim.py          simulator CPU + beban bursty
    linux_cpu.py    sysfs cpufreq, /proc/stat, thermal, RAPL, baterai
    nvidia_gpu.py   (eksperimental) nvidia-smi
tests/  examples/  systemd/  models/
```

## Roadmap

- Deep RL (DQN/PPO) dengan state kontinu; dukungan AMD GPU (`power_dpm_force_performance_level`) dan iGPU Intel
- Kontrol per-cluster (P-core/E-core), kontrol batch size & jumlah thread
- Reward berbasis energi per inferensi (joule/inferensi)

## Lisensi

MIT

from dataclasses import dataclass


@dataclass
class Observation:
    """Satu pembacaan sensor dari backend (CPU/GPU/simulator)."""

    util: float           # 0..1, utilisasi pada frekuensi saat ini
    temp_c: float         # suhu (derajat C)
    power_w: float        # daya (Watt) - terukur atau estimasi
    latency_ratio: float  # latensi inferensi / target latensi (>1 = melanggar SLA)

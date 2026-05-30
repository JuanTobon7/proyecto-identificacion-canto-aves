from dataclasses import dataclass, asdict

from pyparsing import Optional

@dataclass
class ButterworthParams:
    """Parámetros resultantes del entrenamiento."""
    order: int
    cutoff_freq: float
    filter_type: str          # 'low', 'high', 'band'
    low_freq: Optional[float] = None   # solo para bandpass
    high_freq: Optional[float] = None  # solo para bandpass
    auto_detected: bool = False
    spectral_energy_p95_hz: Optional[float] = None  # referencia del análisis
    sample_rate: int = 0
 
    def to_dict(self) -> dict:
        return asdict(self)
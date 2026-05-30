from dataclasses import dataclass
from dataclasses import asdict
from dataclasses import field

@dataclass
class AudioStats:
    """Estadísticas detalladas de un único archivo de audio."""
 
    # Identificación
    file_path: str = ""
    file_name: str = ""
    duration_sec: float = 0.0
    sample_rate: int = 0
    n_samples: int = 0
    channels: int = 1
 
    # Amplitud
    amp_mean: float = 0.0
    amp_std: float = 0.0
    amp_max: float = 0.0
    amp_min: float = 0.0
    amp_rms: float = 0.0
    amp_peak_to_rms_db: float = 0.0   # crest factor en dB
 
    # Energía total (dominio temporal)
    energy: float = 0.0
 
    # Silencio
    silence_ratio: float = 0.0        # fracción de muestras < umbral
    max_silence_run_sec: float = 0.0  # silencio continuo más largo
 
    # Frecuencias (FFT)
    spectral_centroid: float = 0.0
    spectral_bandwidth: float = 0.0
    dominant_freq_hz: float = 0.0
    band_energies: dict = field(default_factory=dict)  # {label: energy}
 
    # Calidad
    clipping_samples: int = 0         # muestras en ±1.0 (saturación)
    dc_offset: float = 0.0
 
    # Veredicto
    is_outlier: bool = False
    outlier_reasons: list[str] = field(default_factory=list)
 
    def to_dict(self) -> dict:
        return asdict(self)
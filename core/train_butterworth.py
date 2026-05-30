from core.fft import FFTProcessor
from core.filter_butterworth import FilterButterworth
from typing import Optional
from dataclasses import dataclass, asdict
from core.audio_converter import AudioConverter
from core.buttherworth_params import ButterworthParams
import numpy as np

class TrainModelButterworth:
    """
    Configura un FilterButterworth óptimo a partir de una señal representativa.
 
    Parámetros
    ----------
    cutoff_freq   : frecuencia de corte en Hz.
                    Si es None, se detecta automáticamente desde la señal.
    order         : orden del filtro (más alto = corte más abrupto, más costo).
                    Rango recomendado: 2–8. Default: 4.
    filter_type   : 'low' | 'high' | 'band'.
                    'band' requiere low_freq y high_freq.
    low_freq      : frecuencia inferior para filtro pasa-banda.
    high_freq     : frecuencia superior para filtro pasa-banda.
    energy_percentile : percentil de energía espectral acumulada usado para
                    la detección automática de cutoff (default: 95).
    min_cutoff_hz : límite inferior al detectar cutoff automáticamente.
    max_cutoff_hz : límite superior al detectar cutoff automáticamente.
                    Si es None, se usa nyquist * 0.9.
    """
 
    def __init__(
        self,
        cutoff_freq: Optional[float] = None,
        order: int = 4,
        filter_type: str = "low",
        low_freq: Optional[float] = None,
        high_freq: Optional[float] = None,
        energy_percentile: float = 95.0,
        min_cutoff_hz: float = 200.0,
        max_cutoff_hz: Optional[float] = None,
    ):
        if filter_type not in ("low", "high", "band"):
            raise ValueError(f"filter_type debe ser 'low', 'high' o 'band', no '{filter_type}'.")
        if filter_type == "band" and (low_freq is None or high_freq is None):
            raise ValueError("Para filter_type='band' debes indicar low_freq y high_freq.")
        if order < 1:
            raise ValueError("El orden del filtro debe ser >= 1.")
        if energy_percentile <= 0 or energy_percentile >= 100:
            raise ValueError("energy_percentile debe estar entre 0 y 100 (exclusivo).")
 
        self.cutoff_freq = cutoff_freq
        self.order = order
        self.filter_type = filter_type
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.energy_percentile = energy_percentile
        self.min_cutoff_hz = min_cutoff_hz
        self.max_cutoff_hz = max_cutoff_hz
        self.filter_butterworth: Optional[FilterButterworth] = None
        self._fft = FFTProcessor()
        self.last_params: Optional[ButterworthParams] = None
 
    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
 
    def train(self, signal: np.ndarray, sr: int) -> FilterButterworth:
        """
        Analiza *signal* (mono float32) y retorna un FilterButterworth
        configurado con los parámetros óptimos.
 
        Parámetros
        ----------
        signal : array mono float32, señal representativa del corpus.
        sr     : sample rate en Hz.
 
        Retorna
        -------
        FilterButterworth listo para usar.
        """
        if sr <= 0:
            raise ValueError(f"Sample rate inválido: {sr}")
        if signal is None or len(signal) == 0:
            raise ValueError("La señal está vacía.")
 
        # Asegurar mono float32
        y = AudioConverter.to_mono_float32(signal, sr)
 
        # Resolver parámetros
        if self.filter_type == "band":
            params = self._resolve_bandpass_params(y, sr)
        elif self.cutoff_freq is not None:
            params = ButterworthParams(
                order=self.order,
                cutoff_freq=self.cutoff_freq,
                filter_type=self.filter_type,
                auto_detected=False,
                sample_rate=sr,
            )
        else:
            params = self.filter_butterworth._auto_detect_cutoff(y, sr)
 
        self.last_params = params
        self._print_params(params)
 
        # Construir filtro
        self.filter_butterworth = FilterButterworth(order=params.order)
        self.filter_butterworth.filter_type = params.filter_type
        return self.filter_butterworth
  
    def _resolve_bandpass_params(self, y: np.ndarray, sr: int) -> ButterworthParams:
        """Resuelve parámetros para filtro pasa-banda."""
        low = self.low_freq
        high = self.high_freq
 
        # Si no se fijaron manualmente, detectar automáticamente
        if low is None or high is None:
            freqs, magnitude = self._fft.compute_fft(y, sr)
            dc_mask = freqs > 0
            freqs = freqs[dc_mask]
            magnitude = magnitude[dc_mask]
 
            energy = magnitude ** 2
            cumulative_energy = np.cumsum(energy)
            if cumulative_energy[-1] > 0:
                cum_norm = cumulative_energy / cumulative_energy[-1]
                idx_low = np.searchsorted(cum_norm, 0.05)
                idx_high = np.searchsorted(cum_norm, 0.95)
                nyquist = sr / 2
                low = float(np.clip(freqs[idx_low], self.min_cutoff_hz, nyquist * 0.8))
                high = float(np.clip(freqs[idx_high], low * 1.5, nyquist * 0.9))
            else:
                low = self.min_cutoff_hz
                high = min(sr * 0.4, 8000.0)
 
        return ButterworthParams(
            order=self.order,
            cutoff_freq=(low + high) / 2,  # centro de banda como referencia
            filter_type="band",
            low_freq=low,
            high_freq=high,
            auto_detected=(self.low_freq is None or self.high_freq is None),
            sample_rate=sr,
        )
        
    @staticmethod
    def _print_params(params: ButterworthParams) -> None:
        mode = "AUTO" if params.auto_detected else "FIJO"
        print(f"\n[TrainModelButterworth] Parámetros ({mode}):")
        print(f"  Tipo          : {params.filter_type}")
        print(f"  Orden         : {params.order}")
        if params.filter_type == "band":
            print(f"  Low freq      : {params.low_freq:.1f} Hz")
            print(f"  High freq     : {params.high_freq:.1f} Hz")
        else:
            print(f"  Cutoff freq   : {params.cutoff_freq:.1f} Hz")
        if params.spectral_energy_p95_hz:
            print(f"  Energía p95   : {params.spectral_energy_p95_hz:.1f} Hz")
        print(f"  Sample rate   : {params.sample_rate} Hz")
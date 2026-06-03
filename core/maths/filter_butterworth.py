import numpy as np
from scipy.signal import butter, sosfiltfilt
from core.dto.buttherworth_params import ButterworthParams
from core.maths.fft import FFTProcessor

class FilterButterworth:
    def __init__(self, order: int = 4):
        self.order = order
        self._fft = FFTProcessor()
        self.filter_type = None

    def apply_lowpass(
        self,
        signal: np.ndarray,
        sr: int,
        cutoff_freq: float
    ) -> np.ndarray:
        """
        Filtro pasa-bajas Butterworth
        """

        normalized_cutoff = self._normalize_cutoff(sr, cutoff_freq)

        sos = butter(
            self.order,
            normalized_cutoff,
            btype='low',
            analog=False,
            output='sos'
        )

        filtered_signal = sosfiltfilt(sos, signal)

        return filtered_signal

    def apply_highpass(
        self,
        signal: np.ndarray,
        sr: int,
        cutoff_freq: float
    ) -> np.ndarray:
        """
        Filtro pasa-altas Butterworth
        """

        normalized_cutoff = self._normalize_cutoff(sr, cutoff_freq)

        sos = butter(
            self.order,
            normalized_cutoff,
            btype='high',
            analog=False,
            output='sos'
        )

        filtered_signal = sosfiltfilt(sos, signal)

        return filtered_signal

    def apply_bandpass(
        self,
        signal: np.ndarray,
        sr: int,
        low_freq: float,
        high_freq: float
    ) -> np.ndarray:
        """
        Filtro pasa-banda Butterworth
        """

        low, high = self._normalize_band(sr, low_freq, high_freq)

        sos = butter(
            self.order,
            [low, high],
            btype='bandpass',
            output='sos'
        )

        filtered_signal = sosfiltfilt(sos, signal)

        return filtered_signal

    @staticmethod
    def _normalize_cutoff(sr: int, cutoff_freq: float) -> float:
        nyquist = sr / 2
        cutoff = float(cutoff_freq)
        if cutoff <= 0:
            raise ValueError(f"cutoff_freq debe ser mayor que 0, no {cutoff_freq}.")
        if cutoff >= nyquist:
            raise ValueError(
                f"cutoff_freq debe ser menor que Nyquist ({nyquist:.1f} Hz), no {cutoff_freq}."
            )
        return cutoff / nyquist

    @staticmethod
    def _normalize_band(sr: int, low_freq: float, high_freq: float) -> tuple[float, float]:
        nyquist = sr / 2
        low = float(low_freq)
        high = float(high_freq)
        if low <= 0:
            raise ValueError(f"low_freq debe ser mayor que 0, no {low_freq}.")
        if high >= nyquist:
            raise ValueError(
                f"high_freq debe ser menor que Nyquist ({nyquist:.1f} Hz), no {high_freq}."
            )
        if low >= high:
            raise ValueError(
                f"low_freq debe ser menor que high_freq, no {low_freq} >= {high_freq}."
            )
        return low / nyquist, high / nyquist
    
    def _auto_detect_cutoff(self, y: np.ndarray, sr: int) -> ButterworthParams:
        """
        Calcula el percentil de energía espectral acumulada y lo usa
        como frecuencia de corte para el filtro pasa-bajas.
        """
        freqs, magnitude = self._fft.compute_fft(y, sr)
 
        if len(freqs) == 0:
            fallback = min(sr * 0.4, 8000.0)
            print(f"[TrainModelButterworth] FFT vacía, usando cutoff fallback: {fallback:.0f} Hz")
            return ButterworthParams(
                order=self.order,
                cutoff_freq=fallback,
                filter_type=self.filter_type,
                auto_detected=True,
                sample_rate=sr,
            )
 
        # Ignorar DC (freq=0)
        dc_mask = freqs > 0
        freqs = freqs[dc_mask]
        magnitude = magnitude[dc_mask]
 
        energy = magnitude ** 2
        cumulative_energy = np.cumsum(energy)
        if cumulative_energy[-1] == 0:
            cutoff = self.min_cutoff_hz
        else:
            cumulative_norm = cumulative_energy / cumulative_energy[-1]
            threshold = self.energy_percentile / 100.0
            idx = np.searchsorted(cumulative_norm, threshold)
            idx = min(idx, len(freqs) - 1)
            cutoff = float(freqs[idx])
 
        # Aplicar límites
        nyquist = sr / 2
        max_hz = self.max_cutoff_hz if self.max_cutoff_hz else nyquist * 0.9
        cutoff = float(np.clip(cutoff, self.min_cutoff_hz, max_hz))
 
        return ButterworthParams(
            order=self.order,
            cutoff_freq=cutoff,
            filter_type=self.filter_type,
            auto_detected=True,
            spectral_energy_p95_hz=cutoff,
            sample_rate=sr,
        )
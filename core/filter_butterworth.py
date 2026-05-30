import numpy as np
from scipy.signal import butter, filtfilt


class FilterButterworth:
    def __init__(self, order: int = 4):
        self.order = order

    def apply_lowpass(
        self,
        signal: np.ndarray,
        sr: int,
        cutoff_freq: float
    ) -> np.ndarray:
        """
        Filtro pasa-bajas Butterworth
        """

        nyquist = sr / 2
        normalized_cutoff = cutoff_freq / nyquist

        b, a = butter(
            self.order,
            normalized_cutoff,
            btype='low',
            analog=False
        )

        filtered_signal = filtfilt(b, a, signal)

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

        nyquist = sr / 2
        normalized_cutoff = cutoff_freq / nyquist

        b, a = butter(
            self.order,
            normalized_cutoff,
            btype='high',
            analog=False
        )

        filtered_signal = filtfilt(b, a, signal)

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

        nyquist = sr / 2

        low = low_freq / nyquist
        high = high_freq / nyquist

        b, a = butter(
            self.order,
            [low, high],
            btype='band'
        )

        filtered_signal = filtfilt(b, a, signal)

        return filtered_signal
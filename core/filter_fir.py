import numpy as np
from scipy.signal import firwin, lfilter


class FilterFir:
    def __init__(
        self,
        sr: int,
        num_taps: int = 101,
        window: str = "hamming"
    ):
        """
        sr: sample rate
        num_taps: número de coeficientes FIR
        window: hamming, hann, blackman, boxcar
        """

        self.sr = sr
        self.num_taps = num_taps
        self.window = window

        self.coeffs = None
        self.buffer = None

    def design_lowpass(
        self,
        cutoff_freq: float
    ):
        """
        Diseña un FIR pasa-bajas
        """

        self.coeffs = firwin(
            numtaps=self.num_taps,
            cutoff=cutoff_freq,
            fs=self.sr,
            window=self.window
        )

        self.buffer = np.zeros(
            len(self.coeffs),
            dtype=np.float32
        )

    def design_highpass(
        self,
        cutoff_freq: float
    ):
        """
        Diseña un FIR pasa-altas
        """

        self.coeffs = firwin(
            numtaps=self.num_taps,
            cutoff=cutoff_freq,
            fs=self.sr,
            pass_zero=False,
            window=self.window
        )

        self.buffer = np.zeros(
            len(self.coeffs),
            dtype=np.float32
        )

    def design_bandpass(
        self,
        low_freq: float,
        high_freq: float
    ):
        """
        Diseña un FIR pasa-banda
        """

        self.coeffs = firwin(
            numtaps=self.num_taps,
            cutoff=[low_freq, high_freq],
            fs=self.sr,
            pass_zero=False,
            window=self.window
        )

        self.buffer = np.zeros(
            len(self.coeffs),
            dtype=np.float32
        )

    def process_sample(
        self,
        input_sample: float
    ) -> float:
        """
        Procesa una sola muestra
        """

        if self.coeffs is None:
            raise ValueError(
                "Primero diseña el filtro"
            )

        self.buffer[1:] = self.buffer[:-1]
        self.buffer[0] = input_sample

        output_sample = np.sum(
            self.coeffs * self.buffer
        )

        return float(output_sample)

    def process_signal(
        self,
        signal: np.ndarray
    ) -> np.ndarray:
        """
        Procesamiento manual
        """

        output = np.zeros_like(signal)

        for i, sample in enumerate(signal):
            output[i] = self.process_sample(sample)

        return output

    def process_signal_fast(
        self,
        signal: np.ndarray
    ) -> np.ndarray:
        """
        Procesamiento rápido usando lfilter
        """

        if self.coeffs is None:
            raise ValueError(
                "Primero diseña el filtro"
            )

        return lfilter(
            self.coeffs,
            1.0,
            signal
        )
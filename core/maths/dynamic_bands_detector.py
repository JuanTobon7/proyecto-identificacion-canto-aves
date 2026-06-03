import numpy as np
from pathlib import Path
from scipy.signal import stft
from scipy.ndimage import gaussian_filter1d


class DynamicBandsDetector:
    """
    Detecta subbandas dinámicas usando la distribución de energía
    del espectro promedio del audio.
    """

    @staticmethod
    def detect_bands_from_audio(
        audio: np.ndarray | str | Path,
        sr: int,
        low_freq: float,
        high_freq: float,
        n_bands: int = 8,
        smoothing_sigma: float = 3.0,
    ) -> list[tuple[float, float]]:
        """
        Genera exactamente n_bands bandas adaptativas.

        Cada banda contiene aproximadamente la misma cantidad
        de energía espectral.
        """

        # Si es un path, cargar el audio
        if isinstance(audio, (str, Path)):
            try:
                import librosa
                audio, _ = librosa.load(str(audio), sr=sr, mono=True)
            except Exception:
                return [(low_freq, high_freq)]

        # Validar que audio es un array válido
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0 or audio.ndim != 1:
            return [(low_freq, high_freq)]

        # Espectrograma
        freqs, _, zxx = stft(
            audio,
            fs=sr,
            nperseg=1024,
            noverlap=512
        )

        # Magnitud promedio
        spectrum = np.mean(np.abs(zxx), axis=1)

        # Limitar al rango de interés
        mask = (freqs >= low_freq) & (freqs <= high_freq)

        freqs = freqs[mask]
        spectrum = spectrum[mask]

        if len(freqs) < 2:
            return [(low_freq, high_freq)]

        # Suavizar para eliminar fluctuaciones pequeñas
        spectrum = gaussian_filter1d(
            spectrum,
            sigma=smoothing_sigma
        )

        total_energy = np.sum(spectrum)

        if total_energy <= 0:
            step = (high_freq - low_freq) / n_bands

            return [
                (
                    low_freq + i * step,
                    low_freq + (i + 1) * step
                )
                for i in range(n_bands)
            ]

        # Energía acumulada normalizada
        cumulative = np.cumsum(spectrum)
        cumulative /= cumulative[-1]

        # Bordes de cuantiles
        band_edges = [low_freq]

        for i in range(1, n_bands):
            target = i / n_bands

            idx = np.searchsorted(
                cumulative,
                target
            )

            idx = min(idx, len(freqs) - 1)

            band_edges.append(
                float(freqs[idx])
            )

        band_edges.append(high_freq)

        # Construcción de bandas
        bands = []

        for i in range(len(band_edges) - 1):
            low = band_edges[i]
            high = band_edges[i + 1]

            if high > low:
                bands.append((low, high))

        return bands

    @staticmethod
    def get_band_labels(
        bands: list[tuple[float, float]]
    ) -> list[str]:
        return [
            f"{int(low)}-{int(high)}Hz"
            for low, high in bands
        ]
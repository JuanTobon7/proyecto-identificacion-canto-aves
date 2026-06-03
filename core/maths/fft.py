import numpy as np
from typing import Tuple

class FFTProcessor:
    @staticmethod
    def compute_fft(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula FFT (magnitud) del audio mono o estéreo.
        Retorna: (freqs, magnitude)
        """

        # Convertir a mono si es estéreo
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        n = len(y)

        if n == 0:
            return np.array([]), np.array([])

        # FFT positiva
        fft_vals = np.fft.rfft(y)

        magnitude = np.abs(fft_vals) / n

        # Vector de frecuencias
        freqs = np.fft.rfftfreq(n, 1.0 / sr)

        return freqs, magnitude

    @staticmethod
    def compute_band_energies(
        y: np.ndarray,
        sr: int,
        bands: list[tuple[float, float]]
    ) -> np.ndarray:
        """
        Calcula energía por bandas de frecuencia usando FFT.

        bands:
        [
            (0, 500),
            (500, 1000),
            ...
        ]
        """

        freqs, magnitude = FFTProcessor.compute_fft(y, sr)

        energies = []

        for low, high in bands:
            mask = (freqs >= low) & (freqs < high)

            energy = (
                np.sum(magnitude[mask] ** 2)
                if np.any(mask)
                else 0.0
            )

            energies.append(energy)

        return np.array(energies, dtype=np.float32)
    
    @staticmethod
    def build_subbands(
        low: float,
        high: float,
        n: int,
        dynamic_bands: list[dict] | list[tuple[float, float]] | None = None,
    ) -> list[tuple[float, float]]:
        """
        Divide el rango [low, high] en n sub-bandas de igual ancho.

        Si se pasan bandas dinámicas válidas, se usan directamente.
 
        Usado durante la predicción para construir el banco de filtros
        que coincide con el profile_vector del modelo entrenado.
        """
        normalized = FFTProcessor._normalize_dynamic_bands(dynamic_bands, n)
        if normalized is not None:
            return normalized

        edges = np.linspace(low, high, n + 1)
        return [(float(edges[i]), float(edges[i + 1])) for i in range(n)]

    @staticmethod
    def _normalize_dynamic_bands(
        dynamic_bands: list[dict] | list[tuple[float, float]] | None,
        expected_count: int,
    ) -> list[tuple[float, float]] | None:
        if not dynamic_bands:
            return None

        normalized: list[tuple[float, float]] = []
        for band in dynamic_bands:
            try:
                if isinstance(band, dict):
                    low = float(band["low"])
                    high = float(band["high"])
                else:
                    low = float(band[0])
                    high = float(band[1])
            except (TypeError, KeyError, IndexError, ValueError):
                continue
            if high > low:
                normalized.append((low, high))

        if len(normalized) == expected_count:
            return normalized
        return None
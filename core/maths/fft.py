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
    def build_subbands(low: float, high: float, n: int) -> list[tuple[float, float]]:
        """
        Divide el rango [low, high] en n sub-bandas de igual ancho.
 
        Usado durante la predicción para construir el banco de filtros
        que coincide con el profile_vector del modelo entrenado.
        """
        edges = np.linspace(low, high, n + 1)
        return [(float(edges[i]), float(edges[i + 1])) for i in range(n)]
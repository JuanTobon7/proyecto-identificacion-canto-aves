import numpy as np
from core.maths.fft import FFTProcessor


class DynamicBandsDetector:
    """
    Detecta subbandas dinámicas del audio basadas en cambios de energía espectral.
    """

    @staticmethod
    def detect_bands_from_audio(
        audio: np.ndarray,
        sr: int,
        low_freq: float,
        high_freq: float,
        n_bands: int = 8,
        percentile_threshold: float = 30.0,
    ) -> list[tuple[float, float]]:
        """
        Detecta automáticamente subbandas dinámicas del audio analizado.

        Divide la banda [low_freq, high_freq] en n_bands iniciales,
        luego detecta los puntos de máxima energía y ajusta los límites.

        Args:
            audio: señal de audio
            sr: sample rate
            low_freq, high_freq: rango de frecuencias
            n_bands: número de subbandas a crear
            percentile_threshold: percentil para detectar transiciones

        Returns:
            Lista de tuplas (low, high) con las subbandas detectadas
        """
        # Crear subbandas iniciales uniformes
        initial_bands = FFTProcessor.build_subbands(low_freq, high_freq, n_bands, None)

        # Calcular energía en cada banda inicial
        energies = FFTProcessor.compute_band_energies(audio, sr, initial_bands)

        if np.sum(energies) == 0:
            return initial_bands

        # Normalizar energías
        norm_energies = energies / np.sum(energies)

        # Detectar transiciones de energía
        diffs = np.abs(np.diff(norm_energies))
        threshold = np.percentile(diffs, percentile_threshold)

        # Puntos donde hay cambios significativos
        transitions = np.where(diffs > threshold)[0]

        if len(transitions) == 0:
            return initial_bands

        # Construir nuevas subbandas basadas en transiciones
        band_edges = [low_freq]

        for trans_idx in transitions:
            # Usar el punto medio entre bandas que transicionan
            edge = (initial_bands[trans_idx][1] + initial_bands[trans_idx + 1][0]) / 2.0
            if band_edges[-1] < edge < high_freq:
                band_edges.append(edge)

        band_edges.append(high_freq)

        # Agrupar subbandas si hay demasiadas
        dynamic_bands = [(band_edges[i], band_edges[i + 1])
                        for i in range(len(band_edges) - 1)]

        # Si tenemos menos de n_bands, expandir; si más, consolidar
        if len(dynamic_bands) < 3:
            return initial_bands
        if len(dynamic_bands) > n_bands:
            dynamic_bands = DynamicBandsDetector._consolidate_bands(
                dynamic_bands, n_bands
            )

        return dynamic_bands

    @staticmethod
    def _consolidate_bands(
        bands: list[tuple[float, float]],
        target_count: int
    ) -> list[tuple[float, float]]:
        """
        Consolida bandas a un número objetivo agrupándolas.
        """
        if len(bands) <= target_count:
            return bands

        consolidated = []
        bands_per_group = len(bands) / target_count

        for i in range(target_count):
            start_idx = int(i * bands_per_group)
            end_idx = int((i + 1) * bands_per_group)

            if start_idx >= len(bands):
                break
            if end_idx > len(bands):
                end_idx = len(bands)

            low = bands[start_idx][0]
            high = bands[end_idx - 1][1]
            consolidated.append((low, high))

        return consolidated

    @staticmethod
    def get_band_labels(bands: list[tuple[float, float]]) -> list[str]:
        """
        Genera etiquetas legibles para las subbandas.
        """
        return [f"{int(low)}-{int(high)}Hz" for low, high in bands]

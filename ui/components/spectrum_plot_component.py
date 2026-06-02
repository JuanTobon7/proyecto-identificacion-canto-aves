from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QFrame, QVBoxLayout


class SpectrumPlotComponent(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("PlotCard")
        self.setMinimumHeight(460)
        self.figure = Figure(figsize=(8, 5.4), facecolor="#ffffff")
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.subplots(2, 1, sharex=False)
        self.figure.subplots_adjust(hspace=0.48, left=0.09, right=0.98, top=0.94, bottom=0.10)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

    def set_data(
        self,
        original_freqs: np.ndarray,
        original_magnitude: np.ndarray,
        filtered_freqs: np.ndarray,
        filtered_magnitude: np.ndarray,
        profile_vector: np.ndarray | None = None,
        std_energy_vector: np.ndarray | None = None,
        subband_frequencies: list[tuple[float, float]] | None = None,
        original_band_energies: np.ndarray | None = None,
        filtered_band_energies: np.ndarray | None = None,
    ) -> None:
        self.figure.clear()
        self.axes = self.figure.subplots(2, 1, sharex=False)
        original_mag = self._normalize_magnitude(original_magnitude)
        filtered_mag = self._normalize_magnitude(filtered_magnitude)

        # Eje 1: Espectro original con perfil y sub-bandas
        self.axes[0].plot(original_freqs, original_mag, color="#d97706", linewidth=1.6, label="Espectro original")
        self.axes[0].fill_between(original_freqs, 0.0, original_mag, color="#f59e0b", alpha=0.14)
        
        # Agregar visualización de reconocimiento en eje original
        if profile_vector is not None and subband_frequencies is not None and len(profile_vector) > 0:
            self._plot_recognition_data(
                self.axes[0],
                profile_vector,
                std_energy_vector,
                subband_frequencies,
                original_band_energies,
                "original"
            )
        
        self.axes[0].set_title("Espectro original", color="#0f172a", fontweight="bold")
        self.axes[0].set_ylabel("Magnitud", color="#334155")
        self.axes[0].tick_params(labelbottom=False)
        self.axes[0].legend(loc="upper right", fontsize=8)

        # Eje 2: Espectro filtrado con perfil y sub-bandas
        self.axes[1].plot(filtered_freqs, filtered_mag, color="#2563eb", linewidth=1.8, label="Espectro filtrado")
        self.axes[1].fill_between(filtered_freqs, 0.0, filtered_mag, color="#60a5fa", alpha=0.16)
        
        # Agregar visualización de reconocimiento en eje filtrado
        if profile_vector is not None and subband_frequencies is not None and len(profile_vector) > 0:
            self._plot_recognition_data(
                self.axes[1],
                profile_vector,
                std_energy_vector,
                subband_frequencies,
                filtered_band_energies,
                "filtered"
            )
        
        self.axes[1].set_title("Espectro filtrado", color="#0f172a", fontweight="bold")
        self.axes[1].set_ylabel("Magnitud", color="#334155")
        self.axes[1].set_xlabel("Frecuencia (Hz)", color="#334155")
        self.axes[1].legend(loc="upper right", fontsize=8)

        for axis in self.axes:
            axis.set_facecolor("#f8fafc")
            axis.grid(True, color="#cbd5e1", alpha=0.7, linestyle="--", linewidth=0.7)
            axis.tick_params(colors="#334155", labelsize=9)
            for spine in axis.spines.values():
                spine.set_color("#94a3b8")
            axis.set_ylim(bottom=0.0)
            max_freq = self._max_frequency(original_freqs, filtered_freqs)
            if max_freq > 0:
                axis.set_xlim(0, max_freq)

        self.figure.subplots_adjust(hspace=0.48, left=0.09, right=0.98, top=0.94, bottom=0.10)
        self.canvas.draw_idle()

    def _plot_recognition_data(
        self,
        axis,
        profile_vector: np.ndarray,
        std_energy_vector: np.ndarray | None,
        subband_frequencies: list[tuple[float, float]],
        band_energies: np.ndarray | None,
        source: str,
    ) -> None:
        """Grafica líneas de sub-bandas, perfil vector y energía normalizada."""
        
        # Colores según el source
        if source == "original":
            color_lines = "#d97706"
            color_profile = "#ea580c"
            color_energy = "#f59e0b"
        else:
            color_lines = "#2563eb"
            color_profile = "#1d4ed8"
            color_energy = "#60a5fa"
        
        # Líneas verticales delimitando sub-bandas
        for low_freq, high_freq in subband_frequencies:
            axis.axvline(low_freq, color=color_lines, linestyle=":", linewidth=0.9, alpha=0.5)
        
        # Línea final de la última sub-banda
        if subband_frequencies:
            axis.axvline(subband_frequencies[-1][1], color=color_lines, linestyle=":", linewidth=0.9, alpha=0.5)
        
        # Calcular frecuencias centrales para plotear el perfil
        band_centers = np.array([(low + high) / 2.0 for low, high in subband_frequencies])
        
        # Normalizar el profile_vector
        profile_normalized = self._normalize_magnitude(profile_vector)
        
        # Plotear profile_vector como línea
        axis.plot(
            band_centers,
            profile_normalized,
            color=color_profile,
            linewidth=2.0,
            marker="o",
            markersize=5,
            label="Perfil umbral",
            alpha=0.9
        )
        
        # Graficar área de desviación estándar (perfil ± std)
        if std_energy_vector is not None and len(std_energy_vector) == len(profile_vector):
            std_normalized = self._normalize_magnitude(std_energy_vector)
            upper = profile_normalized + std_normalized
            lower = np.maximum(0, profile_normalized - std_normalized)
            axis.fill_between(band_centers, lower, upper, color=color_profile, alpha=0.15, label="Rango ±std")
        
        # Graficar energía actual normalizada si está disponible
        if band_energies is not None and len(band_energies) == len(profile_vector):
            energy_normalized = self._normalize_magnitude(band_energies)
            axis.bar(
                band_centers,
                energy_normalized,
                width=(band_centers[1] - band_centers[0]) * 0.6 if len(band_centers) > 1 else 100,
                color=color_energy,
                alpha=0.4,
                label="Energía actual"
            )

    @staticmethod
    def _normalize_magnitude(magnitude: np.ndarray) -> np.ndarray:
        magnitude = np.asarray(magnitude, dtype=np.float64)
        if magnitude.size == 0:
            return magnitude
        peak = float(np.max(magnitude)) if np.max(magnitude) > 0 else 1.0
        return magnitude / peak

    @staticmethod
    def _max_frequency(original_freqs: np.ndarray, filtered_freqs: np.ndarray) -> float:
        max_freq = 0.0
        if original_freqs.size > 0:
            max_freq = max(max_freq, float(np.max(original_freqs)))
        if filtered_freqs.size > 0:
            max_freq = max(max_freq, float(np.max(filtered_freqs)))
        return max_freq

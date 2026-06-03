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
        subband_frequencies: list[tuple[float, float]] | None = None,
        filtered_freqs: np.ndarray | None = None,
        filtered_magnitude: np.ndarray | None = None,
        comparison_spectrum_profiles: list[dict[str, np.ndarray]] | None = None,
    ) -> None:
        self.figure.clear()
        self.axes = self.figure.subplots(2, 1, sharex=False)
        filtered_freqs = np.asarray(filtered_freqs, dtype=np.float64) if filtered_freqs is not None else np.array([])
        filtered_mag = self._normalize_magnitude(filtered_magnitude) if filtered_magnitude is not None else np.array([])

        # Eje 1: Espectro filtrado con bandas dinámicas
        if filtered_freqs.size > 0 and filtered_mag.size > 0:
            self.axes[0].plot(filtered_freqs, filtered_mag, color="#2563eb", linewidth=1.8, label="Espectro filtrado")
            self.axes[0].fill_between(filtered_freqs, 0.0, filtered_mag, color="#60a5fa", alpha=0.14)

        if subband_frequencies:
            self._plot_subband_lines(self.axes[0], subband_frequencies)

        self.axes[0].set_title("Espectro filtrado", color="#0f172a", fontweight="bold")
        self.axes[0].set_ylabel("Magnitud", color="#334155")
        self.axes[0].legend(loc="upper right", fontsize=8)

        # Eje 2: Comparación espectral contra las otras especies del modelo
        if filtered_freqs.size > 0 and filtered_mag.size > 0:
            self.axes[1].plot(filtered_freqs, filtered_mag, color="#2563eb", linewidth=1.6, label="Espectro filtrado")
        if comparison_spectrum_profiles:
            colors = ["#f97316", "#16a34a", "#8b5cf6", "#ef4444", "#0f766e"]
            for index, profile in enumerate(comparison_spectrum_profiles):
                species = str(profile.get("species", f"Especie {index + 1}"))
                freqs = np.asarray(profile.get("freqs", []), dtype=np.float64)
                magnitude = np.asarray(profile.get("magnitude", []), dtype=np.float64)
                if freqs.size == 0 or magnitude.size == 0:
                    continue
                color = colors[index % len(colors)]
                self.axes[1].plot(freqs, self._normalize_magnitude(magnitude), linewidth=1.4, linestyle="--", color=color, label=species)

        self.axes[1].set_title("Comparación espectral", color="#0f172a", fontweight="bold")
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
            max_freq = self._max_frequency(filtered_freqs, None)
            if max_freq > 0:
                axis.set_xlim(0, max_freq)

        if filtered_freqs.size > 0:
            max_freq = float(np.max(filtered_freqs))
            self.axes[0].set_xlim(0, max_freq)
            self.axes[1].set_xlim(0, max_freq)

        self.figure.subplots_adjust(hspace=0.48, left=0.09, right=0.98, top=0.94, bottom=0.10)
        self.canvas.draw_idle()

    def _plot_subband_lines(self, axis, subband_frequencies: list[tuple[float, float]]) -> None:
        color_lines = "#1d4ed8"
        for low_freq, high_freq in subband_frequencies:
            axis.axvline(low_freq, color=color_lines, linestyle="-", linewidth=1.2, alpha=0.55)
        if subband_frequencies:
            axis.axvline(subband_frequencies[-1][1], color=color_lines, linestyle="-", linewidth=1.2, alpha=0.55)

    @staticmethod
    def _normalize_magnitude(magnitude: np.ndarray) -> np.ndarray:
        magnitude = np.asarray(magnitude, dtype=np.float64)
        if magnitude.size == 0:
            return magnitude
        peak = float(np.max(magnitude)) if np.max(magnitude) > 0 else 1.0
        return magnitude / peak

    @staticmethod
    def _max_frequency(original_freqs: np.ndarray, filtered_freqs: np.ndarray | None) -> float:
        max_freq = 0.0
        if original_freqs.size > 0:
            max_freq = max(max_freq, float(np.max(original_freqs)))
        if filtered_freqs is not None and filtered_freqs.size > 0:
            max_freq = max(max_freq, float(np.max(filtered_freqs)))
        return max_freq

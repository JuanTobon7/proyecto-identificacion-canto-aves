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

    def set_data(self, original_freqs: np.ndarray, original_magnitude: np.ndarray, filtered_freqs: np.ndarray, filtered_magnitude: np.ndarray) -> None:
        self.figure.clear()
        self.axes = self.figure.subplots(2, 1, sharex=False)
        original_mag = self._normalize_magnitude(original_magnitude)
        filtered_mag = self._normalize_magnitude(filtered_magnitude)

        self.axes[0].plot(original_freqs, original_mag, color="#d97706", linewidth=1.6)
        self.axes[0].fill_between(original_freqs, 0.0, original_mag, color="#f59e0b", alpha=0.14)
        self.axes[0].set_title("Espectro original", color="#0f172a", fontweight="bold")
        self.axes[0].set_ylabel("Magnitud", color="#334155")
        self.axes[0].tick_params(labelbottom=False)

        self.axes[1].plot(filtered_freqs, filtered_mag, color="#2563eb", linewidth=1.8)
        self.axes[1].fill_between(filtered_freqs, 0.0, filtered_mag, color="#60a5fa", alpha=0.16)
        self.axes[1].set_title("Espectro filtrado", color="#0f172a", fontweight="bold")
        self.axes[1].set_ylabel("Magnitud", color="#334155")
        self.axes[1].set_xlabel("Frecuencia (Hz)", color="#334155")

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

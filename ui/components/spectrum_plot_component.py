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
        self.figure = Figure(figsize=(8, 5.4), facecolor="#0f172a")
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.subplots(2, 1, sharex=False)
        self.figure.subplots_adjust(hspace=0.48, left=0.09, right=0.98, top=0.94, bottom=0.10)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

    def set_data(self, original_freqs: np.ndarray, original_magnitude: np.ndarray, filtered_freqs: np.ndarray, filtered_magnitude: np.ndarray) -> None:
        self.figure.clear()
        self.axes = self.figure.subplots(2, 1, sharex=False)
        original_db = self._to_db(original_magnitude)
        filtered_db = self._to_db(filtered_magnitude)

        self.axes[0].plot(original_freqs, original_db, color="#fbbf24", linewidth=1.6)
        self.axes[0].fill_between(original_freqs, original_db, original_db.min(initial=-120.0), color="#f59e0b", alpha=0.16)
        self.axes[0].set_title("Espectro original", color="#e5e7eb", fontweight="bold")
        self.axes[0].set_ylabel("Magnitud (dB)", color="#cbd5e1")
        self.axes[0].tick_params(labelbottom=False)

        self.axes[1].plot(filtered_freqs, filtered_db, color="#c084fc", linewidth=1.8)
        self.axes[1].fill_between(filtered_freqs, filtered_db, filtered_db.min(initial=-120.0), color="#a855f7", alpha=0.18)
        self.axes[1].set_title("Espectro filtrado", color="#e5e7eb", fontweight="bold")
        self.axes[1].set_ylabel("Magnitud (dB)", color="#cbd5e1")
        self.axes[1].set_xlabel("Frecuencia (Hz)", color="#cbd5e1")

        for axis in self.axes:
            axis.set_facecolor("#111827")
            axis.grid(True, color="#475569", alpha=0.45, linestyle="--", linewidth=0.7)
            axis.tick_params(colors="#cbd5e1", labelsize=9)
            for spine in axis.spines.values():
                spine.set_color("#334155")
            axis.set_ylim(-120, 5)
            if original_freqs.size > 0:
                axis.set_xlim(0, float(max(original_freqs.max(), filtered_freqs.max(initial=0.0))))

        self.figure.subplots_adjust(hspace=0.48, left=0.09, right=0.98, top=0.94, bottom=0.10)
        self.canvas.draw_idle()

    @staticmethod
    def _to_db(magnitude: np.ndarray) -> np.ndarray:
        magnitude = np.asarray(magnitude, dtype=np.float64)
        if magnitude.size == 0:
            return magnitude
        reference = float(np.max(magnitude)) if np.max(magnitude) > 0 else 1.0
        return 20.0 * np.log10(np.maximum(magnitude / reference, 1e-8))

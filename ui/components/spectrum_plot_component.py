from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QFrame, QVBoxLayout


class SpectrumPlotComponent(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("PlotCard")
        self.figure = Figure(figsize=(8, 4), facecolor="#0f172a")
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.subplots(2, 1, sharex=False)
        self.figure.tight_layout(pad=2.0)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

    def set_data(self, original_freqs: np.ndarray, original_magnitude: np.ndarray, filtered_freqs: np.ndarray, filtered_magnitude: np.ndarray) -> None:
        self.figure.clear()
        self.axes = self.figure.subplots(2, 1, sharex=False)
        self.axes[0].plot(original_freqs, original_magnitude, color="#f59e0b", linewidth=0.8)
        self.axes[0].set_title("Espectro original", color="#e5e7eb")
        self.axes[1].plot(filtered_freqs, filtered_magnitude, color="#a78bfa", linewidth=0.8)
        self.axes[1].set_title("Espectro filtrado", color="#e5e7eb")

        for axis in self.axes:
            axis.set_facecolor("#111827")
            axis.grid(True, color="#334155", alpha=0.35)
            axis.tick_params(colors="#cbd5e1")

        self.figure.tight_layout(pad=2.0)
        self.canvas.draw_idle()

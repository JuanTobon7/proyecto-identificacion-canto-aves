from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QFrame, QVBoxLayout


class EnergyVectorComponent(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("PlotCard")
        self.figure = Figure(figsize=(8, 3), facecolor="#0f172a")
        self.canvas = FigureCanvas(self.figure)
        self.axis = self.figure.add_subplot(111)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

    def set_data(self, original: np.ndarray, filtered: np.ndarray, labels: list[str]) -> None:
        self.axis.clear()
        index = np.arange(len(labels))
        width = 0.4
        self.axis.bar(index - width / 2, original, width=width, color="#38bdf8", label="Original")
        self.axis.bar(index + width / 2, filtered, width=width, color="#22c55e", label="Filtrada")
        self.axis.set_xticks(index)
        self.axis.set_xticklabels(labels, rotation=25, ha="right", fontsize=8, color="#cbd5e1")
        self.axis.set_title("Vector de energia", color="#e5e7eb")
        self.axis.set_facecolor("#111827")
        self.axis.grid(True, axis="y", color="#334155", alpha=0.35)
        self.axis.tick_params(colors="#cbd5e1")
        self.axis.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e5e7eb")
        self.figure.tight_layout(pad=2.0)
        self.canvas.draw_idle()

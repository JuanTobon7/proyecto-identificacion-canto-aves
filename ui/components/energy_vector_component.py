from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QFrame, QVBoxLayout


class EnergyVectorComponent(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("PlotCard")
        self.figure = Figure(figsize=(8, 3), facecolor="#ffffff")
        self.canvas = FigureCanvas(self.figure)
        self.axis = self.figure.add_subplot(111)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

    def set_data(
        self,
        current: np.ndarray,
        labels: list[str],
        comparison_profiles: list[dict[str, np.ndarray]] | None = None,
    ) -> None:
        self.axis.clear()
        index = np.arange(len(labels), dtype=np.float64)
        self.axis.plot(index, current, color="#2563eb", linewidth=2.0, marker="o", markersize=4, label="Espectro filtrado")
        if comparison_profiles:
            colors = ["#f97316", "#16a34a", "#8b5cf6", "#ef4444", "#0f766e"]
            for idx, profile in enumerate(comparison_profiles):
                species = str(profile.get("species", f"Especie {idx + 1}"))
                vector = np.asarray(profile.get("vector", []), dtype=np.float64)
                if vector.size != index.size or vector.size == 0:
                    continue
                color = colors[idx % len(colors)]
                self.axis.plot(index, vector, color=color, linewidth=1.8, linestyle="--", marker="s", markersize=4, label=species)
        self.axis.set_xticks(index)
        self.axis.set_xticklabels(labels, rotation=25, ha="right", fontsize=8, color="#334155")
        self.axis.set_title("Comparación de vectores de energía", color="#0f172a")
        self.axis.set_facecolor("#f8fafc")
        self.axis.grid(True, axis="y", color="#cbd5e1", alpha=0.6)
        self.axis.tick_params(colors="#334155")
        self.axis.legend(facecolor="#ffffff", edgecolor="#cbd5e1", labelcolor="#0f172a")
        self.figure.tight_layout(pad=2.0)
        self.canvas.draw_idle()

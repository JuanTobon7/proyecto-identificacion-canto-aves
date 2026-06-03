from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QFrame, QVBoxLayout


class EnergyVectorComponent(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("PlotCard")
        self.figure = Figure(figsize=(14, 5), facecolor="#ffffff")
        self.canvas = FigureCanvas(self.figure)
        self.axes = None

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

    def set_data(
        self,
        current: np.ndarray,
        labels: list[str],
        comparison_profiles: list[dict[str, np.ndarray]] | None = None,
    ) -> None:
        self.figure.clear()

        # Determine number of comparison profiles to display
        num_comparisons = len(comparison_profiles) if comparison_profiles else 0
        num_subplots = num_comparisons + 1  # +1 para el ave analizada

        # Create subplots: one for the current species and one for each comparison
        self.axes = self.figure.subplots(1, num_subplots, sharey=True)
        if num_subplots == 1:
            self.axes = [self.axes]

        colors_species = ["#f97316", "#16a34a", "#8b5cf6", "#ef4444", "#0f766e"]

        # Plot the analyzed bird (current)
        ax = self.axes[0]
        index = np.arange(len(labels), dtype=np.float64)
        ax.plot(index, current, color="#2563eb", linewidth=2.2, marker="o", markersize=5, label="Analizado")
        ax.fill_between(index, 0, current, color="#2563eb", alpha=0.15)
        ax.set_xticks(index)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7, color="#334155")
        ax.set_title("Ave Analizada\n(tiempo real)", color="#0f172a", fontsize=9, fontweight="bold")
        ax.set_facecolor("#f8fafc")
        ax.grid(True, axis="y", color="#cbd5e1", alpha=0.5)
        ax.tick_params(colors="#334155", labelsize=8)

        # Plot comparison species with their own band labels
        if comparison_profiles:
            for idx, profile in enumerate(comparison_profiles):
                if idx + 1 >= len(self.axes):
                    break

                ax = self.axes[idx + 1]
                species = str(profile.get("species", f"Especie {idx + 1}"))
                vector = np.asarray(profile.get("vector", []), dtype=np.float64)
                band_labels = profile.get("band_labels", [f"B{i+1}" for i in range(len(vector))])

                if vector.size == 0:
                    continue

                color = colors_species[idx % len(colors_species)]
                index_comp = np.arange(len(vector), dtype=np.float64)

                ax.plot(index_comp, vector, color=color, linewidth=2.2, marker="s", markersize=5, label=species)
                ax.fill_between(index_comp, 0, vector, color=color, alpha=0.15)
                ax.set_xticks(index_comp)
                ax.set_xticklabels(band_labels, rotation=25, ha="right", fontsize=7, color="#334155")
                ax.set_title(f"{species}\n(referencia)", color="#0f172a", fontsize=9, fontweight="bold")
                ax.set_facecolor("#f8fafc")
                ax.grid(True, axis="y", color="#cbd5e1", alpha=0.5)
                ax.tick_params(colors="#334155", labelsize=8)

        # Set common ylabel
        self.axes[0].set_ylabel("Energía", color="#334155", fontsize=9)

        self.figure.suptitle("Comparación de vectores de energía por subbandas",
                            color="#0f172a", fontsize=10, fontweight="bold", y=1.00)
        self.figure.tight_layout(pad=1.5)
        self.canvas.draw_idle()

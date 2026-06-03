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

        # Use comparison_profiles if available, otherwise use current + labels
        if comparison_profiles and len(comparison_profiles) > 0:
            num_subplots = len(comparison_profiles)
            profiles_to_plot = comparison_profiles
        else:
            num_subplots = 1
            profiles_to_plot = [{
                "species": "Ave Analizada",
                "vector": current,
                "band_labels": labels,
            }]

        # Create subplots ensuring axes is always a list
        if num_subplots > 1:
            axes_result = self.figure.subplots(1, num_subplots, sharey=True)
            self.axes = list(axes_result) if isinstance(axes_result, np.ndarray) else [axes_result]
        else:
            self.axes = [self.figure.add_subplot(111)]

        colors_species = ["#2563eb", "#f97316", "#16a34a", "#8b5cf6", "#ef4444", "#0f766e"]

        # Plot each species
        for idx, profile in enumerate(profiles_to_plot):
            if idx >= len(self.axes):
                break

            ax = self.axes[idx]
            species = str(profile.get("species", f"Especie {idx + 1}"))
            vector = np.asarray(profile.get("vector", []), dtype=np.float64)
            band_labels = profile.get("band_labels", [f"Banda {i+1}" for i in range(len(vector))])

            if vector.size == 0:
                continue

            color = colors_species[idx % len(colors_species)]
            index_comp = np.arange(len(vector), dtype=np.float64)

            ax.plot(index_comp, vector, color=color, linewidth=2.2, marker="o", markersize=5)
            ax.fill_between(index_comp, 0, vector, color=color, alpha=0.15)
            ax.set_xticks(index_comp)
            ax.set_xticklabels(band_labels, rotation=35, ha="right", fontsize=8, color="#334155")

            # Set title
            if idx == 0 and comparison_profiles and len(comparison_profiles) > 0:
                ax.set_title(f"{species}\n(tiempo real)", color="#0f172a", fontsize=9, fontweight="bold")
            else:
                ax.set_title(f"{species}", color="#0f172a", fontsize=9, fontweight="bold")

            ax.set_facecolor("#f8fafc")
            ax.grid(True, axis="y", color="#cbd5e1", alpha=0.5)
            ax.tick_params(colors="#334155", labelsize=8)
            ax.set_ylim(bottom=0)

        # Set common ylabel
        if self.axes:
            self.axes[0].set_ylabel("Energía", color="#334155", fontsize=9)

        self.figure.suptitle("Energía de cada ave en sus subbandas características",
                            color="#0f172a", fontsize=10, fontweight="bold", y=0.98)
        self.figure.tight_layout(pad=1.5)
        self.canvas.draw_idle()

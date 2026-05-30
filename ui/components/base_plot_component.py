from __future__ import annotations

from abc import ABC, abstractmethod

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget


class BasePlotComponent(QWidget, ABC):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(7, 3), facecolor="#0f172a")
        self.canvas = FigureCanvas(self.figure)
        self.axis = self.figure.add_subplot(111)
        self.axis.set_title(title, color="#e5e7eb")
        self.axis.set_facecolor("#111827")
        self.axis.tick_params(colors="#cbd5e1")
        self.axis.grid(True, color="#334155", alpha=0.35)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def clear_plot(self) -> None:
        self.axis.clear()
        self.axis.set_facecolor("#111827")
        self.axis.tick_params(colors="#cbd5e1")
        self.axis.grid(True, color="#334155", alpha=0.35)
        self.canvas.draw_idle()

    @abstractmethod
    def set_data(self, *args, **kwargs) -> None:
        raise NotImplementedError

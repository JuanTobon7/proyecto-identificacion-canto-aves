from __future__ import annotations

from abc import ABC, abstractmethod

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget


class BasePlotComponent(QWidget, ABC):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(7, 3), facecolor="#ffffff")
        self.canvas = FigureCanvas(self.figure)
        self.axis = self.figure.add_subplot(111)
        self.axis.set_title(title, color="#0f172a")
        self.axis.set_facecolor("#f8fafc")
        self.axis.tick_params(colors="#334155")
        self.axis.grid(True, color="#cbd5e1", alpha=0.6)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def clear_plot(self) -> None:
        self.axis.clear()
        self.axis.set_facecolor("#f8fafc")
        self.axis.tick_params(colors="#334155")
        self.axis.grid(True, color="#cbd5e1", alpha=0.6)
        self.canvas.draw_idle()

    @abstractmethod
    def set_data(self, *args, **kwargs) -> None:
        raise NotImplementedError

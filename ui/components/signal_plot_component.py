from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QFrame, QVBoxLayout


class SignalPlotComponent(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("PlotCard")
        self.figure = Figure(figsize=(8, 4), facecolor="#ffffff")
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.subplots(2, 1, sharex=True)
        self.figure.tight_layout(pad=2.0)
        self._cursor_lines: list[object] = []
        self._duration_sec = 0.0

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

    def set_data(self, original: np.ndarray, filtered: np.ndarray, sample_rate: int) -> None:
        self.figure.clear()
        self.axes = self.figure.subplots(2, 1, sharex=True)
        time_original = np.arange(original.size) / sample_rate if sample_rate else np.arange(original.size)
        time_filtered = np.arange(filtered.size) / sample_rate if sample_rate else np.arange(filtered.size)
        self._duration_sec = float(max(time_original[-1] if time_original.size else 0.0, time_filtered[-1] if time_filtered.size else 0.0))

        self.axes[0].plot(time_original, original, color="#0284c7", linewidth=0.8)
        self.axes[0].set_title("Señal original", color="#0f172a")
        self.axes[1].plot(time_filtered, filtered, color="#16a34a", linewidth=0.8)
        self.axes[1].set_title("Señal filtrada", color="#0f172a")

        for axis in self.axes:
            axis.set_facecolor("#f8fafc")
            axis.grid(True, color="#cbd5e1", alpha=0.6)
            axis.tick_params(colors="#334155")
            for spine in axis.spines.values():
                spine.set_color("#94a3b8")

        self._cursor_lines = [axis.axvline(0.0, color="#f97316", linestyle="--", linewidth=1.1, alpha=0.95) for axis in self.axes]
        self._set_cursor_visible(False)

        self.figure.tight_layout(pad=2.0)
        self.canvas.draw_idle()

    def set_playback_position(self, position_sec: float) -> None:
        if not self._cursor_lines:
            return
        cursor_position = float(max(0.0, position_sec))
        self._set_cursor_visible(True)
        for line in self._cursor_lines:
            line.set_xdata([cursor_position, cursor_position])
        self.canvas.draw_idle()

    def clear_playback_position(self) -> None:
        self._set_cursor_visible(False)
        self.canvas.draw_idle()

    def _set_cursor_visible(self, visible: bool) -> None:
        for line in self._cursor_lines:
            line.set_visible(visible)

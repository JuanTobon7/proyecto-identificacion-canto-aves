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

        # Almacenar xlim originales para cada eje
        self._original_xlim = [None, None]

        # Variables para pan (arrastre)
        self._press = None
        self._xpress = None

        # Conectar eventos para zoom y pan
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

    def _on_scroll(self, event) -> None:
        """Zoom interactivo con scroll del mouse en el eje X (frecuencia)."""
        if event.inaxes is None or event.button is None:
            return

        # Encontrar cuál eje es
        ax_idx = None
        if event.inaxes == self.axes[0]:
            ax_idx = 0
        elif event.inaxes == self.axes[1]:
            ax_idx = 1
        else:
            return

        ax = self.axes[ax_idx]

        # Obtener posición actual del mouse
        xdata = event.xdata
        if xdata is None:
            return

        # Zoom factor
        zoom_factor = 1.5 if event.button == "up" else 1.0 / 1.5

        # Rango actual
        cur_xlim = ax.get_xlim()
        cur_center = (cur_xlim[0] + cur_xlim[1]) / 2.0
        cur_width = cur_xlim[1] - cur_xlim[0]

        # Nuevo rango centrado en la posición del mouse
        new_width = cur_width / zoom_factor
        new_left = xdata - (xdata - cur_xlim[0]) * (new_width / cur_width)
        new_right = xdata + (cur_xlim[1] - xdata) * (new_width / cur_width)

        # Limitar a bordes originales
        if self._original_xlim[ax_idx]:
            orig_left, orig_right = self._original_xlim[ax_idx]
            new_left = max(new_left, orig_left)
            new_right = min(new_right, orig_right)

        ax.set_xlim(new_left, new_right)
        self.canvas.draw_idle()

    def _on_press(self, event) -> None:
        """Detectar click del botón izquierdo para pan."""
        if event.inaxes is None or event.button != 1:
            return

        self._press = event.inaxes
        self._xpress = event.xdata

    def _on_release(self, event) -> None:
        """Liberar el pan."""
        self._press = None
        self._xpress = None

    def _on_motion(self, event) -> None:
        """Manejar arrastre para pan horizontal."""
        if self._press is None or event.xdata is None:
            return

        ax = self._press
        if ax not in self.axes:
            return

        # Calcular el desplazamiento
        dx = event.xdata - self._xpress
        cur_xlim = ax.get_xlim()
        new_left = cur_xlim[0] - dx
        new_right = cur_xlim[1] - dx

        # Limitar a bordes originales
        ax_idx = 0 if ax == self.axes[0] else 1
        if self._original_xlim[ax_idx]:
            orig_left, orig_right = self._original_xlim[ax_idx]
            if new_left < orig_left:
                diff = orig_left - new_left
                new_left = orig_left
                new_right += diff
            if new_right > orig_right:
                diff = new_right - orig_right
                new_right = orig_right
                new_left -= diff

        ax.set_xlim(new_left, new_right)
        self.canvas.draw_idle()

    def set_data(
        self,
        subband_frequencies: list[tuple[float, float]] | None = None,
        filtered_freqs: np.ndarray | None = None,
        filtered_magnitude: np.ndarray | None = None,
        comparison_spectrum_profiles: list[dict[str, np.ndarray]] | None = None,
    ) -> None:
        self.figure.clear()
        self.axes = self.figure.subplots(2, 1, sharex=False)
        filtered_freqs = np.asarray(filtered_freqs, dtype=np.float64) if filtered_freqs is not None else np.array([])
        filtered_mag = self._normalize_magnitude(filtered_magnitude) if filtered_magnitude is not None else np.array([])

        # Eje 1: Espectro filtrado con bandas dinámicas
        if filtered_freqs.size > 0 and filtered_mag.size > 0:
            self.axes[0].plot(filtered_freqs, filtered_mag, color="#2563eb", linewidth=1.8, label="Espectro filtrado")

        if subband_frequencies:
            self._plot_subband_lines(self.axes[0], subband_frequencies, filtered_freqs, filtered_mag)

        self.axes[0].set_title("Espectro filtrado", color="#0f172a", fontweight="bold")
        self.axes[0].set_ylabel("Magnitud", color="#334155")
        self.axes[0].legend(loc="upper right", fontsize=8)

        # Eje 2: Comparación espectral contra las otras especies del modelo
        if filtered_freqs.size > 0 and filtered_mag.size > 0:
            self.axes[1].plot(filtered_freqs, filtered_mag, color="#2563eb", linewidth=1.6, label="Espectro filtrado")
        if comparison_spectrum_profiles:
            colors = ["#f97316", "#16a34a", "#8b5cf6", "#ef4444", "#0f766e"]
            for index, profile in enumerate(comparison_spectrum_profiles):
                species = str(profile.get("species", f"Especie {index + 1}"))
                freqs = np.asarray(profile.get("freqs", []), dtype=np.float64)
                magnitude = np.asarray(profile.get("magnitude", []), dtype=np.float64)
                if freqs.size == 0 or magnitude.size == 0:
                    continue
                color = colors[index % len(colors)]
                self.axes[1].plot(freqs, self._normalize_magnitude(magnitude), linewidth=1.4, linestyle="--", color=color, label=species)

        self.axes[1].set_title("Comparación espectral", color="#0f172a", fontweight="bold")
        self.axes[1].set_ylabel("Magnitud", color="#334155")
        self.axes[1].set_xlabel("Frecuencia (Hz)", color="#334155")
        self.axes[1].legend(loc="upper right", fontsize=8)

        for axis in self.axes:
            axis.set_facecolor("#f8fafc")
            axis.grid(True, color="#cbd5e1", alpha=0.7, linestyle="--", linewidth=0.7)
            axis.tick_params(colors="#334155", labelsize=9)
            for spine in axis.spines.values():
                spine.set_color("#94a3b8")
            axis.set_ylim(bottom=0.0)
            max_freq = self._max_frequency(filtered_freqs, None)
            if max_freq > 0:
                axis.set_xlim(0, max_freq)

        if filtered_freqs.size > 0:
            max_freq = float(np.max(filtered_freqs))
            self.axes[0].set_xlim(0, max_freq)
            self.axes[1].set_xlim(0, max_freq)
            self._original_xlim = [(0, max_freq), (0, max_freq)]
        else:
            self._original_xlim = [None, None]

        self.figure.subplots_adjust(hspace=0.48, left=0.09, right=0.98, top=0.94, bottom=0.10)
        self.canvas.draw_idle()

    def _plot_subband_lines(self, axis, subband_frequencies: list[tuple[float, float]],
                           filtered_freqs: np.ndarray | None = None,
                           filtered_mag: np.ndarray | None = None) -> None:
        colors_bands = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#14b8a6", "#f97316"]

        # Get axis limits for filling
        ylim = axis.get_ylim()
        if ylim[1] == 0:  # If ylim not set yet
            ylim = (0, 1)

        # Fill each subband with a different color
        for idx, (low_freq, high_freq) in enumerate(subband_frequencies):
            color = colors_bands[idx % len(colors_bands)]
            axis.axvspan(low_freq, high_freq, alpha=0.08, color=color)

            # Draw vertical lines at band edges
            axis.axvline(low_freq, color=color, linestyle="-", linewidth=1.2, alpha=0.55)

            # Add frequency label at top of band
            mid_freq = (low_freq + high_freq) / 2.0
            freq_label = f"{int(low_freq)}-{int(high_freq)}Hz"
            axis.text(mid_freq, ylim[1] * 0.95, freq_label,
                     ha="center", va="top", fontsize=7, color=color, fontweight="bold", alpha=0.8)

        # Draw the last edge
        if subband_frequencies:
            last_high = subband_frequencies[-1][1]
            color = colors_bands[(len(subband_frequencies) - 1) % len(colors_bands)]
            axis.axvline(last_high, color=color, linestyle="-", linewidth=1.2, alpha=0.55)

    @staticmethod
    def _normalize_magnitude(magnitude: np.ndarray) -> np.ndarray:
        magnitude = np.asarray(magnitude, dtype=np.float64)
        if magnitude.size == 0:
            return magnitude
        peak = float(np.max(magnitude)) if np.max(magnitude) > 0 else 1.0
        return magnitude / peak

    @staticmethod
    def _max_frequency(original_freqs: np.ndarray, filtered_freqs: np.ndarray | None) -> float:
        max_freq = 0.0
        if original_freqs.size > 0:
            max_freq = max(max_freq, float(np.max(original_freqs)))
        if filtered_freqs is not None and filtered_freqs.size > 0:
            max_freq = max(max_freq, float(np.max(filtered_freqs)))
        return max_freq

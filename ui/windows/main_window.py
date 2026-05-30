from __future__ import annotations

from PySide6.QtWidgets import QApplication, QFrame, QMainWindow, QMessageBox, QScrollArea, QSplitter, QVBoxLayout, QWidget

from ui.components.bird_info_component import BirdInfoComponent
from ui.components.butterworth_info_component import ButterworthInfoComponent
from ui.components.energy_vector_component import EnergyVectorComponent
from ui.components.metrics_component import MetricsComponent
from ui.components.prediction_card_component import PredictionCardComponent
from ui.components.signal_plot_component import SignalPlotComponent
from ui.components.sidebar_component import SidebarComponent
from ui.components.spectrum_plot_component import SpectrumPlotComponent
from ui.controllers.prediction_controller import PredictionController
from ui.dtos.prediction_result import PredictionResult


class MainWindow(QMainWindow):
    def __init__(self, controller: PredictionController | None = None) -> None:
        super().__init__()
        self.controller = controller or PredictionController()
        self.setWindowTitle("Clasificador de cantos de aves")
        self.resize(1500, 980)

        self.sidebar = SidebarComponent()
        self.prediction_card = PredictionCardComponent()
        self.bird_info = BirdInfoComponent()
        self.signal_plot = SignalPlotComponent()
        self.spectrum_plot = SpectrumPlotComponent()
        self.energy_vector = EnergyVectorComponent()
        self.filter_info = ButterworthInfoComponent()
        self.metrics = MetricsComponent()
        self._last_result: PredictionResult | None = None

        self._build_layout()
        self._bind_controller()
        self._load_models()

    def _build_layout(self) -> None:
        root = QSplitter()
        root.setChildrenCollapsible(False)

        root.addWidget(self.sidebar)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.addWidget(self.prediction_card)
        content_layout.addWidget(self.bird_info)
        content_layout.addWidget(self.signal_plot)
        content_layout.addWidget(self.spectrum_plot)
        content_layout.addWidget(self.energy_vector)
        content_layout.addWidget(self.filter_info)
        content_layout.addWidget(self.metrics)
        content_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)

        root.addWidget(scroll)
        root.setStretchFactor(0, 0)
        root.setStretchFactor(1, 1)

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.addWidget(root)
        self.setCentralWidget(container)

    def _bind_controller(self) -> None:
        self.sidebar.processRequested.connect(self._handle_process)
        self.sidebar.playRequested.connect(self._handle_play)
        self.sidebar.playFilteredRequested.connect(self._handle_play_filtered)
        self.sidebar.recordRequested.connect(self._handle_record)
        self.controller.prediction_started.connect(lambda: self.sidebar.set_busy(True))
        self.controller.prediction_finished.connect(self._apply_result)
        self.controller.prediction_failed.connect(self._show_error)
        self.controller.playback_started.connect(lambda: self.sidebar.set_busy(True, "Reproduciendo audio..."))
        self.controller.playback_started.connect(lambda: self.signal_plot.set_playback_position(0.0))
        self.controller.playback_position_changed.connect(self.signal_plot.set_playback_position)
        self.controller.playback_finished.connect(self.signal_plot.clear_playback_position)
        self.controller.playback_finished.connect(lambda: self.sidebar.set_busy(False))
        self.controller.recording_started.connect(lambda: self.sidebar.set_busy(True, "Grabando 3 segundos..."))
        self.controller.recording_finished.connect(self._apply_result)

    def _load_models(self) -> None:
        self.sidebar.set_models(self.controller.available_models())

    def _handle_process(self, model_name: str, audio_path: str) -> None:
        if not model_name:
            self._show_error("No hay modelo seleccionado.")
            return
        if not audio_path:
            self._show_error("Selecciona un archivo de audio.")
            return
        self.controller.run_prediction(model_name, audio_path)

    def _handle_play(self, audio_path: str) -> None:
        if not audio_path:
            self._show_error("Selecciona un archivo de audio para reproducir.")
            return
        self.controller.play_audio(audio_path)

    def _handle_record(self, model_name: str) -> None:
        if not model_name:
            self._show_error("Selecciona un modelo antes de grabar.")
            return
        self.controller.record_and_predict(model_name, seconds=3.0)

    def _handle_play_filtered(self) -> None:
        if self._last_result is None or self._last_result.filtered_signal.size == 0:
            self._show_error("Primero procesa o graba un audio para reproducir el filtrado.")
            return
        self.controller.play_signal(self._last_result.filtered_signal, self._last_result.sample_rate)

    def _apply_result(self, result: PredictionResult) -> None:
        self.sidebar.set_busy(False)
        self._last_result = result
        self.prediction_card.set_result(result)
        self.bird_info.set_bird_info(result.bird_info)
        self.signal_plot.set_data(result.original_signal, result.filtered_signal, result.sample_rate)
        self.signal_plot.clear_playback_position()
        self.spectrum_plot.set_data(result.original_freqs, result.original_magnitude, result.filtered_freqs, result.filtered_magnitude)
        self.energy_vector.set_data(result.original_energy_vector, result.filtered_energy_vector, result.band_labels)
        self.filter_info.set_parameters(result.butterworth_params)
        self.metrics.set_metrics(result.original_stats.to_dict(), result.filtered_stats.to_dict())

    def _show_error(self, message: str) -> None:
        self.sidebar.set_busy(False)
        self.signal_plot.clear_playback_position()
        QMessageBox.critical(self, "Error de inferencia", message)

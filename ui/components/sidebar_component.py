from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDoubleSpinBox, QFrame, QFormLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout

from config.app_settings import AppSettings
from ui.components.audio_selector_component import AudioSelectorComponent
from ui.components.loading_component import LoadingComponent
from ui.components.model_selector_component import ModelSelectorComponent


class SidebarComponent(QFrame):
    processRequested = Signal(str, str)
    playRequested = Signal(str)
    playFilteredRequested = Signal()
    recordRequested = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("SidebarCard")
        self.model_selector = ModelSelectorComponent()
        self.audio_selector = AudioSelectorComponent()
        self.butterworth_order = QSpinBox()
        self.butterworth_low = QDoubleSpinBox()
        self.butterworth_high = QDoubleSpinBox()
        self.fft_points = QSpinBox()
        self.process_button = QPushButton("Procesar")
        self.play_button = QPushButton("Escuchar sin procesar")
        self.play_filtered_button = QPushButton("Escuchar procesado")
        self.record_button = QPushButton("Grabar 3s y analizar")
        self.loading = LoadingComponent()

        self._configure_parameter_controls()

        layout = QVBoxLayout(self)
        layout.addWidget(self.model_selector)
        layout.addWidget(self.audio_selector)
        layout.addWidget(self._build_parameter_box())
        action_row = QHBoxLayout()
        action_row.addWidget(self.play_button)
        action_row.addWidget(self.play_filtered_button)
        action_row.addWidget(self.record_button)
        layout.addLayout(action_row)
        layout.addWidget(self.process_button)
        layout.addWidget(self.loading)
        layout.addStretch(1)

        self.process_button.clicked.connect(self._emit_process)
        self.play_button.clicked.connect(self._emit_play)
        self.play_filtered_button.clicked.connect(self.playFilteredRequested.emit)
        self.record_button.clicked.connect(self._emit_record)

    def _configure_parameter_controls(self) -> None:
        self.butterworth_order.setRange(1, 12)
        self.butterworth_order.setValue(AppSettings.BUTTERWORTH_ORDER_DEFAULT)

        for spinbox in (self.butterworth_low, self.butterworth_high):
            spinbox.setRange(0.0, 48000.0)
            spinbox.setDecimals(1)
            spinbox.setSingleStep(50.0)

        self.butterworth_low.setValue(AppSettings.BUTTERWORTH_LOW_FREQ_DEFAULT)
        self.butterworth_high.setValue(AppSettings.BUTTERWORTH_HIGH_FREQ_DEFAULT)

        self.fft_points.setRange(128, 32768)
        self.fft_points.setSingleStep(128)
        self.fft_points.setValue(AppSettings.FFT_POINTS_DEFAULT)

    def _build_parameter_box(self) -> QFrame:
        box = QFrame()
        box.setObjectName("InfoCard")
        layout = QFormLayout(box)
        layout.addRow(QLabel("Butterworth orden"), self.butterworth_order)
        layout.addRow(QLabel("Butterworth baja (Hz)"), self.butterworth_low)
        layout.addRow(QLabel("Butterworth alta (Hz)"), self.butterworth_high)
        layout.addRow(QLabel("FFT puntos"), self.fft_points)
        return box

    def set_models(self, models: list[str]) -> None:
        self.model_selector.set_models(models)

    def selected_model(self) -> str:
        return self.model_selector.selected_model()

    def selected_audio_path(self) -> str:
        return self.audio_selector.selected_audio_path()

    def set_audio_path(self, audio_path: str) -> None:
        self.audio_selector.set_audio_path(audio_path)

    def analysis_parameters(self) -> dict[str, float | int]:
        return {
            "butterworth_order": int(self.butterworth_order.value()),
            "butterworth_low_freq": float(self.butterworth_low.value()),
            "butterworth_high_freq": float(self.butterworth_high.value()),
            "fft_points": int(self.fft_points.value()),
        }

    def set_busy(self, loading: bool, message: str = "Procesando audio...") -> None:
        self.loading.set_loading(loading, message)
        self.process_button.setEnabled(not loading)

    def _emit_process(self) -> None:
        self.processRequested.emit(self.selected_model(), self.selected_audio_path())

    def _emit_play(self) -> None:
        self.playRequested.emit(self.selected_audio_path())

    def _emit_record(self) -> None:
        self.recordRequested.emit(self.selected_model())

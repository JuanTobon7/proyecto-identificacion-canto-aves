from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QPushButton, QVBoxLayout

from ui.components.audio_selector_component import AudioSelectorComponent
from ui.components.loading_component import LoadingComponent
from ui.components.model_selector_component import ModelSelectorComponent


class SidebarComponent(QFrame):
    processRequested = Signal(str, str)
    playRequested = Signal(str)
    recordRequested = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("SidebarCard")
        self.model_selector = ModelSelectorComponent()
        self.audio_selector = AudioSelectorComponent()
        self.process_button = QPushButton("Procesar")
        self.play_button = QPushButton("Reproducir")
        self.record_button = QPushButton("Grabar 3s y analizar")
        self.loading = LoadingComponent()

        layout = QVBoxLayout(self)
        layout.addWidget(self.model_selector)
        layout.addWidget(self.audio_selector)
        action_row = QHBoxLayout()
        action_row.addWidget(self.play_button)
        action_row.addWidget(self.record_button)
        layout.addLayout(action_row)
        layout.addWidget(self.process_button)
        layout.addWidget(self.loading)
        layout.addStretch(1)

        self.process_button.clicked.connect(self._emit_process)
        self.play_button.clicked.connect(self._emit_play)
        self.record_button.clicked.connect(self._emit_record)

    def set_models(self, models: list[str]) -> None:
        self.model_selector.set_models(models)

    def selected_model(self) -> str:
        return self.model_selector.selected_model()

    def selected_audio_path(self) -> str:
        return self.audio_selector.selected_audio_path()

    def set_busy(self, loading: bool, message: str = "Procesando audio...") -> None:
        self.loading.set_loading(loading, message)
        self.process_button.setEnabled(not loading)

    def _emit_process(self) -> None:
        self.processRequested.emit(self.selected_model(), self.selected_audio_path())

    def _emit_play(self) -> None:
        self.playRequested.emit(self.selected_audio_path())

    def _emit_record(self) -> None:
        self.recordRequested.emit(self.selected_model())

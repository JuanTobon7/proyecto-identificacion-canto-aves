from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLineEdit, QLabel, QPushButton, QWidget


class AudioSelectorComponent(QWidget):
    audioPathChanged = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.label = QLabel("Audio")
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Selecciona un archivo wav, mp3 o flac")
        self.browse_button = QPushButton("Buscar")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        layout.addWidget(self.path_edit, 1)
        layout.addWidget(self.browse_button)

        self.path_edit.textChanged.connect(self.audioPathChanged.emit)
        self.browse_button.clicked.connect(self._browse)

    def selected_audio_path(self) -> str:
        return self.path_edit.text().strip()

    def set_audio_path(self, audio_path: str | Path) -> None:
        self.path_edit.setText(str(audio_path))

    def _browse(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar audio",
            "",
            "Audio (*.wav *.mp3 *.flac)",
        )
        if file_path:
            self.set_audio_path(file_path)
            self.audioPathChanged.emit(file_path)

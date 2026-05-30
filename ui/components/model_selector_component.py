from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QWidget


class ModelSelectorComponent(QWidget):
    modelChanged = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.label = QLabel("Modelo")
        self.combo = QComboBox()
        self.refresh_button = QPushButton("Actualizar")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        layout.addWidget(self.combo, 1)
        layout.addWidget(self.refresh_button)

        self.combo.currentTextChanged.connect(self.modelChanged.emit)

    def set_models(self, models: list[str]) -> None:
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItems(models)
        self.combo.blockSignals(False)
        if models:
            self.modelChanged.emit(self.selected_model())

    def selected_model(self) -> str:
        return self.combo.currentText().strip()

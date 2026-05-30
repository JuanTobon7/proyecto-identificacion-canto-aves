from __future__ import annotations

from PySide6.QtWidgets import QFrame, QLabel, QTextEdit, QVBoxLayout

from core.dto.cards_bird import BirdInfo


class BirdInfoComponent(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("BirdInfoCard")
        self.title = QLabel("Información del ave")
        self.text = QTextEdit()
        self.text.setReadOnly(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.title)
        layout.addWidget(self.text)

    def set_bird_info(self, bird_info: BirdInfo | None) -> None:
        if bird_info is None:
            self.text.setPlainText("Sin información disponible.")
            return
        self.text.setPlainText(str(bird_info))

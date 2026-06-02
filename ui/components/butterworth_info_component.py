from __future__ import annotations

from PySide6.QtWidgets import QFrame, QLabel, QTextEdit, QVBoxLayout


class ButterworthInfoComponent(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("InfoCard")
        self.title = QLabel("Parametros del filtro")
        self.text = QTextEdit()
        self.text.setReadOnly(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.title)
        layout.addWidget(self.text)

    def set_parameters(self, params: dict) -> None:
        if not params:
            self.text.setPlainText("Sin parametros disponibles.")
            return
        lines = []
        for key, value in params.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.4f}")
            else:
                lines.append(f"{key}: {value}")
        self.text.setPlainText("\n".join(lines))

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QLabel, QTextEdit, QHBoxLayout, QVBoxLayout


class MetricsComponent(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("InfoCard")
        self.original_title = QLabel("Metricas originales")
        self.filtered_title = QLabel("Metricas filtradas")
        self.original_text = QTextEdit()
        self.filtered_text = QTextEdit()
        for widget in (self.original_text, self.filtered_text):
            widget.setReadOnly(True)

        layout = QHBoxLayout(self)
        left = QVBoxLayout()
        left.addWidget(self.original_title)
        left.addWidget(self.original_text)
        right = QVBoxLayout()
        right.addWidget(self.filtered_title)
        right.addWidget(self.filtered_text)
        layout.addLayout(left)
        layout.addLayout(right)

    def set_metrics(self, original_metrics: dict, filtered_metrics: dict) -> None:
        self.original_text.setPlainText(self._format_metrics(original_metrics))
        self.filtered_text.setPlainText(self._format_metrics(filtered_metrics))

    @staticmethod
    def _format_metrics(metrics: dict) -> str:
        if not metrics:
            return "Sin datos."
        lines = []
        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.6f}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)

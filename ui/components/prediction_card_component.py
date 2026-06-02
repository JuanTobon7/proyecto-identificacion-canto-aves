from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QListWidget, QListWidgetItem, QVBoxLayout

from ui.dtos.prediction_result import PredictionResult


class PredictionCardComponent(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("PredictionCard")
        self.species_label = QLabel("Especie predicha")
        self.species_value = QLabel("--")
        self.summary_label = QLabel("Resumen")
        self.summary_value = QLabel("--")
        self.confidence_label = QLabel("Confianza")
        self.confidence_value = QLabel("--")
        self.rank_list = QListWidget()

        self.species_value.setWordWrap(True)
        self.summary_value.setWordWrap(True)
        self.rank_list.setMinimumHeight(120)

        layout = QVBoxLayout(self)
        layout.addWidget(self.species_label)
        layout.addWidget(self.species_value)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.summary_value)
        grid = QGridLayout()
        grid.addWidget(self.confidence_label, 0, 0)
        grid.addWidget(self.confidence_value, 0, 1)
        layout.addLayout(grid)
        layout.addWidget(QLabel("Ranking"))
        layout.addWidget(self.rank_list)

    def set_result(self, result: PredictionResult | None) -> None:
        if result is None:
            self.species_value.setText("--")
            self.summary_value.setText("--")
            self.confidence_value.setText("--")
            self.rank_list.clear()
            return

        candidate_species = result.ranking[0].get("species", "--") if result.ranking else "--"
        candidate_score = result.ranking[0].get("score", 0.0) if result.ranking else 0.0
        self.summary_value.setText(
            f"Candidato: {candidate_species} | Score: {candidate_score:.4f} | Umbral: {result.rejection_threshold * 100:.1f}%"
        )

        if result.rejected:
            self.species_value.setText("Rechazado")
            self.confidence_value.setText(f"{result.confidence * 100:.1f}%")
        else:
            self.species_value.setText(result.predicted_species or "--")
            self.confidence_value.setText(f"{result.confidence * 100:.1f}%")
        self.rank_list.clear()
        for item in result.ranking[:5]:
            label = f"{item.get('species', '--')}: {item.get('score', 0.0):.4f}"
            self.rank_list.addItem(QListWidgetItem(label))

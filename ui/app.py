from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from ui.controllers.prediction_controller import PredictionController
from ui.windows.main_window import MainWindow


APP_STYLE = """
QWidget {
    background: #0f172a;
    color: #e5e7eb;
    font-family: Segoe UI;
    font-size: 10pt;
}
QFrame#SidebarCard, QFrame#PredictionCard, QFrame#BirdInfoCard, QFrame#PlotCard, QFrame#InfoCard {
    background: #111827;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 12px;
}
QPushButton {
    background: #38bdf8;
    color: #082f49;
    border: none;
    border-radius: 10px;
    padding: 10px 14px;
    font-weight: 600;
}
QPushButton:disabled {
    background: #475569;
    color: #cbd5e1;
}
QLineEdit, QComboBox, QTextEdit, QListWidget {
    background: #0b1220;
    color: #e5e7eb;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 8px;
}
QProgressBar {
    border: 1px solid #334155;
    border-radius: 8px;
    text-align: center;
    background: #0b1220;
}
QProgressBar::chunk {
    background-color: #22c55e;
    border-radius: 8px;
}
"""


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLE)
    window = MainWindow(controller=PredictionController())
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

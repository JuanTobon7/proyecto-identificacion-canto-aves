from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from ui.controllers.prediction_controller import PredictionController
from ui.windows.main_window import MainWindow


APP_STYLE = """
QWidget {
    background: #f8fafc;
    color: #0f172a;
    font-family: Segoe UI;
    font-size: 10pt;
}
QFrame#SidebarCard, QFrame#PredictionCard, QFrame#BirdInfoCard, QFrame#PlotCard, QFrame#InfoCard {
    background: #ffffff;
    border: 1px solid #dbe4ee;
    border-radius: 14px;
    padding: 12px;
}
QPushButton {
    background: #2563eb;
    color: #ffffff;
    border: none;
    border-radius: 10px;
    padding: 10px 14px;
    font-weight: 600;
}
QPushButton:disabled {
    background: #cbd5e1;
    color: #64748b;
}
QLineEdit, QComboBox, QTextEdit, QListWidget {
    background: #ffffff;
    color: #0f172a;
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    padding: 8px;
}
QSpinBox, QDoubleSpinBox {
    background: #ffffff;
    color: #0f172a;
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    padding: 6px 8px;
}
QListWidget::item:selected, QComboBox QAbstractItemView::item:selected {
    background: #dbeafe;
    color: #0f172a;
}
QProgressBar {
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    text-align: center;
    background: #ffffff;
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

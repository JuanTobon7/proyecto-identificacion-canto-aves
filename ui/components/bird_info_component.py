from __future__ import annotations

from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QFrame, QLabel, QTextBrowser, QVBoxLayout

from core.dto.cards_bird import BirdInfo


class BirdInfoComponent(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("BirdInfoCard")
        self.title = QLabel("Información del ave")
        self.text = QTextBrowser()
        self.text.setOpenExternalLinks(True)
        self.text.setMinimumHeight(240)

        layout = QVBoxLayout(self)
        layout.addWidget(self.title)
        layout.addWidget(self.text)

    def set_bird_info(self, bird_info: BirdInfo | None) -> None:
        if bird_info is None:
            self.text.setHtml("<p>Sin información disponible.</p>")
            return
        image_html = f'<p><img src="{bird_info.img}" alt="{bird_info.nombre_comun_ingles}" style="max-width: 100%; height: auto; border-radius: 10px;"></p>' if bird_info.img else ""
        self.text.setHtml(
            "<div style='font-size: 13px; line-height: 1.5;'>"
            f"<h3 style='margin: 0 0 8px 0;'>{bird_info.nombre_comun_espanol} ({bird_info.nombre_comun_ingles})</h3>"
            f"<p><b>Científico:</b> {bird_info.nombre_cientifico}<br>"
            f"<b>Familia:</b> {bird_info.familia}<br>"
            f"<b>Orden:</b> {bird_info.orden}</p>"
            f"{image_html}"
            f"<p><b>Descripción:</b> {bird_info.descripcion}</p>"
            f"<p><b>Distribución:</b> {bird_info.distribucion}</p>"
            f"<p><b>Hábitat:</b> {', '.join(bird_info.habitat or [])}</p>"
            f"<p><b>Dieta:</b> {', '.join(bird_info.dieta or [])}</p>"
            f"<p><b>Estado de conservación:</b> {bird_info.estado_conservacion}</p>"
            f"<p><b>Longitud:</b> {bird_info.longitud_cm} cm<br>"
            f"<b>Peso:</b> {bird_info.peso_g_min} - {bird_info.peso_g_max} g</p>"
            f"<p><b>Vocalizaciones:</b> {bird_info.vocalizaciones.descripcion}</p>"
            f"<p><b>Tipo:</b> {', '.join(bird_info.vocalizaciones.tipo_vocalizacion)}<br>"
            f"<b>Frecuencia dominante:</b> {bird_info.vocalizaciones.frecuencias_hz.frecuencia_dominante} Hz<br>"
            f"<b>Rango principal:</b> {bird_info.vocalizaciones.frecuencias_hz.rango_principal.min} - {bird_info.vocalizaciones.frecuencias_hz.rango_principal.max} Hz<br>"
            f"<b>Notas:</b> {bird_info.vocalizaciones.frecuencias_hz.notas}</p>"
            "</div>"
        )
        self.text.moveCursor(QTextCursor.MoveOperation.Start)
        self.text.verticalScrollBar().setValue(0)

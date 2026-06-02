from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QTextCursor
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
        
        # Imagen del ave
        self.image_label = QLabel()
        self.image_label.setStyleSheet("border-radius: 10px;")
        self.image_label.setAlignment(self.image_label.alignment() | 0x0001 | 0x0080)  # Center alignment

        layout = QVBoxLayout(self)
        layout.addWidget(self.title)
        layout.addWidget(self.image_label)
        layout.addWidget(self.text)

    def _load_image_from_url(self, url: str) -> QPixmap | None:
        """Descarga y carga una imagen desde una URL remota."""
        try:
            response = urlopen(url, timeout=5)
            image_data = response.read()
            pixmap = QPixmap()
            pixmap.loadFromData(image_data)
            if not pixmap.isNull():
                # Redimensionar a ancho máximo de 300px manteniendo proporción
                scaled_pixmap = pixmap.scaledToWidth(300, Qt.SmoothTransformation)
                return scaled_pixmap
        except Exception as e:
            print(f"Error al descargar imagen remota: {e}")
        return None

    def _load_image_from_path(self, image_path: str) -> QPixmap | None:
        """Carga una imagen desde una ruta local."""
        try:
            # Convertir rutas relativas a absolutas desde la raíz del proyecto
            if image_path.startswith("/") or image_path.startswith("./"):
                # Limpiar la ruta: quitar / y ./
                clean_path = image_path.lstrip("/").lstrip("./")
                # Ruta relativa - buscar desde la raíz del proyecto
                local_path = Path(__file__).parent.parent.parent / clean_path
            else:
                local_path = Path(image_path)
            
            if local_path.exists():
                pixmap = QPixmap(str(local_path))
                if not pixmap.isNull():
                    # Redimensionar a ancho máximo de 300px manteniendo proporción
                    scaled_pixmap = pixmap.scaledToWidth(300, Qt.SmoothTransformation)
                    return scaled_pixmap
                else:
                    print(f"No se pudo cargar la imagen: {local_path}")
            else:
                print(f"Archivo de imagen no existe: {local_path}")
        except Exception as e:
            print(f"Error al cargar imagen local: {e}")
        return None

    def _load_image(self, url_or_path: str) -> QPixmap | None:
        """Carga una imagen desde URL remota o ruta local."""
        if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
            return self._load_image_from_url(url_or_path)
        else:
            return self._load_image_from_path(url_or_path)

    def set_bird_info(self, bird_info: BirdInfo | None) -> None:
        if bird_info is None:
            self.text.setHtml("<p>Sin información disponible.</p>")
            self.image_label.setPixmap(QPixmap())
            return
        
        # Cargar y mostrar imagen
        if bird_info.img:
            pixmap = self._load_image(bird_info.img)
            if pixmap:
                self.image_label.setPixmap(pixmap)
            else:
                self.image_label.setText("No se pudo cargar la imagen")
        else:
            self.image_label.setPixmap(QPixmap())
        
        self.text.setHtml(
            "<div style='font-size: 13px; line-height: 1.5;'>"
            f"<h3 style='margin: 0 0 8px 0;'>{bird_info.nombre_comun_espanol} ({bird_info.nombre_comun_ingles})</h3>"
            f"<p><b>Científico:</b> {bird_info.nombre_cientifico}<br>"
            f"<b>Familia:</b> {bird_info.familia}<br>"
            f"<b>Orden:</b> {bird_info.orden}</p>"
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

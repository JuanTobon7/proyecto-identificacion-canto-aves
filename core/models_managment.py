"""
ModelsManagement
================
Abstracción para persistir y recuperar objetos de modelo (o cualquier artefacto
serializable) de forma independiente al tipo de modelo concreto.

Uso básico
----------
    mm = ModelsManagement(base_dir="models")

    # Guardar
    mm.save("butterworth_lp_500hz", filter_obj, metadata={"order": 4, "cutoff": 500})

    # Cargar
    filter_obj = mm.get("butterworth_lp_500hz")

    # Listar
    for entry in mm.list_models():
        print(entry)

    # Borrar
    mm.delete("butterworth_lp_500hz")
"""

from __future__ import annotations

import json
import os
import pickle
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ModelsManagement:
    """Gestiona el ciclo de vida de modelos serializados en disco."""

    MODELS_DIR = Path("models")
    INDEX_FILE = "index.json"

    def __init__(self, base_dir: str | Path | None = None):
        self.base_dir = Path(base_dir) if base_dir else self.MODELS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.base_dir / self.INDEX_FILE
        self._index: dict[str, dict] = self._load_index()

    # ------------------------------------------------------------------
    # Operaciones públicas
    # ------------------------------------------------------------------

    def save(
        self,
        name: str,
        model: Any,
        metadata: dict | None = None,
        overwrite: bool = True,
    ) -> Path:
        """
        Serializa *model* con pickle y registra la entrada en el índice.

        Parámetros
        ----------
        name      : identificador único del modelo (sin extensión).
        model     : objeto serializable.
        metadata  : dict con información adicional (hiperparámetros, etc.).
        overwrite : si False lanza ValueError cuando el nombre ya existe.

        Retorna la ruta del archivo generado.
        """
        if not overwrite and name in self._index:
            raise ValueError(
                f"El modelo '{name}' ya existe. "
                "Usa overwrite=True para sobreescribirlo."
            )

        file_path = self.base_dir / f"{name}.pkl"
        with open(file_path, "wb") as fh:
            pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)

        checksum = self._md5(file_path)
        self._index[name] = {
            "name": name,
            "file": str(file_path),
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "checksum_md5": checksum,
            "metadata": metadata or {},
        }
        self._save_index()
        return file_path

    def get(self, name: str) -> Any:
        """
        Carga y retorna el modelo identificado por *name*.
        Lanza KeyError si no existe y FileNotFoundError si el pkl fue borrado.
        """
        if name not in self._index:
            raise KeyError(f"Modelo '{name}' no encontrado en el índice.")

        file_path = Path(self._index[name]["file"])
        if not file_path.exists():
            raise FileNotFoundError(
                f"El archivo '{file_path}' no existe en disco. "
                "Puede haber sido borrado manualmente."
            )

        with open(file_path, "rb") as fh:
            return pickle.load(fh)

    def delete(self, name: str, remove_file: bool = True) -> None:
        """
        Elimina el modelo del índice y, opcionalmente, borra el archivo pkl.
        """
        if name not in self._index:
            raise KeyError(f"Modelo '{name}' no encontrado en el índice.")

        if remove_file:
            file_path = Path(self._index[name]["file"])
            if file_path.exists():
                file_path.unlink()

        del self._index[name]
        self._save_index()

    def exists(self, name: str) -> bool:
        """Retorna True si el modelo está registrado en el índice."""
        return name in self._index

    def info(self, name: str) -> dict:
        """Retorna los metadatos almacenados del modelo."""
        if name not in self._index:
            raise KeyError(f"Modelo '{name}' no encontrado en el índice.")
        return dict(self._index[name])

    def list_models(self) -> list[dict]:
        """Retorna una lista con la info de todos los modelos registrados."""
        return list(self._index.values())

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _load_index(self) -> dict:
        if self._index_path.exists():
            with open(self._index_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return {}

    def _save_index(self) -> None:
        with open(self._index_path, "w", encoding="utf-8") as fh:
            json.dump(self._index, fh, indent=2, ensure_ascii=False)

    @staticmethod
    def _md5(path: Path) -> str:
        h = hashlib.md5()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
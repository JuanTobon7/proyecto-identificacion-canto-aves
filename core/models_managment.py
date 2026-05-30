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
        data: dict,
        metadata: dict | None = None,
        overwrite: bool = True,
    ) -> Path:
        """
        Guarda un artefacto JSON (no ML, no pickle).

        Parámetros
        ----------
        name      : identificador único.
        data      : dict serializable a JSON.
        metadata  : información extra.
        overwrite : sobreescribir si existe.

        Retorna
        -------
        Ruta del archivo generado.
        """

        if not overwrite and name in self._index:
            raise ValueError(
                f"El recurso '{name}' ya existe. "
                "Usa overwrite=True para sobreescribirlo."
            )

        file_path = self.base_dir / f"{name}.json"

        with open(file_path, "w", encoding="utf-8") as fh:
            json.dump(
                data,
                fh,
                indent=2,
                ensure_ascii=False
            )

        checksum = self._md5(file_path)

        self._index[name] = {
            "name": name,
            "file": str(file_path),
            "type": "json",
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "checksum_md5": checksum,
            "metadata": metadata or {},
        }

        self._save_index()
        return file_path

    def get_json(self, name: str) -> dict:
        """
        Carga un artefacto JSON.
        """
        file_path = self._resolve_file_path(name)

        if not file_path.exists():
            raise FileNotFoundError(
                f"El archivo '{file_path}' no existe."
            )

        with open(file_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

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
        return name in self._index or (self.base_dir / f"{name}.json").exists()

    def info(self, name: str) -> dict:
        """Retorna los metadatos almacenados del modelo."""
        if name not in self._index:
            raise KeyError(f"Modelo '{name}' no encontrado en el índice.")
        return dict(self._index[name])

    def list_models(self) -> list[dict]:
        """Retorna una lista con la info de todos los modelos registrados."""
        models = list(self._index.values())
        seen_names = {entry.get("name") for entry in models if entry.get("name")}

        for file_path in sorted(self.base_dir.glob("*.json")):
            if file_path.name == self.INDEX_FILE:
                continue
            model_name = file_path.stem
            if model_name in seen_names:
                continue
            models.append({
                "name": model_name,
                "file": str(file_path),
                "type": "json",
                "saved_at": None,
                "checksum_md5": self._md5(file_path),
                "metadata": {},
            })

        return models

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _load_index(self) -> dict:
        if self._index_path.exists():
            with open(self._index_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return {}

    def _resolve_file_path(self, name: str) -> Path:
        if name in self._index:
            return Path(self._index[name]["file"])

        direct_path = Path(name)
        if direct_path.suffix == ".json" and direct_path.exists():
            return direct_path

        file_path = self.base_dir / f"{name}.json"
        if file_path.exists():
            return file_path

        raise KeyError(f"Recurso '{name}' no encontrado.")

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
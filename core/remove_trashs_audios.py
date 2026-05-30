"""
RemoveTrashAudios
=================
Marca audios problemáticos en un archivo JSON y permite borrarlos
de forma controlada ejecutando el script de limpieza manualmente.

Flujo de trabajo
----------------
1. Durante el análisis, el AudioAnalyzer llama a:
       trash = RemoveTrashAudios()
       trash.add("dataset_aves_normalice/XC12345.wav", reason="outlier σ=4.2")

2. Los archivos marcados se acumulan en `trash_audios.json`.

3. Cuando el usuario quiere borrar, ejecuta explícitamente:
       trash.delete_marked()           # borra todos los marcados
   o  trash.delete_marked(dry_run=True)  # solo muestra qué borraría

4. Para revisar antes de borrar:
       for entry in trash.list():
           print(entry)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from typing import List

class RemoveTrashAudios:
    """Gestiona la lista negra de audios que deben ser descartados."""

    DEFAULT_JSON = Path("trash_audios.json")

    def __init__(self, json_path: str | Path | None = None):
        self._path = Path(json_path) if json_path else self.DEFAULT_JSON
        self._entries: dict[str, dict] = self._load()

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def add(
        self,
        audio_file_path: str | Path,
        reason: str = "",
        metadata: dict | None = None,
    ) -> None:
        """
        Marca *audio_file_path* para eliminación posterior.

        Parámetros
        ----------
        audio_file_path : ruta al archivo de audio.
        reason          : texto libre que describe por qué se descarta.
        metadata        : cualquier info extra (stats, índices, etc.).
        """
        key = str(Path(audio_file_path).resolve())
        if key in self._entries:
            # Actualiza la razón si ya estaba marcado
            self._entries[key]["reason"] = reason
            self._entries[key]["updated_at"] = datetime.now(timezone.utc).isoformat()
        else:
            self._entries[key] = {
                "path": key,
                "reason": reason,
                "marked_at": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {},
            }
        self._save()

    def remove_mark(self, audio_file_path: str | Path) -> None:
        """Quita el archivo de la lista negra (sin borrarlo del disco)."""
        key = str(Path(audio_file_path).resolve())
        if key in self._entries:
            del self._entries[key]
            self._save()

    def is_marked(self, audio_file_path: str | Path) -> bool:
        key = str(Path(audio_file_path).resolve())
        return key in self._entries

    def list(self) -> list[dict]:
        """Retorna la lista de entradas marcadas."""
        return list(self._entries.values())

    def count(self) -> int:
        return len(self._entries)

    def clear_marks(self) -> None:
        """Vacía la lista negra sin borrar nada del disco."""
        self._entries.clear()
        self._save()

    def delete_marked(self, dry_run: bool = False) -> list[str]:
        """
        Borra del disco todos los archivos marcados.

        Parámetros
        ----------
        dry_run : si True, imprime lo que haría pero no borra nada.

        Retorna
        -------
        Lista de rutas procesadas (borradas o que hubiera borrado en dry_run).
        """
        processed: list[str] = []
        not_found: list[str] = []

        for key, entry in list(self._entries.items()):
            path = Path(entry["path"])
            if path.exists():
                if dry_run:
                    print(f"[DRY-RUN] Borraría: {path}  —  {entry['reason']}")
                else:
                    path.unlink()
                    print(f"[DELETED] {path}")
                    del self._entries[key]
                processed.append(str(path))
            else:
                print(f"[NOT FOUND] {path} (ya no existe en disco)")
                not_found.append(str(path))
                if not dry_run:
                    del self._entries[key]

        if not dry_run:
            self._save()
            print(
                f"\n✓ Eliminados {len(processed)} archivos. "
                f"{len(not_found)} no encontrados (limpiados del índice)."
            )
        else:
            print(f"\n[DRY-RUN] Se borrarían {len(processed)} archivos.")

        return processed

    # ------------------------------------------------------------------
    # Iteración
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[dict]:
        return iter(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        if self._path.exists():
            with open(self._path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return {}

    def _save(self) -> None:
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(self._entries, fh, indent=2, ensure_ascii=False)
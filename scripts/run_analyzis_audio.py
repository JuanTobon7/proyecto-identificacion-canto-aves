"""
run_analysis.py
===============

Ejecución
---------
    # Análisis completo (marca outliers, guarda modelo)
    python run_analysis.py

    # Solo previsualizar qué se borraría (sin borrar)
    python run_analysis.py --dry-run

    # Borrar los audios ya marcados en trash_audios.json
    python run_analysis.py --delete-trash
"""

import argparse
import json
from pathlib import Path

from core.audio_analyzer import AudioAnalyzer
from core.remove_trashs_audios import RemoveTrashAudios


def main() -> None:
    parser = argparse.ArgumentParser(description="Análisis y limpieza de corpus de audio.")
    parser.add_argument(
        "--normalized-dir",
        default=None,
        help="Carpeta con audios limpios (por defecto: RoutesPath.BANK_AUDIOS_NORMALIZED).",
    )
    parser.add_argument(
        "--std-threshold",
        type=float,
        default=3.0,
        help="Número de σ para considerar outlier (default: 3.0).",
    )
    parser.add_argument(
        "--butterworth-order",
        type=int,
        default=4,
        help="Orden del filtro Butterworth (default: 4).",
    )
    parser.add_argument(
        "--butterworth-cutoff",
        type=float,
        default=8000.0,
        help="Frecuencia de corte en Hz del filtro (default: 8000).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Muestra qué archivos se borrarían sin borrarlos.",
    )
    parser.add_argument(
        "--delete-trash",
        action="store_true",
        help="Borra los archivos marcados en trash_audios.json y sale.",
    )
    parser.add_argument(
        "--trash-json",
        default="trash_audios.json",
        help="Ruta al JSON de archivos basura (default: trash_audios.json).",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directorio donde se guardan los modelos (default: models/).",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="Si se indica, guarda el reporte completo en este archivo JSON.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Modo: solo borrar lo ya marcado
    # ------------------------------------------------------------------
    if args.delete_trash:
        trash = RemoveTrashAudios(json_path=args.trash_json)
        print(f"Archivos marcados en '{args.trash_json}': {trash.count()}")
        if trash.count() == 0:
            print("Nada que borrar.")
            return
        trash.delete_marked(dry_run=False)
        return

    # ------------------------------------------------------------------
    # Modo: análisis completo
    # ------------------------------------------------------------------
    analyzer = AudioAnalyzer(
        normalized_dir=args.normalized_dir,
        std_threshold=args.std_threshold,
        butterworth_order=args.butterworth_order,
        butterworth_cutoff_hz=args.butterworth_cutoff,
        trash_json=args.trash_json,
        models_dir=args.models_dir,
    )

    report = analyzer.run()

    # Guardar reporte JSON si se pidió
    if args.report_json:
        report_path = Path(args.report_json)
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        print(f"Reporte guardado en: {report_path}")

    # Si se pidió dry-run, mostrar qué se borraría
    if args.dry_run:
        print("\n--- DRY-RUN: archivos que se borrarían ---")
        analyzer.trash.delete_marked(dry_run=True)


if __name__ == "__main__":
    main()
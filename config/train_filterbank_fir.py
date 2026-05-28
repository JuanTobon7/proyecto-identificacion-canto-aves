"""
Atajo para entrenar el modelo de banco de filtros FIR.

Ejecuta exactamente el mismo flujo de filterbank_train, pero por defecto usa:
- dataset_aves_fir/
- bands=6
- models/model_filterbank_fir.json

Uso:
  python -m config.train_filterbank_fir
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from config.filterbank_train import build_frequency_bands, collect_species, compute_profile


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    default_dataset = base_dir / "dataset_aves_fir"
    default_output = base_dir / "models" / "model_filterbank_fir.json"

    parser = argparse.ArgumentParser(description="Entrena el modelo FIR de banco de filtros.")
    parser.add_argument("--dataset", type=Path, default=default_dataset, help="Carpeta con audios FIR por especie")
    parser.add_argument("--bands", type=int, default=6, help="Número de subbandas")
    parser.add_argument("--min-hz", type=float, default=50.0)
    parser.add_argument("--max-hz", type=float, default=16000.0)
    parser.add_argument("--output", type=Path, default=default_output, help="Archivo JSON de salida")
    args = parser.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {args.dataset}")

    bands = build_frequency_bands(args.bands, args.min_hz, args.max_hz)
    profiles = []
    species_list = []

    for species_dir in sorted(args.dataset.iterdir()):
        if not species_dir.is_dir():
            continue
        samples = collect_species(species_dir, bands)
        profile = compute_profile(samples)
        profile["species_key"] = species_dir.name
        profiles.append(profile)
        species_list.append(species_dir.name)
        logger.info("Construido perfil FIR para %s (muestras=%d)", species_dir.name, profile.get("sample_count", 0))

    payload = {
        "model_type": f"filterbank_fir_{args.bands}",
        "kind": "filterbank_energy_thresholds",
        "bands": bands,
        "feature_summary": {"band_count": args.bands, "source": "dataset_aves_fir"},
        "species": species_list,
        "species_profiles": profiles,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    logger.info("Modelo FIR guardado en %s", args.output)


if __name__ == "__main__":
    main()

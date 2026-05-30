"""
Script de entrenamiento compacto.

Genera un JSON con un modelo por especie para un tipo de filtro:
- butterworth : usa FilterButterworth (orden configurable)
- fir         : diseña un FilterFir por especie (num_taps configurable)

Salida: un único archivo JSON que contiene la colección `models`.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa

from core.repo.birds_repo import BirdRepository
from core.maths.fft import FFTProcessor
from core.maths.statistics import Statistics
from core.maths.energy_vector import EnergyVector
from core.maths.filter_butterworth import FilterButterworth
from core.maths.filter_fir import FilterFir
from core.audio_converter import AudioConverter


def process_species_audios(
    species: str,
    files: list[Path],
    bird_info,
    model_type: str,
    bands: int,
    sample_rate: int,
    order: int,
    num_taps: int,
) -> dict:
    """Procesa los audios de una especie y devuelve su perfil estadístico."""
    fft = FFTProcessor()
    low = float(bird_info.vocalizaciones.frecuencias_hz.rango_principal.min)
    high = float(bird_info.vocalizaciones.frecuencias_hz.rango_principal.max)

    vectors: list[np.ndarray] = []

    for p in files:
        vec = _compute_energy_vector_for_file(
            path=p,
            model_type=model_type,
            low=low,
            high=high,
            sample_rate=sample_rate,
            order=order,
            num_taps=num_taps,
            bands=bands,
            fft=fft,
        )
        if vec is not None:
            vectors.append(vec)

    if len(vectors) == 0:
        profile = np.zeros(bands, dtype=np.float32)
        stdv = np.zeros(bands, dtype=np.float32)
    else:
        profile = Statistics.mean_vector(vectors).astype(np.float32)
        stdv = Statistics.std_vector(vectors).astype(np.float32)

    params = {
        "low_freq": low,
        "high_freq": high,
    }
    if model_type == "butterworth":
        params.update({"order": int(order), "type": "butterworth"})
    else:
        params.update({"num_taps": int(num_taps), "type": "fir"})

    return {
        "species": species,
        "params": params,
        "profile_vector": profile.tolist(),
        "std_energy_vector": stdv.tolist(),
    }


def _compute_energy_vector_for_file(
    path: Path,
    model_type: str,
    low: float,
    high: float,
    sample_rate: int,
    order: int,
    num_taps: int,
    bands: int,
    fft: FFTProcessor,
) -> np.ndarray | None:
    try:
        audio, sr = sf.read(str(path), always_2d=False)
    except Exception as exc:
        print(f"  - no pude leer {path.name}: {exc}")
        return None

    y = AudioConverter.to_mono_float32(np.asarray(audio))
    if y.size == 0:
        return None

    if sr != sample_rate:
        try:
            y = AudioConverter.resample(y, sr, sample_rate)
            sr = sample_rate
        except Exception as exc:
            print(f"  - fallo remuestreo en {path.name}: {exc}")
            return None

    filtered = _filter_signal(
        signal=y,
        sr=sr,
        model_type=model_type,
        low=low,
        high=high,
        order=order,
        num_taps=num_taps,
        file_name=path.name,
    )
    if filtered is None:
        return None

    subbands = fft.build_subbands(low, high, bands)
    energies = fft.compute_band_energies(filtered, sr, subbands)
    return EnergyVector.compute(energies)


def _filter_signal(
    signal: np.ndarray,
    sr: int,
    model_type: str,
    low: float,
    high: float,
    order: int,
    num_taps: int,
    file_name: str,
) -> np.ndarray | None:
    try:
        if model_type == "butterworth":
            return FilterButterworth(order=order).apply_bandpass(signal, sr, low, high)

        fir = FilterFir(sr=sr, num_taps=max(3, num_taps))
        fir.design_bandpass(low, high)
        return fir.process_signal_fast(signal)
    except Exception as exc:
        print(f"  - fallo filtrado {model_type} en {file_name}: {exc}")
        return None


def main():
    p = argparse.ArgumentParser(description="Entrena modelos filterbank por especie (Butterworth/FIR)")
    p.add_argument("--model-type", choices=["butterworth", "fir"], default="butterworth")
    p.add_argument("--bands", type=int, default=8, help="Número de sub-bandas por modelo")
    p.add_argument("--sample-rate", type=int, default=22050)
    p.add_argument("--order", type=int, default=4, help="Orden para Butterworth")
    p.add_argument("--num-taps", type=int, default=101, help="Taps para FIR")
    p.add_argument("--output", type=str, default=None, help="Ruta de salida JSON (default models/model_{type}.json)")
    args = p.parse_args()

    repo = BirdRepository(env="training")
    audios_by_species = repo.get_audios_by_species()

    now = datetime.now(timezone.utc).isoformat()

    models = []
    total = len(audios_by_species)
    processed = 0
    for species, files in audios_by_species.items():
        processed += 1
        print(f"[{processed}/{total}] Procesando especie: {species} -> {len(files)} archivos")
        if len(files) == 0:
            print("  - sin archivos, se omite")
            continue

        bird = repo.get_by_species(species)
        entry = process_species_audios(
            species=species,
            files=files,
            bird_info=bird,
            model_type=args.model_type,
            bands=max(1, int(args.bands)),
            sample_rate=int(args.sample_rate),
            order=int(args.order),
            num_taps=int(args.num_taps),
        )
        models.append(entry)

    collection = {
        "kind": f"{args.model_type}_collection",
        "model_type": f"filterbank_{args.model_type}",
        "bands": int(args.bands),
        "sample_rate": int(args.sample_rate),
        "created_at": now,
        "models": models,
    }

    out_path = Path(args.output) if args.output else Path("models") / f"model_filterbank_{args.model_type}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(collection, fh, indent=2, ensure_ascii=False)

    print(f"Guardado modelo: {out_path}")


if __name__ == "__main__":
    main()
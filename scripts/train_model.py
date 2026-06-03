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
from typing import Any

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
from core.maths.dynamic_bands_detector import DynamicBandsDetector

def _load_species_analysis(analysis_path: Path, species: str) -> dict[str, Any] | None:
    if not analysis_path.exists():
        return None

    try:
        with open(analysis_path, "r", encoding="utf-8") as fh:
            analysis = json.load(fh)
    except Exception as exc:
        print(f"  - no pude leer el análisis espectral {analysis_path.name}: {exc}")
        return None

    for entry in analysis.get("analysis_results", []):
        if entry.get("species") == species:
            return entry
    return None


def _bands_to_json(bands: list[tuple[float, float]] | None) -> list[dict[str, float]]:
    if not bands:
        return []
    return [{"low": float(low), "high": float(high)} for low, high in bands]


def _band_metadata_from_profile(
    low: float,
    high: float,
    bands: int,
    sample_rate: int,
    analysis_entry: dict[str, Any] | None,
) -> dict[str, float | int]:
    bandwidth_hz = float(max(0.0, high - low))
    nyquist_hz = float(sample_rate / 2.0)

    fft_bins_in_range = 0
    if analysis_entry:
        spectrum_profile = analysis_entry.get("spectrum_profile", {})
        frequencies = np.asarray(spectrum_profile.get("frequencies", []), dtype=np.float64)
        if frequencies.size > 0:
            fft_bins_in_range = int(np.sum((frequencies >= low) & (frequencies <= high)))

    points_per_band = float(fft_bins_in_range / bands) if bands > 0 else 0.0
    nyquist_percent = float((bandwidth_hz / nyquist_hz) * 100.0) if nyquist_hz > 0 else 0.0

    return {
        "bandwidth_hz": bandwidth_hz,
        "fft_bins_in_range": int(fft_bins_in_range),
        "points_per_band": points_per_band,
        "nyquist_percent": nyquist_percent,
    }

def process_species_audios(
    species: str,
    files: list[Path],
    bird_info,
    model_type: str,
    bands: int,
    sample_rate: int,
    order: int,
    num_taps: int,
    dynamic_bands: list[tuple[float, float]] | None = None,
    band_metadata: dict[str, float | int] | None = None,
) -> dict:
    """Procesa los audios de una especie y devuelve su perfil estadístico."""
    fft = FFTProcessor()
    low = float(bird_info.vocalizaciones.frecuencias_hz.rango_principal.min)
    high = float(bird_info.vocalizaciones.frecuencias_hz.rango_principal.max)

    species_bands = dynamic_bands if dynamic_bands else None
    band_count = len(species_bands) if species_bands else bands

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
            bands=band_count,
            dynamic_bands=species_bands,
            fft=fft,
        )
        if vec is not None:
            vectors.append(vec)

    if len(vectors) == 0:
        profile = np.zeros(band_count, dtype=np.float32)
        stdv = np.zeros(band_count, dtype=np.float32)
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
        "bands": int(band_count),
        "dynamic_bands": _bands_to_json(species_bands),
        "band_metadata": band_metadata or {},
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
    dynamic_bands: list[tuple[float, float]] | None,
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

    subbands = dynamic_bands if dynamic_bands else fft.build_subbands(low, high, bands)
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
    p.add_argument(
        "--analysis",
        type=str,
        default=str(Path("models") / "bird_spectrum_analysis.json"),
        help="Ruta al JSON con dynamic_bands por especie",
    )
    p.add_argument("--output", type=str, default=None, help="Ruta de salida JSON (default models/model_{type}.json)")
    args = p.parse_args()

    repo = BirdRepository(env="training")
    audios_by_species = repo.get_audios_by_species()
    analysis_path = Path(args.analysis)

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
        analysis_entry = _load_species_analysis(analysis_path, species)
        
        dynamic_bands = DynamicBandsDetector.detect_bands_from_audio(files[0], 
            int(args.sample_rate), 
            float(bird.vocalizaciones.frecuencias_hz.rango_principal.min), 
            float(bird.vocalizaciones.frecuencias_hz.rango_principal.max), 
            n_bands=int(args.bands))
        
        if dynamic_bands:
            print(f"  - usando {len(dynamic_bands)} bandas dinámicas desde {analysis_path.name}")
        else:
            print("  - sin bandas dinámicas, usando división uniforme")
        band_metadata = _band_metadata_from_profile(
            low=float(bird.vocalizaciones.frecuencias_hz.rango_principal.min),
            high=float(bird.vocalizaciones.frecuencias_hz.rango_principal.max),
            bands=max(1, int(args.bands)),
            sample_rate=int(args.sample_rate),
            analysis_entry=analysis_entry,
        )
        entry = process_species_audios(
            species=species,
            files=files,
            bird_info=bird,
            model_type=args.model_type,
            bands=max(1, int(args.bands)),
            sample_rate=int(args.sample_rate),
            order=int(args.order),
            num_taps=int(args.num_taps),
            dynamic_bands=dynamic_bands,
            band_metadata=band_metadata,
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
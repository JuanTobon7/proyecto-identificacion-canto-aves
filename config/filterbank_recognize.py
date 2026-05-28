"""
Reconocedor simple por vectores de energía de subbandas (banco de filtros).

Uso:
  python -m config.filterbank_recognize --model models/model_filterbank.json --file path/to.wav

Principio: calcula vector de energías por banda, compara con cada perfil
usando suma de diferencias absolutas (L1) y opcionalmente usando std para
rechazo.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.io import wavfile

from core.fft_filter import apply_hann_window, compute_fft


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def to_mono_float32(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        scale = float(max(abs(info.min), info.max))
        audio = audio.astype(np.float32) / scale if scale > 0 else audio.astype(np.float32)
    else:
        audio = audio.astype(np.float32)
    return audio


def extract_band_energies(freqs: np.ndarray, magnitude: np.ndarray, bands: List[List[float]]) -> np.ndarray:
    energies = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        energies.append(float(np.sum(magnitude[mask] ** 2)) if np.any(mask) else 0.0)
    return np.asarray(energies, dtype=np.float32)


def process_audio(path: Path, bands: List[List[float]]):
    sr, audio = wavfile.read(str(path))
    audio = to_mono_float32(audio)
    windowed = apply_hann_window(audio)
    freqs, magnitude = compute_fft(windowed, sr)
    energies = extract_band_energies(freqs, magnitude, bands)
    return energies, freqs, magnitude


def resolve_model_path(model_path: Path) -> Path:
    if model_path.exists():
        return model_path

    candidates = []
    stem = model_path.stem
    parent = model_path.parent

    if stem.endswith("_fir"):
        candidates.append(parent / f"{stem[:-4]}.json")
    if stem.startswith("model_"):
        candidates.append(parent / f"{stem}.json")
        candidates.append(parent / f"{stem.replace('_fir', '', 1)}.json")

    for candidate in candidates:
        if candidate.exists():
            logger.warning("Modelo %s no encontrado; usando %s", model_path, candidate)
            return candidate

    available = sorted(parent.glob("*.json"))
    if available:
        logger.warning("Modelo %s no encontrado; usando %s", model_path, available[0])
        return available[0]

    raise FileNotFoundError(f"No se encontró ningún modelo en {parent}")


def recognize(model_path: Path, audio_path: Path):
    model_path = resolve_model_path(model_path)
    with model_path.open("r", encoding="utf-8") as fh:
        model = json.load(fh)

    bands = model.get("bands") or model.get("frequency_bands")
    if not bands:
        raise ValueError("Modelo inválido: no contiene bandas")

    energies, freqs, magnitude = process_audio(audio_path, bands)

    best = None
    best_score = float("inf")
    results = []
    for profile in model.get("species_profiles", []):
        mean_vec = np.asarray(profile.get("mean_energy_vector", []), dtype=np.float32)
        std_vec = np.asarray(profile.get("std_energy_vector", []), dtype=np.float32)
        if mean_vec.size != energies.size:
            score = float("inf")
        else:
            # L1 score as described: sum absolute differences
            score = float(np.sum(np.abs(energies - mean_vec)))
        results.append((profile.get("species_key"), score, float(profile.get("rejection_threshold", float("nan")))))
        if score < best_score:
            best_score = score
            best = profile

    logger.info("Top candidate: %s (score=%.6f)", best.get("species_key") if best else None, best_score)
    # Provide full ranked results
    for name, score, thr in sorted(results, key=lambda x: x[1]):
        logger.info("  %s -> score=%.6f thr=%s", name, score, thr)

    return best, freqs, magnitude, energies


def main():
    parser = argparse.ArgumentParser(description="Reconocedor por bancos de filtros (energías por banda)")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--file", type=Path, required=True)
    args = parser.parse_args()

    best, freqs, magnitude, energies = recognize(args.model, args.file)
    print("Energies:", energies)
    if best:
        print("Best:", best.get("species_key"))


if __name__ == "__main__":
    main()

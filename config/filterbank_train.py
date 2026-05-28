"""
Entrenador de banco de filtros (energías por subbanda) tal como se describe en la especificación.

Genera un JSON con, por especie:
 - mean_energy_vector
 - std_energy_vector
 - sample_count
 - rejection_threshold (mean + 2*std sobre distancias L2)

Uso:
  python -m config.filterbank_train --dataset ../dataset_aves --bands 6 --output models/filterbank_model.json

"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

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


def build_frequency_bands(n_bands: int, min_hz: float = 50.0, max_hz: float = 16000.0) -> List[Tuple[float, float]]:
    edges = np.linspace(min_hz, max_hz, n_bands + 1)
    bands = [(float(edges[i]), float(edges[i + 1])) for i in range(n_bands)]
    return bands


def extract_band_energies_from_magnitude(freqs: np.ndarray, magnitude: np.ndarray, bands: List[Tuple[float, float]]) -> np.ndarray:
    energies = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        if np.any(mask):
            energies.append(float(np.sum(magnitude[mask] ** 2)))
        else:
            energies.append(0.0)
    return np.asarray(energies, dtype=np.float32)


def process_file(path: Path, bands: List[Tuple[float, float]]) -> np.ndarray:
    sr, audio = wavfile.read(str(path))
    audio = to_mono_float32(audio)
    if audio.size == 0:
        return np.zeros(len(bands), dtype=np.float32)
    windowed = apply_hann_window(audio)
    freqs, magnitude = compute_fft(windowed, sr)
    return extract_band_energies_from_magnitude(freqs, magnitude, bands)


def collect_species(species_dir: Path, bands: List[Tuple[float, float]]) -> List[np.ndarray]:
    samples = []
    for wav in sorted(species_dir.glob("*.wav")):
        try:
            vec = process_file(wav, bands)
            samples.append(vec)
        except Exception:
            logger.exception("Error procesando %s", wav)
    return samples


def compute_profile(samples: List[np.ndarray]) -> Dict[str, object]:
    if not samples:
        return {
            "sample_count": 0,
            "mean_energy_vector": [],
            "std_energy_vector": [],
            "profile_vector": [],
        }
    mat = np.asarray(samples, dtype=np.float32)
    mean_e = np.mean(mat, axis=0)
    std_e = np.std(mat, axis=0)
    # profile_vector: L2-normalized mean energy (matches the description)
    norm = np.linalg.norm(mean_e)
    profile = mean_e / norm if norm > 0 else mean_e
    # compute rejection threshold as mean+2std of distances (L2)
    dists = [float(np.linalg.norm((row / (np.linalg.norm(row) if np.linalg.norm(row)>0 else 1.0)) - profile)) for row in mat]
    mean_d = float(np.mean(dists)) if dists else 0.0
    std_d = float(np.std(dists)) if dists else 0.0
    threshold = float(mean_d + 2.0 * std_d)
    return {
        "sample_count": int(mat.shape[0]),
        "mean_energy_vector": mean_e.astype(float).tolist(),
        "std_energy_vector": std_e.astype(float).tolist(),
        "profile_vector": profile.astype(float).tolist(),
        "distance_mean": mean_d,
        "distance_std": std_d,
        "rejection_threshold": threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena banco de filtros y genera vectores de umbral de energía por especie.")
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parent.parent / "dataset_aves", help="Carpeta con subcarpetas por especie")
    parser.add_argument("--bands", type=int, default=6, help="Número de subbandas a usar")
    parser.add_argument("--min-hz", type=float, default=50.0)
    parser.add_argument("--max-hz", type=float, default=16000.0)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "models" / "model_filterbank.json")
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
        logger.info("Construido perfil para %s (muestras=%d)", species_dir.name, profile.get("sample_count", 0))

    payload = {
        "model_type": f"filterbank_{args.bands}",
        "kind": "filterbank_energy_thresholds",
        "bands": bands,
        "feature_summary": {"band_count": args.bands},
        "species": species_list,
        "species_profiles": profiles,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    logger.info("Modelo guardado en %s", args.output)


if __name__ == "__main__":
    main()

"""Genera plantillas espectrales deterministas para clasificar aves sin machine learning.

El flujo es:
1. Cargar audios del dataset suavizado
2. Aplicar FFT y calcular magnitudes
3. Calcular energías por banda
4. Construir plantillas promedio por especie
5. Guardar JSON con perfiles y metadatos

Uso:
    python -m config.train_model [--output DIR] [--model-type {cosine,weighted,both}]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.io import wavfile

from core.fft_filter import apply_hann_window, compute_fft


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset_aves_soften"
INFO_PATH = BASE_DIR / "general_info_aves.json"
OUTPUT_DIR = BASE_DIR / "models"


FREQUENCY_BANDS = [
	(50, 500),
	(500, 1000),
	(1000, 2000),
	(2000, 4000),
	(4000, 8000),
	(8000, 16000),
]


def normalize_species_name(name: Optional[str]) -> str:
	if not name:
		return ""
	# Ensure we operate on a string even if the source uses non-string values
	name_str = str(name)
	return name_str.strip().replace(" ", "_").replace("/", "_")


def band_name(low_hz: float, high_hz: float) -> str:
	return f"{int(low_hz)}_{int(high_hz)}_hz"


def band_center(low_hz: float, high_hz: float) -> float:
	return float((low_hz + high_hz) / 2.0)


def to_mono_float32(audio: np.ndarray) -> np.ndarray:
	if audio.size == 0:
		return audio.astype(np.float32, copy=False)
	if audio.ndim > 1:
		audio = np.mean(audio, axis=1)
	if np.issubdtype(audio.dtype, np.integer):
		info = np.iinfo(audio.dtype)
		scale = float(max(abs(info.min), info.max))
		audio = audio.astype(np.float32) / scale if scale > 0 else audio.astype(np.float32)
	else:
		audio = audio.astype(np.float32)
	return audio


def load_species_info(info_path: Path) -> Dict[str, Dict[str, object]]:
	if not info_path.exists():
		raise FileNotFoundError(f"No se encontró el archivo de especies: {info_path}")

	with info_path.open("r", encoding="utf-8") as handle:
		payload = json.load(handle)

	index: Dict[str, Dict[str, object]] = {}
	for record in payload.get("aves", []):
		common_en = normalize_species_name(record.get("nombre_comun_ingles", ""))
		if not common_en:
			continue
		index[common_en] = record
		index[normalize_species_name(record.get("nombre_comun_espanol", ""))] = record
		index[normalize_species_name(record.get("nombre_cientifico", ""))] = record
	return index


def extract_band_features(freqs: np.ndarray, magnitude: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	energy_vector: List[float] = []
	mean_magnitude_vector: List[float] = []

	for low_hz, high_hz in FREQUENCY_BANDS:
		mask = (freqs >= low_hz) & (freqs < high_hz)
		if np.any(mask):
			band_energy = float(np.sum(magnitude[mask] ** 2))
			band_mean_magnitude = float(np.mean(magnitude[mask]))
		else:
			band_energy = 0.0
			band_mean_magnitude = 0.0
		energy_vector.append(band_energy)
		mean_magnitude_vector.append(band_mean_magnitude)

	return np.asarray(energy_vector, dtype=np.float32), np.asarray(mean_magnitude_vector, dtype=np.float32)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
	if vector.size == 0:
		return vector.astype(np.float32, copy=False)
	norm = float(np.linalg.norm(vector))
	if norm > 0:
		return (vector / norm).astype(np.float32, copy=False)
	return vector.astype(np.float32, copy=False)


def weighted_vector(vector: np.ndarray, std_vector: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
	if vector.size == 0:
		return vector.astype(np.float32, copy=False)
	weights = 1.0 / np.maximum(std_vector, epsilon)
	return (vector * weights).astype(np.float32, copy=False)


def euclidean_distance(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
	if vector_a.size == 0 or vector_b.size == 0:
		return 0.0
	return float(np.linalg.norm(vector_a - vector_b))


def compute_species_distance_stats(energy_matrix: np.ndarray, profile_vector: np.ndarray, profile_std_vector: np.ndarray, scoring_method: str) -> Tuple[float, float, float]:
	distances: List[float] = []
	for sample_vector in energy_matrix:
		normalized_sample = normalize_vector(np.asarray(sample_vector, dtype=np.float32))
		if scoring_method == "weighted":
			reference_vector = weighted_vector(profile_vector, profile_std_vector)
			sample_reference = weighted_vector(normalized_sample, profile_std_vector)
		else:
			reference_vector = profile_vector
			sample_reference = normalized_sample
		distances.append(euclidean_distance(sample_reference, reference_vector))

	distance_mean = float(np.mean(distances)) if distances else 0.0
	distance_std = float(np.std(distances)) if distances else 0.0
	rejection_threshold = float(distance_mean + 2.0 * distance_std)
	return distance_mean, distance_std, rejection_threshold


def process_audio_file(audio_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
	try:
		sample_rate, audio = wavfile.read(str(audio_path))
		audio = to_mono_float32(audio)
		if audio.size == 0:
			return None

		windowed = apply_hann_window(audio)
		freqs, magnitude = compute_fft(windowed, sample_rate)
		if freqs.size == 0:
			logger.warning("FFT vacío para %s", audio_path.name)
			return None

		energy_vector, mean_magnitude_vector = extract_band_features(freqs, magnitude)
		return normalize_vector(energy_vector), mean_magnitude_vector, magnitude
	except Exception as exc:
		logger.exception("Error procesando %s: %s", audio_path.name, exc)
		return None


def collect_species_samples(species_dir: Path) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
	energy_vectors: List[np.ndarray] = []
	mean_magnitude_vectors: List[np.ndarray] = []
	sample_count = 0

	for audio_file in sorted(species_dir.glob("*.wav")):
		result = process_audio_file(audio_file)
		if result is None:
			continue
		energy_vector, mean_magnitude_vector, _ = result
		energy_vectors.append(energy_vector)
		mean_magnitude_vectors.append(mean_magnitude_vector)
		sample_count += 1
		logger.info("  ✓ %s", audio_file.name)

	return energy_vectors, mean_magnitude_vectors, sample_count


def build_dataset_summary(sample_count: int, band_energy_totals: np.ndarray, band_magnitude_totals: np.ndarray) -> Dict[str, object]:
	band_centers_hz = [band_center(low, high) for low, high in FREQUENCY_BANDS]
	band_names = [band_name(low, high) for low, high in FREQUENCY_BANDS]
	band_statistics = []
	for index, (low_hz, high_hz) in enumerate(FREQUENCY_BANDS):
		band_statistics.append({
			"name": band_names[index],
			"low_hz": low_hz,
			"high_hz": high_hz,
			"center_hz": band_centers_hz[index],
			"mean_energy": float(band_energy_totals[index] / sample_count) if sample_count else 0.0,
			"mean_magnitude": float(band_magnitude_totals[index] / sample_count) if sample_count else 0.0,
		})

	return {
		"pipeline": ["FFT", "magnitudes", "energias por banda", "vectores de energia"],
		"feature_type": "band_energy",
		"band_names": band_names,
		"band_centers_hz": band_centers_hz,
		"band_statistics": band_statistics,
		"sample_count": int(sample_count),
	}


def build_species_profile(species_key: str, species_dir: Path, species_info: Dict[str, Dict[str, object]], scoring_method: str) -> Optional[Dict[str, object]]:
	energy_vectors, mean_magnitude_vectors, sample_count = collect_species_samples(species_dir)
	if sample_count == 0:
		return None

	energy_matrix = np.asarray(energy_vectors, dtype=np.float32)
	mean_magnitude_matrix = np.asarray(mean_magnitude_vectors, dtype=np.float32)
	energy_mean = np.mean(energy_matrix, axis=0) if energy_matrix.size else np.zeros(len(FREQUENCY_BANDS), dtype=np.float32)
	energy_std = np.std(energy_matrix, axis=0) if energy_matrix.size else np.zeros(len(FREQUENCY_BANDS), dtype=np.float32)
	mean_magnitude_mean = np.mean(mean_magnitude_matrix, axis=0) if mean_magnitude_matrix.size else np.zeros(len(FREQUENCY_BANDS), dtype=np.float32)
	profile_vector = normalize_vector(np.asarray(energy_mean, dtype=np.float32))
	profile_std_vector = np.asarray(energy_std, dtype=np.float32)
	distance_mean, distance_std, rejection_threshold = compute_species_distance_stats(energy_matrix, profile_vector, profile_std_vector, scoring_method)

	info = species_info.get(species_key, {})
	frequency_info = info.get("vocalizaciones", {}).get("frecuencias_Hz", {}) if isinstance(info, dict) else {}
	range_info = frequency_info.get("rango_principal", {}) if isinstance(frequency_info, dict) else {}

	return {
		"species_key": species_key,
		"common_name_en": info.get("nombre_comun_ingles", species_key) if isinstance(info, dict) else species_key,
		"common_name_es": info.get("nombre_comun_espanol", "") if isinstance(info, dict) else "",
		"scientific_name": info.get("nombre_cientifico", "") if isinstance(info, dict) else "",
		"family": info.get("familia", "") if isinstance(info, dict) else "",
		"order": info.get("orden", "") if isinstance(info, dict) else "",
		"image_url": info.get("img", "") if isinstance(info, dict) else "",
		"vocalization_frequency_range": {
			"min_hz": float(range_info.get("min", 0.0)) if isinstance(range_info, dict) else 0.0,
			"max_hz": float(range_info.get("max", 0.0)) if isinstance(range_info, dict) else 0.0,
			"dominant_hz": float(frequency_info.get("frecuencia_dominante", 0.0)) if isinstance(frequency_info, dict) else 0.0,
		},
		"sample_count": int(sample_count),
		"profile_vector": profile_vector.tolist(),
		"mean_energy_vector": energy_mean.astype(float).tolist(),
		"std_energy_vector": energy_std.astype(float).tolist(),
		"distance_mean": distance_mean,
		"distance_std": distance_std,
		"rejection_threshold": rejection_threshold,
		"mean_magnitude_vector": mean_magnitude_mean.astype(float).tolist(),
	}


def load_dataset(scoring_method: str) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
	if not DATASET_PATH.exists():
		raise FileNotFoundError(f"Dataset no encontrado: {DATASET_PATH}")

	species_info = load_species_info(INFO_PATH)
	profiles: List[Dict[str, object]] = []
	band_energy_totals = np.zeros(len(FREQUENCY_BANDS), dtype=np.float64)
	band_magnitude_totals = np.zeros(len(FREQUENCY_BANDS), dtype=np.float64)
	total_samples = 0

	logger.info("Cargando dataset desde %s", DATASET_PATH)
	for species_dir in sorted(DATASET_PATH.iterdir()):
		if not species_dir.is_dir():
			continue

		species_key = species_dir.name
		logger.info("Procesando especie: %s", species_key)
		profile = build_species_profile(species_key, species_dir, species_info, scoring_method)
		if profile is None:
			logger.warning("  No hay muestras válidas para %s", species_key)
			continue

		profiles.append(profile)
		sample_count = int(profile["sample_count"])
		total_samples += sample_count
		band_energy_totals += np.asarray(profile["mean_energy_vector"], dtype=np.float64) * sample_count
		band_magnitude_totals += np.asarray(profile["mean_magnitude_vector"], dtype=np.float64) * sample_count

	if not profiles:
		raise ValueError("No se pudieron construir perfiles espectrales")

	dataset_summary = build_dataset_summary(total_samples, band_energy_totals, band_magnitude_totals)
	logger.info("Perfiles construidos: %d especies, %d muestras", len(profiles), total_samples)
	return profiles, dataset_summary


def build_model_payload(scoring_method: str, profiles: List[Dict[str, object]], dataset_summary: Dict[str, object]) -> Dict[str, object]:
	return {
		"model_type": f"spectral_{scoring_method}",
		"kind": "deterministic_spectral_template",
		"scoring_method": scoring_method,
		"frequency_bands": FREQUENCY_BANDS,
		"feature_summary": dataset_summary,
		"species": [profile["species_key"] for profile in profiles],
		"species_profiles": profiles,
		"matching": {
			"distance_metric": "euclidean",
			"threshold_strategy": "mean_plus_2std",
			"ambiguity_margin": 0.05,
			"temperature": 0.35,
		},
	}


def save_model(payload: Dict[str, object], output_dir: Path, model_name: str) -> Path:
	output_dir.mkdir(parents=True, exist_ok=True)
	model_path = output_dir / f"model_{model_name}.json"
	with model_path.open("w", encoding="utf-8") as handle:
		json.dump(payload, handle, indent=2, ensure_ascii=False)
	return model_path


def build_and_save_models(model_type: str, output_dir: Path, profiles: List[Dict[str, object]], dataset_summary: Dict[str, object]) -> List[Path]:
	model_variants = ["cosine", "weighted"] if model_type == "both" else [model_type]
	created_paths: List[Path] = []

	for variant in model_variants:
		if variant not in {"cosine", "weighted"}:
			raise ValueError(f"Modelo no soportado: {variant}")
		payload = build_model_payload(variant, profiles, dataset_summary)
		created_paths.append(save_model(payload, output_dir, f"spectral_{variant}"))
		logger.info("Modelo guardado: %s", created_paths[-1])

	return created_paths


def main() -> None:
	parser = argparse.ArgumentParser(description="Genera plantillas espectrales para clasificar aves sin machine learning.")
	parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Directorio de salida para los modelos JSON.")
	parser.add_argument(
		"--model-type",
		choices=["cosine", "weighted", "both"],
		default="both",
		help="Variante de scoring a guardar.",
	)
	args = parser.parse_args()

	try:
		profiles, dataset_summary = load_dataset(args.model_type if args.model_type in {"cosine", "weighted"} else "cosine")
		created_paths = build_and_save_models(args.model_type, args.output, profiles, dataset_summary)
		logger.info("=" * 60)
		logger.info("Plantillas generadas")
		for path in created_paths:
			logger.info("  %s", path.name)
		logger.info("Especies: %d", len(profiles))
		logger.info("=" * 60)
	except Exception as exc:
		logger.exception("Error generando las plantillas: %s", exc)
		raise


if __name__ == "__main__":
	main()

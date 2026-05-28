"""
Filtro digital para atenuar sonidos de fondo en los audios del dataset.

Implementa dos alternativas:
- IIR Butterworth pasabanda
- FIR pasabanda con ventana de Hamming

Las bandas se ajustan a las frecuencias reportadas en dataset_aves/general_info_aves.json.

Salida por defecto:
- Audios filtrados en la misma carpeta de cada especie
- Nombre: <archivo_original>_<design>.wav
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import filtfilt, firwin
from core.fft_filter import compute_fft, apply_butterworth_filter


logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset_aves"
DATASET_SOFTEN_PATH = BASE_DIR / "dataset_aves_soften"
DATASET_FIR_PATH = BASE_DIR / "dataset_aves_fir"
INFO_PATH = DATASET_PATH / "general_info_aves.json"

DEFAULT_BUTTER_ORDER = 6
DEFAULT_FIR_TAPS = 255


def normalize_species_name(name: str) -> str:
	return name.strip().replace(" ", "_")


def load_species_bands(info_path: Path) -> Dict[str, Tuple[float, float]]:
	"""Lee el JSON de metadatos y construye bandas pasabanda por carpeta."""
	if not info_path.exists():
		raise FileNotFoundError(f"No existe el archivo de metadatos: {info_path}")

	with info_path.open("r", encoding="utf-8") as f:
		data = json.load(f)

	species_bands: Dict[str, Tuple[float, float]] = {}

	for bird in data.get("aves", []):
		common_en = bird.get("nombre_comun_ingles", "")
		folder_name = normalize_species_name(common_en)

		freq_info = bird.get("vocalizaciones", {}).get("frecuencias_Hz", {})
		main_range = freq_info.get("rango_principal", {})

		min_freq = float(main_range.get("min", 0))
		max_freq = float(main_range.get("max", 0))

		if min_freq <= 0 or max_freq <= 0 or max_freq <= min_freq:
			continue

		# Margen para no cortar armónicos cercanos y suavizar ruido fuera de banda.
		margin = max(150.0, 0.10 * (max_freq - min_freq))
		low = max(20.0, min_freq - margin)
		high = max_freq + margin

		species_bands[folder_name] = (low, high)

	return species_bands


def ensure_mono(audio: np.ndarray) -> np.ndarray:
	if audio.ndim == 1:
		return audio.astype(np.float32, copy=False)
	return np.mean(audio, axis=1).astype(np.float32, copy=False)


def design_butterworth_bandpass(sample_rate: int, low_hz: float, high_hz: float, order: int = DEFAULT_BUTTER_ORDER):
	# Esta función se delega a core.fft_filter.apply_butterworth_filter.
	raise RuntimeError("use core.fft_filter.apply_butterworth_filter instead")


def design_fir_bandpass(sample_rate: int, low_hz: float, high_hz: float, numtaps: int = DEFAULT_FIR_TAPS):
	nyquist = 0.5 * sample_rate
	low = max(20.0, low_hz) / nyquist
	high = min(high_hz, nyquist * 0.98) / nyquist

	if not 0 < low < high < 1:
		raise ValueError(
			f"Banda inválida para FIR: low={low_hz}, high={high_hz}, sr={sample_rate}"
		)

	return firwin(numtaps, [low, high], pass_zero=False, window="hamming")


def apply_filter(audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float, design: str):
	"""Aplica el filtro seleccionado al audio y devuelve float32 normalizado.

	- para 'butterworth' delega en core.apply_butterworth_filter
	- para 'fir' diseña y aplica FIR localmente
	"""
	if design == "butterworth":
		return apply_butterworth_filter(audio, sample_rate, low_hz, high_hz, order=DEFAULT_BUTTER_ORDER)
	elif design == "fir":
		# Asegurar mono
		audio_mono = ensure_mono(audio)
		taps = design_fir_bandpass(sample_rate, low_hz, high_hz)
		padlen = min(len(audio_mono) - 1, 3 * (len(taps) - 1))
		if padlen < 1:
			filtered = np.convolve(audio_mono, taps, mode="same")
		else:
			filtered = filtfilt(taps, [1.0], audio_mono, padlen=padlen)
		peak = np.max(np.abs(filtered)) if filtered.size else 0.0
		if peak > 0:
			filtered = 0.98 * filtered / peak
		return filtered.astype(np.float32, copy=False)
	else:
		raise ValueError(f"Diseño no soportado: {design}")


def process_audio_file(audio_path: Path, output_path: Path, low_hz: float, high_hz: float, design: str):
	sample_rate, audio = wavfile.read(str(audio_path))
	filtered = apply_filter(audio, sample_rate, low_hz, high_hz, design)
	wavfile.write(str(output_path), sample_rate, np.int16(np.clip(filtered, -1.0, 1.0) * 32767))


def iter_audio_files(species_dir: Path) -> Iterable[Path]:
	for ext in ("*.wav",):
		yield from sorted(species_dir.glob(ext))


def build_output_path(audio_file: Path, design: str, output_root: Path) -> Path:
	return output_root / audio_file.parent.name / f"{audio_file.stem}_{design}.wav"


def prepare_output_roots() -> None:
	DATASET_SOFTEN_PATH.mkdir(parents=True, exist_ok=True)
	DATASET_FIR_PATH.mkdir(parents=True, exist_ok=True)


def get_designs(selected_design: str):
	return [selected_design] if selected_design != "both" else ["butterworth", "fir"]


def process_species_directory(species_dir: Path, band: Tuple[float, float], selected_design: str, overwrite: bool) -> tuple[int, int]:
	low_hz, high_hz = band
	logger.info("Procesando especie %s con banda %.1f-%.1f Hz", species_dir.name, low_hz, high_hz)

	audio_files = [p for p in iter_audio_files(species_dir) if not p.stem.endswith(("_butterworth", "_fir"))]
	if not audio_files:
		logger.info("  No hay audios para procesar.")
		return 0, 0

	processed = 0
	generated = 0
	for audio_file in audio_files:
		processed += 1
		for design in get_designs(selected_design):
			output_root = DATASET_SOFTEN_PATH if design == "butterworth" else DATASET_FIR_PATH
			output_file = build_output_path(audio_file, design, output_root)
			output_file.parent.mkdir(parents=True, exist_ok=True)

			if output_file.exists() and not overwrite:
				logger.info("  Ya existe: %s (se omite)", output_file.name)
				continue

			try:
				process_audio_file(audio_file, output_file, low_hz, high_hz, design)
				generated += 1
				logger.info("  ✓ %s -> %s", audio_file.name, output_file.name)
			except Exception:
				logger.exception("Error procesando %s", audio_file.name)

	return processed, generated


def main():
	parser = argparse.ArgumentParser(
		description="Atenúa el ruido de fondo aplicando filtros pasabanda por especie."
	)
	parser.add_argument(
		"--design",
		choices=["butterworth", "fir", "both"],
		default="butterworth",
		help="Tipo de filtro a aplicar.",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Sobrescribe archivos existentes.",
	)
	args = parser.parse_args()

	if not DATASET_PATH.exists():
		raise FileNotFoundError(f"No existe la carpeta del dataset: {DATASET_PATH}")

	prepare_output_roots()

	species_bands = load_species_bands(INFO_PATH)
	logger.info("Bandas cargadas desde general_info_aves.json")

	total_input = 0
	total_output = 0

	for species_dir in sorted(DATASET_PATH.iterdir()):
		if not species_dir.is_dir():
			continue

		band = species_bands.get(species_dir.name)
		if band is None:
			logger.warning("No se encontró banda para %s; se omite.", species_dir.name)
			continue

		processed, generated = process_species_directory(species_dir, band, args.design, args.overwrite)
		total_input += processed
		total_output += generated

	logger.info("=" * 60)
	logger.info("Resumen")
	logger.info("  Audios de entrada: %d", total_input)
	logger.info("  Archivos generados: %d", total_output)
	logger.info("  Diseño aplicado: %s", args.design)
	logger.info("=" * 60)


if __name__ == "__main__":
	main()

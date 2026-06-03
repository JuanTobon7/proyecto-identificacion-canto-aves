"""
Compara el espectro de 3 especies de aves en una única gráfica.
Muestra también las subbandas dinámicas y vectores de energía de cada ave.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from config.routes_path import RoutesPath
from core.audio_converter import AudioConverter
from core.birds_data_loader import BirdDataLoader
from core.maths.dynamic_bands_detector import DynamicBandsDetector
from core.maths.energy_vector import EnergyVector
from core.maths.fft import FFTProcessor
from core.maths.filter_butterworth import FilterButterworth


WHITE_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "#f8fafc",
    "axes.edgecolor": "#94a3b8",
    "axes.labelcolor": "#0f172a",
    "axes.titlecolor": "#0f172a",
    "xtick.color": "#334155",
    "ytick.color": "#334155",
    "grid.color": "#cbd5e1",
    "text.color": "#0f172a",
    "legend.facecolor": "white",
    "legend.edgecolor": "#cbd5e1",
}


def load_audio(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    """Carga un archivo de audio."""
    audio, sample_rate = librosa.load(str(path), sr=None, mono=False)
    audio = AudioConverter.to_mono_float32(np.asarray(audio))
    if target_sr > 0 and sample_rate != target_sr:
        audio = AudioConverter.resample(audio, sample_rate, target_sr)
        sample_rate = target_sr
    return audio, sample_rate


def compute_average_spectrum(
    audio: np.ndarray, sample_rate: int, fft_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula el espectro promedio normalizado."""
    freqs, magnitude = FFTProcessor.compute_fft(audio, sample_rate)

    if freqs.size == 0 or magnitude.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    peak = float(np.max(magnitude)) if np.max(magnitude) > 0 else 1.0
    normalized = magnitude / peak
    freq_grid = np.linspace(0.0, sample_rate / 2.0, fft_points)
    interpolated = np.interp(freq_grid, freqs, normalized, left=0.0, right=0.0)
    return freq_grid, interpolated


def analyze_species(
    audio_paths: list[Path],
    low_freq: float,
    high_freq: float,
    sample_rate: int,
    band_count: int,
    fft_points: int,
) -> dict:
    """Analiza una especie y retorna espectro, subbandas y energía."""
    spectrum_values: list[np.ndarray] = []
    filtered_spectrum_values: list[np.ndarray] = []
    energy_values: list[np.ndarray] = []
    freq_grid: np.ndarray | None = None

    # Crear filtro butterworth
    butterworth = FilterButterworth(order=4)

    for audio_path in audio_paths[:10]:  # Usar máximo 10 archivos
        try:
            audio, sr = load_audio(audio_path, sample_rate)
        except Exception:
            continue

        if audio.size == 0 or sr <= 0:
            continue

        # Espectro sin filtrar
        current_freq_grid, spectrum = compute_average_spectrum(audio, sr, fft_points)
        if current_freq_grid.size == 0:
            continue

        freq_grid = current_freq_grid
        spectrum_values.append(spectrum)

        # Espectro filtrado
        try:
            filtered_audio = butterworth.apply_bandpass(audio, sr, low_freq, high_freq)
            _, filtered_spectrum = compute_average_spectrum(filtered_audio, sr, fft_points)
            filtered_spectrum_values.append(filtered_spectrum)
        except Exception:
            filtered_spectrum_values.append(spectrum)

        # Detectar subbandas dinámicas
        dynamic_bands = DynamicBandsDetector.detect_bands_from_audio(
            audio, sr, low_freq, high_freq, band_count
        )

        band_energies = FFTProcessor.compute_band_energies(audio, sr, dynamic_bands)
        energy_values.append(EnergyVector.compute(band_energies).astype(np.float64))

    if not spectrum_values or not energy_values or freq_grid is None:
        return {
            "freq_grid": np.array([], dtype=np.float64),
            "mean_spectrum": np.array([], dtype=np.float64),
            "mean_filtered_spectrum": np.array([], dtype=np.float64),
            "dynamic_bands": [],
            "band_labels": [],
            "mean_energy": np.array([], dtype=np.float64),
        }

    spectrum_stack = np.vstack(spectrum_values)
    filtered_spectrum_stack = np.vstack(filtered_spectrum_values)
    energy_stack = np.vstack(energy_values)

    # Las bandas dinámicas ya se detectaron en el primer archivo
    # que fue procesado exitosamente arriba
    if energy_stack.shape[0] == 0:
        return {
            "freq_grid": np.array([], dtype=np.float64),
            "mean_spectrum": np.array([], dtype=np.float64),
            "mean_filtered_spectrum": np.array([], dtype=np.float64),
            "dynamic_bands": [],
            "band_labels": [],
            "mean_energy": np.array([], dtype=np.float64),
        }

    # Detectar bandas dinámicas del primer audio procesado exitosamente
    first_successful_audio = None
    for audio_path in audio_paths[:10]:
        try:
            audio, sr = load_audio(audio_path, sample_rate)
            if audio.size > 0 and sr > 0:
                first_successful_audio = audio
                break
        except Exception:
            continue

    if first_successful_audio is not None:
        dynamic_bands = DynamicBandsDetector.detect_bands_from_audio(
            first_successful_audio,
            sample_rate,
            low_freq,
            high_freq,
            band_count,
        )
    else:
        # Fallback a bandas uniformes
        step = (high_freq - low_freq) / band_count
        dynamic_bands = [
            (low_freq + i * step, low_freq + (i + 1) * step)
            for i in range(band_count)
        ]

    band_labels = [f"{int(low)}-{int(high)}Hz" for low, high in dynamic_bands]

    return {
        "freq_grid": freq_grid,
        "mean_spectrum": spectrum_stack.mean(axis=0),
        "mean_filtered_spectrum": filtered_spectrum_stack.mean(axis=0),
        "dynamic_bands": dynamic_bands,
        "band_labels": band_labels,
        "mean_energy": energy_stack.mean(axis=0),
    }


def plot_comparison(
    species_data: dict[str, dict],
    output_path: Path,
) -> None:
    """Crea gráfica comparativa con espectros, subbandas y energías."""
    plt.style.use("default")
    plt.rcParams.update(WHITE_STYLE)

    # Crear figura con espacios para gráfica y tablas
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1.5], hspace=0.4)

    # Gráfica 1: Espectros combinados
    ax_spectrum = fig.add_subplot(gs[0])

    colors = ["#1d4ed8", "#dc2626", "#059669", "#d97706", "#7c3aed"]
    max_freq = 0

    for idx, (species_name, data) in enumerate(species_data.items()):
        freq_grid = data["freq_grid"]
        mean_spectrum = data["mean_spectrum"]
        mean_filtered_spectrum = data.get("mean_filtered_spectrum", np.array([]))

        if freq_grid.size == 0 or mean_spectrum.size == 0:
            continue

        color = colors[idx % len(colors)]

        # Espectro sin filtrar (línea sólida)
        ax_spectrum.plot(
            freq_grid,
            mean_spectrum,
            color=color,
            linewidth=2.5,
            label=f"{species_name} (sin filtro)",
            alpha=0.85,
            linestyle="-",
        )

        # Espectro filtrado (línea punteada)
        if mean_filtered_spectrum.size > 0:
            ax_spectrum.plot(
                freq_grid,
                mean_filtered_spectrum,
                color=color,
                linewidth=2.0,
                label=f"{species_name} (filtrado)",
                alpha=0.6,
                linestyle="--",
            )

        max_freq = max(max_freq, float(freq_grid.max()))

        # Mostrar subbandas
        for low, high in data["dynamic_bands"]:
            ax_spectrum.axvspan(low, high, alpha=0.05, color=color)

    ax_spectrum.set_title(
        "Comparación de Espectros de Aves",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax_spectrum.set_xlabel("Frecuencia (Hz)", fontsize=11)
    ax_spectrum.set_ylabel("Magnitud normalizada", fontsize=11)
    ax_spectrum.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
    ax_spectrum.set_xlim(0, max_freq)
    ax_spectrum.set_ylim(bottom=0)
    ax_spectrum.legend(loc="upper right", fontsize=10, framealpha=0.95)

    # Tabla 1: Subbandas
    ax_bands = fig.add_subplot(gs[1])
    ax_bands.axis("off")

    bands_data = []
    max_bands = max(len(data["band_labels"]) for data in species_data.values())

    for band_idx in range(max_bands):
        row = [f"Banda {band_idx + 1}"]
        for species_name, data in species_data.items():
            if band_idx < len(data["band_labels"]):
                row.append(data["band_labels"][band_idx])
            else:
                row.append("--")
        bands_data.append(row)

    table_bands = ax_bands.table(
        cellText=bands_data,
        colLabels=["Banda"] + list(species_data.keys()),
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table_bands.auto_set_font_size(False)
    table_bands.set_fontsize(9)
    table_bands.scale(1, 1.8)

    # Estilo de tabla
    for i in range(len(species_data) + 1):
        table_bands[(0, i)].set_facecolor("#1d4ed8")
        table_bands[(0, i)].set_text_props(weight="bold", color="white")

    ax_bands.text(0.5, 1.15, "Subbandas Dinámicas", ha="center", fontsize=12, fontweight="bold", transform=ax_bands.transAxes)

    # Tabla 2: Vectores de Energía
    ax_energy = fig.add_subplot(gs[2])
    ax_energy.axis("off")

    energy_data = []
    max_bands = max(len(data["band_labels"]) for data in species_data.values())

    for band_idx in range(max_bands):
        row = []
        # Primera columna con etiqueta de banda
        first_species = list(species_data.values())[0]
        if band_idx < len(first_species["band_labels"]):
            row.append(first_species["band_labels"][band_idx])
        else:
            row.append(f"Banda {band_idx + 1}")

        # Energía para cada especie
        for species_name, data in species_data.items():
            if band_idx < len(data["mean_energy"]):
                energy_val = data["mean_energy"][band_idx]
                row.append(f"{energy_val:.4f}")
            else:
                row.append("--")

        energy_data.append(row)

    table_energy = ax_energy.table(
        cellText=energy_data,
        colLabels=["Banda"] + list(species_data.keys()),
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table_energy.auto_set_font_size(False)
    table_energy.set_fontsize(9)
    table_energy.scale(1, 1.8)

    # Estilo de tabla
    for i in range(len(species_data) + 1):
        table_energy[(0, i)].set_facecolor("#059669")
        table_energy[(0, i)].set_text_props(weight="bold", color="white")

    ax_energy.text(
        0.5,
        1.15,
        "Vectores de Energía",
        ha="center",
        fontsize=12,
        fontweight="bold",
        transform=ax_energy.transAxes,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] Gráfica comparativa -> {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compara el espectro de 3 especies de aves en una única gráfica."
    )
    parser.add_argument(
        "--dataset-dir",
        default=RoutesPath.BANK_AUDIOS_NORMALIZED,
        help="Directorio base del dataset.",
    )
    parser.add_argument(
        "--output",
        default="reports/species_comparison.png",
        help="Ruta de salida para la gráfica.",
    )
    parser.add_argument(
        "--species",
        nargs=3,
        required=True,
        help="3 nombres de especies a comparar (ej: Dusky_Antbird Great_Tinamou Barred_Antshrike)",
    )
    parser.add_argument(
        "--band-count",
        type=int,
        default=8,
        help="Número de subbandas dinámicas.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Sample rate para los audios.",
    )
    parser.add_argument(
        "--fft-points",
        type=int,
        default=4096,
        help="Puntos para el espectro.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    birds = BirdDataLoader(RoutesPath.AVES_INFO).load()
    bird_lookup = {bird.nombre_comun_ingles: bird for bird in birds}

    species_data = {}

    for species_name in args.species:
        if species_name not in bird_lookup:
            print(f"[skip] {species_name}: no encontrada en la base de datos")
            continue

        bird = bird_lookup[species_name]
        species_dir = dataset_dir / species_name
        audio_paths = sorted(species_dir.glob("*.wav"))

        if not audio_paths:
            print(f"[skip] {species_name}: sin audios")
            continue

        low_freq = float(bird.vocalizaciones.frecuencias_hz.rango_principal.min)
        high_freq = float(bird.vocalizaciones.frecuencias_hz.rango_principal.max)

        print(f"[procesando] {species_name}...")
        analysis = analyze_species(
            audio_paths,
            low_freq,
            high_freq,
            args.sample_rate,
            args.band_count,
            args.fft_points,
        )

        if analysis["freq_grid"].size > 0:
            species_data[species_name] = analysis

    if len(species_data) < 2:
        print("[error] Se necesitan al menos 2 especies válidas para comparar")
        return 1

    output_path = Path(args.output)
    plot_comparison(species_data, output_path)

    if output_path.exists():
        print(f"[ok] Comparativa generada exitosamente")
        return 0
    else:
        print("[error] No se pudo generar la gráfica")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

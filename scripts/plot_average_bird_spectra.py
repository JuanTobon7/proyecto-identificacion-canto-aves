"""
Genera gráficos por especie con:
- espectro promedio
- vector de energía promedio
- desviación estándar de ambos

La salida se guarda en PNG con tema blanco.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from config.routes_path import RoutesPath
from core.audio_converter import AudioConverter
from core.birds_data_loader import BirdDataLoader
from core.maths.energy_vector import EnergyVector
from core.maths.fft import FFTProcessor


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


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "species"


def format_frequency_label(low: float, high: float) -> str:
    if max(low, high) >= 1000:
        return f"{low / 1000:.1f}-{high / 1000:.1f} kHz"
    return f"{low:.0f}-{high:.0f} Hz"


def load_audio(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    audio, sample_rate = librosa.load(str(path), sr=None, mono=False)
    audio = AudioConverter.to_mono_float32(np.asarray(audio))
    if target_sr > 0 and sample_rate != target_sr:
        audio = AudioConverter.resample(audio, sample_rate, target_sr)
        sample_rate = target_sr
    return audio, sample_rate


def compute_average_spectrum(audio: np.ndarray, sample_rate: int, fft_points: int) -> tuple[np.ndarray, np.ndarray]:
    freqs, magnitude = FFTProcessor.compute_fft(audio, sample_rate)
    if freqs.size == 0 or magnitude.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    peak = float(np.max(magnitude)) if np.max(magnitude) > 0 else 1.0
    normalized = magnitude / peak
    freq_grid = np.linspace(0.0, sample_rate / 2.0, fft_points)
    interpolated = np.interp(freq_grid, freqs, normalized, left=0.0, right=0.0)
    return freq_grid, interpolated


def summarize_species(
    audio_paths: list[Path],
    low_freq: float,
    high_freq: float,
    sample_rate: int,
    band_count: int,
    fft_points: int,
) -> dict[str, np.ndarray]:
    bands = FFTProcessor.build_subbands(low_freq, high_freq, band_count)
    spectrum_values: list[np.ndarray] = []
    energy_values: list[np.ndarray] = []
    freq_grid: np.ndarray | None = None

    for audio_path in audio_paths:
        try:
            audio, sr = load_audio(audio_path, sample_rate)
        except Exception:
            continue

        if audio.size == 0 or sr <= 0:
            continue

        current_freq_grid, spectrum = compute_average_spectrum(audio, sr, fft_points)
        if current_freq_grid.size == 0:
            continue

        freq_grid = current_freq_grid
        spectrum_values.append(spectrum)

        band_energies = FFTProcessor.compute_band_energies(audio, sr, bands)
        energy_values.append(EnergyVector.compute(band_energies).astype(np.float64))

    if not spectrum_values or not energy_values or freq_grid is None:
        return {
            "freq_grid": np.array([], dtype=np.float64),
            "mean_spectrum": np.array([], dtype=np.float64),
            "std_spectrum": np.array([], dtype=np.float64),
            "mean_energy": np.array([], dtype=np.float64),
            "std_energy": np.array([], dtype=np.float64),
            "bands": np.array([], dtype=object),
        }

    spectrum_stack = np.vstack(spectrum_values)
    energy_stack = np.vstack(energy_values)
    band_labels = np.array([format_frequency_label(low, high) for low, high in bands], dtype=object)

    return {
        "freq_grid": freq_grid,
        "mean_spectrum": spectrum_stack.mean(axis=0),
        "std_spectrum": spectrum_stack.std(axis=0),
        "mean_energy": energy_stack.mean(axis=0),
        "std_energy": energy_stack.std(axis=0),
        "bands": band_labels,
    }
def plot_species_report(
    species_name: str,
    summary: dict[str, np.ndarray],
    output_path: Path,
    title_prefix: str = "Espectro promedio y vector de energía",
) -> None:
    freq_grid = summary["freq_grid"]
    mean_spectrum = summary["mean_spectrum"]
    std_spectrum = summary["std_spectrum"]
    mean_energy = summary["mean_energy"]
    std_energy = summary["std_energy"]
    band_labels = summary["bands"]

    if freq_grid.size == 0 or mean_spectrum.size == 0 or mean_energy.size == 0:
        return

    plt.style.use("default")
    plt.rcParams.update(WHITE_STYLE)

    fig, (ax_spectrum, ax_energy) = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)
    fig.suptitle(f"{title_prefix} - {species_name}", fontsize=16, fontweight="bold")
    fig.patch.set_facecolor("white")

    ax_spectrum.plot(freq_grid, mean_spectrum, color="#1d4ed8", linewidth=1.8, label="Promedio")
    ax_spectrum.fill_between(
        freq_grid,
        np.maximum(mean_spectrum - std_spectrum, 0.0),
        mean_spectrum + std_spectrum,
        color="#60a5fa",
        alpha=0.20,
        label="± desviación estándar",
    )
    ax_spectrum.set_title("Espectro promedio")
    ax_spectrum.set_xlabel("Frecuencia (Hz)")
    ax_spectrum.set_ylabel("Magnitud")
    ax_spectrum.grid(True, linestyle="--", linewidth=0.7)
    ax_spectrum.set_ylim(bottom=0.0)
    ax_spectrum.set_xlim(0, float(freq_grid.max()))
    ax_spectrum.legend(loc="best")

    indices = np.arange(len(mean_energy))
    ax_energy.bar(
        indices,
        mean_energy,
        yerr=std_energy,
        capsize=4,
        color="#2563eb",
        edgecolor="#1e3a8a",
        linewidth=0.8,
        alpha=0.95,
        label="Promedio",
    )
    ax_energy.set_xticks(indices)
    ax_energy.set_xticklabels(band_labels, rotation=25, ha="right")
    ax_energy.set_title("Vector de energía promedio")
    ax_energy.set_xlabel("Bandas de frecuencia")
    ax_energy.set_ylabel("Energía normalizada")
    ax_energy.grid(True, axis="y", linestyle="--", linewidth=0.7)
    ax_energy.legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Genera gráficos blancos de espectro promedio y vector de energía por especie."
    )
    parser.add_argument(
        "--dataset-dir",
        default=RoutesPath.BANK_AUDIOS_NORMALIZED,
        help="Directorio base del dataset por especie.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/average_spectra",
        help="Directorio de salida para los PNG.",
    )
    parser.add_argument(
        "--species",
        action="append",
        default=None,
        help="Filtra una especie concreta. Puede repetirse.",
    )
    parser.add_argument(
        "--band-count",
        type=int,
        default=8,
        help="Número de bandas para el vector de energía.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Sample rate objetivo para remuestrear los audios.",
    )
    parser.add_argument(
        "--fft-points",
        type=int,
        default=4096,
        help="Puntos de interpolación para el espectro promedio.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    birds = BirdDataLoader(RoutesPath.AVES_INFO).load()
    species_filter = set(args.species) if args.species else None

    generated = 0
    for bird in birds:
        species_name = bird.nombre_comun_ingles
        if species_filter and species_name not in species_filter:
            continue

        species_dir = dataset_dir / species_name
        audio_paths = sorted(species_dir.glob("*.wav"))
        if not audio_paths:
            print(f"[skip] {species_name}: sin audios en {species_dir}")
            continue

        low_freq = float(bird.vocalizaciones.frecuencias_hz.rango_principal.min)
        high_freq = float(bird.vocalizaciones.frecuencias_hz.rango_principal.max)
        summary = summarize_species(
            audio_paths=audio_paths,
            low_freq=low_freq,
            high_freq=high_freq,
            sample_rate=args.sample_rate,
            band_count=max(1, int(args.band_count)),
            fft_points=max(128, int(args.fft_points)),
        )

        output_path = output_dir / f"{sanitize_filename(species_name)}_average_spectrum_energy.png"
        plot_species_report(species_name, summary, output_path)

        if output_path.exists():
            generated += 1
            print(f"[ok] {species_name} -> {output_path}")
        else:
            print(f"[skip] {species_name}: no se pudo generar gráfico")

    print(f"Generados {generated} gráficos en {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
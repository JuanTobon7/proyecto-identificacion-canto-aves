"""
Script para analizar el espectro promedio de cada ave.

Genera un JSON con:
- Espectro promedio para cada especie
- Frecuencias de mínimos y máximos detectados
- Bandas dinámicamente determinadas basadas en esos extremos
- Información detallada del análisis espectral
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

from core.repo.birds_repo import BirdRepository
from core.maths.fft import FFTProcessor
from core.audio_converter import AudioConverter


def load_and_process_audio(path: Path, target_sr: int) -> Tuple[np.ndarray, int] | None:
    """Carga un archivo de audio y lo resampling si es necesario."""
    try:
        audio, sr = sf.read(str(path), always_2d=False)
        y = AudioConverter.to_mono_float32(np.asarray(audio))
        
        if y.size == 0:
            return None
        
        if sr != target_sr:
            try:
                y = AudioConverter.resample(y, sr, target_sr)
                sr = target_sr
            except Exception:
                return None
        
        return y, sr
    except Exception:
        return None


def compute_average_spectrum(
    files: list[Path],
    sample_rate: int,
    freq_range: Tuple[float, float] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula el espectro promedio de un conjunto de archivos.
    
    Retorna: (frequencies, average_magnitude, std_magnitude)
    """
    fft_processor = FFTProcessor()
    spectra = []
    
    for file_path in files:
        result = load_and_process_audio(file_path, sample_rate)
        if result is None:
            continue
        
        y, sr = result
        freqs, magnitude = fft_processor.compute_fft(y, sr)
        
        if freq_range:
            low, high = freq_range
            mask = (freqs >= low) & (freqs <= high)
            freqs = freqs[mask]
            magnitude = magnitude[mask]
        
        spectra.append(magnitude)
    
    if not spectra:
        return np.array([]), np.array([]), np.array([])
    
    # Interpolar todos los espectros al mismo tamaño (del primero)
    target_length = len(spectra[0])
    aligned_spectra = []
    
    for spec in spectra:
        if len(spec) != target_length:
            spec_interp = np.interp(
                np.linspace(0, 1, target_length),
                np.linspace(0, 1, len(spec)),
                spec
            )
            aligned_spectra.append(spec_interp)
        else:
            aligned_spectra.append(spec)
    
    aligned_spectra = np.array(aligned_spectra)
    avg_magnitude = np.mean(aligned_spectra, axis=0)
    std_magnitude = np.std(aligned_spectra, axis=0)
    
    return freqs, avg_magnitude, std_magnitude


def find_spectral_peaks_and_valleys(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
    num_bands: int = 8,
    height_threshold: float = None,
) -> Tuple[list[dict], list[dict]]:
    """
    Detecta picos (máximos) y valles (mínimos) en el espectro.
    
    Retorna: (peaks, valleys) donde cada elemento es:
        {"frequency": float, "magnitude": float, "index": int}
    """
    if height_threshold is None:
        height_threshold = np.max(magnitude) * 0.1
    
    # Detectar picos
    peaks, properties = find_peaks(magnitude, height=height_threshold, distance=5)
    peaks_list = [
        {
            "frequency": float(frequencies[i]),
            "magnitude": float(magnitude[i]),
            "index": int(i),
        }
        for i in peaks
    ]
    
    # Detectar valles (picos inversos)
    inverted_mag = -magnitude
    valleys, _ = find_peaks(inverted_mag, distance=5)
    valleys_list = [
        {
            "frequency": float(frequencies[i]),
            "magnitude": float(magnitude[i]),
            "index": int(i),
        }
        for i in valleys
    ]
    
    # Ordenar por magnitud (descendente para picos, ascendente para valles)
    peaks_list.sort(key=lambda x: x["magnitude"], reverse=True)
    valleys_list.sort(key=lambda x: x["magnitude"])
    
    return peaks_list, valleys_list


def determine_dynamic_bands(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
    peaks: list[dict],
    valleys: list[dict],
    num_bands: int = 8,
) -> list[Tuple[float, float]]:
    """
    Determina dinámicamente n bandas usando los picos y valles detectados.
    
    Estrategia:
    1. Usar valles como límites naturales de bandas
    2. Si hay pocos valles, usar los picos para definir bandas
    3. Completar con división uniforme si es necesario
    """
    freq_min = float(np.min(frequencies))
    freq_max = float(np.max(frequencies))
    
    if len(valleys) >= num_bands - 1:
        # Usar los valles más significativos como bordes
        valley_freqs = sorted([v["frequency"] for v in valleys[:num_bands - 1]])
        band_edges = [freq_min] + valley_freqs + [freq_max]
    elif len(peaks) >= num_bands:
        # Usar picos como centros de bandas aproximados
        peak_freqs = sorted([p["frequency"] for p in peaks[:num_bands]])
        # Crear bandas alrededor de los picos
        band_edges = [freq_min]
        for i, peak_freq in enumerate(peak_freqs[:-1]):
            midpoint = (peak_freq + peak_freqs[i + 1]) / 2
            band_edges.append(midpoint)
        band_edges.append(freq_max)
    else:
        # Fallback: división uniforme
        band_edges = list(np.linspace(freq_min, freq_max, num_bands + 1))
    
    # Asegurar que tenemos exactamente num_bands + 1 bordes
    if len(band_edges) > num_bands + 1:
        band_edges = [band_edges[i] for i in np.linspace(0, len(band_edges) - 1, num_bands + 1, dtype=int)]
    elif len(band_edges) < num_bands + 1:
        band_edges = list(np.linspace(freq_min, freq_max, num_bands + 1))
    
    # Crear bandas a partir de los bordes
    bands = [
        (float(band_edges[i]), float(band_edges[i + 1]))
        for i in range(num_bands)
    ]
    
    return bands


def analyze_species_spectra(
    species: str,
    files: list[Path],
    bird_info,
    sample_rate: int,
    num_bands: int = 8,
) -> dict:
    """Analiza el espectro de una especie y retorna sus parámetros dinámicos."""
    
    if not files:
        return {
            "species": species,
            "num_audios": 0,
            "error": "No audio files found",
        }
    
    # Obtener rango de frecuencias de la especie
    low = float(bird_info.vocalizaciones.frecuencias_hz.rango_principal.min)
    high = float(bird_info.vocalizaciones.frecuencias_hz.rango_principal.max)
    
    print(f"  Analizando espectro en rango [{low}, {high}] Hz...")
    
    # Calcular espectro promedio
    freqs, avg_mag, std_mag = compute_average_spectrum(
        files,
        sample_rate,
        freq_range=(low, high),
    )
    
    if freqs.size == 0:
        return {
            "species": species,
            "num_audios": len(files),
            "error": "Could not process audio files",
        }
    
    # Detectar picos y valles
    peaks, valleys = find_spectral_peaks_and_valleys(
        freqs,
        avg_mag,
        num_bands=num_bands,
        height_threshold=np.max(avg_mag) * 0.15,
    )
    
    # Determinar bandas dinámicamente
    dynamic_bands = determine_dynamic_bands(
        freqs,
        avg_mag,
        peaks,
        valleys,
        num_bands=num_bands,
    )
    
    # Estadísticas del espectro
    spectrum_stats = {
        "freq_min_hz": float(np.min(freqs)),
        "freq_max_hz": float(np.max(freqs)),
        "freq_dominante_hz": float(freqs[np.argmax(avg_mag)]),
        "magnitud_pico": float(np.max(avg_mag)),
        "magnitud_minima": float(np.min(avg_mag)),
        "magnitud_media": float(np.mean(avg_mag)),
    }
    
    return {
        "species": species,
        "num_audios": len(files),
        "frequency_range": {"min": low, "max": high},
        "spectrum_stats": spectrum_stats,
        "num_peaks": len(peaks),
        "num_valleys": len(valleys),
        "peaks": peaks[:10],  # Top 10 picos
        "valleys": valleys[:5],  # Top 5 valles
        "dynamic_bands": [
            {"low": float(b[0]), "high": float(b[1])} for b in dynamic_bands
        ],
        "spectrum_profile": {
            "frequencies": freqs.tolist(),
            "magnitude_mean": avg_mag.tolist(),
            "magnitude_std": std_mag.tolist(),
        },
    }


def main():
    p = argparse.ArgumentParser(
        description="Analiza espectro promedio de cada ave y determina bandas dinámicamente"
    )
    p.add_argument("--bands", type=int, default=8, help="Número de bandas a determinar")
    p.add_argument("--sample-rate", type=int, default=22050, help="Frecuencia de muestreo")
    p.add_argument("--output", type=str, default=None, help="Ruta de salida JSON")
    args = p.parse_args()
    
    repo = BirdRepository(env="training")
    audios_by_species = repo.get_audios_by_species()
    
    now = datetime.now(timezone.utc).isoformat()
    
    analysis_results = []
    total = len(audios_by_species)
    processed = 0
    
    for species, files in audios_by_species.items():
        processed += 1
        print(f"[{processed}/{total}] Analizando especie: {species} -> {len(files)} archivos")
        
        if len(files) == 0:
            print("  - sin archivos, se omite")
            continue
        
        bird = repo.get_by_species(species)
        if bird is None:
            print("  - información de especie no encontrada")
            continue
        
        result = analyze_species_spectra(
            species=species,
            files=files,
            bird_info=bird,
            sample_rate=int(args.sample_rate),
            num_bands=int(args.bands),
        )
        analysis_results.append(result)
    
    collection = {
        "analysis_type": "bird_spectrum_analysis",
        "num_bands": int(args.bands),
        "sample_rate": int(args.sample_rate),
        "created_at": now,
        "analysis_results": analysis_results,
    }
    
    out_path = Path(args.output) if args.output else Path("models") / "bird_spectrum_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(collection, fh, indent=2, ensure_ascii=False)
    
    print(f"\nAnálisis completado. Guardado en: {out_path}")
    print(f"Especies analizadas: {len(analysis_results)}")


if __name__ == "__main__":
    main()
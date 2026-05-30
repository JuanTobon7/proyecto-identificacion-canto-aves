from pathlib import Path
import argparse
import json
import numpy as np
import soundfile as sf

from core.models_managment import ModelsManagement
from core.repo.birds_repo import BirdRepository
from core.audio_converter import AudioConverter
from core.butterworth_controller import ButterworthController
from core.maths.filter_fir import FilterFir
from core.maths.fft import FFTProcessor
from core.maths.energy_vector import EnergyVector
from core.maths.statistics import Statistics
from config.frecuency_bands import FrequencyBands


def build_representative_signal(files: list[Path], max_files: int = 5):
    """Return concatenated signal and sample rate from up to max_files valid files."""
    audio_parts = []
    sr_ref = None
    for f in files[:max_files]:
        try:
            data, sr = sf.read(str(f), always_2d=False)
        except Exception:
            continue
        if sr_ref is None:
            sr_ref = sr
        if sr != sr_ref:
            # Mixed sample rates — skip this file
            continue
        y = AudioConverter.to_mono_float32(data)
        if y.size == 0:
            continue
        audio_parts.append(y)
    if not audio_parts:
        return None, None
    return np.concatenate(audio_parts), sr_ref


def build_energy_vectors(files: list[Path], bands: list[tuple[float, float]], max_files: int = 5):
    """Compute band energy vectors for up to max_files files. Returns list of vectors and sample rate.
    """
    vectors = []
    sr_ref = None
    for f in files[:max_files]:
        try:
            data, sr = sf.read(str(f), always_2d=False)
        except Exception:
            continue
        if sr_ref is None:
            sr_ref = sr
        if sr != sr_ref:
            continue
        y = AudioConverter.to_mono_float32(data)
        energies = FFTProcessor.compute_band_energies(y, sr, bands)
        vec = EnergyVector.compute(energies)
        vectors.append(vec)
    return vectors, sr_ref


def evaluate_band_energy(y: np.ndarray, sr: int, low: float, high: float):
    freqs, magnitude = FFTProcessor.compute_fft(y, sr)
    mask_band = (freqs >= low) & (freqs <= high)
    band_energy = float(np.sum(magnitude[mask_band] ** 2)) if np.any(mask_band) else 0.0
    total_energy = float(np.sum(magnitude ** 2)) if magnitude.size else 0.0
    return band_energy, total_energy


def train(args):
    repo = BirdRepository(env="training")
    audios_by_species = repo.get_audios_by_species()
    mm = ModelsManagement(base_dir=args.models_dir)

    butter_models = []
    fir_models = []

    # prepare frequency bands from config
    band_tuples = [(lo, hi) for lo, hi, _ in FrequencyBands().get_bands()]

    for species, files in audios_by_species.items():
        if not files:
            continue
        print(f"Training for species: {species} — files: {len(files)}")

        # compute per-file energy vectors (for mean/std)
        vectors, sr = build_energy_vectors(files, band_tuples, max_files=args.max_files)
        if not vectors or sr is None:
            print(f"  [WARN] No valid representative signal for {species}")
            continue

        # aggregate profile
        profile_vector = Statistics.mean_vector(vectors)
        std_vector = Statistics.std_vector(vectors)
        sample_count = len(vectors)

        # get annotated band from bird info
        bird = repo.get_by_species(species)
        band = bird.vocalizaciones.frecuencias_hz.rango_principal
        low = float(band.min) * (1.0 - args.margin)
        high = float(band.max) * (1.0 + args.margin)
        nyquist = sr / 2
        low = max(1.0, low)
        high = min(nyquist * 0.99, high)

        model_record = {
            "species": species,
            "sr": sr,
            "requested_band": [float(band.min), float(band.max)],
            "train_band": [low, high],
            "metrics": {},
        }

        # Baseline energies using concatenated representative signal
        y_concat, _ = build_representative_signal(files, max_files=args.max_files)
        if y_concat is None:
            pre_band_energy, pre_total = 0.0, 0.0
        else:
            pre_band_energy, pre_total = evaluate_band_energy(y_concat, sr, low, high)
        model_record["metrics"]["pre_band_energy"] = pre_band_energy
        model_record["metrics"]["pre_total_energy"] = pre_total

        # Attach profile
        model_record["profile_vector"] = profile_vector.tolist()
        model_record["std_energy_vector"] = std_vector.tolist()
        model_record["sample_count"] = sample_count

        if args.model in ("butterworth", "both"):
            try:
                ctrl = ButterworthController(order=args.butterworth_order, filter_type="band", low_freq=low, high_freq=high)
                fb = ctrl.build(signal=y_concat, sr=sr)
                params = ctrl.last_params
                # Apply filter for evaluation
                y_bw = fb.apply_bandpass(signal=y_concat, sr=sr, low_freq=params.low_freq, high_freq=params.high_freq)
                bw_band_energy, bw_total = evaluate_band_energy(y_bw, sr, low, high)
                model_record_bw = {
                    "type": "butterworth",
                    "params": params.to_dict(),
                    "pre_band_energy": pre_band_energy,
                    "post_band_energy": bw_band_energy,
                    "post_total_energy": bw_total,
                    "profile_vector": profile_vector.tolist(),
                    "std_energy_vector": std_vector.tolist(),
                    "sample_count": sample_count,
                }
                model_record_bw["species"] = species
                butter_models.append(model_record_bw)
                print(f"  Collected Butterworth params for: {species}")
            except Exception as exc:
                print(f"  [ERROR] Butterworth train failed for {species}: {exc}")

        if args.model in ("fir", "both"):
            try:
                fir = FilterFir(sr=sr, num_taps=args.num_taps, window=args.window)
                fir.design_bandpass(low_freq=low, high_freq=high)
                y_fir = fir.process_signal_fast(y_concat)
                fir_band_energy, fir_total = evaluate_band_energy(y_fir, sr, low, high)
                model_record_fir = {
                    "type": "fir",
                    "sr": sr,
                    "num_taps": args.num_taps,
                    "window": args.window,
                    "low_freq": low,
                    "high_freq": high,
                    "coeffs": (fir.coeffs.tolist() if hasattr(fir, "coeffs") and fir.coeffs is not None else []),
                    "pre_band_energy": pre_band_energy,
                    "post_band_energy": fir_band_energy,
                    "post_total_energy": fir_total,
                    "profile_vector": profile_vector.tolist(),
                    "std_energy_vector": std_vector.tolist(),
                    "sample_count": sample_count,
                }
                model_record_fir["species"] = species
                fir_models.append(model_record_fir)
                print(f"  Collected FIR params for: {species}")
            except Exception as exc:
                print(f"  [ERROR] FIR train failed for {species}: {exc}")

    # Save aggregated models (one per type)
    if args.model in ("butterworth", "both") and butter_models:
        agg = {
            "kind": "butterworth_collection",
            "sample_rate": butter_models[0].get("params", {}).get("sample_rate"),
            "models": butter_models,
        }
        mm.save("butterworth_all", agg, metadata={"type": "butterworth_collection"})
        print("Saved aggregated Butterworth model: models/butterworth_all.json")

    if args.model in ("fir", "both") and fir_models:
        agg = {
            "kind": "fir_collection",
            "models": fir_models,
        }
        mm.save("fir_all", agg, metadata={"type": "fir_collection"})
        print("Saved aggregated FIR model: models/fir_all.json")

    print("Training complete.")


def main():
    parser = argparse.ArgumentParser(description="Entrena modelos de filtro por especie (Butterworth / FIR)")
    parser.add_argument("--model", choices=["fir", "butterworth", "both"], default="both")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--max-files", type=int, default=5, help="Máx archivos por especie para construir señal representativa")
    parser.add_argument("--margin", type=float, default=0.2, help="Margen relativo alrededor del rango anotado (ej: 0.2 = ±20%%)")
    parser.add_argument("--butterworth-order", type=int, default=4)
    parser.add_argument("--num-taps", type=int, default=101)
    parser.add_argument("--window", default="hamming")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
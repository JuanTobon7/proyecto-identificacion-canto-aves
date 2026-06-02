from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np

from config.frecuency_bands import FrequencyBands
from core.audio_converter import AudioConverter
from core.butterworth_controller import ButterworthController
from core.dto.audio_stats import AudioStats
from core.maths.energy_vector import EnergyVector
from core.maths.fft import FFTProcessor
from core.maths.filter_butterworth import FilterButterworth
from core.maths.filter_fir import FilterFir


class AudioService:
    def __init__(self) -> None:
        self.converter = AudioConverter()
        self.fft = FFTProcessor()
        self.frequency_bands = FrequencyBands()

    def load_audio(self, audio_path: str | Path) -> tuple[np.ndarray, int]:
        raw_audio, sample_rate = librosa.load(str(audio_path), sr=None, mono=False)
        return self.converter.to_mono_float32(np.asarray(raw_audio)), int(sample_rate)

    def resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr or audio.size == 0:
            return audio.astype(np.float32, copy=False)
        return librosa.resample(audio.astype(np.float32, copy=False), orig_sr=orig_sr, target_sr=target_sr)

    def spectrum(self, audio: np.ndarray, sr: int, fft_points: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        freqs, magnitude = self.fft.compute_fft(audio, sr)
        if fft_points is None or fft_points <= 0 or freqs.size == 0 or magnitude.size == 0:
            return freqs, magnitude

        freq_grid = np.linspace(0.0, sr / 2.0, int(fft_points))
        interpolated = np.interp(freq_grid, freqs, magnitude, left=0.0, right=0.0)
        return freq_grid, interpolated

    def build_model_subbands(self, model: dict[str, Any]) -> tuple[list[tuple[float, float]], list[str]]:
        low_freq, high_freq = self._model_band(model)
        profile_vector = np.asarray(model.get("profile_vector", []), dtype=np.float32)
        band_count = max(1, int(profile_vector.size))
        edges = np.linspace(low_freq, high_freq, band_count + 1)
        bands = [(float(edges[index]), float(edges[index + 1])) for index in range(band_count)]
        labels = [f"{edges[index]:.0f}-{edges[index + 1]:.0f}" for index in range(band_count)]
        return bands, labels

    def energy_vector(self, audio: np.ndarray, sr: int, model: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
        bands, labels = self.build_model_subbands(model)
        band_energies = self.fft.compute_band_energies(audio, sr, bands)
        return EnergyVector.compute(band_energies), labels

    def compute_stats(
        self,
        audio: np.ndarray,
        sr: int,
        band_energies: np.ndarray | None = None,
        band_labels: list[str] | None = None,
    ) -> AudioStats:
        stats = AudioStats(sample_rate=sr, n_samples=int(audio.size), duration_sec=float(audio.size / sr if sr else 0.0))
        if audio.size == 0 or sr <= 0:
            return stats

        stats.channels = 1
        stats.amp_mean = float(np.mean(audio))
        stats.amp_std = float(np.std(audio))
        stats.amp_max = float(np.max(audio))
        stats.amp_min = float(np.min(audio))
        stats.amp_rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = max(abs(stats.amp_max), abs(stats.amp_min))
        if stats.amp_rms > 0:
            stats.amp_peak_to_rms_db = float(20 * np.log10(peak / stats.amp_rms))
        stats.energy = float(np.sum(audio ** 2))
        stats.dc_offset = stats.amp_mean
        stats.clipping_samples = int(np.sum(np.abs(audio) >= 0.9999))
        silence_mask = np.abs(audio) < 0.01
        stats.silence_ratio = float(np.mean(silence_mask))
        stats.max_silence_run_sec = float(self._max_run(silence_mask) / sr)

        freqs, magnitude = self.fft.compute_fft(audio, sr)
        if len(freqs) > 0:
            total_mag = float(np.sum(magnitude))
            if total_mag > 0:
                stats.spectral_centroid = float(np.sum(freqs * magnitude) / total_mag)
                stats.spectral_bandwidth = float(np.sqrt(np.sum(((freqs - stats.spectral_centroid) ** 2) * magnitude) / total_mag))
            stats.dominant_freq_hz = float(freqs[np.argmax(magnitude)])

        if band_energies is not None and band_labels is not None:
            stats.band_energies = {label: float(value) for label, value in zip(band_labels, band_energies)}

        return stats

    def filter_audio(
        self,
        audio: np.ndarray,
        sr: int,
        model: dict[str, Any],
        model_kind: str,
        order_override: int | None = None,
        low_freq_override: float | None = None,
        high_freq_override: float | None = None,
    ) -> np.ndarray:
        if model_kind == "fir_collection" or model.get("type") == "fir":
            fir = FilterFir(sr=sr, num_taps=max(1, len(model.get("coeffs", []))))
            fir.coeffs = np.asarray(model.get("coeffs", []), dtype=np.float64)
            return fir.process_signal_fast(audio)

        params = model.get("params", {})
        butterworth = FilterButterworth(order=int(order_override if order_override is not None else params.get("order", 4)))
        low_freq = float(low_freq_override if low_freq_override is not None else params.get("low_freq", 0.0))
        high_freq = float(high_freq_override if high_freq_override is not None else params.get("high_freq", 0.0))
        return butterworth.apply_bandpass(
            signal=audio,
            sr=sr,
            low_freq=low_freq,
            high_freq=high_freq,
        )

    def preview_butterworth_params(
        self,
        audio: np.ndarray,
        sr: int,
        model: dict[str, Any],
        model_kind: str,
        order_override: int | None = None,
        low_freq_override: float | None = None,
        high_freq_override: float | None = None,
    ) -> dict[str, Any]:
        if audio.size == 0 or sr <= 0:
            return {}
        if model_kind != "butterworth_collection" and model.get("type") != "butterworth":
            return dict(model.get("params", {}))

        params = model.get("params", {})
        controller = ButterworthController(
            order=int(order_override if order_override is not None else params.get("order", 4)),
            filter_type="band",
            low_freq=low_freq_override if low_freq_override is not None else float(params.get("low_freq", 0.0)),
            high_freq=high_freq_override if high_freq_override is not None else float(params.get("high_freq", 0.0)),
        )
        controller.build(signal=audio, sr=sr)
        if controller.last_params is None:
            return dict(params)

        detected = controller.last_params.to_dict()
        detected["source"] = "audio"
        return detected

    @staticmethod
    def _model_band(model: dict[str, Any]) -> tuple[float, float]:
        if model.get("type") == "fir":
            return float(model["low_freq"]), float(model["high_freq"])
        params = model.get("params", {})
        return float(params["low_freq"]), float(params["high_freq"])

    @staticmethod
    def _max_run(mask: np.ndarray) -> int:
        max_run = 0
        current = 0
        for value in mask:
            if value:
                current += 1
                max_run = max(max_run, current)
            else:
                current = 0
        return max_run

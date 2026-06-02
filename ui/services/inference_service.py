from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from core.audio_converter import AudioConverter
from core.models_managment import ModelsManagement
from core.predict import Predict
from core.repo.birds_repo import BirdRepository
from config.app_settings import AppSettings
from ui.dtos.prediction_result import PredictionResult
from ui.services.audio_service import AudioService


class InferenceService:
    def __init__(self, model_dir: str | Path = "models") -> None:
        self.model_dir = str(model_dir)
        self.models_mgmt = ModelsManagement(base_dir=model_dir)
        self.audio_service = AudioService()
        self.bird_repo = BirdRepository(env="training")

    def available_models(self) -> list[str]:
        models: list[str] = []
        for entry in self.models_mgmt.list_models():
            model_name = str(entry.get("name", ""))
            if not model_name:
                continue
            try:
                model = self.models_mgmt.get_json(model_name)
            except Exception:
                continue
            if str(model.get("kind", "")).endswith("collection") or isinstance(model.get("models"), list):
                models.append(model_name)
        return sorted(models)

    def process(
        self,
        model_name: str,
        audio_path: str | Path,
        butterworth_order: int | None = None,
        butterworth_low_freq: float | None = None,
        butterworth_high_freq: float | None = None,
        fft_points: int | None = None,
    ) -> PredictionResult:
        collection = self.models_mgmt.get_json(model_name)
        model_kind = str(collection.get("kind", ""))
        model_sr = self._resolve_sample_rate(collection)

        audio, original_sr = self.audio_service.load_audio(audio_path)
        return self.process_audio(
            model_name,
            audio_path,
            audio,
            original_sr,
            collection,
            model_kind,
            model_sr,
            butterworth_order=butterworth_order,
            butterworth_low_freq=butterworth_low_freq,
            butterworth_high_freq=butterworth_high_freq,
            fft_points=fft_points,
        )

    def process_audio(
        self,
        model_name: str,
        audio_path: str | Path,
        audio: np.ndarray,
        original_sr: int,
        collection: dict[str, Any] | None = None,
        model_kind: str | None = None,
        model_sr: int | None = None,
        butterworth_order: int | None = None,
        butterworth_low_freq: float | None = None,
        butterworth_high_freq: float | None = None,
        fft_points: int | None = None,
    ) -> PredictionResult:
        if collection is None:
            collection = self.models_mgmt.get_json(model_name)
        if model_kind is None:
            model_kind = str(collection.get("kind", ""))
        if model_sr is None:
            model_sr = self._resolve_sample_rate(collection)

        if model_sr > 0 and original_sr != model_sr:
            audio = self.audio_service.resample_audio(audio, original_sr, model_sr)
            sample_rate = model_sr
        else:
            sample_rate = original_sr

        predictor = Predict(model_name=model_name, model_dir=self.model_dir)
        ranking = predictor.predict_proba(audio, sample_rate)
        allowed_species = set(self.bird_repo.species_names())
        ranking = [item for item in ranking if item.get("species") in allowed_species]
        if not ranking:
            raise ValueError("El modelo cargado no contiene especies presentes en general_info_aves.json.")

        confidence = self.confidence_margin(ranking)
        top_species = ranking[0]["species"]
        selected_model = self._select_model(collection, top_species)
        rejection_threshold = self._rejection_threshold(collection, selected_model)
        rejected = confidence < rejection_threshold
        predicted_species = "Rechazado" if rejected else top_species

        effective_low_freq = butterworth_low_freq if butterworth_low_freq is not None else self._model_band(selected_model)[0]
        effective_high_freq = butterworth_high_freq if butterworth_high_freq is not None else self._model_band(selected_model)[1]

        filtered = self.audio_service.filter_audio(
            audio,
            sample_rate,
            selected_model,
            model_kind,
            order_override=butterworth_order,
            low_freq_override=effective_low_freq,
            high_freq_override=effective_high_freq,
        )
        original_freqs, original_magnitude = self.audio_service.spectrum(audio, sample_rate, fft_points=fft_points)
        filtered_freqs, filtered_magnitude = self.audio_service.spectrum(filtered, sample_rate, fft_points=fft_points)
        original_vector, band_labels = self.audio_service.energy_vector(audio, sample_rate, selected_model)
        filtered_vector, _ = self.audio_service.energy_vector(filtered, sample_rate, selected_model)

        original_energies = self.audio_service.fft.compute_band_energies(audio, sample_rate, self.audio_service.build_model_subbands(selected_model)[0])
        filtered_energies = self.audio_service.fft.compute_band_energies(filtered, sample_rate, self.audio_service.build_model_subbands(selected_model)[0])

        bird_info = self.bird_repo.get_by_species(top_species)
        original_stats = self.audio_service.compute_stats(audio, sample_rate, original_energies, band_labels)
        filtered_stats = self.audio_service.compute_stats(filtered, sample_rate, filtered_energies, band_labels)
        butterworth_params = self.audio_service.preview_butterworth_params(
            audio,
            sample_rate,
            selected_model,
            model_kind,
            order_override=butterworth_order,
            low_freq_override=effective_low_freq,
            high_freq_override=effective_high_freq,
        )
        if not butterworth_params:
            butterworth_params = self._model_parameters(selected_model)
        if butterworth_order is not None:
            butterworth_params["order"] = int(butterworth_order)
        butterworth_params.setdefault("rejection_threshold", rejection_threshold)
        butterworth_params.setdefault("fft_points", fft_points if fft_points is not None else 0)

        # Datos para visualización de reconocimiento
        profile_vector = np.array(selected_model.get("profile_vector", []), dtype=np.float32)
        std_energy_vector = np.array(selected_model.get("std_energy_vector", []), dtype=np.float32)
        model_params = selected_model.get("params", {})
        low_freq = float(model_params.get("low_freq", effective_low_freq))
        high_freq = float(model_params.get("high_freq", effective_high_freq))
        n_bands = len(profile_vector) if len(profile_vector) > 0 else 8
        
        from core.maths.fft import FFTProcessor
        subband_frequencies = FFTProcessor.build_subbands(low_freq, high_freq, n_bands)

        return PredictionResult(
            model_name=model_name,
            model_kind=model_kind,
            audio_path=str(audio_path),
            sample_rate=sample_rate,
            predicted_species=predicted_species,
            confidence=confidence,
            rejection_threshold=rejection_threshold,
            rejected=rejected,
            ranking=ranking,
            bird_info=bird_info,
            original_signal=audio,
            filtered_signal=filtered,
            original_freqs=original_freqs,
            original_magnitude=original_magnitude,
            filtered_freqs=filtered_freqs,
            filtered_magnitude=filtered_magnitude,
            original_energy_vector=original_vector,
            filtered_energy_vector=filtered_vector,
            band_labels=band_labels,
            butterworth_params=butterworth_params,
            original_stats=original_stats,
            filtered_stats=filtered_stats,
            profile_vector=profile_vector,
            std_energy_vector=std_energy_vector,
            subband_frequencies=subband_frequencies,
            original_band_energies=original_energies,
            filtered_band_energies=filtered_energies,
        )

    def _select_model(self, collection: dict[str, Any], species: str) -> dict[str, Any]:
        for model in collection.get("models", []):
            if model.get("species") == species:
                return model
        return collection.get("models", [{}])[0]

    @staticmethod
    def _resolve_sample_rate(collection: dict[str, Any]) -> int:
        sample_rate = collection.get("sample_rate")
        if sample_rate:
            return int(sample_rate)
        for model in collection.get("models", []):
            params = model.get("params") or {}
            if params.get("sample_rate"):
                return int(params["sample_rate"])
            if model.get("sr"):
                return int(model["sr"])
        return 0

    @staticmethod
    def confidence_margin(ranking):

        if len(ranking) < 2:
            return 1.0

        best = ranking[0]["score"]
        second = ranking[1]["score"]

        return float(np.clip((best - second) / 0.2, 0, 1))


    @staticmethod
    def _softmax_confidence(ranking: list[dict[str, Any]]) -> float:
        if not ranking:
            return 0.0
        scores = np.array([float(item.get("score", 0.0)) for item in ranking], dtype=np.float64)
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores)
        denominator = float(np.sum(exp_scores))
        if denominator <= 0:
            return 0.0
        return float(exp_scores[0] / denominator)

    def _rejection_threshold(self, collection: dict[str, Any], model: dict[str, Any]) -> float:
        for source in (model, model.get("params", {}), collection):
            if not isinstance(source, dict):
                continue
            candidate = source.get("rejection_threshold")
            if candidate is None:
                candidate = source.get("confidence_threshold")
            if candidate is None:
                continue
            try:
                threshold = float(candidate)
            except (TypeError, ValueError):
                continue
            if threshold > 0:
                return threshold
        return float(AppSettings.CONFIDENCE_REJECTION_THRESHOLD)

    @staticmethod
    def _model_parameters(model: dict[str, Any]) -> dict[str, Any]:
        if model.get("type") == "fir":
            return {
                "type": "fir",
                "low_freq": model.get("low_freq"),
                "high_freq": model.get("high_freq"),
                "num_taps": len(model.get("coeffs", [])),
                "window": model.get("window"),
            }
        return dict(model.get("params", {}))

    @staticmethod
    def _model_band(model: dict[str, Any]) -> tuple[float, float]:
        params = model.get("params", {})
        return float(params.get("low_freq", 0.0)), float(params.get("high_freq", 0.0))

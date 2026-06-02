from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from ui.dtos.prediction_result import PredictionResult
from ui.services.inference_service import InferenceService
from ui.services.audio_runtime_service import AudioRuntimeService


class PredictionController(QObject):
    prediction_started = Signal()
    prediction_finished = Signal(object)
    prediction_failed = Signal(str)
    playback_started = Signal()
    playback_position_changed = Signal(float)
    playback_finished = Signal()
    recording_started = Signal()
    recording_finished = Signal(object)

    def __init__(self, inference_service: InferenceService | None = None) -> None:
        super().__init__()
        self.inference_service = inference_service or InferenceService()
        self.audio_runtime = AudioRuntimeService()

    def available_models(self) -> list[str]:
        return self.inference_service.available_models()

    def run_prediction(
        self,
        model_name: str,
        audio_path: str | Path,
        butterworth_order: int | None = None,
        butterworth_low_freq: float | None = None,
        butterworth_high_freq: float | None = None,
        fft_points: int | None = None,
    ) -> PredictionResult | None:
        self.prediction_started.emit()
        try:
            result = self.inference_service.process(
                model_name=model_name,
                audio_path=audio_path,
                butterworth_order=butterworth_order,
                butterworth_low_freq=butterworth_low_freq,
                butterworth_high_freq=butterworth_high_freq,
                fft_points=fft_points,
            )
        except Exception as exc:
            self.prediction_failed.emit(str(exc))
            return None

        self.prediction_finished.emit(result)
        return result

    def play_audio(self, audio_path: str | Path) -> None:
        def _worker() -> None:
            try:
                self.playback_started.emit()
                self.audio_runtime.play_file(
                    audio_path,
                    on_position=lambda position: self.playback_position_changed.emit(position),
                )
                self.playback_finished.emit()
            except Exception as exc:
                self.prediction_failed.emit(str(exc))

        threading.Thread(target=_worker, daemon=True).start()

    def play_signal(self, audio, sample_rate: int) -> None:
        def _worker() -> None:
            try:
                self.playback_started.emit()
                self.audio_runtime.play_signal(
                    audio,
                    sample_rate,
                    on_position=lambda position: self.playback_position_changed.emit(position),
                )
                self.playback_finished.emit()
            except Exception as exc:
                self.prediction_failed.emit(str(exc))

        threading.Thread(target=_worker, daemon=True).start()

    def record_and_predict(
        self,
        model_name: str,
        seconds: float = 3.0,
        butterworth_order: int | None = None,
        butterworth_low_freq: float | None = None,
        butterworth_high_freq: float | None = None,
        fft_points: int | None = None,
    ) -> None:
        def _worker() -> None:
            try:
                self.recording_started.emit()
                audio, sample_rate, recorded_path = self.audio_runtime.record_seconds(seconds=seconds)
                result = self.inference_service.process_audio(
                    model_name=model_name,
                    audio_path=recorded_path,
                    audio=audio,
                    original_sr=sample_rate,
                    butterworth_order=butterworth_order,
                    butterworth_low_freq=butterworth_low_freq,
                    butterworth_high_freq=butterworth_high_freq,
                    fft_points=fft_points,
                )
                self.recording_finished.emit(result)
            except Exception as exc:
                self.prediction_failed.emit(str(exc))

        threading.Thread(target=_worker, daemon=True).start()

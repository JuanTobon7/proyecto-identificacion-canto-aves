from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import numpy as np
import sounddevice as sd

from core.audio_converter import AudioConverter
from ui.services.audio_service import AudioService


class AudioRuntimeService:
    def __init__(self) -> None:
        self.audio_service = AudioService()

    def play_signal(
        self,
        audio: np.ndarray,
        sample_rate: int,
        on_position: Callable[[float], None] | None = None,
    ) -> None:
        if audio.size == 0 or sample_rate <= 0:
            raise ValueError("No se pudo reproducir el audio seleccionado.")

        samples = AudioConverter.to_mono_float32(np.asarray(audio)).astype(np.float32, copy=False).reshape(-1)
        total_samples = int(samples.size)
        if total_samples == 0:
            raise ValueError("No se pudo reproducir el audio seleccionado.")

        playback_state = {"index": 0, "last_emit": -1.0}
        blocksize = min(4096, max(512, sample_rate // 10))

        def _callback(outdata, frames, time, status) -> None:  # noqa: ARG001
            start = playback_state["index"]
            end = min(start + frames, total_samples)
            chunk = samples[start:end]
            outdata.fill(0)
            if chunk.size > 0:
                outdata[: chunk.size, 0] = chunk

            playback_state["index"] = end
            current_position = float(end / sample_rate)
            if on_position is not None and (
                current_position - playback_state["last_emit"] >= 0.05 or end >= total_samples
            ):
                playback_state["last_emit"] = current_position
                on_position(current_position)

            if end >= total_samples:
                raise sd.CallbackStop()

        duration_sec = total_samples / sample_rate
        with sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            callback=_callback,
            blocksize=blocksize,
            latency="low",
        ):
            if on_position is not None:
                on_position(0.0)
            sd.sleep(int(math.ceil(duration_sec * 1000.0)) + 250)

    def play_file(
        self,
        audio_path: str | Path,
        on_position: Callable[[float], None] | None = None,
    ) -> None:
        audio, sample_rate = self.audio_service.load_audio(audio_path)
        self.play_signal(audio, sample_rate, on_position=on_position)

    def record_seconds(self, seconds: float = 3.0, sample_rate: int = 48000) -> tuple[np.ndarray, int]:
        if seconds <= 0:
            raise ValueError("La duracion de grabacion debe ser mayor que cero.")
        duration = float(seconds)
        frames = int(duration * sample_rate)
        recording = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
        return recording.reshape(-1).astype(np.float32, copy=False), sample_rate

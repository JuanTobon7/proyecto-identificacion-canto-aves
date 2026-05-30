from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.dto.audio_stats import AudioStats
from core.dto.cards_bird import BirdInfo


@dataclass
class PredictionResult:
    model_name: str
    model_kind: str
    audio_path: str
    sample_rate: int
    predicted_species: str
    confidence: float
    ranking: list[dict[str, Any]] = field(default_factory=list)
    bird_info: BirdInfo | None = None
    original_signal: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    filtered_signal: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    original_freqs: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    original_magnitude: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    filtered_freqs: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    filtered_magnitude: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    original_energy_vector: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    filtered_energy_vector: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    band_labels: list[str] = field(default_factory=list)
    butterworth_params: dict[str, Any] = field(default_factory=dict)
    original_stats: AudioStats = field(default_factory=AudioStats)
    filtered_stats: AudioStats = field(default_factory=AudioStats)

"""Funciones FFT y filtros reutilizables para el proyecto.

Incluye:
- compute_fft(y, sr): devuelve frecuencias y magnitud
- design_butterworth_bandpass / apply_butterworth_filter: diseño y aplicación de IIR SOS
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt


def compute_fft(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula FFT (magnitud) del audio mono o estéreo (se convierte a mono).

    Retorna (freqs, magnitude)
    """
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    n = len(y)
    if n == 0:
        return np.array([]), np.array([])
    fft_vals = np.fft.rfft(y)
    magnitude = np.abs(fft_vals) / n
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    return freqs, magnitude


def design_butterworth_bandpass_sr(sample_rate: int, low_hz: float, high_hz: float, order: int = 6):
    """Diseña un filtro Butterworth en salida SOS con normalización por Nyquist."""
    nyquist = 0.5 * sample_rate
    low = max(20.0, low_hz) / nyquist
    high = min(high_hz, nyquist * 0.98) / nyquist
    if not 0 < low < high < 1:
        raise ValueError(f"Banda inválida: {low_hz}-{high_hz} Hz para sr={sample_rate}")
    return butter(order, [low, high], btype="bandpass", output="sos")


def apply_butterworth_filter(audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float, order: int = 6) -> np.ndarray:
    """Aplica filtro Butterworth (zero-phase) y devuelve array float32 normalizado."""
    sos = design_butterworth_bandpass_sr(sample_rate, low_hz, high_hz, order=order)
    # Asegurar mono
    if audio.ndim > 1:
        audio_mono = np.mean(audio, axis=1)
    else:
        audio_mono = audio
    filtered = sosfiltfilt(sos, audio_mono)
    peak = np.max(np.abs(filtered)) if filtered.size else 0.0
    if peak > 0:
        filtered = 0.98 * filtered / peak
    return filtered.astype(np.float32, copy=False)

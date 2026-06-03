from core.models_managment import ModelsManagement
from core.maths.fft import FFTProcessor
from core.maths.statistics import Statistics
from core.maths.energy_vector import EnergyVector
from core.maths.filter_butterworth import FilterButterworth
from core.maths.dynamic_bands_detector import DynamicBandsDetector
from core.audio_converter import AudioConverter

import numpy as np


class Predict:
    """
    Clasificador de especies usando perfiles espectrales Butterworth.

    Para cada modelo en la colección:
      1. Aplica el filtro pasa-banda de sus params al audio de entrada.
            2. Construye sub-bandas usando dynamic_bands si el modelo las trae,
                 o sub-bandas uniformes como fallback.
      3. Calcula la firma espectral (EnergyVector) sobre esas sub-bandas.
      4. Calcula distancia L1 (suma de diferencias absolutas) con el perfil del modelo.
    La especie con menor distancia es la predicción (según guía: min{EXC, EXD}).
    """

    def __init__(self, model_name: str, model_dir: str):
        self.models_mgmt = ModelsManagement(base_dir=model_dir)
        self.fft_processor = FFTProcessor()
        self.butterworth   = FilterButterworth(order=4)

        collection          = self.models_mgmt.get_json(model_name)
        self._sample_rate   = collection["sample_rate"]
        self._models        = collection["models"]

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def predict(self, audio: np.ndarray, sr: int) -> str:
        """Retorna el nombre de la especie con menor distancia (mínima diferencia absoluta)."""
        audio = self._prepare(audio, sr)

        best_species = ""
        best_distance = np.inf

        for model in self._models:
            distance = self._compute_l1_distance(audio, model)
            if distance < best_distance:
                best_distance = distance
                best_species = model["species"]

        return best_species

    def predict_proba(self, audio: np.ndarray, sr: int) -> list[dict]:
        """Retorna todos los scores ordenados de mayor a menor.
        Score se calcula como: score = 1 / (1 + distancia_L1)
        Mayor score = menor distancia = mejor match.
        """
        audio = self._prepare(audio, sr)

        results = []
        for m in self._models:
            l1_distance = self._compute_l1_distance(audio, m)
            # Convertir distancia a score: 0 distancia → score=1, distancia alta → score≈0
            score = 1.0 / (1.0 + l1_distance)
            results.append({
                "species": m["species"],
                "score": float(score),
                "distance": float(l1_distance)
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Orquestación interna — sin matemática directa
    # ------------------------------------------------------------------

    def _prepare(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Mono float32 + remuestreo si hace falta."""
        audio = AudioConverter.to_mono_float32(audio)
        if sr != self._sample_rate:
            audio = AudioConverter.resample(audio, sr, self._sample_rate)
        return audio

    def _compute_l1_distance(self, audio: np.ndarray, model: dict) -> float:
        """
        Calcula la distancia L1 (suma de diferencias absolutas) entre
        el vector de energía del audio y el perfil del modelo.

        Retorna la distancia L1 real (a menor distancia, mejor match).
        """
        params   = model["params"]
        low_freq = params["low_freq"]
        high_freq = params["high_freq"]
        profile  = np.array(model["profile_vector"], dtype=np.float64)
        n_bands  = len(profile)

        try:
            filtered = self.butterworth.apply_bandpass(audio, self._sample_rate, low_freq, high_freq)
        except ValueError:
            return np.inf

        # Detectar subbandas dinámicas del audio analizado
        audio_bands = DynamicBandsDetector.detect_bands_from_audio(
            filtered, self._sample_rate, low_freq, high_freq, n_bands
        )

        # Calcular energías en las subbandas dinámicas del audio
        energies = FFTProcessor.compute_band_energies(filtered, self._sample_rate, audio_bands)
        energy_vec = EnergyVector.compute(energies).astype(np.float64)

        # Si el número de bandas detectadas no coincide con el perfil, interpolar
        if len(energy_vec) != len(profile):
            energy_vec = self._interpolate_energy_vector(energy_vec, len(profile))

        # Calcular distancia L1 (suma de diferencias absolutas)
        l1_distance = Statistics.absolute_difference(energy_vec, profile)

        return l1_distance

    def _interpolate_energy_vector(self, vec: np.ndarray, target_len: int) -> np.ndarray:
        """
        Interpola un vector de energía a la longitud objetivo.
        """
        if len(vec) == target_len:
            return vec
        x_old = np.linspace(0, 1, len(vec))
        x_new = np.linspace(0, 1, target_len)
        return np.interp(x_new, x_old, vec)

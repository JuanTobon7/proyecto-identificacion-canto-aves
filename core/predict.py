from core.models_managment import ModelsManagement
from core.maths.fft import FFTProcessor
from core.maths.statistics import Statistics
from core.maths.energy_vector import EnergyVector
from core.maths.filter_butterworth import FilterButterworth
from core.audio_converter import AudioConverter

import numpy as np


class Predict:
    """
    Clasificador de especies usando perfiles espectrales Butterworth.

    Para cada modelo en la colección:
      1. Aplica el filtro pasa-banda de sus params al audio de entrada.
      2. Construye sub-bandas uniformes iguales al largo de profile_vector.
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
            distance = self._compute_distance(audio, model)
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
            distance = self._compute_distance(audio, m)
            # Convertir distancia a score: 0 distancia → score=1, infinita distancia → score≈0
            score = 1.0 / (1.0 + distance)
            results.append({"species": m["species"], "score": float(score)})
        
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

    def _compute_distance(self, audio: np.ndarray, model: dict) -> float:
        """
        Calcula la distancia L1 (suma de diferencias absolutas) entre el vector de energía
        del audio y el perfil del modelo.
        
        Según la guía:
          EX = [EX1 EX2 ... EXn]  (vector de energía del audio)
          EC = [EC1 EC2 ... ECn]  (perfil de la clase)
          EXC = Σ|EXi - ECi|  (distancia a minimizar)
        
        Retorna la distancia L1 (a menor distancia, mejor match).
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

        sub_bands   = FFTProcessor.build_subbands(low_freq, high_freq, n_bands)
        energies    = FFTProcessor.compute_band_energies(filtered, self._sample_rate, sub_bands)
        energy_vec  = EnergyVector.compute(energies).astype(np.float64)

        # Calcular distancia L1: Σ|EXi - ECi|
        distance = Statistics.absolute_difference(energy_vec, profile)

        return distance
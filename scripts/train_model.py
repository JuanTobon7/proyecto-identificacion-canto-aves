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
      4. Compara con el perfil del modelo vía similitud coseno + z-score.
    La especie con mayor score combinado es la predicción.
    """

    _W_COSINE = 0.6
    _W_ZSCORE  = 0.4

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
        """Retorna el nombre de la especie con mayor puntaje."""
        audio = self._prepare(audio, sr)

        best_species = ""
        best_score   = -np.inf

        for model in self._models:
            score = self._score_model(audio, model)
            if score > best_score:
                best_score   = score
                best_species = model["species"]

        return best_species

    def predict_proba(self, audio: np.ndarray, sr: int) -> list[dict]:
        """Retorna todos los scores ordenados de mayor a menor."""
        audio = self._prepare(audio, sr)

        results = [
            {"species": m["species"], "score": float(self._score_model(audio, m))}
            for m in self._models
        ]
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

    def _score_model(self, audio: np.ndarray, model: dict) -> float:
        """Orquesta el cálculo de score para un modelo; la matemática vive en sus clases."""
        params   = model["params"]
        low_freq = params["low_freq"]
        high_freq = params["high_freq"]
        profile  = np.array(model["profile_vector"],    dtype=np.float64)
        std_vec  = np.array(model["std_energy_vector"], dtype=np.float64)
        n_bands  = len(profile)

        try:
            filtered = self.butterworth.apply_bandpass(audio, self._sample_rate, low_freq, high_freq)
        except ValueError:
            return -np.inf

        sub_bands   = FFTProcessor.build_subbands(low_freq, high_freq, n_bands)
        energies    = FFTProcessor.compute_band_energies(filtered, self._sample_rate, sub_bands)
        energy_vec  = EnergyVector.compute(energies).astype(np.float64)

        cosine_sim  = Statistics.cosine_similarity(energy_vec, profile)
        zscore_sim  = Statistics.zscore_similarity(energy_vec, profile, std_vec)

        return self._W_COSINE * cosine_sim + self._W_ZSCORE * zscore_sim
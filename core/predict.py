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

    Estrategia
    ----------
    Para cada modelo almacenado:
      1. Aplica el filtro pasa-banda definido en sus params.
      2. Divide la banda en N sub-bandas uniformes (igual al largo de profile_vector).
      3. Calcula la energía normalizada por sub-banda (EnergyVector).
      4. Compara el vector resultante con el profile_vector del modelo usando
         similitud coseno y distancia z-score ponderada.
    El modelo con mayor score combinado gana.
    """

    # Peso relativo de cada métrica (deben sumar 1.0)
    _W_COSINE = 0.6
    _W_ZSCORE = 0.4

    def __init__(self, model_name: str, model_dir: str):
        self.model_dir = model_dir
        self.models_mgmt = ModelsManagement(base_dir=model_dir)
        self.fft_processor = FFTProcessor()
        self.butterworth = FilterButterworth(order=4)
        self.audio_converter = AudioConverter()

        # Carga la colección de modelos desde disco
        collection = self.models_mgmt.get_json(model_name)
        self._sample_rate: int = collection["sample_rate"]
        self._models: list[dict] = collection["models"]

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def predict(self, audio: np.ndarray, sr: int) -> str:
        """
        Clasifica el audio y retorna el nombre de la especie predicha.

        Parámetros
        ----------
        audio : señal de audio como np.ndarray (mono o estéreo).
        sr    : tasa de muestreo del audio entrante.

        Retorna
        -------
        Nombre de la especie con mayor puntaje.
        """
        # 1. Normalizar audio a mono float32
        audio = AudioConverter.to_mono_float32(audio)

        # 2. Remuestrear si la tasa de muestreo no coincide con la del modelo
        if sr != self._sample_rate:
            audio = self._resample(audio, sr, self._sample_rate)
            sr = self._sample_rate

        # 3. Calcular score para cada modelo y elegir el mejor
        best_species: str = ""
        best_score: float = -np.inf

        scores: list[tuple[str, float]] = []

        for model in self._models:
            score = self._score_model(audio, sr, model)
            scores.append((model["species"], score))
            if score > best_score:
                best_score = score
                best_species = model["species"]

        return best_species

    def predict_proba(self, audio: np.ndarray, sr: int) -> list[dict]:
        """
        Igual que predict() pero retorna todos los scores ordenados de mayor a menor.

        Retorna
        -------
        Lista de dicts {"species": str, "score": float} ordenada descendentemente.
        """
        audio = AudioConverter.to_mono_float32(audio)
        if sr != self._sample_rate:
            audio = self._resample(audio, sr, self._sample_rate)
            sr = self._sample_rate

        results = []
        for model in self._models:
            score = self._score_model(audio, sr, model)
            results.append({"species": model["species"], "score": float(score)})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Lógica interna
    # ------------------------------------------------------------------

    def _score_model(self, audio: np.ndarray, sr: int, model: dict) -> float:
        """
        Calcula un puntaje combinado (coseno + z-score) entre el audio
        y el perfil espectral del modelo.
        """
        params = model["params"]
        low_freq: float = params["low_freq"]
        high_freq: float = params["high_freq"]
        profile: np.ndarray = np.array(model["profile_vector"], dtype=np.float64)
        std_vec: np.ndarray = np.array(model["std_energy_vector"], dtype=np.float64)
        n_bands: int = len(profile)

        # 1. Filtrar señal con el pasa-banda del modelo
        try:
            filtered = self.butterworth.apply_bandpass(audio, sr, low_freq, high_freq)
        except ValueError:
            # Si la señal es muy corta o los parámetros inválidos, score mínimo
            return -np.inf

        # 2. Dividir la banda en n_bands sub-bandas uniformes
        sub_bands = self._build_subbands(low_freq, high_freq, n_bands)

        # 3. Calcular energía por sub-banda sobre la señal filtrada
        band_energies = self.fft_processor.compute_band_energies(filtered, sr, sub_bands)

        # 4. Normalizar → firma espectral (EnergyVector)
        energy_vec = EnergyVector.compute(band_energies).astype(np.float64)

        # 5. Similitud coseno
        cosine_sim = self._cosine_similarity(energy_vec, profile)

        # 6. Distancia z-score ponderada (convertida a similitud)
        zscore_sim = self._zscore_similarity(energy_vec, profile, std_vec)

        # 7. Score combinado
        score = self._W_COSINE * cosine_sim + self._W_ZSCORE * zscore_sim
        return score

    @staticmethod
    def _build_subbands(low: float, high: float, n: int) -> list[tuple[float, float]]:
        """
        Divide el rango [low, high] en n sub-bandas de igual ancho.
        """
        edges = np.linspace(low, high, n + 1)
        return [(edges[i], edges[i + 1]) for i in range(n)]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Similitud coseno entre dos vectores. Retorna valor en [-1, 1]."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def _zscore_similarity(
        observed: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> float:
        """
        Convierte la distancia z-score promedio en una similitud en [0, 1].

        Cuanto más parecido al perfil, más cercano a 1.
        """
        # Evitar división por cero en std
        safe_std = np.where(std > 1e-9, std, 1e-9)
        z_scores = np.abs(observed - mean) / safe_std
        mean_z = float(np.mean(z_scores))
        # Transformación: similitud = 1 / (1 + z_promedio)
        return 1.0 / (1.0 + mean_z)

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Remuestreo simple por interpolación lineal.
        Para producción se recomienda usar librosa.resample o scipy.signal.resample.
        """
        if orig_sr == target_sr:
            return audio
        duration = len(audio) / orig_sr
        target_len = int(duration * target_sr)
        original_indices = np.linspace(0, len(audio) - 1, target_len)
        return np.interp(original_indices, np.arange(len(audio)), audio).astype(np.float32)
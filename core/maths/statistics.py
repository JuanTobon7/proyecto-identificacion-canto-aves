import numpy as np


class Statistics:

    @staticmethod
    def mean_vector(vectors: list[np.ndarray]) -> np.ndarray:
        return np.mean(vectors, axis=0)

    @staticmethod
    def std_vector(vectors: list[np.ndarray]) -> np.ndarray:
        return np.std(vectors, axis=0)

    @staticmethod
    def absolute_difference(a: np.ndarray, b: np.ndarray) -> float:
        """
        Distancia L1 (Manhattan): suma de diferencias absolutas.
        Según guía: EXC = Σ|EXi - ECi|
        Retorna la suma de valores absolutos de las diferencias elemento a elemento.
        """
        return float(np.sum(np.abs(a - b)))
    
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Similitud coseno entre dos vectores.
        Retorna un valor en [-1, 1].
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    @staticmethod
    def zscore_similarity(
        observed: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> float:
        """
        Mide qué tan cerca está `observed` del perfil (mean ± std).
 
        Convierte la distancia z-score promedio en una similitud en [0, 1]:
            similitud = 1 / (1 + mean_z)
        Cuanto más parecido al perfil, más cercano a 1.
        """
        safe_std = np.where(std > 1e-9, std, 1e-9)
        z_scores = np.abs(observed - mean) / safe_std
        mean_z = float(np.mean(z_scores))
        return 1.0 / (1.0 + mean_z)
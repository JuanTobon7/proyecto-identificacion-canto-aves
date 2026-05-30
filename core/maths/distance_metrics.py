import numpy as np


class DistanceMetrics:

    @staticmethod
    def euclidean(a: np.ndarray, b: np.ndarray) -> float:
        return np.linalg.norm(a - b)

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        denominator = np.linalg.norm(a) * np.linalg.norm(b)

        if denominator == 0:
            return 1.0

        return 1 - np.dot(a, b) / denominator
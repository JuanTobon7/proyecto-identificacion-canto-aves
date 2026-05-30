import numpy as np


class Statistics:

    @staticmethod
    def mean_vector(vectors: list[np.ndarray]) -> np.ndarray:
        return np.mean(vectors, axis=0)

    @staticmethod
    def std_vector(vectors: list[np.ndarray]) -> np.ndarray:
        return np.std(vectors, axis=0)
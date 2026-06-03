import numpy as np

class EnergyVector:

    @staticmethod
    def compute(band_energies: np.ndarray) -> np.ndarray:
        """
        Retorna las energías del banco de filtros sin normalizar.
        """
        return np.asarray(band_energies, dtype=np.float32)
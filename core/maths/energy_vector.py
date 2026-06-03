import numpy as np

class EnergyVector:

    @staticmethod
    def compute(band_energies: np.ndarray) -> np.ndarray:
        """
        Normaliza energías del banco de filtros
        para obtener firma espectral.
        """
        total = np.sum(band_energies)

        if total <= 0:
            return np.zeros_like(band_energies)

        return band_energies / total
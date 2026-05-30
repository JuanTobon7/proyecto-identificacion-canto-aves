
class FrequencyBands:
    FREQUENCY_BANDS: list[tuple[float, float, str]] = [
        (0,      500,   "sub_bass"),
        (500,    1000,  "bass"),
        (1000,   2000,  "low_mid"),
        (2000,   4000,  "mid"),
        (4000,   8000,  "high_mid"),
        (8000,   16000, "presence"),
        (16000,  22050, "brilliance"),
    ]
    
    def get_bands(self) -> list[tuple[float, float, str]]:
        """Retorna la lista de bandas de frecuencia con sus nombres."""
        return self.FREQUENCY_BANDS
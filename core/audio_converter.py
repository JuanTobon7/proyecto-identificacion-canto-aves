import numpy as np

class AudioConverter:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def process(self, audio_data):
        # Placeholder for audio processing logic
        # This could include normalization, noise reduction, etc.
        return audio_data
    
    def convert_to_mono(self, audio_data):
        if audio_data.ndim > 1:
            return np.mean(audio_data, axis=1)
        return audio_data
    
    def to_mono_float32(self, audio: np.ndarray) -> np.ndarray:
        """Convierte audio a mono float32 normalizando según el tipo de dato original."""
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if np.issubdtype(audio.dtype, np.integer):
            info = np.iinfo(audio.dtype)
            scale = float(max(abs(info.min), info.max))
            audio = audio.astype(np.float32) / scale if scale > 0 else audio.astype(np.float32)
        else:
            audio = audio.astype(np.float32)
        return audio
    
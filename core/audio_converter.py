from config.routes_path import RoutesPath
import numpy as np

class AudioConverter:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate,
        self.base_to_train = RoutesPath.BANK_AUDIOS_NORMALIZED
        
    def convert_to_mono(self, audio_data):
        if audio_data.ndim > 1:
            return np.mean(audio_data, axis=1)
        return audio_data
    
    @staticmethod
    def to_mono_float32(audio: np.ndarray) -> np.ndarray:
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
    
    def convert_to_audio_from_mono_float32(self, mono_audio: np.ndarray, original_dtype) -> np.ndarray:
        """Convierte audio mono float32 de vuelta a su tipo de dato original."""
        if np.issubdtype(original_dtype, np.integer):
            info = np.iinfo(original_dtype)
            scale = float(max(abs(info.min), info.max))
            return (mono_audio * scale).clip(info.min, info.max).astype(original_dtype)
        else:
            return mono_audio.astype(original_dtype)
        
    
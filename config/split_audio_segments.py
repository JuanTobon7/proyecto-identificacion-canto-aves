"""
Script para dividir audios en segmentos de 2 segundos
Patrón de nombre: ML_ID_N (donde N es el número del segmento)
"""

import os
from pathlib import Path
import librosa
import soundfile as sf
import logging
import numpy as np

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuración
DATASET_PATH = Path(__file__).parent.parent / "dataset_aves"
SEGMENT_DURATION_MS = 2000  # 2 segundos en milisegundos

def split_audio(audio_path, output_dir, segment_duration=SEGMENT_DURATION_MS):
    """
    Divide un audio en segmentos de duración especificada.
    
    Args:
        audio_path: Ruta del archivo de audio
        output_dir: Directorio donde guardar los segmentos
        segment_duration: Duración del segmento en milisegundos
    """
    try:
        # Obtener nombre del archivo sin extensión
        file_stem = audio_path.stem
        file_ext = audio_path.suffix
        
        logger.info(f"Procesando: {audio_path.name}")
        
        # Cargar el audio
        y, sr = librosa.load(str(audio_path), sr=None)
        
        # Calcular duración total en muestras
        total_samples = len(y)
        segment_samples = int(sr * (segment_duration / 1000))  # Convertir ms a segundos
        
        # Calcular número de segmentos
        num_segments = (total_samples + segment_samples - 1) // segment_samples
        
        total_duration_s = total_samples / sr
        logger.info(f"  Duración: {total_duration_s:.2f}s -> {num_segments} segmentos")
        
        # Dividir en segmentos
        for segment_num in range(1, num_segments + 1):
            start_sample = (segment_num - 1) * segment_samples
            end_sample = min(segment_num * segment_samples, total_samples)
            
            # Extraer segmento
            segment_audio = y[start_sample:end_sample]
            segment_duration_s = len(segment_audio) / sr
            
            # Crear nombre del archivo: ML_ID_N (en WAV)
            output_filename = f"{file_stem}_{segment_num}.wav"
            output_path = output_dir / output_filename
            
            # Guardar segmento en formato WAV
            sf.write(str(output_path), segment_audio, sr)
            logger.info(f"  ✓ {output_filename} ({segment_duration_s:.2f}s)")
        
        return num_segments
        
    except Exception as e:
        logger.error(f"Error procesando {audio_path.name}: {str(e)}")
        return 0

def main():
    """Procesa todos los audios en el dataset."""
    
    if not DATASET_PATH.exists():
        logger.error(f"Dataset path no existe: {DATASET_PATH}")
        return
    
    logger.info(f"Iniciando división de audios en {DATASET_PATH}")
    logger.info(f"Duración de segmento: {SEGMENT_DURATION_MS}ms (2 segundos)\n")
    
    total_audios = 0
    total_segmentos = 0
    
    # Procesar cada carpeta de especie
    for species_dir in sorted(DATASET_PATH.iterdir()):
        if not species_dir.is_dir():
            continue
        
        species_name = species_dir.name
        logger.info(f"\n📁 Procesando especie: {species_name}")
        
        # Encontrar todos los archivos de audio
        audio_files = sorted(species_dir.glob("*.mp3"))
        
        if not audio_files:
            logger.warning(f"  No se encontraron archivos MP3 en {species_name}")
            continue
        
        logger.info(f"  Encontrados: {len(audio_files)} archivos\n")
        
        species_total = 0
        # Procesar cada audio
        for audio_file in audio_files:
            num_segments = split_audio(audio_file, species_dir)
            species_total += num_segments
            total_audios += 1
        
        total_segmentos += species_total
        logger.info(f"✓ {species_name}: {species_total} segmentos generados\n")
    
    logger.info("=" * 60)
    logger.info(f"Resumen:")
    logger.info(f"  Total de audios procesados: {total_audios}")
    logger.info(f"  Total de segmentos generados: {total_segmentos}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
